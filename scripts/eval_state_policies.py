#!/usr/bin/env python3
"""
Unified state-policy evaluator for ReYOLOv8.

Evaluates the same model checkpoint under multiple hidden-state reset policies
while keeping all other settings identical:
  - Same GT loading
  - Same NMS / confidence / IoU thresholds
  - Same AP implementation (all-points interpolation)
  - Same class mapping
  - Same scene ordering

Policies:
  1  reset_every_frame        — hidden reset at every frame
  2  reset_every_clip         — hidden reset every RESET_INTERVAL frames
  3  continuous_scene         — hidden resets only between scenes
  4  continuous_scene_excl_warmup — continuous_scene, warmup frames excluded from AP
  5  periodic_reset           — hidden reset every RESET_INTERVAL frames (same as reset_every_clip)
  6  scene_boundary_reset     — alias for continuous_scene

Usage (from /home/loki/event/ReYOLOv8):
  WANDB_MODE=disabled /home/loki/anaconda3/envs/reyolov8/bin/python \\
    scripts/eval_state_policies.py \\
    --weights runs/train/mtevent_640x480_fixed_c11/weights/best.pt \\
    --h5 preprocessed_datasets/vtei_mtevent_50ms_5bin_640x480/images/val/mtevent_val.h5 \\
    --labels preprocessed_datasets/vtei_mtevent_50ms_5bin_640x480/labels/val \\
    --nc 17 --clip_length 11 --reset_interval 11 --warmup 5 \\
    --out_dir benchmark_results/state_policy_audit/c11
"""
import sys, os, argparse, json, time
from pathlib import Path
import numpy as np
import torch
import h5py

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics.yolo.utils import ops

# All-points AP (same as eval_merged_klt.py)
def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[ii + 1] - mrec[ii]) * mpre[ii + 1]))


def compute_map(all_preds, all_gts, nc, iou_thresh=0.5):
    from collections import defaultdict
    tp_fp = defaultdict(list)
    n_gt = defaultdict(int)
    for preds, gts in zip(all_preds, all_gts):
        for g in gts:
            n_gt[int(g[4])] += 1
        if not preds:
            continue
        preds_s = sorted(preds, key=lambda x: -x[4])
        gt_used = set()
        for p in preds_s:
            px1, py1, px2, py2, pconf, pcls = p
            pcls = int(pcls)
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gts):
                gx1, gy1, gx2, gy2, gcls = g
                if int(gcls) != pcls or j in gt_used:
                    continue
                ix1 = max(px1, gx1); iy1 = max(py1, gy1)
                ix2 = min(px2, gx2); iy2 = min(py2, gy2)
                iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
                inter = iw * ih
                union = (px2-px1)*(py2-py1) + (gx2-gx1)*(gy2-gy1) - inter
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_thresh and best_j >= 0:
                tp_fp[pcls].append((preds_s.index(p), 1, pconf))  # TP
                gt_used.add(best_j)
            else:
                tp_fp[pcls].append((preds_s.index(p), 0, pconf))  # FP

    ap_per_class = {}
    for cls_id in range(nc):
        entries = tp_fp.get(cls_id, [])
        if n_gt[cls_id] == 0:
            ap_per_class[cls_id] = float('nan')
            continue
        if not entries:
            ap_per_class[cls_id] = 0.0
            continue
        entries.sort(key=lambda x: -x[2])
        tps = np.cumsum([e[1] for e in entries])
        fps = np.cumsum([1 - e[1] for e in entries])
        rec = tps / n_gt[cls_id]
        pre = tps / (tps + fps + 1e-9)
        ap_per_class[cls_id] = compute_ap(rec, pre)
    valid = [v for v in ap_per_class.values() if not np.isnan(v)]
    mAP = float(np.mean(valid)) if valid else 0.0
    return ap_per_class, mAP


def load_gts(labels_dir, H, W):
    """Return list-of-lists: gts[frame_idx] = [(x1,y1,x2,y2,cls), ...]"""
    lbl_files = sorted(Path(labels_dir).glob("*.npy"))
    all_gts = []
    scene_boundaries = []
    offset = 0
    for lf in lbl_files:
        lbls = np.load(lf, allow_pickle=True)
        scene_boundaries.append((offset, offset + len(lbls)))
        for frame_labels in lbls:
            gts_frame = []
            for ann in frame_labels:
                c = int(ann[0])
                cx, cy, w, h = ann[1]*W, ann[2]*H, ann[3]*W, ann[4]*H
                gts_frame.append((cx - w/2, cy - h/2, cx + w/2, cy + h/2, c))
            all_gts.append(gts_frame)
        offset += len(lbls)
    return all_gts, scene_boundaries


def run_policy(model, h5_path, labels_dir, device, policy, clip_length,
               reset_interval, conf_thresh, iou_thresh, ap_iou, warmup,
               nc, imgsz):
    """
    Run a single state-reset policy and return per-frame predictions + metadata.

    Returns: (all_preds, all_gts_used, frame_included_mask, reset_count, n_warmup_excl)
    """
    H, W = imgsz

    all_gts, scene_boundaries = load_gts(labels_dir, H, W)
    N = len(all_gts)

    f_h5 = h5py.File(h5_path, 'r')
    all_preds = []
    frame_included = []
    reset_count = 0
    n_warmup_excl = 0
    frame_in_scene = 0

    hidden = {"0": None, "1": None, "2": None, "3": None}
    current_scene = 0
    scene_start, scene_end = scene_boundaries[0]

    with torch.no_grad():
        for idx in range(N):
            # Detect scene boundary
            if idx >= scene_end:
                current_scene += 1
                scene_start, scene_end = scene_boundaries[current_scene]
                frame_in_scene = 0
                # Scene-boundary reset (always done for all policies)
                hidden = {"0": None, "1": None, "2": None, "3": None}
                reset_count += 1

            # Apply policy-specific reset
            include = True
            if policy == "reset_every_frame":
                hidden = {"0": None, "1": None, "2": None, "3": None}
                reset_count += 1 if frame_in_scene > 0 else 0

            elif policy in ("reset_every_clip", "periodic_reset"):
                if frame_in_scene % reset_interval == 0 and frame_in_scene > 0:
                    hidden = {"0": None, "1": None, "2": None, "3": None}
                    reset_count += 1

            elif policy in ("continuous_scene", "scene_boundary_reset"):
                pass  # reset only at scene boundaries (handled above)

            elif policy == "continuous_scene_excl_warmup":
                if frame_in_scene < warmup:
                    include = False
                    n_warmup_excl += 1

            # Forward pass
            fr = np.array(f_h5['1mp'][idx], dtype=np.float32)
            inp = torch.tensor(fr).unsqueeze(0).to(device)

            # Pad to 32-multiple
            import torch.nn.functional as F
            ph = (32 - inp.shape[2] % 32) % 32
            pw = (32 - inp.shape[3] % 32) % 32
            if ph > 0 or pw > 0:
                inp = F.pad(inp, (0, pw, 0, ph))

            out, hidden = model(inp, hidden)
            dets = ops.non_max_suppression(out, conf_thres=conf_thresh,
                                            iou_thres=iou_thresh,
                                            multi_label=False, max_det=300)[0]
            preds_frame = []
            for d in dets:
                x1, y1, x2, y2, conf, cls = d.cpu().numpy()
                # Scale back to original resolution
                scale_h = H / inp.shape[2]
                scale_w = W / inp.shape[3]
                preds_frame.append((x1*scale_w, y1*scale_h, x2*scale_w, y2*scale_h,
                                    float(conf), int(cls)))

            all_preds.append(preds_frame if include else [])
            frame_included.append(include)
            frame_in_scene += 1

    f_h5.close()

    # AP computation (only included frames)
    preds_eval = [p for p, inc in zip(all_preds, frame_included) if inc]
    gts_eval   = [g for g, inc in zip(all_gts,   frame_included) if inc]

    ap_per_cls, mAP = compute_map(preds_eval, gts_eval, nc, iou_thresh=ap_iou)
    n_frames_eval = sum(frame_included)

    return {
        "policy": policy,
        "mAP50": mAP,
        "ap_per_class": {str(k): v for k, v in ap_per_cls.items()},
        "n_frames_total": N,
        "n_frames_evaluated": n_frames_eval,
        "n_warmup_excluded": n_warmup_excl,
        "reset_count": reset_count,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--nc", type=int, default=17)
    parser.add_argument("--clip_length", type=int, default=11)
    parser.add_argument("--reset_interval", type=int, default=11,
                        help="frames between resets for reset_every_clip / periodic_reset")
    parser.add_argument("--warmup", type=int, default=5,
                        help="warmup frames to exclude per scene (continuous_scene_excl_warmup)")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--ap_iou", type=float, default=0.5)
    parser.add_argument("--out_dir", default="benchmark_results/state_policy_audit/run")
    parser.add_argument("--policies", nargs="+",
                        default=["reset_every_frame", "reset_every_clip",
                                 "continuous_scene", "continuous_scene_excl_warmup"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, nargs=2, default=[480, 640],
                        help="H W of frames in H5")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"Loading model: {args.weights}")
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    model = ckpt["model"].to(device).float()
    model.eval()

    results = {}
    for policy in args.policies:
        print(f"\n[{policy}]  clip_length={args.clip_length}  "
              f"reset_interval={args.reset_interval}  warmup={args.warmup}")
        t0 = time.time()
        r = run_policy(model, args.h5, args.labels, device, policy,
                       args.clip_length, args.reset_interval,
                       args.conf, args.iou, args.ap_iou, args.warmup,
                       args.nc, tuple(args.imgsz))
        r["runtime_s"] = time.time() - t0
        results[policy] = r
        print(f"  mAP50={r['mAP50']:.4f}  frames={r['n_frames_evaluated']}"
              f"  resets={r['reset_count']}  warmup_excl={r['n_warmup_excluded']}"
              f"  runtime={r['runtime_s']:.1f}s")

    # Save JSON
    out_json = Path(args.out_dir) / "results.json"
    with open(out_json, "w") as f:
        json.dump({
            "weights": args.weights,
            "h5": args.h5,
            "labels": args.labels,
            "clip_length": args.clip_length,
            "reset_interval": args.reset_interval,
            "warmup": args.warmup,
            "conf": args.conf,
            "iou": args.iou,
            "ap_iou": args.ap_iou,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to {out_json}")

    # Summary table
    policies_ordered = ["reset_every_frame", "reset_every_clip",
                        "continuous_scene", "continuous_scene_excl_warmup"]
    print("\n" + "=" * 72)
    print(f"{'Policy':<35s} {'mAP50':>8s} {'Frames':>8s} {'Resets':>8s}")
    print("-" * 72)
    for p in policies_ordered:
        if p in results:
            r = results[p]
            print(f"{p:<35s} {r['mAP50']:>8.4f} {r['n_frames_evaluated']:>8d} "
                  f"{r['reset_count']:>8d}")
    print("=" * 72)


if __name__ == "__main__":
    main()

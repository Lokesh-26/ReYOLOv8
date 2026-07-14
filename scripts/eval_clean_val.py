#!/usr/bin/env python3
"""
Clean-validation evaluator for ReYOLOv8.

Evaluates checkpoints on non-leaked validation scenes only.

Clean val scenes (NOT present in train):
  3, 4, 5, 6, 7, 8, 14, 26, 33, 35  (10 scenes, 3164 frames)

Leaked scenes excluded from clean eval:
  10, 21, 23  (1074 frames, 25.3% of full val)

Strictly held-out subset (4 scenes):
  14, 26, 33, 35  (originally intended validation scenes)

Evaluation settings (identical to eval_state_policies.py):
  conf=0.001, NMS IoU=0.45, AP matching IoU=0.50 (AP50)
  AP: all-points area-under-curve interpolation

Policies:
  reset_every_clip      — reset every clip_length frames (training-matched)
  continuous_scene      — reset only at scene boundaries (deployment)

Usage:
  WANDB_MODE=disabled /home/loki/anaconda3/envs/reyolov8/bin/python \\
    scripts/eval_clean_val.py \\
    --weights runs/train/mtevent_640x480_fixed_c21/weights/best.pt \\
    --h5 preprocessed_datasets/vtei_mtevent_50ms_5bin_640x480/images/val/mtevent_val.h5 \\
    --labels preprocessed_datasets/vtei_mtevent_50ms_5bin_640x480/labels/val \\
    --nc 17 --clip_length 21 --out_dir benchmark_results/clean_validation/c21
"""
import sys, os, argparse, json, time, glob
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import h5py
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from ultralytics.yolo.utils import ops

# Scenes that appear identically in both train and val splits — exclude from clean eval
LEAKED_SCENES = {10, 21, 23}
# Strictly held-out scenes (never in train under any configuration)
STRICT_HELD_OUT = {14, 26, 33, 35}


def compute_ap(recall, precision):
    """All-points area-under-curve AP (same as eval_state_policies.py)."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[ii + 1] - mrec[ii]) * mpre[ii + 1]))


def compute_map(all_preds, all_gts, nc, iou_thresh=0.5):
    """Compute per-class and overall mAP50."""
    tp_fp = defaultdict(list)
    n_gt = defaultdict(int)

    for preds_frame, gts_frame in zip(all_preds, all_gts):
        for gt in gts_frame:
            n_gt[gt[4]] += 1
        matched = set()
        preds_s = sorted(preds_frame, key=lambda x: -x[4])
        for p in preds_s:
            px1, py1, px2, py2, pconf, pcls = p
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gts_frame):
                if g[4] != pcls or j in matched:
                    continue
                gx1, gy1, gx2, gy2 = g[:4]
                ix1 = max(px1, gx1); iy1 = max(py1, gy1)
                ix2 = min(px2, gx2); iy2 = min(py2, gy2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                union = ((px2-px1)*(py2-py1) + (gx2-gx1)*(gy2-gy1) - inter)
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_thresh and best_j >= 0:
                matched.add(best_j)
                tp_fp[pcls].append((1, pconf))
            else:
                tp_fp[pcls].append((0, pconf))

    ap_per_class = {}
    aps = []
    for cls_id in range(nc):
        if cls_id not in n_gt:
            continue
        n = n_gt[cls_id]
        if cls_id not in tp_fp:
            ap_per_class[cls_id] = 0.0
            aps.append(0.0)
            continue
        entries = sorted(tp_fp[cls_id], key=lambda x: -x[1])
        tp_arr = np.array([e[0] for e in entries])
        cum_tp = np.cumsum(tp_arr)
        cum_fp = np.cumsum(1 - tp_arr)
        rec = cum_tp / max(n, 1)
        pre = cum_tp / np.maximum(cum_tp + cum_fp, 1e-10)
        ap = compute_ap(rec, pre)
        ap_per_class[cls_id] = ap
        aps.append(ap)

    mAP = float(np.mean(aps)) if aps else 0.0
    return ap_per_class, mAP


def load_scene_info(labels_dir, leaked_scenes=LEAKED_SCENES):
    """
    Parse label dir, return:
      scene_list: [(scene_id, n_frames, h5_start, h5_end, is_clean), ...]
    Ordered by label filename (= H5 frame order).
    """
    lbl_files = sorted(Path(labels_dir).glob("scene_*.npy"))
    scenes = []
    h5_offset = 0
    for lf in lbl_files:
        sid = int(lf.stem.split("_")[1])
        data = np.load(lf, allow_pickle=True)
        n = len(data)
        scenes.append({
            "scene_id": sid,
            "n_frames": n,
            "h5_start": h5_offset,
            "h5_end": h5_offset + n,
            "labels": data,
            "is_clean": sid not in leaked_scenes,
            "is_strict": sid in STRICT_HELD_OUT,
        })
        h5_offset += n
    return scenes


def run_policy_filtered(model, h5_path, scenes, device, policy, clip_length,
                        reset_interval, conf_thresh, iou_thresh, ap_iou,
                        nc, imgsz, scene_filter=None):
    """
    Run eval policy on a filtered subset of scenes.
    scene_filter: set of scene_ids to include (None = all clean scenes).
    """
    H, W = imgsz
    if scene_filter is None:
        scene_filter = {s["scene_id"] for s in scenes if s["is_clean"]}

    all_preds = []
    all_gts = []
    per_scene_preds = {}
    per_scene_gts = {}
    per_scene_meta = {}

    f_h5 = h5py.File(h5_path, 'r')
    h5_data = f_h5['1mp']

    with torch.no_grad():
        for scene in scenes:
            sid = scene["scene_id"]
            if sid not in scene_filter:
                continue

            frame_in_scene = 0
            hidden = {"0": None, "1": None, "2": None, "3": None}
            reset_count = 0

            s_preds = []
            s_gts = []

            for local_idx in range(scene["n_frames"]):
                global_idx = scene["h5_start"] + local_idx

                # Reset policy within scene
                if policy == "reset_every_frame":
                    if frame_in_scene > 0:
                        hidden = {"0": None, "1": None, "2": None, "3": None}
                        reset_count += 1
                elif policy in ("reset_every_clip", "periodic_reset"):
                    if frame_in_scene > 0 and frame_in_scene % reset_interval == 0:
                        hidden = {"0": None, "1": None, "2": None, "3": None}
                        reset_count += 1
                # continuous_scene: reset only at start of each scene (handled above)

                # Load frame
                fr = np.array(h5_data[global_idx], dtype=np.float32)
                inp = torch.tensor(fr).unsqueeze(0).to(device)
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
                    scale_h = H / inp.shape[2]
                    scale_w = W / inp.shape[3]
                    preds_frame.append((x1*scale_w, y1*scale_h,
                                        x2*scale_w, y2*scale_h,
                                        float(conf), int(cls)))

                gts_frame = []
                for ann in scene["labels"][local_idx]:
                    c = int(ann[0])
                    cx, cy, bw, bh = ann[1]*W, ann[2]*H, ann[3]*W, ann[4]*H
                    gts_frame.append((cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2, c))

                s_preds.append(preds_frame)
                s_gts.append(gts_frame)
                frame_in_scene += 1

            # Per-scene AP
            sc_ap_per_cls, sc_mAP = compute_map(s_preds, s_gts, nc, iou_thresh=ap_iou)
            per_scene_preds[sid] = s_preds
            per_scene_gts[sid] = s_gts
            per_scene_meta[sid] = {
                "n_frames": scene["n_frames"],
                "mAP50": sc_mAP,
                "ap_per_class": {str(k): v for k, v in sc_ap_per_cls.items()},
                "reset_count": reset_count,
            }

            all_preds.extend(s_preds)
            all_gts.extend(s_gts)

    f_h5.close()

    ap_per_cls, mAP = compute_map(all_preds, all_gts, nc, iou_thresh=ap_iou)
    n_frames = sum(len(s) for s in per_scene_preds.values())

    return {
        "mAP50": mAP,
        "ap_per_class": {str(k): v for k, v in ap_per_cls.items()},
        "n_frames": n_frames,
        "n_scenes": len(per_scene_preds),
        "per_scene": per_scene_meta,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--nc", type=int, default=17)
    parser.add_argument("--clip_length", type=int, default=11)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--ap_iou", type=float, default=0.5)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, nargs=2, default=[480, 640])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"Loading model: {args.weights}")
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    model = ckpt["model"].to(device).float()
    model.eval()

    scenes = load_scene_info(args.labels)
    clean_scenes = {s["scene_id"] for s in scenes if s["is_clean"]}
    strict_scenes = {s["scene_id"] for s in scenes if s["is_strict"]}

    print(f"All val scenes: {sorted(s['scene_id'] for s in scenes)}")
    print(f"Clean val scenes ({len(clean_scenes)}): {sorted(clean_scenes)}")
    print(f"Strict held-out ({len(strict_scenes)}): {sorted(strict_scenes)}")
    print(f"Leaked (excluded): {sorted(LEAKED_SCENES)}")

    # Count frames
    clean_frames = sum(s["n_frames"] for s in scenes if s["is_clean"])
    strict_frames = sum(s["n_frames"] for s in scenes if s["is_strict"])
    print(f"Clean frames: {clean_frames}, Strict frames: {strict_frames}")

    results = {"weights": args.weights, "clip_length": args.clip_length,
               "conf": args.conf, "iou": args.iou, "ap_iou": args.ap_iou,
               "clean_scenes": sorted(clean_scenes),
               "leaked_scenes": sorted(LEAKED_SCENES),
               "strict_scenes": sorted(strict_scenes),
               "results": {}}

    for policy in ["reset_every_clip", "continuous_scene"]:
        reset_interval = args.clip_length
        print(f"\n[{policy}]  clip_length={args.clip_length}  reset_interval={reset_interval}")
        t0 = time.time()

        # Clean val (10 scenes)
        r_clean = run_policy_filtered(
            model, args.h5, scenes, device, policy,
            args.clip_length, reset_interval,
            args.conf, args.iou, args.ap_iou, args.nc, tuple(args.imgsz),
            scene_filter=clean_scenes
        )
        r_clean["runtime_s"] = time.time() - t0

        # Strict held-out (4 scenes)
        t1 = time.time()
        r_strict = run_policy_filtered(
            model, args.h5, scenes, device, policy,
            args.clip_length, reset_interval,
            args.conf, args.iou, args.ap_iou, args.nc, tuple(args.imgsz),
            scene_filter=strict_scenes
        )
        r_strict["runtime_s"] = time.time() - t1

        results["results"][policy] = {
            "clean_val": r_clean,
            "strict_val": r_strict,
        }

        print(f"  Clean val  ({r_clean['n_scenes']} scenes, {r_clean['n_frames']} frames): "
              f"mAP50={r_clean['mAP50']:.4f}")
        print(f"  Strict val ({r_strict['n_scenes']} scenes, {r_strict['n_frames']} frames): "
              f"mAP50={r_strict['mAP50']:.4f}")

        # Per-scene breakdown
        print(f"  Per-scene (clean val):")
        for sid in sorted(r_clean["per_scene"].keys()):
            meta = r_clean["per_scene"][sid]
            print(f"    scene {sid:3d}: {meta['n_frames']:4d} frames, "
                  f"mAP50={meta['mAP50']:.4f}")

    out_file = Path(args.out_dir) / "results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")

    print("\n" + "="*70)
    print(f"{'Policy':<35} {'Clean mAP50':>12} {'Strict mAP50':>13}")
    print("-"*70)
    for policy in ["reset_every_clip", "continuous_scene"]:
        rc = results["results"][policy]["clean_val"]["mAP50"]
        rs = results["results"][policy]["strict_val"]["mAP50"]
        print(f"{policy:<35} {rc:>12.4f} {rs:>13.4f}")
    print("="*70)


if __name__ == "__main__":
    main()

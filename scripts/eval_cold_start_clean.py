#!/usr/bin/env python3
"""
Cold-start analysis on the clean 10-scene validation set.

Evaluates how well recurrent models perform at scene boundaries (cold hidden state)
vs. mid-scene (warmed hidden state).

For each model, runs continuous_scene policy on the clean val, collects
per-frame predictions and GT, then computes:
  - AP50 on all frames
  - AP50 excluding the first 1, 5, 10 frames of each scene
  - Recall during only the first 1, 5, 10 frames of each scene
  - Time to first correct detection (per scene and overall)
  - False-positive count during the initial N-frame windows
  - Per-scene early-frame recall

Clean val scenes (10 scenes, 3164 frames):
  3, 4, 5, 6, 7, 8, 14, 26, 33, 35

Usage:
  /home/loki/anaconda3/envs/reyolov8/bin/python scripts/eval_cold_start_clean.py \
    --configs configs.json

where configs.json maps model names to checkpoint paths and clip lengths.

Or call with individual flags (single model):
  /home/loki/anaconda3/envs/reyolov8/bin/python scripts/eval_cold_start_clean.py \
    --weights runs/train/mtevent_640x480_fixed_c11/weights/best.pt \
    --clip_length 11 --model_name C11 \
    --h5 preprocessed_datasets/vtei_mtevent_50ms_5bin_640x480/images/val/mtevent_val.h5 \
    --labels preprocessed_datasets/vtei_mtevent_50ms_5bin_640x480/labels/val \
    --out_dir benchmark_results/clean_validation/cold_start
"""

import sys, os, argparse, json, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import h5py
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from ultralytics.yolo.utils import ops

LEAKED_SCENES = {10, 21, 23}
CLEAN_SCENES  = {3, 4, 5, 6, 7, 8, 14, 26, 33, 35}
EARLY_WINDOWS = [1, 5, 10]

IOU_THRES   = 0.45   # NMS suppression
AP_IOU      = 0.50   # AP matching threshold
CONF_THRES  = 0.001

CLASS_NAMES = [
    "wooden_pallet","small_klt","big_klt","blue_klt","amazon_luggage",
    "ikea_dammang_bin","ikea_vesken_trolley","ikea_sortera_bin",
    "ikea_drona_grey","ikea_drona_blue","ikea_knallig_box","ikea_moppe_drawer",
    "ikea_labbsal_basket","ikea_ivar_box","ikea_skubb_case","ikea_samla_box","human",
]


def compute_ap(recall, precision):
    """All-points area-under-curve AP."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[ii + 1] - mrec[ii]) * mpre[ii + 1]))


def compute_map_from_frames(preds_list, gts_list, nc, iou_thresh=AP_IOU):
    """Compute mAP50 and per-class AP from lists of per-frame predictions and GT."""
    tp_fp = defaultdict(list)
    n_gt  = defaultdict(int)

    for preds_frame, gts_frame in zip(preds_list, gts_list):
        for g in gts_frame:
            n_gt[int(g[4])] += 1
        matched = set()
        for p in sorted(preds_frame, key=lambda x: -x[4]):
            px1, py1, px2, py2, pconf, pcls = p
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gts_frame):
                if int(g[4]) != int(pcls) or j in matched:
                    continue
                gx1, gy1, gx2, gy2 = g[:4]
                inter = max(0, min(px2,gx2)-max(px1,gx1)) * max(0, min(py2,gy2)-max(py1,gy1))
                union = (px2-px1)*(py2-py1) + (gx2-gx1)*(gy2-gy1) - inter
                iou = inter / union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_thresh and best_j >= 0:
                matched.add(best_j)
                tp_fp[int(pcls)].append((1, pconf))
            else:
                tp_fp[int(pcls)].append((0, pconf))

    aps = []
    ap_per_class = {}
    for c in range(nc):
        if c not in n_gt:
            continue
        if c not in tp_fp:
            ap_per_class[c] = 0.0
            aps.append(0.0)
            continue
        entries = sorted(tp_fp[c], key=lambda x: -x[1])
        tp_arr = np.array([e[0] for e in entries])
        cum_tp = np.cumsum(tp_arr)
        cum_fp = np.cumsum(1 - tp_arr)
        rec = cum_tp / max(n_gt[c], 1)
        pre = cum_tp / np.maximum(cum_tp + cum_fp, 1e-10)
        ap = compute_ap(rec, pre)
        ap_per_class[c] = ap
        aps.append(ap)

    return ap_per_class, float(np.mean(aps)) if aps else 0.0


def is_correct_detection(pred, gts_frame, iou_thresh=AP_IOU):
    """Return True if this prediction is a TP (any GT class match with IoU >= thresh)."""
    px1, py1, px2, py2, pconf, pcls = pred
    for g in gts_frame:
        if int(g[4]) != int(pcls):
            continue
        gx1, gy1, gx2, gy2 = g[:4]
        inter = max(0, min(px2,gx2)-max(px1,gx1)) * max(0, min(py2,gy2)-max(py1,gy1))
        union = (px2-px1)*(py2-py1) + (gx2-gx1)*(gy2-gy1) - inter
        iou = inter / union if union > 0 else 0.0
        if iou >= iou_thresh:
            return True
    return False


def load_scene_info(labels_dir):
    lbl_files = sorted(Path(labels_dir).glob("scene_*.npy"),
                       key=lambda p: int(p.stem.split("_")[1]))
    scenes = []
    offset = 0
    for lf in lbl_files:
        sid = int(lf.stem.split("_")[1])
        data = np.load(lf, allow_pickle=True)
        n = len(data)
        scenes.append({
            "scene_id": sid,
            "n_frames": n,
            "h5_start": offset,
            "h5_end": offset + n,
            "labels": data,
            "is_clean": sid not in LEAKED_SCENES and sid in CLEAN_SCENES,
        })
        offset += n
    return scenes


def run_cold_start(model, h5_path, scenes, device, clip_length,
                   nc, imgsz, conf_thresh=CONF_THRES, iou_thresh=IOU_THRES):
    """
    Run continuous_scene policy.
    Returns per-scene list of (preds_frame, gts_frame) tuples with frame index within scene.
    """
    H, W = imgsz
    per_scene_data = {}  # sid -> list of (preds, gts)

    f_h5 = h5py.File(h5_path, 'r')
    h5_data = f_h5['1mp']

    with torch.no_grad():
        for scene in scenes:
            sid = scene["scene_id"]
            if not scene["is_clean"]:
                continue

            hidden = {"0": None, "1": None, "2": None, "3": None}
            scene_frames = []  # list of (preds_frame, gts_frame)

            for local_idx in range(scene["n_frames"]):
                global_idx = scene["h5_start"] + local_idx

                fr = np.array(h5_data[global_idx], dtype=np.float32)
                inp = torch.tensor(fr).unsqueeze(0).to(device)
                ph = (32 - inp.shape[2] % 32) % 32
                pw = (32 - inp.shape[3] % 32) % 32
                if ph > 0 or pw > 0:
                    inp = F.pad(inp, (0, pw, 0, ph))

                out, hidden = model(inp, hidden)
                dets = ops.non_max_suppression(
                    out, conf_thres=conf_thresh, iou_thres=iou_thresh,
                    multi_label=False, max_det=300)[0]

                preds = []
                for d in dets:
                    x1, y1, x2, y2, conf, cls = d.cpu().numpy()
                    sw = W / inp.shape[3]
                    sh = H / inp.shape[2]
                    preds.append((x1*sw, y1*sh, x2*sw, y2*sh, float(conf), int(cls)))

                gts = []
                for ann in scene["labels"][local_idx]:
                    c = int(ann[0])
                    cx, cy, bw, bh = ann[1]*W, ann[2]*H, ann[3]*W, ann[4]*H
                    gts.append((cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2, c))

                scene_frames.append((preds, gts))

            per_scene_data[sid] = scene_frames

    f_h5.close()
    return per_scene_data


def analyze_cold_start(per_scene_data, nc, model_name):
    """
    Compute all cold-start metrics from per-scene frame data.
    Returns a results dict.
    """
    result = {"model": model_name}

    # Flatten for global AP computation
    all_preds, all_gts = [], []
    for scene_frames in per_scene_data.values():
        for preds, gts in scene_frames:
            all_preds.append(preds)
            all_gts.append(gts)

    _, global_map = compute_map_from_frames(all_preds, all_gts, nc)
    result["ap50_all_frames"] = global_map

    # AP excluding first N frames
    for N in EARLY_WINDOWS:
        ex_preds, ex_gts = [], []
        for scene_frames in per_scene_data.values():
            for i, (preds, gts) in enumerate(scene_frames):
                if i >= N:
                    ex_preds.append(preds)
                    ex_gts.append(gts)
        _, ap_ex = compute_map_from_frames(ex_preds, ex_gts, nc)
        result[f"ap50_excl_first_{N}"] = ap_ex

    # Recall in first N frames
    for N in EARLY_WINDOWS:
        tp_total, gt_total, fp_total = 0, 0, 0
        for scene_frames in per_scene_data.values():
            for i, (preds, gts) in enumerate(scene_frames):
                if i >= N:
                    break
                gt_total += len(gts)
                matched_gt = set()
                for p in preds:
                    hit = False
                    px1, py1, px2, py2, pconf, pcls = p
                    for j, g in enumerate(gts):
                        if int(g[4]) != int(pcls) or j in matched_gt:
                            continue
                        gx1, gy1, gx2, gy2 = g[:4]
                        inter = max(0, min(px2,gx2)-max(px1,gx1)) * max(0, min(py2,gy2)-max(py1,gy1))
                        union = (px2-px1)*(py2-py1) + (gx2-gx1)*(gy2-gy1) - inter
                        iou = inter / union if union > 0 else 0.0
                        if iou >= AP_IOU:
                            hit = True
                            matched_gt.add(j)
                            break
                    if hit:
                        tp_total += 1
                    else:
                        fp_total += 1
        result[f"recall_first_{N}"] = tp_total / max(gt_total, 1)
        result[f"fp_count_first_{N}"] = fp_total
        result[f"gt_count_first_{N}"] = gt_total
        result[f"tp_count_first_{N}"] = tp_total

    # Time to first correct detection (in frames, 0-indexed)
    ttfd_scenes = {}  # sid -> frame_idx or None
    n_scenes_detected = {N: 0 for N in EARLY_WINDOWS}
    for sid, scene_frames in per_scene_data.items():
        ttfd = None
        for i, (preds, gts) in enumerate(scene_frames):
            if any(is_correct_detection(p, gts) for p in preds):
                ttfd = i
                break
        ttfd_scenes[sid] = ttfd

        for N in EARLY_WINDOWS:
            if ttfd is not None and ttfd < N:
                n_scenes_detected[N] += 1

    ttfd_vals = [v for v in ttfd_scenes.values() if v is not None]
    result["ttfd_per_scene"] = {str(k): v for k, v in ttfd_scenes.items()}
    result["ttfd_mean_frames"] = float(np.mean(ttfd_vals)) if ttfd_vals else None
    result["ttfd_median_frames"] = float(np.median(ttfd_vals)) if ttfd_vals else None
    result["n_scenes_no_detection"] = sum(1 for v in ttfd_scenes.values() if v is None)
    for N in EARLY_WINDOWS:
        result[f"scenes_detected_in_first_{N}"] = n_scenes_detected[N]

    # Per-scene early recall (first 5 frames)
    per_scene_recall = {}
    for sid, scene_frames in per_scene_data.items():
        tp_s, gt_s = 0, 0
        for i, (preds, gts) in enumerate(scene_frames):
            if i >= 5:
                break
            gt_s += len(gts)
            matched = set()
            for p in preds:
                px1, py1, px2, py2, pconf, pcls = p
                for j, g in enumerate(gts):
                    if int(g[4]) != int(pcls) or j in matched:
                        continue
                    gx1, gy1, gx2, gy2 = g[:4]
                    inter = max(0, min(px2,gx2)-max(px1,gx1)) * max(0, min(py2,gy2)-max(py1,gy1))
                    union = (px2-px1)*(py2-py1) + (gx2-gx1)*(gy2-gy1) - inter
                    iou = inter / union if union > 0 else 0.0
                    if iou >= AP_IOU:
                        tp_s += 1
                        matched.add(j)
                        break
        per_scene_recall[str(sid)] = tp_s / max(gt_s, 1)
    result["per_scene_recall_first_5"] = per_scene_recall

    # AP delta: (excl_first_5) - (all_frames) — positive means model is worse at start
    result["delta_ap_excl5_vs_all"] = result["ap50_excl_first_5"] - result["ap50_all_frames"]
    result["delta_ap_excl10_vs_all"] = result["ap50_excl_first_10"] - result["ap50_all_frames"]

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--h5", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--nc", type=int, default=17)
    parser.add_argument("--clip_length", type=int, default=11)
    parser.add_argument("--model_name", default="")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, nargs=2, default=[480, 640])
    args = parser.parse_args()

    if not args.model_name:
        args.model_name = Path(args.weights).parents[1].name

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"Loading: {args.weights}")
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    model = ckpt["model"].to(device).float()
    model.eval()

    scenes = load_scene_info(args.labels)
    clean = [s for s in scenes if s["is_clean"]]
    print(f"Clean scenes: {sorted(s['scene_id'] for s in clean)} ({sum(s['n_frames'] for s in clean)} frames)")

    t0 = time.time()
    per_scene_data = run_cold_start(
        model, args.h5, scenes, device,
        args.clip_length, args.nc, tuple(args.imgsz)
    )
    elapsed = time.time() - t0
    print(f"Inference done in {elapsed:.1f}s")

    result = analyze_cold_start(per_scene_data, args.nc, args.model_name)

    print(f"\n{'='*60}")
    print(f"Model: {args.model_name}")
    print(f"  AP50 all frames:      {result['ap50_all_frames']:.4f}")
    for N in EARLY_WINDOWS:
        ap_ex = result[f"ap50_excl_first_{N}"]
        delta  = result.get(f"delta_ap_excl{N}_vs_all", ap_ex - result['ap50_all_frames'])
        print(f"  AP50 excl first {N:2d}:   {ap_ex:.4f}  (Δ={delta:+.4f})")
    print()
    for N in EARLY_WINDOWS:
        r = result[f"recall_first_{N}"]
        fp = result[f"fp_count_first_{N}"]
        print(f"  Recall first {N:2d} frames: {r:.4f}  (FP={fp})")
    print()
    print(f"  TTFD mean: {result['ttfd_mean_frames']:.1f} frames" if result['ttfd_mean_frames'] else "  TTFD mean: N/A")
    print(f"  TTFD median: {result['ttfd_median_frames']:.1f} frames" if result['ttfd_median_frames'] else "  TTFD median: N/A")
    print(f"  Scenes with detection in first 5 frames: {result['scenes_detected_in_first_5']} / {len(per_scene_data)}")
    print(f"  Scenes with NO detection: {result['n_scenes_no_detection']}")
    print(f"  Δ AP (excl5 - all):  {result['delta_ap_excl5_vs_all']:+.4f}")
    print(f"  Δ AP (excl10 - all): {result['delta_ap_excl10_vs_all']:+.4f}")
    print(f"{'='*60}")

    out_path = Path(args.out_dir) / f"{args.model_name}_cold_start.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

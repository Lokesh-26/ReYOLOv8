#!/usr/bin/env python3
"""
Clean-validation evaluation of the native 15-class C21 model.

Evaluates the mtevent_640x480_15cls_c21_v2 checkpoint on the clean
10-scene validation set using the 15-class label space (KLT merged
during training, not just at eval time).

Also evaluates the 17-class C21 model with KLT eval-time remapping
for direct comparison, using the same clean scenes and evaluator.

Clean val scenes: 3, 4, 5, 6, 7, 8, 14, 26, 33, 35 (3164 frames)

Usage:
  /home/loki/anaconda3/envs/reyolov8/bin/python scripts/eval_15cls_native_clean.py \
    --out_dir benchmark_results/clean_validation/15cls_native
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
IOU_THRES     = 0.45
AP_IOU        = 0.50
CONF_THRES    = 0.001

CLASS_NAMES_15 = [
    "wooden_pallet","klt","amazon_luggage","ikea_dammang_bin",
    "ikea_vesken_trolley","ikea_sortera_bin","ikea_drona_grey","ikea_drona_blue",
    "ikea_knallig_box","ikea_moppe_drawer","ikea_labbsal_basket","ikea_ivar_box",
    "ikea_skubb_case","ikea_samla_box","human",
]


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[ii + 1] - mrec[ii]) * mpre[ii + 1]))


def compute_map(all_preds, all_gts, nc, iou_thresh=AP_IOU):
    tp_fp = defaultdict(list)
    n_gt  = defaultdict(int)
    for preds_frame, gts_frame in zip(all_preds, all_gts):
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

    ap_per_class = {}
    aps = []
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
        ap_per_class[c] = compute_ap(rec, pre)
        aps.append(ap_per_class[c])

    return ap_per_class, float(np.mean(aps)) if aps else 0.0, n_gt


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


def run_eval(model, h5_path, scenes, device, nc, clip_length,
             imgsz, conf_thresh=CONF_THRES, iou_thresh=IOU_THRES):
    """Run continuous_scene policy on clean scenes."""
    H, W = imgsz
    all_preds, all_gts = [], []
    per_scene = {}

    f = h5py.File(h5_path, 'r')
    h5_data = f['1mp']

    with torch.no_grad():
        for scene in scenes:
            sid = scene["scene_id"]
            if not scene["is_clean"]:
                continue

            hidden = {"0": None, "1": None, "2": None, "3": None}
            s_preds, s_gts = [], []

            for local_idx in range(scene["n_frames"]):
                global_idx = scene["h5_start"] + local_idx
                fr = np.array(h5_data[global_idx], dtype=np.float32)
                inp = torch.tensor(fr).unsqueeze(0).to(device)
                ph = (32 - inp.shape[2] % 32) % 32
                pw = (32 - inp.shape[3] % 32) % 32
                if ph or pw:
                    inp = F.pad(inp, (0, pw, 0, ph))

                out, hidden = model(inp, hidden)
                dets = ops.non_max_suppression(
                    out, conf_thres=conf_thresh, iou_thres=iou_thresh,
                    multi_label=False, max_det=300)[0]

                preds = []
                for d in dets:
                    x1, y1, x2, y2, conf, cls = d.cpu().numpy()
                    sw, sh = W / inp.shape[3], H / inp.shape[2]
                    preds.append((x1*sw, y1*sh, x2*sw, y2*sh, float(conf), int(cls)))

                gts = []
                for ann in scene["labels"][local_idx]:
                    c = int(ann[0])
                    cx, cy, bw, bh = ann[1]*W, ann[2]*H, ann[3]*W, ann[4]*H
                    gts.append((cx-bw/2, cy-bh/2, cx+bw/2, cy+bh/2, c))

                s_preds.append(preds)
                s_gts.append(gts)

            ap_cls, sc_map, sc_ngt = compute_map(s_preds, s_gts, nc)
            per_scene[sid] = {
                "n_frames": scene["n_frames"],
                "mAP50": sc_map,
                "ap_per_class": {str(k): v for k, v in ap_cls.items()},
                "n_gt": {str(k): int(v) for k, v in sc_ngt.items()},
            }
            all_preds.extend(s_preds)
            all_gts.extend(s_gts)

    f.close()

    ap_cls, mAP, n_gt = compute_map(all_preds, all_gts, nc)
    n_frames = sum(s["n_frames"] for s in scenes if s["is_clean"])
    n_scenes = len(per_scene)

    return {
        "mAP50": mAP,
        "ap_per_class": {str(k): v for k, v in ap_cls.items()},
        "n_gt_total": {str(k): int(v) for k, v in n_gt.items()},
        "n_frames": n_frames,
        "n_scenes": n_scenes,
        "per_scene": per_scene,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_15cls",
                        default="runs/train/mtevent_640x480_15cls_c21_v2/weights/best.pt")
    parser.add_argument("--h5_15cls",
                        default="preprocessed_datasets/vtei_mtevent_50ms_5bin_640x480_15cls/"
                                "images/val/mtevent_val.h5")
    parser.add_argument("--labels_15cls",
                        default="preprocessed_datasets/vtei_mtevent_50ms_5bin_640x480_15cls/"
                                "labels/val")
    parser.add_argument("--out_dir",
                        default="benchmark_results/clean_validation/15cls_native")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, nargs=2, default=[480, 640])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    results = {}

    # --- Native 15-class model ---
    print(f"\nLoading native 15cls model: {args.weights_15cls}")
    ckpt = torch.load(args.weights_15cls, map_location=device, weights_only=False)
    model15 = ckpt["model"].to(device).float()
    model15.eval()
    best_epoch = ckpt.get("epoch", "?")
    print(f"  Best epoch in checkpoint: {best_epoch}")

    # Check nc
    nc15 = getattr(model15, "nc", None) or 15
    print(f"  Model nc: {nc15}")

    scenes15 = load_scene_info(args.labels_15cls)
    clean15 = [s for s in scenes15 if s["is_clean"]]
    print(f"  Clean scenes: {sorted(s['scene_id'] for s in clean15)} ({sum(s['n_frames'] for s in clean15)} frames)")

    if not Path(args.h5_15cls).exists():
        print(f"ERROR: 15cls val H5 not found at {args.h5_15cls}")
        return

    t0 = time.time()
    r15 = run_eval(model15, args.h5_15cls, scenes15, device,
                   nc15, clip_length=21, imgsz=tuple(args.imgsz))
    r15["elapsed_s"] = time.time() - t0
    r15["weights"] = args.weights_15cls
    r15["best_epoch"] = best_epoch
    results["native_15cls_c21"] = r15

    print(f"\n{'='*60}")
    print(f"Native 15-class C21 (clean val):")
    print(f"  mAP50: {r15['mAP50']:.4f}")
    print(f"  Scenes: {r15['n_scenes']}, Frames: {r15['n_frames']}")
    print(f"  Per-class AP50:")
    for c in range(nc15):
        ap_c = r15["ap_per_class"].get(str(c), 0.0)
        ng   = r15["n_gt_total"].get(str(c), 0)
        print(f"    [{c:2d}] {CLASS_NAMES_15[c]:<25} AP50={ap_c:.4f}  (GT={ng})")
    print(f"  Per-scene mAP50:")
    for sid in sorted(r15["per_scene"].keys()):
        print(f"    scene {sid}: {r15['per_scene'][sid]['mAP50']:.4f}")
    print(f"{'='*60}")

    out_path = Path(args.out_dir) / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

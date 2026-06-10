#!/usr/bin/env python3
"""
Inference + visualization for event streams preprocessed from WP1 bags.

Loads an H5 event voxel file, runs ReYOLOv8 inference frame-by-frame with
recurrent hidden state propagation, renders detections on event frames, and
saves a video + per-frame JSON detections.

Usage (run from /home/loki/event/ReYOLOv8/):
    python scripts/infer_wp1_bags.py \
        --h5      benchmark_results/<name>/images/test/mtevent_test.h5 \
        --weights runs/train/mtevent_17cls_combined_c212/weights/best.pt \
        --out_dir benchmark_results/<name>/detections \
        --device  cuda:0 --conf 0.25
"""
import os
import sys
import json
import math
import argparse

import numpy as np
import h5py
import cv2
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics.nn.autobackend import AutoBackendMemory
from ultralytics.yolo.utils import ops
from ultralytics.yolo.utils.torch_utils import select_device

CLASS_NAMES = [
    "wooden_pallet", "small_klt", "big_klt", "blue_klt",
    "amazon_luggage", "ikea_dammang_bin", "ikea_vesken_trolley",
    "ikea_sortera_bin", "ikea_drona_grey", "ikea_drona_blue",
    "ikea_knallig_box", "ikea_moppe_drawer", "ikea_labbsal_basket",
    "ikea_ivar_box", "ikea_skubb_case", "ikea_samla_box", "human",
]

# Distinct BGR colours per class
PALETTE = [
    (0, 255, 255), (0, 128, 255), (0, 255, 0), (255, 0, 255),
    (255, 128, 0), (0, 0, 255), (255, 255, 0), (128, 0, 255),
    (0, 255, 128), (255, 0, 128), (128, 255, 0), (0, 128, 128),
    (128, 0, 128), (255, 128, 128), (128, 128, 255), (128, 255, 128),
    (0, 200, 255),
]


def pad_to_multiple(tensor, multiple=32):
    """Pad H and W to the next multiple of `multiple` (matches val.py preprocess)."""
    _, _, h, w = tensor.shape
    new_h = math.ceil(h / multiple) * multiple
    new_w = math.ceil(w / multiple) * multiple
    if new_h == h and new_w == w:
        return tensor
    return F.interpolate(tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)


def compute_global_scale(frames, clip_pct=95):
    """Compute a fixed contrast scale from the 95th-percentile of nonzero magnitudes.
    Sampled every 10 frames for speed. Returns scalar float."""
    sample = np.abs(frames[::10].astype(np.float32))
    nonzero = sample[sample > 0]
    return float(np.percentile(nonzero, clip_pct)) if len(nonzero) else 1.0


def event_frame_to_bgr(frame_chw, scale=1.0):
    """Convert (C, H, W) int8 voxel to red/blue polarity BGR image.
    Uses a fixed global scale so contrast is consistent across frames.
    Positive net events → red, negative net events → blue, background black.
    """
    acc = frame_chw.astype(np.float32).sum(axis=0)  # (H, W)
    red  = np.clip( acc / scale * 255, 0, 255).astype(np.uint8)
    blue = np.clip(-acc / scale * 255, 0, 255).astype(np.uint8)
    bgr = np.zeros((*acc.shape, 3), dtype=np.uint8)
    bgr[:, :, 2] = red
    bgr[:, :, 0] = blue
    return bgr


def draw_detections(bgr, dets, orig_h, orig_w, pad_h, pad_w):
    """
    Draw detections on `bgr` (H×W).
    dets: (N, 6) tensor [x1, y1, x2, y2, conf, cls] in padded-image coords.
    Scales back from (pad_h, pad_w) → (orig_h, orig_w).
    """
    if dets is None or len(dets) == 0:
        return bgr
    sx = orig_w / pad_w
    sy = orig_h / pad_h
    for det in dets.cpu().numpy():
        x1, y1, x2, y2, conf, cls = det
        x1 = int(x1 * sx); y1 = int(y1 * sy)
        x2 = int(x2 * sx); y2 = int(y2 * sy)
        x1 = max(0, min(orig_w - 1, x1)); x2 = max(0, min(orig_w - 1, x2))
        y1 = max(0, min(orig_h - 1, y1)); y2 = max(0, min(orig_h - 1, y2))
        cls_id = int(cls)
        color = PALETTE[cls_id % len(PALETTE)]
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else cls_id} {conf:.2f}"
        cv2.putText(bgr, label, (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return bgr


def run(args):
    device = select_device(args.device)

    print(f"[INFO] loading H5: {args.h5}")
    with h5py.File(args.h5, 'r') as f:
        frames = f['1mp'][:]  # (N, C, H, W) int8

    N, C, orig_h, orig_w = frames.shape
    print(f"[INFO] frames shape: {frames.shape}  ({N} frames, {C} channels, {orig_h}×{orig_w})")

    print(f"[INFO] loading weights: {args.weights}")
    model = AutoBackendMemory(args.weights, device=device, fp16=False)
    model.eval()

    # Warmup
    dummy = torch.zeros(1, C, orig_h, orig_w, dtype=torch.float32, device=device)
    dummy_pad = pad_to_multiple(dummy)
    pad_h, pad_w = dummy_pad.shape[2], dummy_pad.shape[3]
    print(f"[INFO] padded inference resolution: {pad_h}×{pad_w}")
    hidden = {"0": None, "1": None, "2": None, "3": None}
    with torch.no_grad():
        model(dummy_pad, hidden)

    os.makedirs(args.out_dir, exist_ok=True)
    frames_dir = os.path.join(args.out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    fps = 1000.0 / 50.0  # 50 ms per voxel window → 20 fps
    video_path = os.path.join(args.out_dir, "detection.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (orig_w, orig_h))

    all_detections = []
    hidden = {"0": None, "1": None, "2": None, "3": None}

    global_scale = compute_global_scale(frames)
    print(f"[INFO] contrast scale (p95 of |nonzero|): {global_scale:.2f}")
    print(f"[INFO] running inference on {N} frames ...")
    with torch.no_grad():
        for t in range(N):
            frame = frames[t]  # (C, H, W) int8
            x = torch.from_numpy(frame.copy()).float().unsqueeze(0).to(device)  # (1, C, H, W)
            x_pad = pad_to_multiple(x)

            raw, hidden = model(x_pad, hidden)

            preds = ops.non_max_suppression(
                raw,
                conf_thres=args.conf,
                iou_thres=args.iou,
                multi_label=True,
                agnostic=False,
                max_det=args.max_det,
            )
            dets = preds[0]  # (N_det, 6): [x1, y1, x2, y2, conf, cls]

            # Build JSON record
            frame_record = {"frame": t, "boxes": []}
            if dets is not None and len(dets) > 0:
                sx = orig_w / pad_w
                sy = orig_h / pad_h
                for det in dets.cpu().numpy():
                    x1, y1, x2, y2, conf, cls = det.tolist()
                    frame_record["boxes"].append([
                        int(cls), round(conf, 4),
                        round(x1 * sx, 1), round(y1 * sy, 1),
                        round(x2 * sx, 1), round(y2 * sy, 1),
                    ])
            all_detections.append(frame_record)

            # Render
            bgr = event_frame_to_bgr(frame, scale=global_scale)
            bgr = draw_detections(bgr, dets, orig_h, orig_w, pad_h, pad_w)

            cv2.imwrite(os.path.join(frames_dir, f"frame_{t:06d}.png"), bgr)
            writer.write(bgr)

            if (t + 1) % 100 == 0:
                n_det = len(dets) if dets is not None else 0
                print(f"  frame {t+1}/{N}  detections={n_det}")

    writer.release()

    json_path = os.path.join(args.out_dir, "detections.json")
    with open(json_path, "w") as f:
        json.dump(all_detections, f)

    total_dets = sum(len(r["boxes"]) for r in all_detections)
    frames_with_dets = sum(1 for r in all_detections if r["boxes"])
    print(f"\n[DONE] {N} frames processed")
    print(f"       {frames_with_dets}/{N} frames had detections  ({total_dets} total boxes)")
    print(f"       video  → {video_path}")
    print(f"       JSON   → {json_path}")
    print(f"       frames → {frames_dir}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5",      required=True, help="path to mtevent_<split>.h5")
    ap.add_argument("--weights", required=True, help="path to best.pt")
    ap.add_argument("--out_dir", required=True, help="output directory")
    ap.add_argument("--device",  default="cuda:0")
    ap.add_argument("--conf",    type=float, default=0.25)
    ap.add_argument("--iou",     type=float, default=0.7)
    ap.add_argument("--max_det", type=int,   default=300)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()

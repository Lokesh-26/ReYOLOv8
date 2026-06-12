#!/usr/bin/env python3
"""
RGB inference on WP1 bags using trained YOLOv8 baseline.

Run with conda python (has ultralytics + torch):
    python scripts/infer_wp1_rgb.py \
        --frames_dir  benchmark_results/<name>/rgb_frames \
        --weights     runs/detect/runs/train/rgb_baseline_proper/weights/best.pt \
        --out_dir     benchmark_results/<name>/rgb_detections \
        --fps         20
"""
import os, json, argparse, glob
import cv2
import numpy as np
from ultralytics import YOLO

CLASS_NAMES = [
    "wooden_pallet","small_klt","big_klt","blue_klt","amazon_luggage",
    "ikea_dammang_bin","ikea_vesken_trolley","ikea_sortera_bin",
    "ikea_drona_grey","ikea_drona_blue","ikea_knallig_box",
    "ikea_moppe_drawer","ikea_labbsal_basket","ikea_ivar_box",
    "ikea_skubb_case","ikea_samla_box","human",
]
PALETTE = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),
           (128,0,0),(0,128,0),(0,0,128),(128,128,0),(128,0,128),(0,128,128),
           (255,128,0),(255,0,128),(0,255,128),(128,255,0),(64,64,255)]


def run(args):
    frames = sorted(glob.glob(os.path.join(args.frames_dir, "*.jpg")) +
                    glob.glob(os.path.join(args.frames_dir, "*.png")))
    if not frames:
        raise RuntimeError(f"No frames found in {args.frames_dir}")
    print(f"[INFO] {len(frames)} frames  weights={args.weights}")

    model = YOLO(args.weights)

    # Get frame size from first image
    sample = cv2.imread(frames[0])
    H, W = sample.shape[:2]

    os.makedirs(args.out_dir, exist_ok=True)
    video_path = os.path.join(args.out_dir, "rgb_detection.mp4")
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))

    all_detections = []
    for t, fpath in enumerate(frames):
        img = cv2.imread(fpath)
        results = model.predict(img, conf=args.conf, iou=args.iou,
                                device=args.device, verbose=False)[0]

        frame_record = {"frame": t, "boxes": []}
        for box in results.boxes:
            cls_id = int(box.cls)
            conf   = float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            frame_record["boxes"].append([cls_id, round(conf, 4),
                                          round(x1,1), round(y1,1),
                                          round(x2,1), round(y2,1)])
            color = PALETTE[cls_id % len(PALETTE)]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"{CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else cls_id} {conf:.2f}"
            cv2.putText(img, label, (int(x1), max(0, int(y1)-4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        all_detections.append(frame_record)
        writer.write(img)

        if (t + 1) % 100 == 0:
            print(f"  frame {t+1}/{len(frames)}  dets={len(frame_record['boxes'])}")

    writer.release()
    json_path = os.path.join(args.out_dir, "rgb_detections.json")
    with open(json_path, "w") as f:
        json.dump(all_detections, f)

    total = sum(len(r["boxes"]) for r in all_detections)
    with_dets = sum(1 for r in all_detections if r["boxes"])
    print(f"\n[DONE] {with_dets}/{len(frames)} frames with detections ({total} total boxes)")
    print(f"       video → {video_path}")
    print(f"       JSON  → {json_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--weights",    required=True)
    ap.add_argument("--out_dir",    required=True)
    ap.add_argument("--conf",  type=float, default=0.25)
    ap.add_argument("--iou",   type=float, default=0.45)
    ap.add_argument("--fps",   type=float, default=20.0)
    ap.add_argument("--device", default="cuda:0")
    run(ap.parse_args())

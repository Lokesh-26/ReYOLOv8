#!/usr/bin/env bash
# Full inference pipeline for the 24_07_new bag → benchmark_results/24_07_new/.
# Event ReYOLoV8 (c1/c5/c11/c21/tc) + YOLOv8s-5ch + RVT-640 (10ch) + RGB YOLOv8.
# Skips steps whose outputs already exist. Run from /home/loki/event/ReYOLOv8/.
set -euo pipefail

PYTHON_ROS=/usr/bin/python3
PYTHON_ML=/home/loki/anaconda3/envs/reyolov8/bin/python
PYTHON_RVT=/home/loki/venvs/rvt/bin/python

BAG="/home/loki/bags/24_07_new/wp1_rgb10ms_human_wagen_pallet_0b_dyn_robot_trailer_side_sparse_20260724_121308.bag"
OUT="benchmark_results/24_07_new"
CONF=0.25
mkdir -p "$OUT"

H5_5CH="$OUT/images/test/mtevent_test.h5"
H5_10CH="$OUT/_rvt10ch/images/test/mtevent_test.h5"
RGB_DIR="$OUT/rgb_frames"

declare -A W
W[c1]=runs/train/mtevent_640x480_c1_clean/weights/best.pt
W[c5]=runs/train/mtevent_640x480_c5_clean/weights/best.pt
W[c11]=runs/train/mtevent_640x480_fixed_c11/weights/best.pt
W[c21]=runs/train/mtevent_640x480_fixed_c21/weights/best.pt
W[tc]=runs/train/mtevent_640x480_tc_c11/weights/best.pt
W[yolov8s]=runs/train/mtevent_640x480_yolov8s_5ch_clean3/weights/best.pt
W_RVT640=paper_weights_2026-07-23/rvt/rvt_640_7akd4jn6.ckpt
W_RGB=runs/detect/runs/train/rgb_baseline_proper/weights/best.pt

# ── preprocessing ───────────────────────────────────────────────────────────
if [ ! -f "$H5_5CH" ]; then
  echo "[prep] 5ch signed VTEI ..."
  $PYTHON_ROS scripts/mtevent_to_reyolo_h5.py --bag_paths "$BAG" --out_root "$OUT" \
    --split test --topic /dvxplorer_left/events --outW 640 --outH 480
fi
if [ ! -f "$H5_10CH" ]; then
  echo "[prep] 10ch interleaved split-pol (RVT-640) ..."
  $PYTHON_ROS scripts/mtevent_to_reyolo_h5.py --bag_paths "$BAG" --out_root "$OUT/_rvt10ch" \
    --split test --topic /dvxplorer_left/events --outW 640 --outH 480 --split_polarity
fi
if [ ! -d "$RGB_DIR" ] || [ -z "$(ls -A "$RGB_DIR" 2>/dev/null)" ]; then
  echo "[prep] extract RGB frames ..."
  $PYTHON_ROS scripts/extract_rgb_frames.py "$BAG" "$RGB_DIR" /rgb/image_raw
fi

# ── event ReYOLoV8 / YOLOv8s ─────────────────────────────────────────────────
for label in c1 c5 c11 c21 tc yolov8s; do
  det="$OUT/detections_$label"
  if [ ! -d "$det" ]; then
    echo "[run] event $label ..."
    $PYTHON_ML scripts/infer_wp1_bags.py --h5 "$H5_5CH" --weights "${W[$label]}" \
      --out_dir "$det" --device cuda:0 --conf "$CONF"
  fi
done

# ── RVT-640 (10ch interleaved) ───────────────────────────────────────────────
if [ ! -d "$OUT/rvt640_detections" ]; then
  echo "[run] RVT-640 ..."
  $PYTHON_RVT scripts/infer_wp1_rvt.py --h5 "$H5_10CH" --weights "$W_RVT640" \
    --out_dir "$OUT/rvt640_detections" --device cuda:0 --conf "$CONF" \
    --height 480 --width 640
fi

# ── RGB YOLOv8 ───────────────────────────────────────────────────────────────
if [ ! -d "$OUT/rgb_detections" ]; then
  echo "[run] RGB YOLOv8 ..."
  $PYTHON_ML scripts/infer_wp1_rgb.py --frames_dir "$RGB_DIR" --weights "$W_RGB" \
    --out_dir "$OUT/rgb_detections" --fps 25 --imgsz 640
fi

echo "ALL DONE → $OUT/"

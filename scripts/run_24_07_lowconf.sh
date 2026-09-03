#!/usr/bin/env bash
# Low-confidence (0.001) re-run of the 24_07_new benchmark, all 15 bags.
# Copy of run_24_07_batch.sh with: CONF=0.001, outputs to /mnt/2tb, paper RGB
# checkpoint, and NO preprocessing — reuses the exact h5/rgb inputs of the
# existing benchmark_results/24_07_new run so the two are directly diffable.
# Run from /home/loki/event/ReYOLOv8/.
set -uo pipefail

PY=/home/loki/anaconda3/envs/reyolov8/bin/python
PY_RVT=/home/loki/venvs/rvt/bin/python

SRC=benchmark_results/24_07_new                  # existing preprocessed inputs (read-only)
AR=/home/loki/bags/annotation/24_07_new/rgb      # existing extracted RGB frames (read-only)
OUT_ROOT=/mnt/2tb/benchmark_results/24_07_lowconf
CONF=0.001

declare -A W
W[c21]=/home/loki/event/event_rgb/checkpoints/event_reyolo/c21_seed0.pt
# md5-identical to runs/train/mtevent_640x480_yolov8s_5ch_clean3/weights/best.pt
W[yolov8s]=/home/loki/event/event_rgb/checkpoints/event_reyolo/ff_seed0.pt
W_RVT640=paper_weights_2026-07-23/rvt/rvt_640_7akd4jn6.ckpt
# paper operating point (mAP50 0.2692), NOT runs/detect/.../rgb_baseline_proper (0.228)
W_RGB=/home/loki/event/event_rgb/checkpoints/rgb/rgb_640_best.pt
RGB_IMGSZ=640   # rgb_640_best.pt is the 640-res run; matches the script default

for src in "$SRC"/*/; do
  name=$(basename "$src")
  h5="$src/images/test/mtevent_test.h5"
  h5r="$src/images/test/rvt10ch_test.h5"
  rgb="$AR/$name/rgb"
  [ ! -f "$h5" ]  && { echo "SKIP $name (no mtevent_test.h5)"; continue; }
  [ ! -f "$h5r" ] && { echo "SKIP $name (no rvt10ch_test.h5)"; continue; }

  echo "===== $name ====="
  out="$OUT_ROOT/$name"; mkdir -p "$out"

  for label in c21 yolov8s; do
    if [ ! -d "$out/detections_$label" ]; then
      echo "  [event $label]"
      $PY scripts/infer_wp1_bags.py --h5 "$h5" --weights "${W[$label]}" \
        --out_dir "$out/detections_$label" --device cuda:0 --conf "$CONF" 2>&1 | tail -1
    fi
  done

  if [ ! -d "$out/rvt640_detections" ]; then
    echo "  [rvt640]"
    $PY_RVT scripts/infer_wp1_rvt.py --h5 "$h5r" --weights "$W_RVT640" \
      --out_dir "$out/rvt640_detections" --device cuda:0 --conf "$CONF" \
      --height 480 --width 640 2>&1 | tail -1
  fi

  if [ -d "$rgb" ] && [ ! -d "$out/rgb_detections" ]; then
    echo "  [rgb]"
    $PY scripts/infer_wp1_rgb.py --frames_dir "$rgb" --weights "$W_RGB" \
      --out_dir "$out/rgb_detections" --fps 25 --conf "$CONF" --imgsz "$RGB_IMGSZ" 2>&1 | tail -1
  fi
done

echo "ALL DONE → $OUT_ROOT/"

#!/usr/bin/env bash
# Run the full benchmark pipeline on all WP1 bags in /home/loki/bags/.
# Runs three models: ReYOLOv8 (event), RVT (event), RGB YOLOv8 (RGB).
# Must be executed from /home/loki/event/ReYOLOv8/
set -euo pipefail

# System Python3 has rosbag + roslz4 (needed for bag reading/LZ4 decompression)
PYTHON_ROS=/usr/bin/python3
# Conda env has ReYOLOv8 / PyTorch / RVT (needed for inference)
PYTHON_ML=/home/loki/anaconda3/envs/reyolov8/bin/python
BAGS_DIR=/home/loki/bags
WEIGHTS_REYOLO=runs/train/mtevent_640x480_fixed_c21/weights/best.pt
WEIGHTS_RVT=/home/loki/event/RVT/dummy/mkts9bwe/checkpoints/best_5ch_finetune.ckpt
WEIGHTS_RGB=runs/detect/runs/train/rgb_baseline_proper/weights/best.pt
TOPIC=/dvxplorer_left/events
CONF=0.25
OUTW=640
OUTH=480

for bag in "$BAGS_DIR"/*.bag; do
    name=$(basename "$bag" .bag)
    out_root="benchmark_results/$name"

    echo "=============================="
    echo "Processing: $name"
    echo "=============================="

    # Step 1: preprocess events → H5 (shared input for both event models)
    $PYTHON_ROS scripts/mtevent_to_reyolo_h5.py \
        --bag_paths "$bag" \
        --out_root  "$out_root" \
        --split     test \
        --topic     "$TOPIC" \
        --outW "$OUTW" --outH "$OUTH"

    # Step 2a: ReYOLOv8 event inference
    $PYTHON_ML scripts/infer_wp1_bags.py \
        --h5      "$out_root/images/test/mtevent_test.h5" \
        --weights "$WEIGHTS_REYOLO" \
        --out_dir "$out_root/detections" \
        --device  cuda:0 \
        --conf    "$CONF"

    # Step 2b: RVT event inference
    $PYTHON_ML scripts/infer_wp1_rvt.py \
        --h5      "$out_root/images/test/mtevent_test.h5" \
        --weights "$WEIGHTS_RVT" \
        --out_dir "$out_root/rvt_detections" \
        --device  cuda:0 \
        --conf    "$CONF"

    # Step 3: extract RGB frames from bag (requires rosbag)
    $PYTHON_ROS scripts/extract_rgb_wp1.py \
        --bag     "$bag" \
        --out_dir "$out_root/rgb_frames" \
        --topic   /rgb/image_raw 2>/dev/null || \
    $PYTHON_ROS scripts/extract_rgb_wp1.py \
        --bag     "$bag" \
        --out_dir "$out_root/rgb_frames" \
        --topic   /camera/image_raw 2>/dev/null || \
        echo "[SKIP] RGB frame extraction failed for $name"

    # Step 4: RGB YOLOv8 inference (only if frames were extracted)
    if [ -d "$out_root/rgb_frames" ] && [ "$(ls -A "$out_root/rgb_frames")" ]; then
        $PYTHON_ML scripts/infer_wp1_rgb.py \
            --frames_dir "$out_root/rgb_frames" \
            --weights    "$WEIGHTS_RGB" \
            --out_dir    "$out_root/rgb_detections" \
            --fps        20
    fi

    echo "Done: $out_root/"
done

echo ""
echo "All bags processed. Results in benchmark_results/"

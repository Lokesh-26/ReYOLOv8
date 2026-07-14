#!/usr/bin/env bash
# Run all paper model weights on bags in /home/loki/bags/03_07/.
# Models: ReYOLOv8 (C1, C5, C11, C21, TC), YOLOv8s baseline, RVT, RGB YOLOv8.
# Processes _renamed.bag files + non-renamed bags that have no renamed counterpart.
# Skips steps that already exist. Run from /home/loki/event/ReYOLOv8/.
set -euo pipefail

PYTHON_ROS=/usr/bin/python3
PYTHON_ML=/home/loki/anaconda3/envs/reyolov8/bin/python

BAGS_DIR=/home/loki/bags/03_07
CONF=0.25
TOPIC=/dvxplorer_left/events

declare -A WEIGHTS
WEIGHTS[c1]=runs/train/mtevent_640x480_c1_clean/weights/best.pt
WEIGHTS[c5]=runs/train/mtevent_640x480_c5_clean/weights/best.pt
WEIGHTS[c11]=runs/train/mtevent_640x480_fixed_c11/weights/best.pt
WEIGHTS[c21]=runs/train/mtevent_640x480_fixed_c21/weights/best.pt
WEIGHTS[tc]=runs/train/mtevent_640x480_tc_c11/weights/best.pt
WEIGHTS[yolov8s]=runs/train/mtevent_640x480_yolov8s_5ch_clean3/weights/best.pt

WEIGHTS_RVT=/home/loki/event/RVT/dummy/mkts9bwe/checkpoints/best_5ch_finetune.ckpt
WEIGHTS_RGB=runs/detect/runs/train/rgb_baseline_proper/weights/best.pt

# Collect bags: _renamed.bag + bags without a renamed counterpart
bags_to_process=()
for bag in "$BAGS_DIR"/*.bag; do
    base=$(basename "$bag" .bag)
    if [[ "$base" == *_renamed ]]; then
        bags_to_process+=("$bag")
    else
        if [ ! -f "$BAGS_DIR/${base}_renamed.bag" ]; then
            bags_to_process+=("$bag")
        fi
    fi
done

total=${#bags_to_process[@]}
echo "Found $total bags to process."

bag_idx=0
for bag in "${bags_to_process[@]}"; do
    bag_idx=$((bag_idx + 1))
    name=$(basename "$bag" .bag)
    out_root="benchmark_results/$name"
    H5="$out_root/images/test/mtevent_test.h5"

    echo ""
    echo "============================== [$bag_idx/$total]"
    echo "Processing: $name"
    echo "=============================="

    # Step 1: preprocess events → H5
    if [ ! -f "$H5" ]; then
        echo "[1/8] Preprocessing events → H5 ..."
        $PYTHON_ROS scripts/mtevent_to_reyolo_h5.py \
            --bag_paths "$bag" \
            --out_root  "$out_root" \
            --split     test \
            --topic     "$TOPIC" \
            --outW 640 --outH 480
    else
        echo "[1/8] H5 exists, skipping."
    fi

    # Steps 2-7: ReYOLOv8 / YOLOv8s models
    step=2
    for label in c1 c5 c11 c21 tc yolov8s; do
        out_det="$out_root/detections_$label"
        if [ ! -d "$out_det" ]; then
            echo "[$step/8] Running $label ..."
            $PYTHON_ML scripts/infer_wp1_bags.py \
                --h5      "$H5" \
                --weights "${WEIGHTS[$label]}" \
                --out_dir "$out_det" \
                --device  cuda:0 \
                --conf    "$CONF"
        else
            echo "[$step/8] $label detections exist, skipping."
        fi
        step=$((step + 1))
    done

    # Step 8: RVT
    if [ ! -d "$out_root/rvt_detections" ]; then
        echo "[8/9] Running RVT ..."
        $PYTHON_ML scripts/infer_wp1_rvt.py \
            --h5      "$H5" \
            --weights "$WEIGHTS_RVT" \
            --out_dir "$out_root/rvt_detections" \
            --device  cuda:0 \
            --conf    "$CONF"
    else
        echo "[8/9] RVT detections exist, skipping."
    fi

    # Step 9a: extract RGB frames
    rgb_dir="$out_root/rgb_frames"
    if [ ! -d "$rgb_dir" ] || [ -z "$(ls -A "$rgb_dir" 2>/dev/null)" ]; then
        echo "[9/9] Extracting RGB frames ..."
        $PYTHON_ROS scripts/extract_rgb_frames.py "$bag" "$rgb_dir" /rgb/image_raw
    else
        echo "[9/9] RGB frames exist, skipping extraction."
    fi

    # Step 9b: RGB YOLOv8 inference
    if [ -d "$rgb_dir" ] && [ -n "$(ls -A "$rgb_dir" 2>/dev/null)" ]; then
        if [ ! -d "$out_root/rgb_detections" ]; then
            echo "[9/9] Running RGB YOLOv8 ..."
            $PYTHON_ML scripts/infer_wp1_rgb.py \
                --frames_dir "$rgb_dir" \
                --weights    "$WEIGHTS_RGB" \
                --out_dir    "$out_root/rgb_detections" \
                --fps        20 \
                || echo "[WARN] RGB inference failed for $name — continuing."
        else
            echo "[9/9] RGB detections exist, skipping."
        fi
    else
        echo "[9/9] No RGB frames extracted, skipping RGB inference."
    fi

    echo "Done → $out_root/"
done

echo ""
echo "All $total bags processed. Results in benchmark_results/"

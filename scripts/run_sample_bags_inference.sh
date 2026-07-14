#!/usr/bin/env bash
# Run all paper model weights on the 3 new sample bags in /home/loki/bags/sample_bags/.
# Outputs go to benchmark_results/<bag_name>/<model_dir>/
# Must be executed from /home/loki/event/ReYOLOv8/
set -euo pipefail

PYTHON_ROS=/usr/bin/python3
PYTHON_ML=/home/loki/anaconda3/envs/reyolov8/bin/python

BAGS_DIR=/home/loki/bags/sample_bags
CONF=0.25
TOPIC=/dvxplorer_left/events

# All 6 paper ReYOLOv8/YOLOv8s weights (label → weights path)
declare -A WEIGHTS
WEIGHTS["c1"]=runs/train/mtevent_640x480_c1_clean/weights/best.pt
WEIGHTS["c5"]=runs/train/mtevent_640x480_c5_clean/weights/best.pt
WEIGHTS["c11"]=runs/train/mtevent_640x480_fixed_c11/weights/best.pt
WEIGHTS["c21"]=runs/train/mtevent_640x480_fixed_c21/weights/best.pt
WEIGHTS["tc"]=runs/train/mtevent_640x480_tc_c11/weights/best.pt
WEIGHTS["yolov8s"]=runs/train/mtevent_640x480_yolov8s_5ch_clean3/weights/best.pt

# RVT (5ch, 320×256 fine-tuned — compatible with infer_wp1_rvt.py)
WEIGHTS_RVT=/home/loki/event/RVT/dummy/mkts9bwe/checkpoints/best_5ch_finetune.ckpt

for bag in "$BAGS_DIR"/*.bag; do
    name=$(basename "$bag" .bag)
    out_root="benchmark_results/$name"

    echo ""
    echo "=============================="
    echo "Processing: $name"
    echo "=============================="

    # Step 1: preprocess events → 640×480 5ch H5
    echo "[1/8] Preprocessing events to H5 ..."
    $PYTHON_ROS scripts/mtevent_to_reyolo_h5.py \
        --bag_paths "$bag" \
        --out_root  "$out_root" \
        --split     test \
        --topic     "$TOPIC" \
        --outW 640 --outH 480

    H5="$out_root/images/test/mtevent_test.h5"

    # Step 2: run each ReYOLOv8/YOLOv8s model
    i=2
    for label in c1 c5 c11 c21 tc yolov8s; do
        echo "[$i/8] Running $label ..."
        $PYTHON_ML scripts/infer_wp1_bags.py \
            --h5      "$H5" \
            --weights "${WEIGHTS[$label]}" \
            --out_dir "$out_root/detections_$label" \
            --device  cuda:0 \
            --conf    "$CONF"
        i=$((i+1))
    done

    # Step 3: RVT (5ch fine-tuned, infers at 320×256)
    echo "[8/8] Running RVT (5ch) ..."
    $PYTHON_ML scripts/infer_wp1_rvt.py \
        --h5      "$H5" \
        --weights "$WEIGHTS_RVT" \
        --out_dir "$out_root/rvt_detections" \
        --device  cuda:0 \
        --conf    "$CONF"

    echo "Done → $out_root/"
done

echo ""
echo "All sample bags processed. Results in benchmark_results/"

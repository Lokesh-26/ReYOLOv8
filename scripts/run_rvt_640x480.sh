#!/bin/bash
# Full pipeline: preprocess MTevent at 640×480 split-pol + train RVT-small.
# Run from /home/loki/event/ReYOLOv8/
#
# Step 1 (needs system python3 with rosbag):
#   bash scripts/run_rvt_640x480.sh preprocess
#
# Step 2 (run after preprocessing completes):
#   bash scripts/run_rvt_640x480.sh train

set -e

REYOLO_DIR=/home/loki/event/ReYOLOv8
RVT_DIR=/home/loki/event/RVT
SCENES_ROOT=/mnt/2tb/MTevent_extracted_min
OUT_DIR=${REYOLO_DIR}/preprocessed_datasets/rvt_mtevent_10ch_640x480
PYTHON_ROSBAG=/usr/bin/python3
PYTHON_RVT=/home/loki/venvs/rvt/bin/python

TRAIN_IDS="9 11 12 13 15 16 17 18 19 20 22 24 25 27 28 29 30 31 32 34 \
           36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 \
           56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75"

VAL_IDS="3 4 5 6 7 8 10 14 21 23 26 33 35"

case "${1}" in
  preprocess)
    echo "=== Preprocessing train split ==="
    ${PYTHON_ROSBAG} ${REYOLO_DIR}/scripts/preprocess_mtevent_to_rvt_640x480.py \
      --scenes_root ${SCENES_ROOT} \
      --scene_ids ${TRAIN_IDS} \
      --out_dir ${OUT_DIR} \
      --split train

    echo "=== Preprocessing val split ==="
    ${PYTHON_ROSBAG} ${REYOLO_DIR}/scripts/preprocess_mtevent_to_rvt_640x480.py \
      --scenes_root ${SCENES_ROOT} \
      --scene_ids ${VAL_IDS} \
      --out_dir ${OUT_DIR} \
      --split val

    echo "=== Preprocessing complete: ${OUT_DIR} ==="
    ;;

  train)
    echo "=== Training RVT-small at 640×480 ==="
    cd ${RVT_DIR}
    WANDB_MODE=disabled ${PYTHON_RVT} train.py \
      +experiment/mtevent=small_10ch_640x480 \
      dataset=mtevent_10ch_640x480 \
      dataset.path=${OUT_DIR} \
      hardware.gpus=0 \
      wandb.group_name=rvt_mtevent_640x480 \
      wandb.project_name=RVT_mtevent
    ;;

  *)
    echo "Usage: $0 {preprocess|train}"
    exit 1
    ;;
esac

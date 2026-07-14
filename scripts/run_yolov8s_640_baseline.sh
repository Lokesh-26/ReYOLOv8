#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Launcher: feed-forward YOLOv8s 640x480 5ch event baseline
#
# Architecture : ultralytics/models/v8/yolov8s_5ch.yaml
#                (no ConvLSTM, 7.08M params, verified)
# Initialisation: scratch (YAML only, no pretrained .pt)
# Data          : vtei_mtevent_640x480 (corrected train H5)
# Resolution    : 640x480, 5 channels, 17 classes
# clip_length   : 1  (each frame independent; no recurrence)
# Epochs        : 150, seed 0, device 0
# Run name      : mtevent_640x480_yolov8s_5ch_clean
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"

PYTHON="/home/loki/anaconda3/envs/reyolov8/bin/python"
LOG_DIR="logs"
PID_DIR="run_pids"
RUN_NAME="mtevent_640x480_yolov8s_5ch_clean"
LOG_FILE="${LOG_DIR}/train_${RUN_NAME}.log"
PID_FILE="${PID_DIR}/train_${RUN_NAME}.pid"

mkdir -p "$LOG_DIR" "$PID_DIR"

START_TIME="$(date '+%Y-%m-%d %H:%M:%S')"
GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

echo "========================================================"
echo " YOLOv8s 640x480 5ch event baseline launcher"
echo " Start time : $START_TIME"
echo " Git commit : $GIT_COMMIT"
echo " Repository : $REPO_DIR"
echo "========================================================"

TRAIN_CMD="WANDB_MODE=disabled $PYTHON train.py \
  --model ultralytics/models/v8/yolov8s_5ch.yaml \
  --data configs/vtei_mtevent_640x480.yaml \
  --hyp configs/default_gen1.yaml \
  --device 0 \
  --batch 4 \
  --imgsz 640 \
  --epochs 150 \
  --channels 5 \
  --clip_length 1 \
  --clip_stride 1 \
  --freeze 0 \
  --seed 0 \
  --name ${RUN_NAME}"

echo "Command:"
echo "  $TRAIN_CMD"
echo ""
echo "Log file: $LOG_FILE"

# Run training
eval "$TRAIN_CMD"
EXIT_CODE=$?

echo ""
echo "========================================================"
echo " Training finished with exit code: $EXIT_CODE"
echo " End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================"

exit $EXIT_CODE

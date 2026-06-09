#!/bin/bash
# Auto-queue: wait for GPU free memory, then launch C1 and C5 sequentially.
# Runs in tmux session `c1_c5_queue`.
# C1 needs roughly 4-6 GB (clip=1); C5 needs roughly 6-9 GB (clip=5).
# We wait for >14000 MiB free (one C21 run has finished) before starting.

PYTHON=/home/loki/anaconda3/envs/reyolov8/bin/python
WORKDIR=/home/loki/event/ReYOLOv8
THRESHOLD_MIB=14000
POLL_SEC=120

cd "$WORKDIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

wait_for_gpu() {
    local needed=$1
    while true; do
        FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader | tr -d ' MiB')
        log "GPU free=${FREE} MiB (need ${needed} MiB)"
        if [ "$FREE" -gt "$needed" ]; then
            log "Sufficient GPU memory available."
            return 0
        fi
        sleep $POLL_SEC
    done
}

# ── C1 ──────────────────────────────────────────────────────────────────────
if [ -d "runs/train/mtevent_640x480_c1" ]; then
    log "mtevent_640x480_c1 already exists — skipping C1 launch."
else
    log "Waiting for GPU memory before launching C1..."
    wait_for_gpu $THRESHOLD_MIB

    log "Launching mtevent_640x480_c1 (clip=1, stride=1, 5ch, 640x480)..."
    log "Dataset: configs/vtei_mtevent_640x480.yaml"
    log "Model:   ultralytics/models/v8/Recurrent/ReYOLOV8s.yaml (from scratch)"
    log "Hyp:     configs/default_gen1.yaml"

    WANDB_MODE=disabled $PYTHON train.py \
        --model ultralytics/models/v8/Recurrent/ReYOLOV8s.yaml \
        --data configs/vtei_mtevent_640x480.yaml \
        --hyp configs/default_gen1.yaml \
        --device 0 --batch 4 --imgsz 640 --epochs 150 \
        --channels 5 --clip_length 1 --clip_stride 1 --freeze 0 \
        --exist_ok --name mtevent_640x480_c1

    log "C1 training finished."
fi

# ── C5 ──────────────────────────────────────────────────────────────────────
if [ -d "runs/train/mtevent_640x480_c5" ]; then
    log "mtevent_640x480_c5 already exists — skipping C5 launch."
else
    log "Waiting for GPU memory before launching C5..."
    wait_for_gpu $THRESHOLD_MIB

    log "Launching mtevent_640x480_c5 (clip=5, stride=3, 5ch, 640x480)..."
    log "Dataset: configs/vtei_mtevent_640x480.yaml"
    log "Model:   ultralytics/models/v8/Recurrent/ReYOLOV8s.yaml (from scratch)"
    log "Hyp:     configs/default_gen1.yaml"

    WANDB_MODE=disabled $PYTHON train.py \
        --model ultralytics/models/v8/Recurrent/ReYOLOV8s.yaml \
        --data configs/vtei_mtevent_640x480.yaml \
        --hyp configs/default_gen1.yaml \
        --device 0 --batch 4 --imgsz 640 --epochs 150 \
        --channels 5 --clip_length 5 --clip_stride 3 --freeze 0 \
        --name mtevent_640x480_c5

    log "C5 training finished."
fi

log "Queue complete."

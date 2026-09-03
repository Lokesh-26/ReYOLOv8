#!/usr/bin/env bash
# Retrain all three detectors on the leakage-free split v2 (configs/mtevent_split_v2.json).
# SEQUENTIAL on one GPU: event-FF -> RGB -> RVT -> RVT(1Mpx-init).  Hyperparameters are unchanged from v1;
# the split is the only variable.  Budget unified to 150 epochs / patience 50 for the two
# YOLO models (RVT trains by steps and has no patience knob -- 25 epochs, best by val/AP,
# matching the v1 run which peaked at epoch 11 of 26).
#
#   tmux new -s splitv2 'bash /home/loki/event/ReYOLOv8/scripts/train_split_v2.sh'
#   tmux attach -t splitv2
set -u
REYOLO=/home/loki/event/ReYOLOv8
RVTDIR=/home/loki/event/RVT
CKPT=/home/loki/event/event_rgb/checkpoints/split_v2
LOG=$REYOLO/logs/split_v2
PY=/home/loki/anaconda3/envs/reyolov8/bin/python
PY_RVT=/home/loki/venvs/rvt/bin/python
mkdir -p "$CKPT" "$LOG"
export WANDB_MODE=disabled PYTHONUNBUFFERED=1
say() { echo "[$(date +%F\ %T)] $*" | tee -a "$LOG/STATUS.txt"; }
# ultralytics silently renames a run whose dir exists (name -> name2), so resolve the
# ACTUAL output dir by mtime instead of assuming the name we asked for.
newest_run() { ls -td "$1"* 2>/dev/null | head -1; }

# ── 1/4  event YOLOv8s-FF, 5ch VTEI, from scratch (5ch stem cannot load COCO) ──
if [ -f "$CKPT/ff_splitv2_seed0.pt" ]; then say "SKIP ff (weights already in $CKPT)"; else
  say "TRAIN 1/4 event-FF  150ep patience50 -> $LOG/ff.log"
  cd $REYOLO && PYTHONPATH=$REYOLO $PY train.py \
    --model ultralytics/models/v8/yolov8s_5ch.yaml \
    --data configs/vtei_mtevent_640x480_v2.yaml \
    --hyp configs/default_gen1.yaml \
    --device 0 --batch 4 --imgsz 640 --epochs 150 --patience 50 \
    --channels 5 --clip_length 1 --clip_stride 1 \
    --freeze 0 --seed 0 --project runs/train --name ff_splitv2_seed0 \
    > "$LOG/ff.log" 2>&1
  R=$(newest_run runs/train/ff_splitv2_seed0)
  if [ -n "$R" ] && [ -f "$R/weights/best.pt" ]; then
    cp "$R/weights/best.pt" "$CKPT/ff_splitv2_seed0.pt"; say "DONE ff -> $CKPT/ff_splitv2_seed0.pt (from $R)"
  else say "FAIL ff (see $LOG/ff.log)"; fi
fi

# ── 2/4  RGB YOLOv8s, COCO-pretrained, 640 (matched resolution) ───────────────
if [ -f "$CKPT/rgb_640_splitv2.pt" ]; then say "SKIP rgb (weights already in $CKPT)"; else
  say "TRAIN 2/4 RGB-640  150ep patience50 -> $LOG/rgb.log"
  cd $REYOLO && $PY scripts/train_rgb_final.py \
    --weights weights/yolov8s.pt --data configs/rgb_clean_v2.yaml \
    --imgsz 640 --epochs 150 --patience 50 --batch 16 --device 0 \
    --name rgb_640_splitv2 > "$LOG/rgb.log" 2>&1
  R=$(newest_run runs/train_rgb/rgb_640_splitv2)
  if [ -n "$R" ] && [ -f "$R/weights/best.pt" ]; then
    cp "$R/weights/best.pt" "$CKPT/rgb_640_splitv2.pt"; say "DONE rgb -> $CKPT/rgb_640_splitv2.pt (from $R)"
  else say "FAIL rgb (see $LOG/rgb.log)"; fi
fi

# ── 3/4  RVT-S, 10ch split-pol histogram ──────────────────────────────────────
if [ -f "$CKPT/rvt_splitv2.ckpt" ]; then say "SKIP rvt (weights already in $CKPT)"; else
  say "TRAIN 3/4 RVT-S  25ep, best by val/AP -> $LOG/rvt.log  (~22 h at v1 speed)"
  cd $RVTDIR && $PY_RVT train.py \
    +experiment/mtevent=small_10ch_640x480 \
    dataset=mtevent_10ch_640x480 \
    dataset.path=$REYOLO/preprocessed_datasets/rvt_mtevent_10ch_640x480_v2 \
    hardware.gpus=0 training.max_epochs=25 \
    wandb.group_name=rvt_splitv2 wandb.project_name=RVT_mtevent \
    > "$LOG/rvt.log" 2>&1
  # keep the highest-val_AP checkpoint of the newest run dir
  BEST=$(ls -t $RVTDIR/dummy/*/checkpoints/*val_AP*.ckpt 2>/dev/null | head -1)
  BEST=$(dirname "$BEST" 2>/dev/null)
  BEST=$(ls "$BEST"/*val_AP*.ckpt 2>/dev/null | sed 's/.*val_AP=\([0-9.]*\)\.ckpt/\1 &/' | sort -rn | head -1 | cut -d' ' -f2-)
  if [ -n "$BEST" ]; then cp "$BEST" "$CKPT/rvt_splitv2.ckpt"; say "DONE rvt -> $CKPT/rvt_splitv2.ckpt (from $(basename "$BEST"))"
  else say "FAIL rvt (see $LOG/rvt.log)"; fi
fi

# ── 4/4  RVT-S, 1Mpx-pretrained init (ablation: isolates initialization) ──────
# Same representation, resolution and split as stage 3 -- ONLY the initialization changes,
# so this row separates pretraining from the 20-ch representation that the zero-shot arm
# also carries. Stem deflated 20->10 with the INTERLEAVED remap (see
# RVT/scripts/adapt_pretrained_to_mtevent_10ch.py); cls_preds reinit for 17 classes.
INIT=/home/loki/event/event_rgb/checkpoints/gen4_zeroshot/rvt-s-1mpx-adapted-mtevent-10ch.ckpt
if [ -f "$CKPT/rvt_splitv2_1mpxinit.ckpt" ]; then say "SKIP rvt-1mpx (weights already in $CKPT)"; else
  say "TRAIN 4/4 RVT-S 1Mpx-init  25ep, best by val/AP -> $LOG/rvt_1mpx.log"
  cd $RVTDIR && $PY_RVT train.py \
    +experiment/mtevent=small_10ch_640x480 \
    dataset=mtevent_10ch_640x480 \
    dataset.path=$REYOLO/preprocessed_datasets/rvt_mtevent_10ch_640x480_v2 \
    hardware.gpus=0 training.max_epochs=25 \
    wandb.artifact_name=rvt-s-1mpx-adapted wandb.artifact_local_file=$INIT \
    wandb.resume_only_weights=True \
    wandb.group_name=rvt_splitv2_1mpxinit wandb.project_name=RVT_mtevent \
    > "$LOG/rvt_1mpx.log" 2>&1 &
  P4=$!
  # fail fast: if the pretrained weights did not actually load, this is just stage 3 again
  for _ in $(seq 1 40); do
    sleep 15
    grep -q "Resuming only the weights" "$LOG/rvt_1mpx.log" && break
    kill -0 $P4 2>/dev/null || break
  done
  if ! grep -q "Resuming only the weights" "$LOG/rvt_1mpx.log"; then
    kill $P4 2>/dev/null; say "FAIL rvt-1mpx: pretrained weights were NOT loaded (see $LOG/rvt_1mpx.log)"
  else
    say "rvt-1mpx: pretrained weights loaded, training"
    wait $P4
    BEST=$(ls -t $RVTDIR/dummy/*/checkpoints/*val_AP*.ckpt 2>/dev/null | head -1)
    BEST=$(dirname "$BEST" 2>/dev/null)
    BEST=$(ls "$BEST"/*val_AP*.ckpt 2>/dev/null | sed 's/.*val_AP=\([0-9.]*\)\.ckpt/\1 &/' | sort -rn | head -1 | cut -d' ' -f2-)
    if [ -n "$BEST" ]; then cp "$BEST" "$CKPT/rvt_splitv2_1mpxinit.ckpt"; say "DONE rvt-1mpx -> $CKPT/rvt_splitv2_1mpxinit.ckpt (from $(basename "$BEST"))"
    else say "FAIL rvt-1mpx (see $LOG/rvt_1mpx.log)"; fi
  fi
fi

cp $REYOLO/configs/mtevent_split_v2.json "$CKPT/"
say "ALL DONE. weights in $CKPT"

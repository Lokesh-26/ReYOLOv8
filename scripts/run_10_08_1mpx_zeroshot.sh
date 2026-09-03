#!/usr/bin/env bash
# 1Mpx (gen4) zero-shot arm — the off-the-shelf event row done at NATIVE resolution.
#
# Why this exists: the GEN1 arm (run_10_08_gen1_zeroshot.sh) only accepts 304x240, so it
# squashes the 640x480 recordings by (2.105 x, 2.0 y) — 4x fewer pixels AND a 5% aspect
# distortion — and eval_10_08.py has to scale the boxes back with its `G` factor.
# RVT-S/1Mpx runs at 640x360, so 640x480 needs NO resize and NO rescale: G = (1, 1).
#
# Differences from the GEN1 arm:
#   - 20ch GROUPED at 640x480 (not 304x240), partition_split 2 -> attention window (8,10).
#     partition_size = in_res/(32*split) is the WINDOW passed to window_partition, so it
#     must be matched to training: 1Mpx trained at (6,10), GEN1 at (8,10). split 1 here
#     would give (15,20) -- 2.5x the trained window, and it cost ~3x AP.
#   - 3 classes (0=pedestrian, 1=two wheeler, 2=car) — person is class 0, not 1
#   - human-only by construction, no pallet row
# Run from /home/loki/event/ReYOLOv8/.
set -uo pipefail

PY=/home/loki/anaconda3/envs/reyolov8/bin/python
PY_RVT=/home/loki/venvs/rvt/bin/python
EVDIR=${EVDIR:-/home/loki/bags/annotation/10_08/events}
OUT_ROOT=${OUT_ROOT:-/mnt/2tb/benchmark_results/10_08_1mpx_zeroshot}
CKPT=${CKPT:-/home/loki/event/event_rgb/checkpoints/gen4_zeroshot/rvt-s-1mpx.ckpt}
CONF=0.001
DT_MS=50

for src in "$EVDIR"/*/; do
  name=$(basename "$src")
  if [ $# -gt 0 ]; then
    match=0; for pat in "$@"; do [[ "$name" == *"$pat"* ]] && match=1; done
    [ $match -eq 0 ] && continue
  fi
  ev="$src/events.h5"; bb="$src/bboxes_2d.json"
  [ ! -f "$ev" ] && { echo "SKIP $name (no events.h5)"; continue; }

  echo "===== $name ====="
  out="$OUT_ROOT/$name"; mkdir -p "$out/images/test"
  h5r="$out/images/test/rvt20_640_test.h5"

  [ ! -f "$h5r" ] && $PY scripts/vtei_from_events_h5.py --events_h5 "$ev" --bboxes "$bb" \
      --out "$h5r" --repr grouped --dt_ms $DT_MS --bins 10 --H 480 --W 640

  if [ ! -d "$out/rvt1mpx_detections" ]; then
    echo "  [rvt-1mpx]"
    $PY_RVT scripts/infer_wp1_rvt.py --h5 "$h5r" --weights "$CKPT" \
      --out_dir "$out/rvt1mpx_detections" --device cuda:0 --conf "$CONF" \
      --dataset gen4 --partition_split 2 --height 480 --width 640 2>&1 | tail -1
  fi
done
echo "ALL DONE → $OUT_ROOT/"

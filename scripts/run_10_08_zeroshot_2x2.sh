#!/usr/bin/env bash
# The two missing cells of the zero-shot 2x2: {GEN1, 1Mpx} weights x {304x240, 640x480} input.
#
# Why: 1Mpx at native 640x480 scored 3x WORSE than GEN1 at squashed 304x240, which is the
# opposite of what "no resize is cleaner" predicts. Both automotive checkpoints learned
# pedestrians as SMALL objects; squashing close-range warehouse humans into 304x240 lands them
# in that size range. This crosses weights against input scale so the two stop being confounded.
#
# Every cell uses the split that yields attention window (8,10): 304x240 -> split 1,
# 640x480 -> split 2. Window size is thus CONSTANT and only weights x input scale vary.
# Both representations already exist -- no preprocessing:
#   304x240 20ch grouped -> 10_08_gen1_zeroshot/<scene>/images/test/rvt20_test.h5
#   640x480 20ch grouped -> 10_08_1mpx_zeroshot/<scene>/images/test/rvt20_640_test.h5
# Run from /home/loki/event/ReYOLOv8/.
set -uo pipefail

PY_RVT=/home/loki/venvs/rvt/bin/python
G1=${G1:-/mnt/2tb/benchmark_results/10_08_gen1_zeroshot}
MPX=${MPX:-/mnt/2tb/benchmark_results/10_08_1mpx_zeroshot}
CK_G1=/home/loki/event/event_rgb/checkpoints/gen1_zeroshot/rvt-s.ckpt
CK_MPX=/home/loki/event/event_rgb/checkpoints/gen4_zeroshot/rvt-s-1mpx.ckpt
CONF=0.001

for src in "$MPX"/*/; do
  name=$(basename "$src")
  echo "===== $name ====="

  # cell: 1Mpx weights on the 304x240 input (the GEN1 arm's own h5)
  h304="$G1/$name/images/test/rvt20_test.h5"
  if [ -f "$h304" ] && [ ! -d "$G1/$name/rvt1mpx304_detections" ]; then
    echo "  [1mpx @304]"
    $PY_RVT scripts/infer_wp1_rvt.py --h5 "$h304" --weights "$CK_MPX" \
      --out_dir "$G1/$name/rvt1mpx304_detections" --device cuda:0 --conf "$CONF" \
      --dataset gen4 --partition_split 1 --height 240 --width 304 2>&1 | tail -1
  fi

  # cell: GEN1 weights on the 640x480 input (the 1Mpx arm's own h5)
  h640="$MPX/$name/images/test/rvt20_640_test.h5"
  if [ -f "$h640" ] && [ ! -d "$MPX/$name/rvtgen1_640_detections" ]; then
    echo "  [gen1 @640]"
    $PY_RVT scripts/infer_wp1_rvt.py --h5 "$h640" --weights "$CK_G1" \
      --out_dir "$MPX/$name/rvtgen1_640_detections" --device cuda:0 --conf "$CONF" \
      --dataset gen1 --partition_split 2 --height 480 --width 640 2>&1 | tail -1
  fi
  # correction arm: GEN1 @304 with split 1 -> window (8,10), which is what GEN1 actually
  # trained with. The long-standing rvtgen1_detections arm used split 2 -> (4,5), i.e. half
  # the trained window. Kept side by side to quantify the window-size effect itself.
  if [ -f "$h304" ] && [ ! -d "$G1/$name/rvtgen1_p1_detections" ]; then
    echo "  [gen1 @304 split1]"
    $PY_RVT scripts/infer_wp1_rvt.py --h5 "$h304" --weights "$CK_G1" \
      --out_dir "$G1/$name/rvtgen1_p1_detections" --device cuda:0 --conf "$CONF" \
      --dataset gen1 --partition_split 1 --height 240 --width 304 2>&1 | tail -1
  fi
done
echo "2x2 DONE"

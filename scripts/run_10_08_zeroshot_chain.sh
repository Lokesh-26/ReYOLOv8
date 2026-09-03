#!/usr/bin/env bash
# Zero-shot chain: finish the 1Mpx arm, fill the 2x2 (weights x input scale), then score.
# All three steps are idempotent. Run under tmux -- plain nohup/setsid launches were being
# reaped here, tmux is what survives.
set -u
cd /home/loki/event/ReYOLOv8
L=logs/split_v2
echo "=== [1/3] 1Mpx zero-shot @640 ==="; bash scripts/run_10_08_1mpx_zeroshot.sh
echo "=== [2/3] 2x2 completion ===";      bash scripts/run_10_08_zeroshot_2x2.sh
echo "=== [3/3] scoring ==="
cd /home/loki/event/event_rgb
PY=/home/loki/anaconda3/envs/reyolov8/bin/python
$PY scripts/eval_10_08.py --results /mnt/2tb/benchmark_results/10_08_lowconf \
  --ann /home/loki/bags/annotation/10_08 --perframe vicon_gt/perframe_10_08.csv \
  --unobs positive --out figures/physical
$PY - <<'PY'
import pandas as pd, os
ZS = {'gen1-rvt':      ('GEN1', '304x240 squashed', '(4,5) MISMATCHED'),
      'gen1-rvt-w810':  ('GEN1', '304x240 squashed', '(8,10) as trained'),
      '1mpx-rvt-304':   ('1Mpx', '304x240 squashed', '(8,10)'),
      'gen1-rvt-640':   ('GEN1', '640x480 native',   '(8,10) as trained'),
      '1mpx-rvt':       ('1Mpx', '640x480 native',   '(8,10)')}
df = pd.read_csv('figures/physical/ap_10_08.csv')
z = df[(df.model.isin(ZS)) & (df.cls == 'person')].copy()
z['weights'] = z.model.map(lambda m: ZS[m][0])
z['input']   = z.model.map(lambda m: ZS[m][1])
z['att_window'] = z.model.map(lambda m: ZS[m][2])
os.makedirs('results', exist_ok=True)
z[['model','weights','input','att_window','scene','cls','ap50','npos','tp','reaction_m']] \
    .sort_values(['model','scene']).to_csv('results/zeroshot_10_08_perscene.csv', index=False)
# score every cell on the SAME scenes, so the 2x2 is paired
common = set.intersection(*[set(z[z.model == m].scene) for m in z.model.unique()])
zc = z[z.scene.isin(common)]
summ = (zc.groupby(['weights','input','att_window'])
          .agg(scenes=('scene','nunique'), mean_ap50=('ap50','mean'),
               median_ap50=('ap50','median'), npos=('npos','sum'), tp=('tp','sum'),
               mean_reaction_m=('reaction_m','mean'))
          .reset_index().sort_values('mean_ap50', ascending=False))
summ.to_csv('results/zeroshot_10_08_summary.csv', index=False)
print(f'\n=== zero-shot person AP50, 10_08, --unobs positive, paired on {len(common)} scenes ===')
print(summ.to_string(index=False))
print('\n=== 2x2 (mean AP50) ===')
print(zc[zc.att_window != '(4,5) MISMATCHED']
      .pivot_table(index='weights', columns='input', values='ap50').round(4).to_string())
print('\nwrote results/zeroshot_10_08_{summary,perscene}.csv')
PY
echo "=== CHAIN DONE ==="

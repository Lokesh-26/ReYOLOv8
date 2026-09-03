#!/usr/bin/env python3
"""Write the RGB train/val/test file lists for a split definition.

The RGB dataset dirs are laid out by the OLD split, so the v2 assignment is expressed as
ultralytics file lists instead of by moving 85k images. Lists are regenerated, not committed
(configs/rgb_v2_*.txt are ~8 MB); configs/rgb_clean_v2.yaml points at them.

    python scripts/make_rgb_v2_lists.py
"""
import collections, glob, json, os

SPLIT = 'configs/mtevent_split_v2.json'  # single source of truth for the split
ROOT = '/home/loki/event/ReYOLOv8/preprocessed_datasets/vtei_rgb_1024x768'

sp = json.load(open(SPLIT))
imgs = collections.defaultdict(list)
for f in glob.glob(f'{ROOT}/images/*/scene*.jpg'):
    imgs[int(os.path.basename(f)[5:8])].append(f)          # scene###_<ts>.jpg

where = {s: split for split in ('train', 'val', 'test') for s in sp[split]}
out = collections.defaultdict(list)
for s, fs in imgs.items():
    out[where[s]] += fs

for split in ('train', 'val', 'test'):
    p = f'configs/rgb_v2_{split}.txt'
    open(p, 'w').write('\n'.join(sorted(out[split])) + '\n')
    n_sc = len({int(os.path.basename(f)[5:8]) for f in out[split]})
    print(f'{split}: {len(out[split])} images, {n_sc} scenes -> {p}')

missing = sorted(set(range(1, 76)) - set(imgs))
print('scenes with no RGB frames:', missing or 'none')

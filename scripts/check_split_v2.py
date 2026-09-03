#!/usr/bin/env python3
"""Sanity-check split v2: disjoint scenes, h5<->label alignment, per-split class counts,
and that both model datasets (VTEI + RVT) see the same scenes."""
import json, glob, os, collections
import h5py, numpy as np

SP = json.load(open('/home/loki/event/ReYOLOv8/configs/mtevent_split_v2.json'))
VTEI = '/mnt/2tb/preprocessed_datasets/vtei_mtevent_50ms_5bin_640x480_v2'
RVT = '/home/loki/event/ReYOLOv8/preprocessed_datasets/rvt_mtevent_10ch_640x480_v2'

seen = collections.Counter()
ok = True
for split in ('train', 'val', 'test'):
    lbl = sorted(glob.glob(f'{VTEI}/labels/{split}/*.npy'))
    ids = [int(os.path.basename(f).split('_')[1].split('.')[0]) for f in lbl]
    rvt_ids = sorted(int(os.path.basename(d).split('_')[1]) for d in glob.glob(f'{RVT}/{split}/scene_*'))
    seen.update(ids)
    n, cls = 0, collections.Counter()
    for f in lbl:
        a = np.load(f, allow_pickle=True); n += len(a)
        for fr in a:
            for row in fr: cls[int(row[0])] += 1
    with h5py.File(f'{VTEI}/images/{split}/mtevent_{split}.h5') as h:
        h5n = h['1mp'].shape[0]
    same = ids == sorted(SP[split]) == rvt_ids
    ok &= same and h5n == n
    print(f'{split:5s} scenes={len(ids)} vtei_frames={n} h5={h5n} {"OK" if h5n==n else "MISALIGNED"} '
          f'scene-lists-match={same}')
    print(f'      classes {dict(sorted(cls.items()))}')
    missing = [c for c in range(17) if c not in cls]
    if missing: print(f'      !! classes absent from {split}: {missing}')
    ok &= not missing or split == 'test'

dup = [s for s, c in seen.items() if c > 1]
print('duplicate scenes across splits:', dup or 'none')
print('coverage 1..75:', 'complete' if set(seen) == set(range(1, 76)) else sorted(set(range(1, 76)) - set(seen)))
print('RESULT:', 'PASS' if ok and not dup else 'FAIL')

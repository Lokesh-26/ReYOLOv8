#!/usr/bin/env python3
"""Assemble per-scene VTEI dirs into split h5 + label dirs (order = ascending scene id,
which is what EventVideoDataset assumes: sorted label filenames index the h5)."""
import json, os, shutil, sys
import h5py, numpy as np

PS = '/mnt/2tb/preprocessed_datasets/vtei_mtevent_50ms_5bin_640x480_perscene'
OUT = sys.argv[1] if len(sys.argv) > 1 else '/mnt/2tb/preprocessed_datasets/vtei_mtevent_50ms_5bin_640x480_v2'
sp = json.load(open('/home/loki/event/ReYOLOv8/configs/mtevent_split_v2.json'))

for split in ('train', 'val', 'test'):
    img_dir, lbl_dir = f'{OUT}/images/{split}', f'{OUT}/labels/{split}'
    os.makedirs(img_dir, exist_ok=True); os.makedirs(lbl_dir, exist_ok=True)
    h5path = f'{img_dir}/mtevent_{split}.h5'
    n = 0
    with h5py.File(h5path, 'w') as out:
        dst = None
        for s in sorted(sp[split]):
            src_h5 = f'{PS}/scene{s}/images/val/mtevent_val.h5'
            src_np = f'{PS}/scene{s}/labels/val/scene_{s:06d}.npy'
            lab = np.load(src_np, allow_pickle=True)
            with h5py.File(src_h5, 'r') as f:
                d = f['1mp']
                assert len(d) == len(lab), f'scene{s}: {len(d)} frames vs {len(lab)} labels'
                if dst is None:
                    dst = out.create_dataset('1mp', shape=(0,) + d.shape[1:], maxshape=(None,) + d.shape[1:],
                                             dtype=d.dtype, chunks=(1,) + d.shape[1:])
                for i in range(0, len(d), 256):          # stream in blocks, RAM-safe
                    blk = d[i:i + 256]
                    dst.resize(dst.shape[0] + len(blk), axis=0)
                    dst[-len(blk):] = blk
            shutil.copyfile(src_np, f'{lbl_dir}/scene_{s:06d}.npy')
            n += len(lab)
        assert dst.shape[0] == n
    print(f'{split}: {len(sp[split])} scenes, {n} frames -> {h5path}', flush=True)

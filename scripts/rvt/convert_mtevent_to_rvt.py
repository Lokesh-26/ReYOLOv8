"""
Convert MTevent preprocessed data (ReYOLOv8 format) to RVT per-scene format.

Input (ReYOLOv8 format):
  preprocessed_datasets/vtei_mtevent_50ms_5bin/
    images/{split}/mtevent_{split}.h5   key='1mp', shape=(N, 5, 256, 320), int8
    labels/{split}/scene_XXXXXX.npy     object array, each elem is (n_boxes, 5)
                                        box format: [class_id, cx_norm, cy_norm, w_norm, h_norm]

Output (RVT format):
  rvt_mtevent/{split}/scene_XXXXXX/
    event_representations_v2/stacked_histogram_dt=50_nbins=5/
      event_representations.h5          key='data', shape=(N, 5, 256, 320), int8
      objframe_idx_2_repr_idx.npy        frame indices with labels
      timestamps_us.npy                  fake: frame_idx * 50000 us
    labels_v2/
      labels.npz                         structured array: t,x,y,w,h,class_id,class_confidence
      timestamps_us.npy                  timestamps of label frames
"""

import argparse
import re
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

H, W = None, None  # inferred from H5 data
DT_US = 50_000  # 50 ms per frame in microseconds
EV_REPR_NAME = 'stacked_histogram_dt=50_nbins=5'  # override with --ev_repr_name

LABEL_DTYPE = np.dtype([
    ('t', np.int64),
    ('x', np.float32),
    ('y', np.float32),
    ('w', np.float32),
    ('h', np.float32),
    ('class_id', np.int64),
    ('class_confidence', np.float32),
])


def scene_sort_key(fname: str) -> int:
    m = re.search(r'(\d+)', fname)
    return int(m.group(1)) if m else 0


def convert_split(src_root: Path, dst_root: Path, split: str):
    h5_path = src_root / 'images' / split / f'mtevent_{split}.h5'
    label_dir = src_root / 'labels' / split

    scene_files = sorted(label_dir.glob('scene_*.npy'), key=lambda p: scene_sort_key(p.name))

    with h5py.File(h5_path, 'r') as h5f:
        all_frames = h5f['1mp'][:]  # (N_total, C, H, W)

    # Infer spatial dimensions and channel count from data
    _, n_channels, frame_H, frame_W = all_frames.shape

    offset = 0
    for scene_path in tqdm(scene_files, desc=split):
        scene_name = scene_path.stem  # e.g. scene_000009
        labels_raw = np.load(scene_path, allow_pickle=True)
        n_frames = len(labels_raw)

        frames = all_frames[offset:offset + n_frames]  # (n_frames, C, H, W)
        offset += n_frames

        out_dir = dst_root / split / scene_name
        repr_dir = out_dir / 'event_representations_v2' / EV_REPR_NAME
        labels_dir = out_dir / 'labels_v2'
        repr_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Write event representations H5
        repr_h5_path = repr_dir / 'event_representations.h5'
        with h5py.File(repr_h5_path, 'w') as f:
            f.create_dataset('data', data=frames, chunks=(1, n_channels, frame_H, frame_W))

        # Fake timestamps: one per frame, 50 ms apart
        repr_timestamps = np.arange(n_frames, dtype=np.int64) * DT_US
        np.save(repr_dir / 'timestamps_us.npy', repr_timestamps)

        # Build labels structured array
        all_boxes = []
        objframe_repr_indices = []
        objframe_label_start = []

        for frame_idx in range(n_frames):
            frame_boxes = labels_raw[frame_idx]
            if not (isinstance(frame_boxes, np.ndarray) and frame_boxes.ndim == 2 and frame_boxes.shape[0] > 0):
                continue

            objframe_repr_indices.append(frame_idx)
            objframe_label_start.append(len(all_boxes))
            t = frame_idx * DT_US

            for box in frame_boxes:
                class_id, cx_n, cy_n, w_n, h_n = float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(box[4])
                x0 = (cx_n - w_n / 2) * frame_W
                y0 = (cy_n - h_n / 2) * frame_H
                w = w_n * frame_W
                h = h_n * frame_H
                all_boxes.append((t, x0, y0, w, h, int(class_id), 1.0))

        if len(all_boxes) == 0:
            # Edge case: scene with no annotations at all — create minimal valid files
            labels_arr = np.zeros(0, dtype=LABEL_DTYPE)
            objframe_idx_2_label_idx = np.zeros(0, dtype=np.int64)
            objframe_idx_2_repr_idx = np.zeros(0, dtype=np.int64)
            label_timestamps = np.zeros(0, dtype=np.int64)
        else:
            labels_arr = np.array(all_boxes, dtype=LABEL_DTYPE)
            objframe_idx_2_label_idx = np.array(objframe_label_start, dtype=np.int64)
            objframe_idx_2_repr_idx = np.array(objframe_repr_indices, dtype=np.int64)
            label_timestamps = objframe_idx_2_repr_idx * DT_US

        np.savez(labels_dir / 'labels.npz',
                 labels=labels_arr,
                 objframe_idx_2_label_idx=objframe_idx_2_label_idx)
        np.save(labels_dir / 'timestamps_us.npy', label_timestamps)
        np.save(repr_dir / 'objframe_idx_2_repr_idx.npy', objframe_idx_2_repr_idx)

    assert offset == len(all_frames), f'Frame count mismatch: {offset} vs {len(all_frames)}'
    print(f'{split}: converted {len(scene_files)} scenes, {offset} frames total')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=Path,
                        default=Path('preprocessed_datasets/vtei_mtevent_50ms_5bin'))
    parser.add_argument('--dst', type=Path,
                        default=Path('preprocessed_datasets/rvt_mtevent'))
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--ev_repr_name', type=str, default=None,
                        help='Override EV_REPR_NAME (directory name for event repr)')
    args = parser.parse_args()

    if args.ev_repr_name is not None:
        global EV_REPR_NAME
        EV_REPR_NAME = args.ev_repr_name

    for split in args.splits:
        h5_path = args.src / 'images' / split / f'mtevent_{split}.h5'
        if not h5_path.exists():
            print(f'Skipping {split}: {h5_path} not found')
            continue
        convert_split(args.src, args.dst, split)


if __name__ == '__main__':
    main()

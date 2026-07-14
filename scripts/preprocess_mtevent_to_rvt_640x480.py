#!/usr/bin/env python3
"""
Preprocess MTevent raw bags → RVT per-scene format at 640×480, 10ch split-pol.

Run with system python3 (/usr/bin/python3) which has rosbag installed.

Usage:
  # Train split (scenes 9-75, excluding leaked 10/21/23 and val 3-8/14/26/33/35):
  /usr/bin/python3 scripts/preprocess_mtevent_to_rvt_640x480.py \
    --scenes_root /mnt/2tb/MTevent_extracted_min \
    --scene_ids 9 11 12 13 15 16 17 18 19 20 22 24 25 27 28 29 30 31 32 34 36 37 \
                38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 \
                60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 \
    --out_dir preprocessed_datasets/rvt_mtevent_10ch_640x480 \
    --split train

  # Val split (all 13 val scenes):
  /usr/bin/python3 scripts/preprocess_mtevent_to_rvt_640x480.py \
    --scenes_root /mnt/2tb/MTevent_extracted_min \
    --scene_ids 3 4 5 6 7 8 10 14 21 23 26 33 35 \
    --out_dir preprocessed_datasets/rvt_mtevent_10ch_640x480 \
    --split val

Writes per-scene RVT structure:
  {out_dir}/{split}/scene_XXXXXX/
    event_representations_v2/stacked_histogram_dt=50_nbins=5_split_pol/
      event_representations.h5   (key='data', shape=(T, 10, 480, 640), int8)
      objframe_idx_2_repr_idx.npy
      timestamps_us.npy
    labels_v2/
      labels.npz  (labels structured array + objframe_idx_2_label_idx)
      timestamps_us.npy
"""
import sys
import os
import re
import glob
import argparse
import json
from pathlib import Path

import numpy as np
import h5py

try:
    import rosbag
except ImportError:
    sys.exit("Run with system python3 (/usr/bin/python3) which has rosbag installed.")

# ── Constants ────────────────────────────────────────────────────────────────
DT_MS       = 50.0
BINS        = 5
TOPIC       = '/dvxplorer_left/events'
CAMERA      = 'ec_left'
CLIP_ABS    = 127
OUT_W, OUT_H = 640, 480
EV_REPR_NAME = 'stacked_histogram_dt=50_nbins=5_split_pol'

CLASS_NAMES = [
    "wooden_pallet", "small_klt", "big_klt", "blue_klt", "amazon_luggage",
    "ikea_dammang_bin", "ikea_vesken_trolley", "ikea_sortera_bin",
    "ikea_drona_grey", "ikea_drona_blue", "ikea_knallig_box", "ikea_moppe_drawer",
    "ikea_labbsal_basket", "ikea_ivar_box", "ikea_skubb_case", "ikea_samla_box", "human",
]
HUMAN_CLASS_ID = 16

LABEL_DTYPE = np.dtype([
    ('t',                np.int64),
    ('x',                np.float32),
    ('y',                np.float32),
    ('w',                np.float32),
    ('h',                np.float32),
    ('class_id',         np.int64),
    ('class_confidence', np.float32),
])


# ── Label I/O ────────────────────────────────────────────────────────────────

def read_jsonl(path):
    labels = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                labels.append(json.loads(line))
    labels.sort(key=lambda x: int(x.get("timestamp", 0)))
    return labels


def nearest_labels_multi(all_object_labels, center_t_ns, label_max_dt_ns=None):
    results = []
    for cls_id, label_list in all_object_labels:
        if not label_list:
            continue
        best, best_dt = None, float('inf')
        for lab in label_list:
            dt = abs(int(lab["timestamp"]) - center_t_ns)
            if dt < best_dt:
                best_dt = dt
                best = lab
        if best is None:
            continue
        if label_max_dt_ns is not None and best_dt > label_max_dt_ns:
            continue
        results.append((cls_id, best))
    return results


def mr6d_to_class_id(n):
    return n - 1


# ── Voxel building ────────────────────────────────────────────────────────────

def build_scene_frames(bag_path, all_object_labels, inW, inH):
    """Read events from bag, build 10ch split-pol voxels at OUT_H×OUT_W.
    Returns frames (T, 10, OUT_H, OUT_W) int8 and per-frame YOLO labels."""
    dt_ns = int(DT_MS * 1e6)
    channels = BINS * 2  # 10 for split-pol

    need_rescale = (inW != OUT_W or inH != OUT_H)
    sx = OUT_W / float(inW)
    sy = OUT_H / float(inH)

    bag = rosbag.Bag(bag_path)

    first_t = None
    win_start = None
    win_end = None
    voxel = None
    frames = []
    labels_obj = []

    def flush_window(center_t):
        nonlocal voxel
        if voxel is None:
            return
        v = np.clip(voxel, 0, CLIP_ABS).astype(np.int8)
        frames.append(v)
        matched = nearest_labels_multi(all_object_labels, center_t)
        if matched:
            rows = []
            for cls_id, lab in matched:
                xmin = float(lab["xmin"]) * (sx if need_rescale else 1.0)
                xmax = float(lab["xmax"]) * (sx if need_rescale else 1.0)
                ymin = float(lab["ymin"]) * (sy if need_rescale else 1.0)
                ymax = float(lab["ymax"]) * (sy if need_rescale else 1.0)
                xmin = max(0., min(xmin, OUT_W - 1.))
                xmax = max(0., min(xmax, OUT_W - 1.))
                ymin = max(0., min(ymin, OUT_H - 1.))
                ymax = max(0., min(ymax, OUT_H - 1.))
                if xmax <= xmin or ymax <= ymin:
                    continue
                cx = (xmin + xmax) / 2. / OUT_W
                cy = (ymin + ymax) / 2. / OUT_H
                w  = (xmax - xmin) / OUT_W
                h  = (ymax - ymin) / OUT_H
                rows.append([cls_id, cx, cy, w, h])
            if rows:
                labels_obj.append(np.array(rows, dtype=np.float32))
            else:
                labels_obj.append(np.zeros((0, 5), dtype=np.float32))
        else:
            labels_obj.append(np.zeros((0, 5), dtype=np.float32))

    for _, msg, _ in bag.read_messages(topics=[TOPIC]):
        for e in msg.events:
            ts = e.ts
            if hasattr(ts, "to_nsec"):
                t_ns = int(ts.to_nsec())
            else:
                t_ns = int(ts.secs) * 1_000_000_000 + int(ts.nsecs)

            x = int(e.x)
            y = int(e.y)
            p = 1 if int(e.polarity) > 0 else 0  # 0 = neg for split-pol

            if first_t is None:
                first_t = t_ns
                win_start = first_t
                win_end = win_start + dt_ns
                voxel = np.zeros((channels, OUT_H, OUT_W), dtype=np.int16)

            while t_ns >= win_end:
                flush_window((win_start + win_end) // 2)
                win_start = win_end
                win_end = win_start + dt_ns
                voxel = np.zeros((channels, OUT_H, OUT_W), dtype=np.int16)

            if need_rescale:
                xo = int(x * sx)
                yo = int(y * sy)
            else:
                xo, yo = x, y

            if xo < 0 or xo >= OUT_W or yo < 0 or yo >= OUT_H:
                continue

            rel = t_ns - win_start
            b = int((rel * BINS) // dt_ns)
            if b < 0: b = 0
            if b >= BINS: b = BINS - 1

            ch = 2 * b if p > 0 else 2 * b + 1
            voxel[ch, yo, xo] += 1

    if voxel is not None:
        flush_window((win_start + win_end) // 2)

    bag.close()

    if frames:
        frames_np = np.stack(frames, axis=0)  # (T, 10, H, W)
    else:
        frames_np = np.zeros((0, channels, OUT_H, OUT_W), dtype=np.int8)
    return frames_np, labels_obj


def infer_wh(bag_path, max_msgs=3):
    max_x = max_y = 0
    bag = rosbag.Bag(bag_path)
    n = 0
    for _, msg, _ in bag.read_messages(topics=[TOPIC]):
        for e in msg.events:
            if int(e.x) > max_x: max_x = int(e.x)
            if int(e.y) > max_y: max_y = int(e.y)
        n += 1
        if n >= max_msgs:
            break
    bag.close()
    return max_x + 1, max_y + 1


# ── Scene loading ────────────────────────────────────────────────────────────

def load_scene_annotations(scene_dir):
    """Load all object annotation JSONs from the scene directory."""
    all_object_labels = []

    # MR6D objects
    mr6d_search_dirs = [scene_dir, os.path.join(scene_dir, "annotation")]
    mr6d_files = []
    for sd in mr6d_search_dirs:
        pattern = os.path.join(sd, f"{CAMERA}_MR6D*_bounding_box_labels_2d.json")
        mr6d_files.extend(glob.glob(pattern))
    mr6d_files = sorted(set(mr6d_files))

    for jf in mr6d_files:
        m = re.search(r"MR6D(\d+)", os.path.basename(jf))
        if m is None:
            continue
        cls_id = mr6d_to_class_id(int(m.group(1)))
        if cls_id < 0 or cls_id >= HUMAN_CLASS_ID:
            continue
        labels = read_jsonl(jf)
        if labels:
            all_object_labels.append((cls_id, labels))

    # Human
    human_candidates = [
        os.path.join(scene_dir, f"human_{CAMERA}_bounding_box_labels_2d.json"),
        os.path.join(scene_dir, "annotation_human", f"human_{CAMERA}_bounding_box_labels_2d.json"),
    ]
    for hc in human_candidates:
        if os.path.isfile(hc):
            labels = read_jsonl(hc)
            if labels:
                all_object_labels.append((HUMAN_CLASS_ID, labels))
            break

    return all_object_labels


# ── RVT writer ───────────────────────────────────────────────────────────────

def write_rvt_scene(scene_id, frames_np, labels_list, out_base):
    """Write one scene in RVT per-scene format."""
    scene_name = f"scene_{scene_id:06d}"
    out_dir = Path(out_base) / scene_name
    repr_dir = out_dir / 'event_representations_v2' / EV_REPR_NAME
    lbl_dir  = out_dir / 'labels_v2'
    repr_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    T = len(frames_np)

    # Write event representations H5
    with h5py.File(repr_dir / 'event_representations.h5', 'w') as f:
        f.create_dataset('data', data=frames_np, chunks=(1, 10, OUT_H, OUT_W))

    # Timestamps: one per frame, 50ms apart
    repr_timestamps = np.arange(T, dtype=np.int64) * int(DT_MS * 1000)
    np.save(repr_dir / 'timestamps_us.npy', repr_timestamps)

    # Build label structured array
    all_boxes = []
    objframe_repr_indices = []
    objframe_label_starts = []

    for frame_idx, frame_labels in enumerate(labels_list):
        if not (isinstance(frame_labels, np.ndarray) and frame_labels.ndim == 2
                and frame_labels.shape[0] > 0):
            continue

        objframe_repr_indices.append(frame_idx)
        objframe_label_starts.append(len(all_boxes))
        t_us = frame_idx * int(DT_MS * 1000)

        for box in frame_labels:
            cls_id, cx_n, cy_n, w_n, h_n = float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(box[4])
            x0 = (cx_n - w_n / 2) * OUT_W
            y0 = (cy_n - h_n / 2) * OUT_H
            w  = w_n * OUT_W
            h  = h_n * OUT_H
            all_boxes.append((t_us, x0, y0, w, h, int(cls_id), 1.0))

    if all_boxes:
        labels_arr = np.array(all_boxes, dtype=LABEL_DTYPE)
        objframe_idx_2_label_idx = np.array(objframe_label_starts, dtype=np.int64)
        objframe_idx_2_repr_idx  = np.array(objframe_repr_indices,  dtype=np.int64)
        label_timestamps = objframe_idx_2_repr_idx * int(DT_MS * 1000)
    else:
        labels_arr = np.zeros(0, dtype=LABEL_DTYPE)
        objframe_idx_2_label_idx = np.zeros(0, dtype=np.int64)
        objframe_idx_2_repr_idx  = np.zeros(0, dtype=np.int64)
        label_timestamps = np.zeros(0, dtype=np.int64)

    np.savez(lbl_dir / 'labels.npz',
             labels=labels_arr,
             objframe_idx_2_label_idx=objframe_idx_2_label_idx)
    np.save(lbl_dir / 'timestamps_us.npy', label_timestamps)
    np.save(repr_dir / 'objframe_idx_2_repr_idx.npy', objframe_idx_2_repr_idx)

    n_labeled = int((labels_arr.shape[0] > 0) if labels_arr.shape[0] else 0)
    return T, len(objframe_repr_indices)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenes_root', required=True,
                        help='Root dir containing sceneN/ subdirectories')
    parser.add_argument('--scene_ids', nargs='+', type=int, required=True,
                        help='Scene IDs to process')
    parser.add_argument('--out_dir', required=True,
                        help='Output root (split subdir will be appended)')
    parser.add_argument('--split', required=True, choices=['train', 'val', 'test'])
    args = parser.parse_args()

    out_base = os.path.join(args.out_dir, args.split)
    os.makedirs(out_base, exist_ok=True)

    total_frames = 0
    skipped = 0

    for scene_id in sorted(args.scene_ids):
        scene_dir = os.path.join(args.scenes_root, f'scene{scene_id}')
        bag_path  = os.path.join(scene_dir, 'left.bag')

        if not os.path.isfile(bag_path):
            print(f'[SKIP] scene{scene_id}: left.bag not found at {bag_path}')
            skipped += 1
            continue

        print(f'[{scene_id:02d}] Loading annotations...', end=' ', flush=True)
        all_object_labels = load_scene_annotations(scene_dir)
        if not all_object_labels:
            print(f'no annotations found, skipping')
            skipped += 1
            continue
        print(f'{len(all_object_labels)} objects. Inferring sensor size...', end=' ', flush=True)

        inW, inH = infer_wh(bag_path)
        print(f'{inW}×{inH} → {OUT_W}×{OUT_H}. Building frames...', end=' ', flush=True)

        frames_np, labels_list = build_scene_frames(bag_path, all_object_labels, inW, inH)
        T = len(frames_np)
        print(f'{T} frames. Writing...', end=' ', flush=True)

        T_out, n_labeled_frames = write_rvt_scene(scene_id, frames_np, labels_list, out_base)
        print(f'done ({n_labeled_frames} labeled frames)')
        total_frames += T_out

    print(f'\nDone. {len(args.scene_ids) - skipped}/{len(args.scene_ids)} scenes, '
          f'{total_frames} total frames → {out_base}')


if __name__ == '__main__':
    main()

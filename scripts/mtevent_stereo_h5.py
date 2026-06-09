#!/usr/bin/env python3
"""
Stereo event preprocessing: left(5ch) + right(5ch) → 10ch H5 for ReYOLOv8.

What it does:
  For each 50ms time window, accumulate events from BOTH left and right cameras
  into separate 5-channel voxel grids, then concatenate them as a single 10-channel
  frame. Labels come from the LEFT camera only (ground truth stays in left frame).

  Output H5 shape: (N, 10, outH, outW)
    channels 0-4  = left camera  (bin0..bin4, combined polarity)
    channels 5-9  = right camera (bin0..bin4, combined polarity)

  Time alignment strategy:
    Left camera defines the time windows (its first event sets t=0).
    Right camera events are binned into those exact same windows.
    Any right-camera time offset is handled via absolute timestamps.

Usage:
  python scripts/mtevent_stereo_h5.py \
    --left_scenes  /mnt/2tb/MTevent_extracted_min/scene9 ... \
    --right_scenes /mnt/2tb/MTevent_extracted_right/scene9 ... \
    --out_root preprocessed_datasets/vtei_mtevent_stereo_5bin \
    --split train \
    --dt_ms 50 --bins 5 --outW 640 --outH 480
"""
import os
import re
import glob
import json
import argparse

import h5py
import numpy as np
import rosbag

try:
    import hdf5plugin
    HAVE_BLOSC = True
except Exception:
    HAVE_BLOSC = False


LEFT_TOPIC  = "/dvxplorer_left/events"
RIGHT_TOPIC = "/dvxplorer_right/events"
LEFT_BAG    = "left.bag"
RIGHT_BAG   = "right.bag"

CLASS_NAMES = [
    "wooden_pallet", "small_klt", "big_klt", "blue_klt", "amazon_luggage",
    "ikea_dammang_bin", "ikea_vesken_trolley", "ikea_sortera_bin",
    "ikea_drona_grey", "ikea_drona_blue", "ikea_knallig_box",
    "ikea_moppe_drawer", "ikea_labbsal_basket", "ikea_ivar_box",
    "ikea_skubb_case", "ikea_samla_box", "human",
]
HUMAN_CLASS_ID = 16


# ── annotation helpers (same as mtevent_to_reyolo_h5.py) ─────────────────────

def read_jsonl(path):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    items.sort(key=lambda d: int(d["timestamp"]))
    return items


def nearest_label(labels, t_ns):
    if not labels:
        return None
    if t_ns <= labels[0]["timestamp"]:
        return labels[0]
    if t_ns >= labels[-1]["timestamp"]:
        return labels[-1]
    lo, hi = 0, len(labels) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        tm = labels[mid]["timestamp"]
        if tm < t_ns:   lo = mid + 1
        elif tm > t_ns: hi = mid - 1
        else:           return labels[mid]
    cands = [labels[i] for i in [hi, lo] if 0 <= i < len(labels)]
    return min(cands, key=lambda d: abs(d["timestamp"] - t_ns)) if cands else None


def load_left_annotations(scene_dir):
    """Return list of (class_id, sorted_label_list) from ec_left annotations."""
    results = []
    for jf in sorted(glob.glob(os.path.join(scene_dir, "ec_left_MR6D*_bounding_box_labels_2d.json"))):
        m = re.search(r"MR6D(\d+)", os.path.basename(jf))
        if not m:
            continue
        cls_id = int(m.group(1)) - 1
        if not (0 <= cls_id < HUMAN_CLASS_ID):
            continue
        labs = read_jsonl(jf)
        if labs:
            results.append((cls_id, labs))
    for hc in [
        os.path.join(scene_dir, "human_ec_left_bounding_box_labels_2d.json"),
        os.path.join(scene_dir, "annotation_human", "human_ec_left_bounding_box_labels_2d.json"),
    ]:
        if os.path.isfile(hc):
            labs = read_jsonl(hc)
            if labs:
                results.append((HUMAN_CLASS_ID, labs))
            break
    return results


def make_yolo_row(lab, cls_id, W, H):
    xmin, xmax = max(0., min(float(lab["xmin"]), W-1)), max(0., min(float(lab["xmax"]), W-1))
    ymin, ymax = max(0., min(float(lab["ymin"]), H-1)), max(0., min(float(lab["ymax"]), H-1))
    if xmax <= xmin or ymax <= ymin:
        return None
    cx = (xmin + xmax) / 2.0 / W
    cy = (ymin + ymax) / 2.0 / H
    w  = (xmax - xmin) / W
    h  = (ymax - ymin) / H
    return [cls_id, cx, cy, w, h]


def labels_at(all_object_labels, t_ns, W, H, max_dt_ns=None):
    rows = []
    for cls_id, labs in all_object_labels:
        lab = nearest_label(labs, t_ns)
        if lab is None:
            continue
        if max_dt_ns is not None and abs(int(lab["timestamp"]) - int(t_ns)) > max_dt_ns:
            continue  # annotation too far in time — object not visible at this frame
        row = make_yolo_row(lab, cls_id, W, H)
        if row:
            rows.append(row)
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 5), dtype=np.float32)


# ── event reading helpers ─────────────────────────────────────────────────────

def infer_wh(bag_path, topic, max_msgs=3):
    mx, my = 0, 0
    bag = rosbag.Bag(bag_path)
    n = 0
    for _, msg, _ in bag.read_messages(topics=[topic]):
        for e in msg.events:
            if int(e.x) > mx: mx = int(e.x)
            if int(e.y) > my: my = int(e.y)
        n += 1
        if n >= max_msgs:
            break
    bag.close()
    return mx + 1, my + 1


def event_time_ns(e):
    ts = e.ts
    if hasattr(ts, "to_nsec"):
        return int(ts.to_nsec())
    return int(ts.secs) * 1_000_000_000 + int(ts.nsecs)


def scale_xy(x, y, inW, inH, outW, outH):
    return int(x * outW / inW), int(y * outH / inH)


# ── core: read one bag into pre-defined time windows ─────────────────────────

def bag_to_voxels(bag_path, topic, win_starts_ns, dt_ns, bins,
                  inW, inH, outW, outH, clip_abs=127):
    """
    Given a list of window start timestamps (ns), fill one voxel per window
    from events in bag_path/topic.

    Returns: np.ndarray (N_windows, bins, outH, outW) int8
             Windows with zero events are returned as all-zeros (right camera
             sometimes starts slightly later than left).
    """
    N = len(win_starts_ns)
    voxels = np.zeros((N, bins, outH, outW), dtype=np.int16)

    bag = rosbag.Bag(bag_path)
    for _, msg, _ in bag.read_messages(topics=[topic]):
        for e in msg.events:
            t = event_time_ns(e)
            # Binary search: which window does this event fall in?
            lo, hi = 0, N - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if win_starts_ns[mid] <= t:
                    lo = mid
                else:
                    hi = mid - 1
            w_idx = lo
            w_start = win_starts_ns[w_idx]
            if t < w_start or t >= w_start + dt_ns:
                continue  # before first window or after last window

            x, y = int(e.x), int(e.y)
            p = 1 if int(e.polarity) > 0 else -1
            if inW != outW or inH != outH:
                x, y = scale_xy(x, y, inW, inH, outW, outH)
            if not (0 <= x < outW and 0 <= y < outH):
                continue

            rel = t - w_start
            b = min(int(rel * bins // dt_ns), bins - 1)
            voxels[w_idx, b, y, x] += p

    bag.close()
    return np.clip(voxels, -clip_abs, clip_abs).astype(np.int8)


# ── per-scene stereo processing ───────────────────────────────────────────────

def process_scene(left_scene_dir, right_scene_dir, dt_ms, bins, outW, outH, max_dt_ns=None):
    """
    Process one scene. Returns:
      stereo_frames : (N, 10, outH, outW) int8   — left(5ch) + right(5ch)
      labels        : (N,) object array           — YOLO labels from left annotations
    """
    dt_ns = int(dt_ms * 1e6)
    left_bag  = os.path.join(left_scene_dir,  LEFT_BAG)
    right_bag = os.path.join(right_scene_dir, RIGHT_BAG)

    if not os.path.isfile(left_bag):
        print(f"  [SKIP] no left bag: {left_bag}")
        return None, None
    if not os.path.isfile(right_bag):
        print(f"  [SKIP] no right bag: {right_bag}")
        return None, None

    # Load left annotations
    all_object_labels = load_left_annotations(left_scene_dir)
    if not all_object_labels:
        print(f"  [SKIP] no annotations in {left_scene_dir}")
        return None, None

    inW_L, inH_L = infer_wh(left_bag,  LEFT_TOPIC)
    inW_R, inH_R = infer_wh(right_bag, RIGHT_TOPIC)
    print(f"  left  camera: {inW_L}×{inH_L}  right camera: {inW_R}×{inH_R}")

    # ── Pass 1: stream left bag to define time windows ────────────────────
    # We collect win_starts (one per 50ms window) and left voxels together.
    win_starts   = []
    left_voxels  = []
    label_list   = []

    bag = rosbag.Bag(left_bag)
    first_t = None
    win_start = None
    win_end   = None
    voxel     = None

    def flush(center_t):
        v = np.clip(voxel, -127, 127).astype(np.int8)
        left_voxels.append(v)
        win_starts.append(win_start)
        label_list.append(labels_at(all_object_labels, center_t, outW, outH, max_dt_ns))

    for _, msg, _ in bag.read_messages(topics=[LEFT_TOPIC]):
        for e in msg.events:
            t = event_time_ns(e)
            if first_t is None:
                first_t = t
                win_start = first_t
                win_end   = win_start + dt_ns
                voxel     = np.zeros((bins, outH, outW), dtype=np.int16)

            while t >= win_end:
                flush((win_start + win_end) // 2)
                win_start = win_end
                win_end   = win_start + dt_ns
                voxel     = np.zeros((bins, outH, outW), dtype=np.int16)

            x, y = int(e.x), int(e.y)
            p = 1 if int(e.polarity) > 0 else -1
            if inW_L != outW or inH_L != outH:
                x, y = scale_xy(x, y, inW_L, inH_L, outW, outH)
            if not (0 <= x < outW and 0 <= y < outH):
                continue
            rel = t - win_start
            b   = min(int(rel * bins // dt_ns), bins - 1)
            voxel[b, y, x] += p

    if voxel is not None:
        flush((win_start + win_end) // 2)
    bag.close()

    if not left_voxels:
        print("  [SKIP] no left frames")
        return None, None

    win_starts_arr = np.array(win_starts, dtype=np.int64)
    left_arr = np.stack(left_voxels, axis=0)   # (N, 5, H, W)

    # ── Pass 2: right bag → same windows ─────────────────────────────────
    right_arr = bag_to_voxels(
        right_bag, RIGHT_TOPIC, win_starts_arr, dt_ns, bins,
        inW_R, inH_R, outW, outH,
    )   # (N, 5, H, W)

    # ── Concatenate along channel axis ────────────────────────────────────
    N = min(len(left_arr), len(right_arr))
    stereo = np.concatenate([left_arr[:N], right_arr[:N]], axis=1)  # (N, 10, H, W)

    labels_arr = np.array(label_list[:N], dtype=object)

    # Keep only frames that have at least one label
    keep = np.array([la.shape[0] > 0 for la in labels_arr])
    if not keep.any():
        print("  [SKIP] no labeled windows")
        return None, None

    print(f"  frames={N}  labeled={keep.sum()}  stereo_shape={stereo.shape}")
    return stereo[keep], labels_arr[keep]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left_scenes",  nargs="+", required=True)
    ap.add_argument("--right_scenes", nargs="+", required=True,
                    help="Must be in same order and count as --left_scenes")
    ap.add_argument("--out_root",  required=True)
    ap.add_argument("--split",     required=True, choices=["train", "val", "test"])
    ap.add_argument("--dt_ms",  type=float, default=50.0)
    ap.add_argument("--bins",   type=int,   default=5)
    ap.add_argument("--outW",   type=int,   default=640)
    ap.add_argument("--outH",   type=int,   default=480)
    ap.add_argument("--label_max_dt_ms", type=float, default=None,
                    help="optional: if nearest label farther than this (ms), treat as no label")
    args = ap.parse_args()

    assert len(args.left_scenes) == len(args.right_scenes), \
        "left_scenes and right_scenes must have the same length"

    out_img_dir = os.path.join(args.out_root, "images", args.split)
    out_lbl_dir = os.path.join(args.out_root, "labels", args.split)
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    # Pair left+right scenes and sort numerically by scene number
    pairs = list(zip(args.left_scenes, args.right_scenes))
    pairs.sort(key=lambda p: int(re.search(r"(\d+)", os.path.basename(p[0])).group(1)))

    h5_path = os.path.join(out_img_dir, f"mtevent_{args.split}.h5")
    channels = args.bins * 2
    frame_shape = (channels, args.outH, args.outW)
    total = 0

    # Write H5 incrementally — avoids accumulating 40+ GB in RAM
    with h5py.File(h5_path, "w") as hf:
        ds = hf.create_dataset(
            "1mp",
            shape=(0,) + frame_shape,
            maxshape=(None,) + frame_shape,
            dtype=np.int8,
            chunks=(1,) + frame_shape,
        )

        for left_dir, right_dir in pairs:
            base = os.path.basename(left_dir.rstrip("/"))
            sid  = int(re.search(r"(\d+)", base).group(1))
            print(f"\n[scene {sid}]  {left_dir}", flush=True)

            max_dt_ns = int(args.label_max_dt_ms * 1e6)
            stereo, labels = process_scene(
                left_dir, right_dir, args.dt_ms, args.bins, args.outW, args.outH, max_dt_ns
            )
            if stereo is None:
                continue

            lbl_path = os.path.join(out_lbl_dir, f"scene_{sid:06d}.npy")
            np.save(lbl_path, labels, allow_pickle=True)
            print(f"  [OK] labels → {lbl_path}", flush=True)

            # Append frames to H5
            n = stereo.shape[0]
            ds.resize(total + n, axis=0)
            ds[total:total + n] = stereo
            total += n
            print(f"  [OK] H5 total so far: {total} frames", flush=True)

    print(f"\n[OK] {h5_path}  total={total}  shape=({total},{channels},{args.outH},{args.outW})")


if __name__ == "__main__":
    main()

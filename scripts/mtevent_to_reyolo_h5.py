#!/usr/bin/env python3
import os
import json
import math
import glob
import argparse
import numpy as np

import rosbag
import h5py
import re

# Optional compression (repo uses hdf5plugin sometimes)
try:
    import hdf5plugin
    HAVE_BLOSC = True
except Exception:
    HAVE_BLOSC = False


def read_jsonl(path):
    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    items.sort(key=lambda d: int(d["timestamp"]))
    return items


def nearest_label(labels, t_ns):
    # labels sorted by timestamp
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
        if tm < t_ns:
            lo = mid + 1
        elif tm > t_ns:
            hi = mid - 1
        else:
            return labels[mid]

    cand = []
    if 0 <= hi < len(labels): cand.append(labels[hi])
    if 0 <= lo < len(labels): cand.append(labels[lo])
    return min(cand, key=lambda d: abs(d["timestamp"] - t_ns)) if cand else None


def to_yolo_array(b, W, H, cls=0):
    """b is dict with xmin/xmax/ymin/ymax in pixels. returns (N,5)."""
    if b is None:
        return np.zeros((0, 5), dtype=np.float32)

    xmin, xmax = float(b["xmin"]), float(b["xmax"])
    ymin, ymax = float(b["ymin"]), float(b["ymax"])

    # clip
    xmin = max(0.0, min(xmin, W - 1.0))
    xmax = max(0.0, min(xmax, W - 1.0))
    ymin = max(0.0, min(ymin, H - 1.0))
    ymax = max(0.0, min(ymax, H - 1.0))

    if xmax <= xmin or ymax <= ymin:
        return np.zeros((0, 5), dtype=np.float32)

    cx = ((xmin + xmax) / 2.0) / W
    cy = ((ymin + ymax) / 2.0) / H
    w  = (xmax - xmin) / W
    h  = (ymax - ymin) / H

    return np.array([[cls, cx, cy, w, h]], dtype=np.float32)


def infer_wh_from_bag(bag_path, topic, max_msgs=3):
    max_x = 0
    max_y = 0
    bag = rosbag.Bag(bag_path)
    n = 0
    for _, msg, _ in bag.read_messages(topics=[topic]):
        for e in msg.events:
            x = int(e.x); y = int(e.y)
            if x > max_x: max_x = x
            if y > max_y: max_y = y
        n += 1
        if n >= max_msgs:
            break
    bag.close()
    return max_x + 1, max_y + 1


def downscale_xy(x, y, inW, inH, outW, outH):
    sx = outW / float(inW)
    sy = outH / float(inH)
    return int(x * sx), int(y * sy)


def downscale_bbox(b, inW, inH, outW, outH):
    sx = outW / float(inW)
    sy = outH / float(inH)
    return {
        "timestamp": b["timestamp"],
        "xmin": float(b["xmin"]) * sx,
        "xmax": float(b["xmax"]) * sx,
        "ymin": float(b["ymin"]) * sy,
        "ymax": float(b["ymax"]) * sy,
    }


def build_frames_from_bag(bag_path, topic, labels, dt_ms, bins, inW, inH, outW, outH,
                          clip_abs=127, label_max_dt_ms=None):
    """
    Returns:
      frames: np.ndarray (L, bins, outH, outW) int8
      labels_obj: np.ndarray (L,) object where each element is (N,5) float32
    """
    dt_ns = int(dt_ms * 1e6)

    # Gather all events into windows streaming
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

        # Quantize / clip to int8
        v = np.clip(voxel, -clip_abs, clip_abs).astype(np.int8)
        frames.append(v)

        # nearest label (optional threshold)
        lab = nearest_label(labels, center_t)
        if label_max_dt_ms is not None and lab is not None:
            if abs(int(lab["timestamp"]) - int(center_t)) > int(label_max_dt_ms * 1e6):
                lab = None

        if lab is not None and (inW != outW or inH != outH):
            lab = downscale_bbox(lab, inW, inH, outW, outH)

        labels_obj.append(to_yolo_array(lab, outW, outH, cls=0))

    for _, msg, _ in bag.read_messages(topics=[topic]):
        for e in msg.events:
            # event timestamp (ns)
            ts = e.ts
            if hasattr(ts, "to_nsec"):
                t_ns = int(ts.to_nsec())
            else:
                t_ns = int(ts.secs) * 1_000_000_000 + int(ts.nsecs)

            x = int(e.x); y = int(e.y)
            p = 1 if int(e.polarity) > 0 else -1

            if first_t is None:
                first_t = t_ns
                win_start = first_t
                win_end = win_start + dt_ns
                voxel = np.zeros((bins, outH, outW), dtype=np.int16)

            # advance windows
            while t_ns >= win_end:
                flush_window((win_start + win_end) // 2)
                win_start = win_end
                win_end = win_start + dt_ns
                voxel = np.zeros((bins, outH, outW), dtype=np.int16)

            # map to output resolution
            if inW != outW or inH != outH:
                xo, yo = downscale_xy(x, y, inW, inH, outW, outH)
            else:
                xo, yo = x, y

            if xo < 0 or xo >= outW or yo < 0 or yo >= outH:
                continue

            rel = t_ns - win_start
            b = int((rel * bins) // dt_ns)
            if b < 0: b = 0
            if b >= bins: b = bins - 1

            voxel[b, yo, xo] += p

    # flush last
    if voxel is not None:
        flush_window((win_start + win_end) // 2)

    bag.close()

    frames = np.stack(frames, axis=0) if frames else np.zeros((0, bins, outH, outW), dtype=np.int8)
    labels_obj = np.array(labels_obj, dtype=object)
    return frames, labels_obj


def write_h5(path, data, key="1mp"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as hf:
        if HAVE_BLOSC:
            hf.create_dataset(key, data=data, chunks=True, maxshape=(None,) + data.shape[1:],
                              **hdf5plugin.Blosc(cname="zstd"))
        else:
            hf.create_dataset(key, data=data, chunks=True, maxshape=(None,) + data.shape[1:])
    print(f"[OK] wrote {path}  {key} shape={data.shape} dtype={data.dtype}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True, help="scene dirs, e.g. /home/loki/datasets/MTevent/scene75 ...")
    ap.add_argument("--out_root", required=True, help="e.g. preprocessed_datasets/vtei_mtevent_50ms_5bin")
    ap.add_argument("--split", required=True, choices=["train","val","test"])
    ap.add_argument("--dt_ms", type=float, default=50.0)
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--topic", default="/dvxplorer_left/events")
    ap.add_argument("--bag_name", default="left.bag")
    ap.add_argument("--labels_rel", default="annotation_human/human_ec_left_bounding_box_labels_2d.json")
    ap.add_argument("--outW", type=int, default=304, help="output width (GEN1=304)")
    ap.add_argument("--outH", type=int, default=240, help="output height (GEN1=240)")
    ap.add_argument("--label_max_dt_ms", type=float, default=None,
                    help="optional: if nearest label farther than this, treat as no label")
    args = ap.parse_args()

    out_img_dir = os.path.join(args.out_root, "images", args.split)
    out_lbl_dir = os.path.join(args.out_root, "labels", args.split)
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    all_frames = []
    # IMPORTANT: label file order defines H5 indexing. We will write labels named scene###.npy
    # and concatenate frames in the same sorted order of those names.
    scene_pairs = []
    for sd in args.scenes:
        # Extract a stable numeric scene id if present
        base = os.path.basename(sd.rstrip("/"))
        scene_pairs.append((base, sd))
    scene_pairs.sort(key=lambda x: x[0])  # same order as glob(sorted)

    for base, scene_dir in scene_pairs:
        bag_path = os.path.join(scene_dir, args.bag_name)
        labels_path = os.path.join(scene_dir, args.labels_rel)
        # skip if either file is missing
        if not os.path.isfile(bag_path):
            print(f"[WARN] skipping {base} because missing bag at {bag_path}")
            continue
        if not os.path.isfile(labels_path):
            print(f"[WARN] skipping {base} because missing labels at {labels_path}")
            continue

        labels = read_jsonl(labels_path)

        inW, inH = infer_wh_from_bag(bag_path, args.topic, max_msgs=3)
        print(f"[INFO] {base}: inferred input W,H = {inW},{inH}")

        frames, labels_obj = build_frames_from_bag(
            bag_path=bag_path,
            topic=args.topic,
            labels=labels,
            dt_ms=args.dt_ms,
            bins=args.bins,
            inW=inW, inH=inH,
            outW=args.outW, outH=args.outH,
            label_max_dt_ms=args.label_max_dt_ms
        )
        matched = sum(1 for a in labels_obj if hasattr(a, "shape") and a.shape[0] > 0)
        if matched == 0:
            print(f"[SKIP] {base}: no labeled windows (matched=0)")
            continue

        # Save labels for this scene
        m = re.search(r"(\d+)", base)
        sid = int(m.group(1)) if m else 0
        lbl_path = os.path.join(out_lbl_dir, f"scene_{sid:06d}.npy")
        np.save(lbl_path, labels_obj, allow_pickle=True)
        print(f"[OK] wrote {lbl_path}  len={len(labels_obj)}")

        all_frames.append(frames)

    # Concatenate all scenes into one big H5 (the loader opens only the first .h5)
    if all_frames:
        big = np.concatenate(all_frames, axis=0)
    else:
        big = np.zeros((0, args.bins, args.outH, args.outW), dtype=np.int8)

    h5_path = os.path.join(out_img_dir, f"mtevent_{args.split}.h5")
    write_h5(h5_path, big, key="1mp")


if __name__ == "__main__":
    main()


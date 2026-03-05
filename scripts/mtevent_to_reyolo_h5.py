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


# ── Multi-class mapping (MR6D dataset) ──────────────────────────────────────────
# MR6D object IDs 1..16 → class IDs 0..15; human → class 16
CLASS_NAMES = [
    "wooden_pallet",              # 0  ← MR6D1
    "small_klt",                  # 1  ← MR6D2
    "big_klt",                    # 2  ← MR6D3
    "blue_klt",                   # 3  ← MR6D4
    "Amazon_basics_luggage",      # 4  ← MR6D5
    "IKEA_Dammang_bin_with_lid",  # 5  ← MR6D6
    "IKEA_vesken_trolley",        # 6  ← MR6D7
    "IKEA_sortera_waste_sorting_bin",  # 7  ← MR6D8
    "IKEA_Drona_grey",            # 8  ← MR6D9
    "IKEA_Drona_blue",            # 9  ← MR6D10
    "IKEA_KNALLIG_wooden_box",    # 10 ← MR6D11
    "IKEA_MOPPE_mini_drawer",     # 11 ← MR6D12
    "IKEA_LABBSAL_basket",        # 12 ← MR6D13
    "IKEA_IVAR_box_on_wheels",    # 13 ← MR6D14
    "IKEA_SKUBB_storage_case",    # 14 ← MR6D15
    "IKEA_SAMLA_transparent_box", # 15 ← MR6D16
    "human",                      # 16
]
HUMAN_CLASS_ID = 16


def mr6d_number_to_class_id(n):
    """MR6D object number (1-16) → class id (0-15)."""
    return n - 1


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
    """Return the label entry closest to *t_ns* (binary search). Returns None if *labels* is empty."""
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


def nearest_labels_multi(all_object_labels, t_ns, label_max_dt_ns=None):
    """
    For every (class_id, sorted_label_list) pair, find the nearest bbox to *t_ns*.

    Returns a list of (class_id, label_dict) tuples that are within the optional
    *label_max_dt_ns* threshold.
    """
    results = []
    for cls_id, labels in all_object_labels:
        lab = nearest_label(labels, t_ns)
        if lab is None:
            continue
        if label_max_dt_ns is not None:
            if abs(int(lab["timestamp"]) - int(t_ns)) > label_max_dt_ns:
                continue
        results.append((cls_id, lab))
    return results


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


def multi_to_yolo_array(matched_labels, W, H, inW=None, inH=None, outW=None, outH=None):
    """
    Convert a list of (class_id, bbox_dict) pairs returned by *nearest_labels_multi*
    into a single (N, 5) float32 array with rows [cls, cx, cy, w, h] in normalised coords.

    If inW/inH differ from outW/outH the bboxes are rescaled accordingly.
    """
    if not matched_labels:
        return np.zeros((0, 5), dtype=np.float32)

    rows = []
    for cls_id, lab in matched_labels:
        if inW is not None and (inW != outW or inH != outH):
            lab = downscale_bbox(lab, inW, inH, outW, outH)
        arr = to_yolo_array(lab, W, H, cls=cls_id)
        if arr.shape[0] > 0:
            rows.append(arr)

    if rows:
        return np.concatenate(rows, axis=0)
    return np.zeros((0, 5), dtype=np.float32)


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


def build_frames_from_bag(bag_path, topic, all_object_labels, dt_ms, bins, inW, inH, outW, outH,
                          clip_abs=127, label_max_dt_ms=None, split_polarity=False):
    """
    Args:
      all_object_labels: list of (class_id, sorted_label_list) tuples – one entry per object/class.
    Returns:
      frames: np.ndarray (L, bins, outH, outW) int8
      labels_obj: np.ndarray (L,) object where each element is (N,5) float32
    """
    dt_ns = int(dt_ms * 1e6)
    label_max_dt_ns = int(label_max_dt_ms * 1e6) if label_max_dt_ms is not None else None
    channels = bins * 2 if split_polarity else bins

    need_rescale = (inW != outW or inH != outH)

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

        # Collect nearest bbox for every object/class
        matched = nearest_labels_multi(all_object_labels, center_t, label_max_dt_ns)
        labels_obj.append(
            multi_to_yolo_array(matched, outW, outH,
                                inW=inW if need_rescale else None,
                                inH=inH if need_rescale else None,
                                outW=outW, outH=outH)
        )

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
                voxel = np.zeros((channels, outH, outW), dtype=np.int16)

            # advance windows
            while t_ns >= win_end:
                flush_window((win_start + win_end) // 2)
                win_start = win_end
                win_end = win_start + dt_ns
                voxel = np.zeros((channels, outH, outW), dtype=np.int16)

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

            if split_polarity:
                ch = 2 * b if p > 0 else 2 * b + 1
                voxel[ch, yo, xo] += 1
            else:
                voxel[b, yo, xo] += p

    # flush last
    if voxel is not None:
        flush_window((win_start + win_end) // 2)

    bag.close()

    frames = np.stack(frames, axis=0) if frames else np.zeros((0, channels, outH, outW), dtype=np.int8)
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
    ap.add_argument("--split_polarity", action="store_true",
                    help="If set, output channels=bins*2 with separate pos/neg per bin.")
    ap.add_argument("--topic", default="/dvxplorer_left/events")
    ap.add_argument("--bag_name", default="left.bag")
    ap.add_argument("--camera", default="ec_left",
                    help="Camera prefix used in annotation filenames, e.g. ec_left or ec_right")
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
        if not os.path.isfile(bag_path):
            print(f"[WARN] skipping {base} because missing bag at {bag_path}")
            continue

        # ── Collect ALL annotation JSON files for the chosen camera ──────────
        all_object_labels = []   # list of (class_id, sorted_label_list)

        # 1) MR6D object annotations: {camera}_MR6D{N}_bounding_box_labels_2d.json
        #    Search in scene dir directly AND in annotation/ subdir (both layouts exist)
        mr6d_search_dirs = [scene_dir, os.path.join(scene_dir, "annotation")]
        mr6d_files = []
        for search_dir in mr6d_search_dirs:
            mr6d_pattern = os.path.join(search_dir, f"{args.camera}_MR6D*_bounding_box_labels_2d.json")
            mr6d_files.extend(glob.glob(mr6d_pattern))
        mr6d_files = sorted(set(mr6d_files))  # deduplicate & sort
        for jf in mr6d_files:
            fname = os.path.basename(jf)
            m_obj = re.search(r"MR6D(\d+)", fname)
            if m_obj is None:
                print(f"[WARN]  cannot parse MR6D number from {fname}, skipping")
                continue
            mr6d_num = int(m_obj.group(1))
            cls_id = mr6d_number_to_class_id(mr6d_num)  # N-1
            if cls_id < 0 or cls_id >= HUMAN_CLASS_ID:
                print(f"[WARN]  MR6D{mr6d_num} → cls {cls_id} out of range, skipping {fname}")
                continue
            labels_list = read_jsonl(jf)
            if labels_list:
                all_object_labels.append((cls_id, labels_list))
                print(f"[INFO]   {base}: loaded {len(labels_list)} bboxes from {fname}  (cls={cls_id} {CLASS_NAMES[cls_id]})")

        # 2) Human annotation: human_{camera}_bounding_box_labels_2d.json
        #    Search in scene dir directly AND in annotation_human/ subdir
        human_candidates = [
            os.path.join(scene_dir, f"human_{args.camera}_bounding_box_labels_2d.json"),
            os.path.join(scene_dir, "annotation_human", f"human_{args.camera}_bounding_box_labels_2d.json"),
        ]
        human_json = None
        for hc in human_candidates:
            if os.path.isfile(hc):
                human_json = hc
                break
        if human_json is not None:
            human_labels = read_jsonl(human_json)
            if human_labels:
                all_object_labels.append((HUMAN_CLASS_ID, human_labels))
                print(f"[INFO]   {base}: loaded {len(human_labels)} bboxes from human annotation  (cls={HUMAN_CLASS_ID} human)")
        else:
            print(f"[INFO]   {base}: no human annotation found (searched {human_candidates})")

        if not all_object_labels:
            print(f"[SKIP] {base}: no annotation files found for camera={args.camera}")
            continue

        inW, inH = infer_wh_from_bag(bag_path, args.topic, max_msgs=3)
        print(f"[INFO] {base}: inferred input W,H = {inW},{inH}")

        frames, labels_obj = build_frames_from_bag(
            bag_path=bag_path,
            topic=args.topic,
            all_object_labels=all_object_labels,
            dt_ms=args.dt_ms,
            bins=args.bins,
            inW=inW, inH=inH,
            outW=args.outW, outH=args.outH,
            label_max_dt_ms=args.label_max_dt_ms,
            split_polarity=args.split_polarity,
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
        channels = args.bins * 2 if args.split_polarity else args.bins
        big = np.zeros((0, channels, args.outH, args.outW), dtype=np.int8)

    h5_path = os.path.join(out_img_dir, f"mtevent_{args.split}.h5")
    write_h5(h5_path, big, key="1mp")


if __name__ == "__main__":
    main()


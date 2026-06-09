#!/usr/bin/env python3
"""
Extract RGB frames + annotations from MTevent scenes into MTevent_extracted_rgb/.

Handles both unzipped scene directories and .zip archives.
For zipped scenes: extracts all.bag to a temp dir, processes, then deletes it.

Usage (run with system python3 — needs rosbag):
    python3 scripts/extract_rgb_mtevent.py \
        --mtevent_root /mnt/2tb/MTevent \
        --out_root     /mnt/2tb/MTevent_extracted_rgb \
        --tmp_dir      /mnt/2tb/.tmp_rgb_extract \
        [--scenes 1 2 3 ...]
"""
import os
import re
import sys
import glob
import json
import shutil
import zipfile
import argparse
import tempfile

import numpy as np
import rosbag
from PIL import Image

OUT_W, OUT_H = 1024, 768   # resize from 2048×1536


RGB_TOPICS = ["/rgb/image_raw", "/camera/image_raw"]


def extract_frames_from_bag(bag_path, out_dir):
    """Extract RGB frames to out_dir as JPEG, named by timestamp_ns.
    Tries /rgb/image_raw first, falls back to /camera/image_raw."""
    os.makedirs(out_dir, exist_ok=True)
    bag = rosbag.Bag(bag_path)
    # Auto-detect which topic this bag uses
    available = set(bag.get_type_and_topic_info().topics.keys())
    topic = next((t for t in RGB_TOPICS if t in available), None)
    if topic is None:
        bag.close()
        return 0
    count = 0
    for _, msg, _ in bag.read_messages(topics=[topic]):
        h, w = msg.height, msg.width
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, -1)
        # RGBA8 → RGB, then resize
        rgb = raw[:, :, :3]
        img = Image.fromarray(rgb).resize((OUT_W, OUT_H), Image.BILINEAR)
        ts = msg.header.stamp.secs * 1_000_000_000 + msg.header.stamp.nsecs
        img.save(os.path.join(out_dir, f"{ts}.jpg"), quality=90)
        count += 1
    bag.close()
    return count


def copy_rgb_annotations(src_dir, dst_dir):
    """Copy rgb_* and human_rgb_* 2D bbox annotation JSONs."""
    copied = []
    annotation_dirs = [
        os.path.join(src_dir, "annotation"),
        os.path.join(src_dir, "annotation_human"),
    ]
    for ann_dir in annotation_dirs:
        if not os.path.isdir(ann_dir):
            continue
        for jf in glob.glob(os.path.join(ann_dir, "rgb_*_bounding_box_labels_2d.json")):
            dst = os.path.join(dst_dir, os.path.basename(jf))
            shutil.copy2(jf, dst)
            copied.append(os.path.basename(jf))
        for jf in glob.glob(os.path.join(ann_dir, "human_rgb_bounding_box_labels_2d.json")):
            dst = os.path.join(dst_dir, os.path.basename(jf))
            shutil.copy2(jf, dst)
            copied.append(os.path.basename(jf))
    return copied


def copy_rgb_annotations_from_zip(zf, dst_dir):
    """Extract rgb annotation JSONs directly from a ZipFile object into memory."""
    copied = []
    for name in zf.namelist():
        basename = os.path.basename(name)
        if not name.endswith("_bounding_box_labels_2d.json"):
            continue
        if not (basename.startswith("rgb_") or basename.startswith("human_rgb_")):
            continue
        data = zf.read(name)
        dst = os.path.join(dst_dir, basename)
        with open(dst, "wb") as f:
            f.write(data)
        copied.append(basename)
    return copied


def process_scene_dir(scene_dir, out_scene_dir):
    """Process an already-unzipped scene directory."""
    bag_path = os.path.join(scene_dir, "all.bag")
    if not os.path.isfile(bag_path):
        print(f"  [WARN] no all.bag found at {bag_path}, skipping")
        return False
    rgb_out = os.path.join(out_scene_dir, "rgb")
    n = extract_frames_from_bag(bag_path, rgb_out)
    copied = copy_rgb_annotations(scene_dir, out_scene_dir)
    print(f"  [OK] {n} frames, annotations: {copied}")
    return True


def process_scene_zip(zip_path, out_scene_dir, tmp_dir):
    """Process a zipped scene: extract all.bag to tmp, process, delete."""
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_bag = os.path.join(tmp_dir, "all.bag")

    try:
        # 1. Copy annotation JSONs directly from zip (small, in-memory)
        with zipfile.ZipFile(zip_path, "r") as zf:
            copied = copy_rgb_annotations_from_zip(zf, out_scene_dir)
            # 2. Extract all.bag to tmp
            all_bag_entry = next((n for n in zf.namelist() if os.path.basename(n) == "all.bag"), None)
            if all_bag_entry is None:
                print(f"  [WARN] no all.bag in {zip_path}, skipping")
                return False
            print(f"  extracting all.bag ({zf.getinfo(all_bag_entry).file_size // 1024 // 1024} MB) ...")
            with zf.open(all_bag_entry) as src, open(tmp_bag, "wb") as dst:
                shutil.copyfileobj(src, dst)

        # 3. Extract frames
        rgb_out = os.path.join(out_scene_dir, "rgb")
        n = extract_frames_from_bag(tmp_bag, rgb_out)
        print(f"  [OK] {n} frames, annotations: {copied}")
        return True

    finally:
        if os.path.exists(tmp_bag):
            os.remove(tmp_bag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtevent_root", default="/mnt/2tb/MTevent")
    ap.add_argument("--out_root",     default="/mnt/2tb/MTevent_extracted_rgb")
    ap.add_argument("--tmp_dir",      default="/mnt/2tb/.tmp_rgb_extract")
    ap.add_argument("--scenes", nargs="+", type=int, default=None,
                    help="Scene numbers to process (default: all)")
    args = ap.parse_args()

    # Discover scenes
    if args.scenes:
        scene_nums = args.scenes
    else:
        zips = glob.glob(os.path.join(args.mtevent_root, "scene*.zip"))
        dirs = [d for d in glob.glob(os.path.join(args.mtevent_root, "scene*"))
                if os.path.isdir(d)]
        nums = set()
        for p in zips + dirs:
            m = re.search(r"scene(\d+)", os.path.basename(p))
            if m:
                nums.add(int(m.group(1)))
        scene_nums = sorted(nums)

    print(f"Processing {len(scene_nums)} scenes: {scene_nums[:5]}{'...' if len(scene_nums)>5 else ''}")

    ok = 0
    for n in scene_nums:
        out_scene_dir = os.path.join(args.out_root, f"scene{n}")
        os.makedirs(out_scene_dir, exist_ok=True)

        # Skip if already done
        rgb_dir = os.path.join(out_scene_dir, "rgb")
        if os.path.isdir(rgb_dir) and len(os.listdir(rgb_dir)) > 10:
            print(f"[SKIP] scene{n}: already extracted ({len(os.listdir(rgb_dir))} frames)")
            ok += 1
            continue

        print(f"[scene{n}]")
        scene_dir = os.path.join(args.mtevent_root, f"scene{n}")
        zip_path  = os.path.join(args.mtevent_root, f"scene{n}.zip")

        if os.path.isdir(scene_dir):
            success = process_scene_dir(scene_dir, out_scene_dir)
        elif os.path.isfile(zip_path):
            success = process_scene_zip(zip_path, out_scene_dir, args.tmp_dir)
        else:
            print(f"  [WARN] neither scene dir nor zip found for scene{n}")
            success = False

        if success:
            ok += 1

    # Clean up tmp dir
    if os.path.isdir(args.tmp_dir):
        shutil.rmtree(args.tmp_dir, ignore_errors=True)

    print(f"\n[DONE] {ok}/{len(scene_nums)} scenes extracted → {args.out_root}")


if __name__ == "__main__":
    main()

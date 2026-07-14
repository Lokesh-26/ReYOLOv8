#!/usr/bin/env python3
"""
Convert MTevent_extracted_rgb/ scenes to a YOLO image+label dataset.

Matches each RGB frame (by timestamp) to the nearest annotation entry,
outputs JPEG images and YOLO .txt label files.

Same train/val/test split as the event model (vtei_mtevent_640x480.yaml).

Usage (run with system python3):
    python3 scripts/rgb_to_yolo.py \
        --scenes_root /mnt/2tb/MTevent_extracted_rgb \
        --out_root    preprocessed_datasets/vtei_rgb_1024x768 \
        --label_max_dt_ms 50
"""
import os
import re
import glob
import json
import shutil
import argparse

import numpy as np

# ── same 17-class mapping as event pipeline ──────────────────────────────────
HUMAN_CLASS_ID = 16
IMG_W, IMG_H = 1024, 768   # must match extract_rgb_mtevent.py output

# Scenes used for each split (mirrors vtei_mtevent_640x480.yaml split)
VAL_SCENES  = {3,4,5,6,7,8,10,14,21,23,26,33,35}
TEST_SCENES = {1, 2}


def split_for_scene(n):
    if n in TEST_SCENES: return "test"
    if n in VAL_SCENES:  return "val"
    return "train"


def read_jsonl(path):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    items.sort(key=lambda d: int(d["timestamp"]))
    return items


def nearest_label(labels, t_ns, max_dt_ns=None):
    if not labels:
        return None
    if t_ns <= int(labels[0]["timestamp"]):
        lab = labels[0]
    elif t_ns >= int(labels[-1]["timestamp"]):
        lab = labels[-1]
    else:
        lo, hi = 0, len(labels) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            tm = int(labels[mid]["timestamp"])
            if tm < t_ns: lo = mid + 1
            elif tm > t_ns: hi = mid - 1
            else: return labels[mid]
        cands = [labels[i] for i in (hi, lo) if 0 <= i < len(labels)]
        lab = min(cands, key=lambda d: abs(int(d["timestamp"]) - t_ns))
    if max_dt_ns and abs(int(lab["timestamp"]) - int(t_ns)) > max_dt_ns:
        return None
    return lab


def bbox_to_yolo(b, cls):
    """Convert pixel bbox dict to normalised YOLO row [cls cx cy w h]."""
    xmin, xmax = float(b["xmin"]), float(b["xmax"])
    ymin, ymax = float(b["ymin"]), float(b["ymax"])
    # annotations are in original 2048×1536 space; normalise by that
    orig_w, orig_h = 2048.0, 1536.0
    cx = ((xmin + xmax) / 2) / orig_w
    cy = ((ymin + ymax) / 2) / orig_h
    w  = (xmax - xmin) / orig_w
    h  = (ymax - ymin) / orig_h
    # Clamp to [0, 1] — some annotations extend slightly beyond image boundary
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w  = max(0.0, min(1.0, w))
    h  = max(0.0, min(1.0, h))
    if w <= 0 or h <= 0:
        return None
    return f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes_root",    default="/mnt/2tb/MTevent_extracted_rgb")
    ap.add_argument("--out_root",       default="preprocessed_datasets/vtei_rgb_1024x768")
    ap.add_argument("--label_max_dt_ms", type=float, default=50.0)
    args = ap.parse_args()

    max_dt_ns = int(args.label_max_dt_ms * 1e6)

    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(args.out_root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.out_root, "labels", split), exist_ok=True)

    scene_dirs = sorted(
        glob.glob(os.path.join(args.scenes_root, "scene*")),
        key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1))
    )

    total_imgs = total_labeled = 0

    for scene_dir in scene_dirs:
        m = re.search(r"scene(\d+)", os.path.basename(scene_dir))
        if not m:
            continue
        scene_num = int(m.group(1))
        split = split_for_scene(scene_num)
        rgb_dir = os.path.join(scene_dir, "rgb")
        if not os.path.isdir(rgb_dir):
            print(f"[SKIP] scene{scene_num}: no rgb/ dir")
            continue

        # Load all annotation files for this scene
        all_object_labels = []  # list of (class_id, sorted_label_list)

        # MR6D objects
        for jf in glob.glob(os.path.join(scene_dir, "rgb_MR6D*_bounding_box_labels_2d.json")):
            m2 = re.search(r"MR6D(\d+)", os.path.basename(jf))
            if not m2:
                continue
            cls_id = int(m2.group(1)) - 1   # MR6D1→0 … MR6D16→15
            if 0 <= cls_id < HUMAN_CLASS_ID:
                labels = read_jsonl(jf)
                if labels:
                    all_object_labels.append((cls_id, labels))

        # Human
        human_jf = os.path.join(scene_dir, "human_rgb_bounding_box_labels_2d.json")
        if os.path.isfile(human_jf):
            labels = read_jsonl(human_jf)
            if labels:
                all_object_labels.append((HUMAN_CLASS_ID, labels))

        if not all_object_labels:
            print(f"[SKIP] scene{scene_num}: no annotations")
            continue

        frames = sorted(glob.glob(os.path.join(rgb_dir, "*.jpg")))
        labeled = 0
        for jpg_path in frames:
            t_ns = int(os.path.splitext(os.path.basename(jpg_path))[0])

            yolo_rows = []
            for cls_id, label_list in all_object_labels:
                lab = nearest_label(label_list, t_ns, max_dt_ns)
                if lab is None:
                    continue
                row = bbox_to_yolo(lab, cls_id)
                if row:
                    yolo_rows.append(row)

            # Only keep frames that have at least one label
            if not yolo_rows:
                continue

            img_name = f"scene{scene_num:03d}_{t_ns}.jpg"
            lbl_name = f"scene{scene_num:03d}_{t_ns}.txt"

            shutil.copy2(jpg_path, os.path.join(args.out_root, "images", split, img_name))
            with open(os.path.join(args.out_root, "labels", split, lbl_name), "w") as f:
                f.write("\n".join(yolo_rows) + "\n")
            labeled += 1

        total_imgs   += len(frames)
        total_labeled += labeled
        print(f"[scene{scene_num:02d}] split={split}  frames={len(frames)}  labeled={labeled}")

    # Write dataset YAML
    yaml_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "vtei_rgb_1024x768.yaml"
    )
    out_abs = os.path.abspath(args.out_root)
    with open(yaml_path, "w") as f:
        f.write(f"path: {out_abs}\n")
        f.write(f"train: {out_abs}/images/train\n")
        f.write(f"val:   {out_abs}/images/val\n")
        f.write(f"test:  {out_abs}/images/test\n\n")
        f.write("nc: 17\nnames:\n")
        names = [
            "wooden_pallet","small_klt","big_klt","blue_klt","amazon_luggage",
            "ikea_dammang_bin","ikea_vesken_trolley","ikea_sortera_bin",
            "ikea_drona_grey","ikea_drona_blue","ikea_knallig_box",
            "ikea_moppe_drawer","ikea_labbsal_basket","ikea_ivar_box",
            "ikea_skubb_case","ikea_samla_box","human",
        ]
        for name in names:
            f.write(f"  - {name}\n")
    print(f"\n[DONE] {total_labeled}/{total_imgs} labeled frames → {args.out_root}")
    print(f"Dataset config written → {yaml_path}")


if __name__ == "__main__":
    main()

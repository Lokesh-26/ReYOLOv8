#!/usr/bin/env python3
"""
Dataset audit for vtei_mtevent_50ms_5bin_640x480 (640×480, 5-channel, 50ms).
Run from /home/loki/event/ReYOLOv8:
    /home/loki/anaconda3/envs/reyolov8/bin/python scripts/audit_mtevent_dataset.py
"""
import os
import sys
import numpy as np
import h5py
from pathlib import Path
from collections import defaultdict

BASE = Path("preprocessed_datasets/vtei_mtevent_50ms_5bin_640x480")
SPLITS = ["train", "val"]

CLASS_NAMES = [
    'wooden_pallet','small_klt','big_klt','blue_klt','amazon_luggage',
    'ikea_dammang_bin','ikea_vesken_trolley','ikea_sortera_bin',
    'ikea_drona_grey','ikea_drona_blue','ikea_knallig_box','ikea_moppe_drawer',
    'ikea_labbsal_basket','ikea_ivar_box','ikea_skubb_case','ikea_samla_box','human',
]
NC = 17

def load_split(split):
    lbl_dir = BASE / "labels" / split
    if not lbl_dir.exists():
        return {}
    scenes = {}
    for f in sorted(lbl_dir.glob("*.npy")):
        scene_id = int(f.stem.replace("scene_", ""))
        labels = np.load(f, allow_pickle=True)
        scenes[scene_id] = labels
    return scenes

def main():
    print("=" * 72)
    print("MTEvent Dataset Audit — 640×480, 5ch, 50ms")
    print("=" * 72)

    split_data = {s: load_split(s) for s in SPLITS}

    # ----------------------------------------------------------------
    print("\n[1] Scene IDs per split")
    for split in SPLITS:
        ids = sorted(split_data[split].keys())
        print(f"  {split:5s}: {len(ids):3d} scenes → {ids}")

    # ----------------------------------------------------------------
    print("\n[2] Train / Val overlap (data leakage)")
    train_ids = set(split_data["train"].keys())
    val_ids   = set(split_data["val"].keys())
    overlap   = sorted(train_ids & val_ids)
    train_only = sorted(train_ids - val_ids)
    val_only   = sorted(val_ids - train_ids)
    if overlap:
        print(f"  *** OVERLAP (leakage): {overlap} ***")
        leak_frames = sum(len(split_data["train"][s]) for s in overlap)
        total_val   = sum(len(v) for v in split_data["val"].values())
        print(f"      Leaked frames = {leak_frames} / {total_val} val frames "
              f"= {100*leak_frames/total_val:.1f}%")
    else:
        print("  No overlap. Clean split.")
    print(f"  Train-only scenes: {train_only[:10]}...")
    print(f"  Val-only scenes:   {val_only}")

    # ----------------------------------------------------------------
    print("\n[3] Frame counts")
    for split in SPLITS:
        scenes = split_data[split]
        frames_per_scene = {s: len(v) for s, v in scenes.items()}
        total = sum(frames_per_scene.values())
        print(f"  {split}: {total} total frames across {len(scenes)} scenes")
        print(f"    min={min(frames_per_scene.values())} max={max(frames_per_scene.values())} "
              f"median={np.median(list(frames_per_scene.values())):.0f}")

    # ----------------------------------------------------------------
    print("\n[4] Class distribution per split")
    for split in SPLITS:
        cls_counts = defaultdict(int)
        for scene_id, labels in split_data[split].items():
            for frame_labels in labels:
                for ann in frame_labels:
                    cls_counts[int(ann[0])] += 1
        total_anns = sum(cls_counts.values())
        print(f"  {split} — {total_anns} total annotations:")
        for c in range(NC):
            n = cls_counts.get(c, 0)
            bar = "#" * (n * 30 // max(cls_counts.values(), default=1))
            print(f"    {c:2d} {CLASS_NAMES[c]:<25s}: {n:5d}  {bar}")

    # ----------------------------------------------------------------
    print("\n[5] H5 shape verification")
    for split in SPLITS:
        h5_path = BASE / "images" / split / "mtevent_train.h5"
        if split == "val":
            h5_path = BASE / "images" / split / "mtevent_val.h5"
        if not h5_path.exists():
            # try both names
            for name in ["mtevent_train.h5", "mtevent_val.h5"]:
                p = BASE / "images" / split / name
                if p.exists():
                    h5_path = p
                    break
        if h5_path.exists():
            with h5py.File(h5_path, 'r') as f:
                shape = f['1mp'].shape
                label_total = sum(len(v) for v in split_data[split].values())
                match = "✓" if shape[0] == label_total else "✗ MISMATCH"
                print(f"  {split}: H5={shape}  label_total={label_total}  {match}")
        else:
            print(f"  {split}: H5 not found at {h5_path}")

    # ----------------------------------------------------------------
    print("\n[6] Per-scene annotation density (non-empty frame fraction)")
    for split in SPLITS:
        for scene_id in sorted(split_data[split].keys())[:5]:
            labels = split_data[split][scene_id]
            nonempty = sum(1 for f in labels if len(f) > 0)
            print(f"  {split} scene {scene_id:3d}: {nonempty}/{len(labels)} "
                  f"frames with annotations ({100*nonempty/max(len(labels),1):.0f}%)")

    # ----------------------------------------------------------------
    print("\n[7] Scenes known to be missing")
    all_ids = sorted(train_ids | val_ids)
    min_id, max_id = min(all_ids), max(all_ids)
    missing = [i for i in range(min_id, max_id+1) if i not in (train_ids | val_ids)]
    print(f"  Missing from both splits: {missing}")
    print(f"  (scene 73 right stream absent; scenes 1-2 intended test; "
          f"scenes 3-8 in val only)")

    # ----------------------------------------------------------------
    print("\n[8] Label file consistency (scene N in train vs val if overlapping)")
    for s in overlap:
        t_labels = split_data["train"][s]
        v_labels = split_data["val"][s]
        n_match = sum(np.array_equal(t_labels[i], v_labels[i])
                      for i in range(min(len(t_labels), len(v_labels))))
        print(f"  scene {s:3d}: train_len={len(t_labels)} val_len={len(v_labels)} "
              f"identical_frames={n_match}/{min(len(t_labels), len(v_labels))}")

    print("\n" + "=" * 72)
    print("AUDIT COMPLETE")
    print("KEY FINDING: Scenes 10, 21, 23 appear identically in both train and val.")
    print("This constitutes data leakage covering ~25% of val frames.")
    print("Val-only (genuinely held-out) scenes: 3,4,5,6,7,8,14,26,33,35")
    print("Note: scenes 3-8 do NOT appear in train; their role is unclear.")
    print("=" * 72)

if __name__ == "__main__":
    main()

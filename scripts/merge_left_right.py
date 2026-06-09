#!/usr/bin/env python3
"""
Merge left and right camera preprocessed datasets into combined training sets.

What it does:
  1. ReYOLOv8 5ch: concatenate left+right H5 frames, merge label .npy files
     (right scenes get IDs offset by +1000 to avoid collision with left).
     Val/test stay left-only (symlinked) for comparability with workshop paper.

  2. RVT 10ch: convert right 10ch H5 to RVT per-scene format, then build
     a combined RVT train dir using symlinks (no data copy).
     Val stays left-only (symlinked).

Run AFTER both preprocessing passes complete:
  vtei_mtevent_right_5bin/images/train/mtevent_train.h5   exists
  vtei_mtevent_right_10ch/images/train/mtevent_train.h5   exists

Usage:
  python scripts/merge_left_right.py [--root PREPROCESSED_ROOT] [--rvt-dir RVT_REPO]
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np


ROOT = Path("preprocessed_datasets")
RVT_DIR = Path("/home/loki/event/RVT")
EV_REPR_NAME = "stacked_histogram_dt=50_nbins=5_split_pol"


# ── helpers ───────────────────────────────────────────────────────────────────

def scene_id(path: Path) -> int:
    m = re.search(r"(\d+)", path.name)
    return int(m.group(1)) if m else 0


def merge_h5(left_h5: Path, right_h5: Path, out_h5: Path):
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(left_h5, "r") as fl, h5py.File(right_h5, "r") as fr:
        left  = fl["1mp"]
        right = fr["1mp"]
        N_l, N_r = left.shape[0], right.shape[0]
        shape_rest = left.shape[1:]
        dtype = left.dtype
        print(f"  left  H5: {N_l} frames {left.shape}")
        print(f"  right H5: {N_r} frames {right.shape}")

        with h5py.File(out_h5, "w") as fo:
            ds = fo.create_dataset("1mp",
                                   shape=(N_l + N_r,) + shape_rest,
                                   dtype=dtype,
                                   chunks=(1,) + shape_rest)
            CHUNK = 256
            for i in range(0, N_l, CHUNK):
                end = min(i + CHUNK, N_l)
                ds[i:end] = left[i:end]
            for i in range(0, N_r, CHUNK):
                end = min(i + CHUNK, N_r)
                ds[N_l+i:N_l+end] = right[i:end]

    print(f"  [OK] wrote {out_h5}  total={N_l+N_r} frames")


def merge_labels(left_lbl: Path, right_lbl: Path, out_lbl: Path, id_offset: int = 1000):
    out_lbl.mkdir(parents=True, exist_ok=True)
    # Left labels: copy as-is
    for f in sorted(left_lbl.glob("scene_*.npy"), key=scene_id):
        shutil.copy2(f, out_lbl / f.name)
    n_left = len(list(left_lbl.glob("scene_*.npy")))

    # Right labels: renumber scene ID by +id_offset
    n_right = 0
    for f in sorted(right_lbl.glob("scene_*.npy"), key=scene_id):
        sid = scene_id(f)
        new_name = f"scene_{sid + id_offset:06d}.npy"
        shutil.copy2(f, out_lbl / new_name)
        n_right += 1

    print(f"  [OK] labels: {n_left} left + {n_right} right → {out_lbl}")


def symlink_dir_contents(src: Path, dst: Path):
    """Symlink every item in src into dst."""
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        link = dst / item.name
        if not link.exists():
            link.symlink_to(item.resolve())


# ── part 1: ReYOLOv8 5ch ──────────────────────────────────────────────────────

def merge_reyolo(root: Path):
    print("\n=== [1/2] ReYOLOv8 5ch merge ===")
    left_root  = root / "vtei_mtevent_50ms_5bin"
    right_root = root / "vtei_mtevent_right_5bin"
    out_root   = root / "vtei_mtevent_combined_5bin"

    # Verify inputs
    right_h5 = right_root / "images" / "train" / "mtevent_train.h5"
    if not right_h5.exists():
        print(f"  [SKIP] {right_h5} not found — run right-camera preprocessing first")
        return False

    # Train: merge H5
    print("\n  Merging train H5 ...")
    merge_h5(
        left_root  / "images" / "train" / "mtevent_train.h5",
        right_h5,
        out_root   / "images" / "train" / "mtevent_train.h5",
    )

    # Train: merge labels
    print("  Merging train labels ...")
    merge_labels(
        left_root  / "labels" / "train",
        right_root / "labels" / "train",
        out_root   / "labels" / "train",
    )

    # Val and test: symlink from left (unchanged)
    for split in ("val", "test"):
        print(f"  Symlinking {split} from left ...")
        symlink_dir_contents(left_root / "images" / split, out_root / "images" / split)
        symlink_dir_contents(left_root / "labels" / split, out_root / "labels" / split)

    # Write dataset YAML
    yaml_path = root.parent / "configs" / "vtei_mtevent_combined.yaml"
    yaml_path.write_text(f"""\
path: {out_root.resolve()}/images
train: {out_root.resolve()}/images/train
val:   {out_root.resolve()}/images/val
test:  {out_root.resolve()}/images/test

nc: 17
names:
  - wooden_pallet          # 0  MR6D1
  - small_klt              # 1  MR6D2
  - big_klt                # 2  MR6D3
  - blue_klt               # 3  MR6D4
  - amazon_luggage         # 4  MR6D5
  - ikea_dammang_bin       # 5  MR6D6
  - ikea_vesken_trolley    # 6  MR6D7
  - ikea_sortera_bin       # 7  MR6D8
  - ikea_drona_grey        # 8  MR6D9
  - ikea_drona_blue        # 9  MR6D10
  - ikea_knallig_box       # 10 MR6D11
  - ikea_moppe_drawer      # 11 MR6D12
  - ikea_labbsal_basket    # 12 MR6D13
  - ikea_ivar_box          # 13 MR6D14
  - ikea_skubb_case        # 14 MR6D15
  - ikea_samla_box         # 15 MR6D16
  - human                  # 16
""")
    print(f"  [OK] wrote {yaml_path}")
    return True


# ── part 2: RVT 10ch ──────────────────────────────────────────────────────────

def merge_rvt(root: Path, rvt_dir: Path):
    print("\n=== [2/2] RVT 10ch merge ===")
    right_h5_root = root / "vtei_mtevent_right_10ch"
    right_rvt     = root / "rvt_mtevent_right_10ch"
    left_rvt      = root / "rvt_mtevent_10ch"
    combined_rvt  = root / "rvt_mtevent_combined_10ch"

    right_h5 = right_h5_root / "images" / "train" / "mtevent_train.h5"
    if not right_h5.exists():
        print(f"  [SKIP] {right_h5} not found — run 10ch right-camera preprocessing first")
        return False

    # Convert right 10ch H5 → RVT per-scene format
    if not right_rvt.exists():
        print(f"\n  Converting right 10ch H5 → RVT format ...")
        convert_script = Path(__file__).parent / "rvt" / "convert_mtevent_to_rvt.py"
        cmd = [
            sys.executable, str(convert_script),
            "--src", str(right_h5_root),
            "--dst", str(right_rvt),
            "--splits", "train",
            "--ev_repr_name", EV_REPR_NAME,
        ]
        subprocess.run(cmd, check=True)
    else:
        print(f"  [SKIP] RVT right already converted: {right_rvt}")

    # Build combined train dir via symlinks (no data copy)
    print(f"\n  Building combined RVT train dir (symlinks) ...")
    combined_train = combined_rvt / "train"
    combined_train.mkdir(parents=True, exist_ok=True)

    # Left scenes: symlink as-is
    n_left = 0
    for scene_dir in sorted((left_rvt / "train").iterdir(), key=scene_id):
        link = combined_train / scene_dir.name
        if not link.exists():
            link.symlink_to(scene_dir.resolve())
        n_left += 1

    # Right scenes: symlink with scene ID +1000
    n_right = 0
    for scene_dir in sorted((right_rvt / "train").iterdir(), key=scene_id):
        sid = scene_id(scene_dir)
        new_name = f"scene_{sid + 1000:06d}"
        link = combined_train / new_name
        if not link.exists():
            link.symlink_to(scene_dir.resolve())
        n_right += 1

    print(f"  [OK] combined train: {n_left} left + {n_right} right scenes → {combined_train}")

    # Val: symlink left val scenes (val stays left-only)
    combined_val = combined_rvt / "val"
    combined_val.mkdir(parents=True, exist_ok=True)
    for scene_dir in (left_rvt / "val").iterdir():
        link = combined_val / scene_dir.name
        if not link.exists():
            link.symlink_to(scene_dir.resolve())
    print(f"  [OK] val symlinked from left-only → {combined_val}")

    print(f"""
  RVT training command (combined 10ch):
    cd {rvt_dir} && WANDB_MODE=disabled /home/loki/venvs/rvt/bin/python train.py \\
      +experiment/mtevent=small_10ch dataset=mtevent \\
      dataset.path={combined_rvt} \\
      training.learning_rate=1e-4 training.weight_decay=0.0282 \\
      training.gradient_clip_val=1.24 training.max_steps=200000 \\
      training.lr_scheduler.total_steps=200000 training.lr_scheduler.pct_start=0.005 \\
      batch_size.train=16 batch_size.eval=4 hardware.gpus=0 \\
      hardware.num_workers.train=2 hardware.num_workers.eval=2 \\
      wandb.group_name=rvt_mtevent_combined_10ch_gen1 \\
      wandb.artifact_name=dummy \\
      wandb.artifact_local_file={rvt_dir}/pretrained/rvt-s-gen1-adapted-mtevent-10ch.ckpt \\
      wandb.resume_only_weights=True \\
      validation.val_check_interval=3500 validation.check_val_every_n_epoch=null \\
      logging.train.high_dim.enable=False logging.validation.high_dim.enable=False
""")
    return True


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",    type=Path, default=ROOT,
                    help="preprocessed_datasets directory")
    ap.add_argument("--rvt-dir", type=Path, default=RVT_DIR,
                    help="RVT repo root")
    ap.add_argument("--skip-reyolo", action="store_true")
    ap.add_argument("--skip-rvt",    action="store_true")
    args = ap.parse_args()

    ok_reyolo = merge_reyolo(args.root) if not args.skip_reyolo else False
    ok_rvt    = merge_rvt(args.root, args.rvt_dir) if not args.skip_rvt else False

    print("\n=== Summary ===")
    print(f"  ReYOLOv8 5ch combined: {'OK' if ok_reyolo else 'SKIPPED'}")
    print(f"  RVT 10ch combined:     {'OK' if ok_rvt    else 'SKIPPED'}")
    if ok_reyolo or ok_rvt:
        print("\nNext: run training commands pointing at the combined datasets.")


if __name__ == "__main__":
    main()

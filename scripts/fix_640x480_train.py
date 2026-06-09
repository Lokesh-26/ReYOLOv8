#!/usr/bin/env python3
"""
Fix 640x480 left-only train dataset by extracting left channels from stereo H5.
The left-only train H5 had wrong event data (misaligned with labels).
The stereo H5 first 5 channels contain correct left-only data (verified identical to val H5).

Run with system python3 (has h5py):
    python3 scripts/fix_640x480_train.py
"""
import h5py, numpy as np, os, shutil
from pathlib import Path

STEREO_H5    = "preprocessed_datasets/vtei_mtevent_stereo_5bin_640x480/images/train/mtevent_train.h5"
STEREO_LBLS  = "preprocessed_datasets/vtei_mtevent_stereo_5bin_640x480/labels/train"
OUT_H5       = "preprocessed_datasets/vtei_mtevent_50ms_5bin_640x480/images/train/mtevent_train.h5"
OUT_LBLS     = "preprocessed_datasets/vtei_mtevent_50ms_5bin_640x480/labels/train"
CHUNK_SIZE   = 200   # frames per chunk to limit RAM

print("=== Fix 640x480 left-only train dataset ===")
print()

# Back up old broken H5
bak = OUT_H5 + ".broken_bak"
if not os.path.exists(bak):
    print(f"Backing up old broken H5 → {bak}")
    os.rename(OUT_H5, bak)
    print(f"  Done ({os.path.getsize(bak)/1e9:.1f} GB)")
else:
    print(f"Backup already exists: {bak}")
    if os.path.exists(OUT_H5):
        os.remove(OUT_H5)
        print("  Removed existing H5 to re-create.")

# Read stereo H5 and write left-only
print(f"\nReading stereo H5: {STEREO_H5}")
with h5py.File(STEREO_H5, 'r') as f_in:
    total = f_in['1mp'].shape[0]
    C, H, W = 5, f_in['1mp'].shape[2], f_in['1mp'].shape[3]
    print(f"  Stereo shape: {f_in['1mp'].shape}  → extracting first 5 channels → ({total}, {C}, {H}, {W})")

    print(f"\nWriting left-only H5: {OUT_H5}")
    with h5py.File(OUT_H5, 'w') as f_out:
        ds = f_out.create_dataset('1mp', shape=(total, C, H, W), dtype='int8',
                                   chunks=(1, C, H, W), compression=None)
        for start in range(0, total, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, total)
            chunk = np.array(f_in['1mp'][start:end, :5, :, :])
            ds[start:end] = chunk
            if (start // CHUNK_SIZE) % 20 == 0:
                print(f"  {end}/{total} frames written ({end/total*100:.0f}%)...")
    print(f"  Done. Size: {os.path.getsize(OUT_H5)/1e9:.1f} GB")

# Verify first frame matches val H5 (spot check)
print("\nVerifying: train frame 0 vs stereo frame 0 (should match)...")
with h5py.File(OUT_H5, 'r') as f_new:
    with h5py.File(STEREO_H5, 'r') as f_stereo:
        frame0_new = np.array(f_new['1mp'][0])
        frame0_stereo = np.array(f_stereo['1mp'][0, :5, :, :])
        print(f"  Match: {np.allclose(frame0_new, frame0_stereo)}")

# Fix train labels
print(f"\nFixing train labels in {OUT_LBLS}")
bak_lbls = OUT_LBLS + "_broken_bak"
if not os.path.exists(bak_lbls):
    shutil.copytree(OUT_LBLS, bak_lbls)
    print(f"  Backed up labels to {bak_lbls}")

# Remove old labels
for f in Path(OUT_LBLS).glob("*.npy"):
    f.unlink()
print(f"  Cleared old labels")

# Copy stereo train labels
n_copied = 0
for src in sorted(Path(STEREO_LBLS).glob("*.npy")):
    dst = Path(OUT_LBLS) / src.name
    shutil.copy2(src, dst)
    n_copied += 1
print(f"  Copied {n_copied} label files from stereo train")

# Verify total frame count
total_lbl = sum(np.load(str(f), allow_pickle=True).shape[0] for f in Path(OUT_LBLS).glob("*.npy"))
print(f"\nVerification:")
print(f"  Train H5 frames: {total}")
print(f"  Train label frames: {total_lbl}")
print(f"  Match: {total == total_lbl}")

print("\n[DONE] Fixed 640x480 left-only train dataset.")
print("Now update config and launch training.")

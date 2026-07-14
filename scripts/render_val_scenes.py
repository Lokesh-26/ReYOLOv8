#!/usr/bin/env python3
"""
Render MTevent val scenes as MP4 videos with GT bounding boxes.
Uses polarity-coloured event visualization (red=positive, blue=negative).

Usage (from /home/loki/event/ReYOLOv8/):
    python scripts/render_val_scenes.py --scenes 14 26 33 35 --out_dir /tmp/val_videos
"""
import os, argparse
import numpy as np
import h5py
import cv2

H5_PATH   = 'preprocessed_datasets/vtei_mtevent_50ms_5bin_640x480/images/val/mtevent_val.h5'
LBL_DIR   = 'preprocessed_datasets/vtei_mtevent_50ms_5bin_640x480/labels/val'
FPS       = 20  # 50ms per frame → 20 fps

CLASS_NAMES = [
    'wooden_pallet','small_klt','big_klt','blue_klt','amazon_luggage',
    'ikea_dammang_bin','ikea_vesken_trolley','ikea_sortera_bin',
    'ikea_drona_grey','ikea_drona_blue','ikea_knallig_box','ikea_moppe_drawer',
    'ikea_labbsal_basket','ikea_ivar_box','ikea_skubb_case','ikea_samla_box','human',
]

PALETTE = [
    (0,255,255),(0,128,255),(0,255,0),(255,0,255),(255,128,0),(0,0,255),
    (255,255,0),(128,0,255),(0,255,128),(255,0,128),(128,255,0),(0,128,128),
    (128,0,128),(255,128,128),(128,128,255),(128,255,128),(0,200,255),
]


def frame_to_bgr(frame_chw, scale):
    """(C,H,W) int8 → BGR: red=positive events, blue=negative, black=no event."""
    acc = frame_chw.astype(np.float32).sum(axis=0)
    red  = np.clip( acc / scale * 255, 0, 255).astype(np.uint8)
    blue = np.clip(-acc / scale * 255, 0, 255).astype(np.uint8)
    bgr  = np.zeros((*acc.shape, 3), dtype=np.uint8)
    bgr[:, :, 2] = red
    bgr[:, :, 0] = blue
    return bgr


def draw_boxes(bgr, labels, H, W):
    for row in labels:
        cls = int(row[0])
        cx, cy, bw, bh = row[1]*W, row[2]*H, row[3]*W, row[4]*H
        x1, y1 = int(cx - bw/2), int(cy - bh/2)
        x2, y2 = int(cx + bw/2), int(cy + bh/2)
        color = PALETTE[cls % len(PALETTE)]
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        label = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else str(cls)
        cv2.putText(bgr, label, (x1, max(y1-4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return bgr


def render_scene(scene_id, frames, labels, out_path):
    N, C, H, W = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, FPS, (W, H))

    # Global contrast scale from p95 of |nonzero| values
    sample = np.abs(frames[::5].astype(np.float32))
    nonzero = sample[sample > 0]
    scale = float(np.percentile(nonzero, 95)) if len(nonzero) else 1.0

    for t in range(N):
        bgr = frame_to_bgr(frames[t], scale)
        frame_labels = labels[t] if t < len(labels) else np.zeros((0, 5))
        if len(frame_labels) > 0:
            bgr = draw_boxes(bgr, frame_labels, H, W)
        # Frame counter overlay
        cv2.putText(bgr, f'Scene {scene_id:02d}  t={t}', (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        writer.write(bgr)

    writer.release()
    print(f'  Scene {scene_id}: {N} frames → {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', nargs='+', type=int, default=[14, 26, 33, 35],
                    help='Scene IDs to render (default: all val scenes 14 26 33 35)')
    ap.add_argument('--out_dir', default='/tmp/val_videos')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Build scene→frame offset map from label files (sorted order matches H5 concat)
    all_label_files = sorted(os.listdir(LBL_DIR))
    scene_ids_in_h5 = [int(f.split('_')[1].split('.')[0]) for f in all_label_files]
    scene_lengths   = [len(np.load(os.path.join(LBL_DIR, f), allow_pickle=True))
                       for f in all_label_files]

    offset_map = {}
    off = 0
    for sid, length in zip(scene_ids_in_h5, scene_lengths):
        offset_map[sid] = (off, length)
        off += length

    print(f'H5 total frames: {off}')
    print(f'Scenes in H5: {scene_ids_in_h5}')

    with h5py.File(H5_PATH, 'r') as f:
        for sid in args.scenes:
            if sid not in offset_map:
                print(f'  Scene {sid}: not found in val set, skipping')
                continue
            start, n = offset_map[sid]
            print(f'  Loading scene {sid}: frames {start}–{start+n-1} ({n} frames)...')
            frames = np.array(f['1mp'][start:start+n])  # (N, C, H, W)

            lbl_file = f'scene_{sid:06d}.npy'
            labels = np.load(os.path.join(LBL_DIR, lbl_file), allow_pickle=True)

            out_path = os.path.join(args.out_dir, f'scene_{sid:02d}_events_gt.mp4')
            render_scene(sid, frames, labels, out_path)

    print(f'\nDone. Videos in {args.out_dir}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Five diagnostic analyses on all 4 clip-length models (C1/C5/C11/C21, 640x480 left).

  (1) Localization error vs inter-frame object displacement
  (2) Recall vs local event density + inactivity duration
  (3) FP persistence after object exit (within-scene IoU-tracker proxy + cold-start)
  (4) AP by object size x clip length
  (5) Per-IoU mAP curve vs clip length

Output: runs/diagnostics/{analysis1..5}/ + qualitative/
Inference is cached per model to runs/diagnostics/cache/.
"""
import sys, os, pickle, csv
sys.path.insert(0, '/home/loki/event/ReYOLOv8')

import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict
from pathlib import Path

from ultralytics.yolo.utils import ops

# ── Config ─────────────────────────────────────────────────────────────────────
MODELS = {
    'C1':  'runs/train/mtevent_640x480_c1/weights/best.pt',
    'C5':  'runs/train/mtevent_640x480_c5/weights/best.pt',
    'C11': 'runs/train/mtevent_640x480_fixed_c11/weights/best.pt',
    'C21': 'runs/train/mtevent_640x480_fixed_c21/weights/best.pt',
}
VAL_H5   = 'preprocessed_datasets/vtei_mtevent_50ms_5bin_640x480/images/val/mtevent_val.h5'
VAL_LBLS = 'preprocessed_datasets/vtei_mtevent_50ms_5bin_640x480/labels/val'
CONF     = 0.001
NMS_IOU  = 0.45
IMG_W, IMG_H = 640, 480

OUT_DIR   = Path('runs/diagnostics')
CACHE_DIR = OUT_DIR / 'cache'
QUAL_DIR  = OUT_DIR / 'qualitative'

CLASS_NAMES = [
    'wooden_pallet','small_klt','big_klt','blue_klt','amazon_luggage',
    'ikea_dammang_bin','ikea_vesken_trolley','ikea_sortera_bin',
    'ikea_drona_grey','ikea_drona_blue','ikea_knallig_box','ikea_moppe_drawer',
    'ikea_labbsal_basket','ikea_ivar_box','ikea_skubb_case','ikea_samla_box','human',
]
NC = len(CLASS_NAMES)

# Object size thresholds (px²) for 640×480 images
SIZE_SMALL  = 32 * 32    # < 1 024 px²
SIZE_MEDIUM = 96 * 96    # 1 024 – 9 216 px²; ≥ SIZE_MEDIUM → large

IOU_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90]
DENSITY_LOW_THRESH = 0.02   # mean |voxel| below this = inactive frame for that box

COLORS = {'C1': '#e74c3c', 'C5': '#f39c12', 'C11': '#2ecc71', 'C21': '#3498db'}

# ── Geometry helpers ───────────────────────────────────────────────────────────

def box_iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]); ub = (b[2]-b[0])*(b[3]-b[1])
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0

def box_center(b):
    return ((b[0]+b[2])*0.5, (b[1]+b[3])*0.5)

def center_dist(a, b):
    return ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5

def box_area(b):
    return max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])

def local_density(voxels, x1, y1, x2, y2):
    """Mean |voxel| inside bounding box. voxels: (C, H, W) int8/float."""
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(voxels.shape[2], int(x2)); y2 = min(voxels.shape[1], int(y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(np.abs(voxels[:, y1:y2, x1:x2]).mean())

def voxels_to_gray(voxels):
    """(C,H,W) int8 → uint8 (H,W,3) for imshow."""
    act = np.abs(voxels.astype(np.float32)).sum(axis=0)
    act = (act / (act.max() + 1e-6) * 255).astype(np.uint8)
    return np.stack([act]*3, axis=-1)

# ── AP computation ─────────────────────────────────────────────────────────────

def _ap(recall, precision):
    mr = np.concatenate(([0.0], recall, [1.0]))
    mp = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mp.size - 1, 0, -1):
        mp[i-1] = max(mp[i-1], mp[i])
    idx = np.where(mr[1:] != mr[:-1])[0]
    return float(np.sum((mr[idx+1] - mr[idx]) * mp[idx+1]))

def compute_map(all_preds, all_gts, nc, iou_thresh=0.5):
    """all_preds/all_gts: lists of per-frame lists of (x1,y1,x2,y2,conf,cls) / (...,cls)."""
    tp_fp  = defaultdict(list)
    n_gt   = defaultdict(int)
    for preds, gts in zip(all_preds, all_gts):
        for g in gts:
            n_gt[int(g[4])] += 1
        if not preds:
            continue
        preds_s = sorted(preds, key=lambda x: -x[4])
        used = set()
        for p in preds_s:
            pc = int(p[5])
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gts):
                if int(g[4]) != pc or j in used:
                    continue
                v = box_iou(p[:4], g[:4])
                if v > best_iou:
                    best_iou, best_j = v, j
            is_tp = float(best_iou >= iou_thresh)
            if is_tp:
                used.add(best_j)
            tp_fp[pc].append((p[4], is_tp))
    ap_dict = {}
    for c in range(nc):
        if n_gt[c] == 0:
            continue
        ent = sorted(tp_fp.get(c, []), key=lambda x: -x[0])
        if not ent:
            ap_dict[c] = 0.0
            continue
        tps = np.cumsum([e[1] for e in ent])
        fps = np.cumsum([1.0 - e[1] for e in ent])
        rec = tps / n_gt[c]
        pre = tps / (tps + fps + 1e-9)
        ap_dict[c] = _ap(rec, pre)
    mAP = float(np.mean(list(ap_dict.values()))) if ap_dict else 0.0
    return ap_dict, mAP

# ── GT cross-frame matching (for displacement + FP-exit tracking) ──────────────

def match_gt_frames(gts_a, gts_b, iou_thresh=0.30):
    """Greedy class-aware GT matching between consecutive frames.
    Returns list of (idx_in_a, idx_in_b, iou)."""
    matched = []; used_b = set()
    for i, a in enumerate(gts_a):
        best_iou, best_j = 0.0, -1
        for j, b in enumerate(gts_b):
            if j in used_b or int(a[4]) != int(b[4]):
                continue
            v = box_iou(a[:4], b[:4])
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0 and best_iou >= iou_thresh:
            matched.append((i, best_j, best_iou))
            used_b.add(best_j)
    return matched

# ── Inference (cached) ─────────────────────────────────────────────────────────

def run_inference(model_name, model_path):
    """Return list of per-frame dicts with predictions, GTs, and raw voxels."""
    cache_file = CACHE_DIR / f'{model_name}_records.pkl'
    if cache_file.exists():
        print(f'  [{model_name}] loading from cache...')
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    print(f'  [{model_name}] running inference...')
    device = torch.device('cuda:0')
    ckpt   = torch.load(model_path, map_location=device, weights_only=False)
    model  = ckpt['model'].to(device).float()
    model.eval()

    lbl_files = sorted(os.listdir(VAL_LBLS))
    f_h5      = h5py.File(VAL_H5, 'r')
    records   = []
    offset    = 0
    hidden    = {"0": None, "1": None, "2": None, "3": None}

    with torch.no_grad():
        for scene_idx, lf in enumerate(lbl_files):
            lbls = np.load(os.path.join(VAL_LBLS, lf), allow_pickle=True)
            n    = len(lbls)
            for i in range(n):
                vox = np.array(f_h5['1mp'][offset + i])          # (5, 480, 640) int8
                inp = torch.tensor(vox.astype(np.float32)).unsqueeze(0).to(device)
                out, hidden = model(inp, hidden)
                dets = ops.non_max_suppression(out, conf_thres=CONF, iou_thres=NMS_IOU,
                                               multi_label=False, max_det=300)[0]
                preds = []
                for d in dets.cpu().numpy():
                    x1, y1, x2, y2, cf, cl = d
                    preds.append((float(x1), float(y1), float(x2), float(y2),
                                  float(cf), int(cl)))
                gts = []
                for g in lbls[i]:
                    c = int(g[0])
                    cx = g[1]*IMG_W; cy = g[2]*IMG_H
                    w  = g[3]*IMG_W; h  = g[4]*IMG_H
                    gts.append((cx-w/2, cy-h/2, cx+w/2, cy+h/2, c))
                records.append({
                    'scene_idx':      scene_idx,
                    'frame_in_scene': i,
                    'global_frame':   offset + i,
                    'preds':          preds,
                    'gts':            gts,
                    'voxels':         vox,
                })
            offset += n
            hidden = {"0": None, "1": None, "2": None, "3": None}

    f_h5.close()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(records, f)
    print(f'    {len(records)} frames cached.')
    return records

# ── Utility: group records by scene ───────────────────────────────────────────

def by_scene(records):
    d = defaultdict(list)
    for r in records:
        d[r['scene_idx']].append(r)
    for v in d.values():
        v.sort(key=lambda x: x['frame_in_scene'])
    return d

# ── Analysis 1: Localization error vs inter-frame displacement ─────────────────

def a1_localization_vs_displacement(model_name, records, out_dir):
    pairs = []   # (displacement_px, detection_iou)

    for sc_frames in by_scene(records).values():
        for fi in range(1, len(sc_frames)):
            prev = sc_frames[fi-1]
            curr = sc_frames[fi]
            gts_p = prev['gts']; gts_c = curr['gts']

            # GT displacement map: gt_index_in_curr → displacement (px)
            disp_map = {}
            for (jp, jc, _) in match_gt_frames(gts_p, gts_c):
                disp_map[jc] = center_dist(box_center(gts_p[jp]), box_center(gts_c[jc]))

            # Match preds to GTs at IoU ≥ 0.5, record (displacement, localization_iou)
            used = set()
            for p in sorted(curr['preds'], key=lambda x: -x[4]):
                best_iou, best_j = 0.0, -1
                for j, g in enumerate(gts_c):
                    if j in used or int(g[4]) != int(p[5]):
                        continue
                    v = box_iou(p[:4], g[:4])
                    if v > best_iou:
                        best_iou, best_j = v, j
                if best_iou >= 0.50:
                    used.add(best_j)
                    if best_j in disp_map:
                        pairs.append((disp_map[best_j], best_iou))

    if not pairs:
        print(f'    A1 {model_name}: no pairs — skipping')
        return None

    disps = np.array([p[0] for p in pairs])
    ious  = np.array([p[1] for p in pairs])

    bins   = [0, 2, 5, 10, 20, 40, np.inf]
    blabels = ['0-2', '2-5', '5-10', '10-20', '20-40', '>40']
    bin_means, bin_counts = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (disps >= lo) & (disps < hi)
        bin_means.append(float(ious[mask].mean()) if mask.sum() else float('nan'))
        bin_counts.append(int(mask.sum()))

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(disps, ious, alpha=0.08, s=4, color=COLORS.get(model_name, 'steelblue'))
    axes[0].set_xlim(0, 60); axes[0].set_ylim(0.4, 1.0)
    axes[0].set_xlabel('GT displacement between frames (px)')
    axes[0].set_ylabel('Detection IoU (TP only)')
    axes[0].set_title(f'{model_name}: Localization IoU vs Object Displacement')
    axes[0].grid(True, alpha=0.3)

    valid = [(l, m, c) for l, m, c in zip(blabels, bin_means, bin_counts) if not np.isnan(m)]
    bars = axes[1].bar([v[0] for v in valid], [v[1] for v in valid],
                       color=COLORS.get(model_name, 'steelblue'), edgecolor='k', alpha=0.85)
    for bar, v in zip(bars, valid):
        axes[1].text(bar.get_x()+bar.get_width()/2, v[1]+0.003,
                     f'n={v[2]}', ha='center', fontsize=8)
    axes[1].set_ylim(0.4, 1.0)
    axes[1].set_xlabel('Displacement bin (px)'); axes[1].set_ylabel('Mean IoU')
    axes[1].set_title('Mean Localization IoU per Displacement Bin')
    axes[1].grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / f'{model_name}_disp_vs_iou.png', dpi=120)
    plt.close()
    np.savetxt(out_dir / f'{model_name}_disp_iou.csv',
               np.column_stack([disps, ious]), delimiter=',',
               header='displacement_px,detection_iou', comments='')
    print(f'    A1 {model_name}: {len(pairs)} TP pairs | '
          + '  '.join(f'{l}:{m:.3f}(n={c})' for l, m, c in zip(blabels, bin_means, bin_counts) if not np.isnan(m)))
    return blabels, bin_means, bin_counts

# ── Analysis 2: Recall vs event density + inactivity duration ─────────────────

def a2_density(model_name, records, out_dir):
    density_rec   = []   # (local_density, was_recalled)
    inactivity_rec = []  # (n_consec_inactive_frames, was_recalled)

    for sc_frames in by_scene(records).values():
        # Forward IoU tracker: track_id → consecutive_inactive_count
        track_inactive = {}    # track_id → int
        prev_track_ids = {}    # gt_idx_in_prev_frame → track_id
        next_tid = [0]

        for fi, frame in enumerate(sc_frames):
            gts   = frame['gts']
            preds = frame['preds']
            vox   = frame['voxels']

            # Recall bitmask
            used_p = set(); recalled = set()
            for j, g in enumerate(gts):
                for k, p in enumerate(preds):
                    if k in used_p or int(p[5]) != int(g[4]):
                        continue
                    if box_iou(p[:4], g[:4]) >= 0.5:
                        recalled.add(j); used_p.add(k); break

            # Assign track IDs by matching to previous frame
            curr_track_ids = {}
            if fi > 0 and prev_gts_:
                for (jp, jc, _) in match_gt_frames(prev_gts_, gts):
                    tid = prev_track_ids.get(jp)
                    if tid is None:
                        tid = next_tid[0]; next_tid[0] += 1
                    curr_track_ids[jc] = tid
            for j in range(len(gts)):
                if j not in curr_track_ids:
                    curr_track_ids[j] = next_tid[0]; next_tid[0] += 1

            for j, g in enumerate(gts):
                dens = local_density(vox, g[0], g[1], g[2], g[3])
                tid  = curr_track_ids[j]

                # Update inactivity counter
                if dens < DENSITY_LOW_THRESH:
                    track_inactive[tid] = track_inactive.get(tid, 0) + 1
                else:
                    track_inactive[tid] = 0

                density_rec.append((dens, int(j in recalled)))
                inactivity_rec.append((track_inactive[tid], int(j in recalled)))

            prev_gts_       = gts
            prev_track_ids  = curr_track_ids

    if not density_rec:
        print(f'    A2 {model_name}: empty — skipping')
        return

    dens = np.array([x[0] for x in density_rec])
    rec  = np.array([x[1] for x in density_rec])
    inac = np.array([x[0] for x in inactivity_rec])
    irec = np.array([x[1] for x in inactivity_rec])

    # Density bins by percentile
    pcts     = [0, 10, 25, 50, 75, 90, 100]
    threshs  = np.percentile(dens, pcts)
    d_labels = [f'p{pcts[i]}-{pcts[i+1]}' for i in range(len(pcts)-1)]
    d_recall = []
    d_counts = []
    for lo, hi in zip(threshs[:-1], threshs[1:]):
        mask = (dens >= lo) & (dens <= hi)
        d_recall.append(float(rec[mask].mean()) if mask.sum() else float('nan'))
        d_counts.append(int(mask.sum()))

    # Inactivity bins (frames)
    i_bins   = [0, 1, 2, 5, 10, 20, np.inf]
    i_labels = ['0', '1', '2-4', '5-9', '10-19', '≥20']
    i_recall = []; i_counts = []
    for lo, hi in zip(i_bins[:-1], i_bins[1:]):
        mask = (inac >= lo) & (inac < hi)
        i_recall.append(float(irec[mask].mean()) if mask.sum() else float('nan'))
        i_counts.append(int(mask.sum()))

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    vd = [(l, r, c) for l, r, c in zip(d_labels, d_recall, d_counts) if not np.isnan(r)]
    axes[0].bar([v[0] for v in vd], [v[1] for v in vd],
                color='darkorange', edgecolor='k', alpha=0.85)
    for v in vd:
        axes[0].text(v[0], v[1]+0.01, f'n={v[2]}', ha='center', fontsize=7, rotation=25)
    axes[0].set_ylim(0, 1.1); axes[0].tick_params(axis='x', rotation=30)
    axes[0].set_xlabel('Local event density (percentile bin)')
    axes[0].set_ylabel('Recall @ IoU 0.5')
    axes[0].set_title(f'{model_name}: Recall vs Local Event Density')
    axes[0].grid(True, axis='y', alpha=0.3)

    vi = [(l, r, c) for l, r, c in zip(i_labels, i_recall, i_counts) if not np.isnan(r)]
    axes[1].bar([v[0] for v in vi], [v[1] for v in vi],
                color='forestgreen', edgecolor='k', alpha=0.85)
    for v in vi:
        axes[1].text(v[0], v[1]+0.01, f'n={v[2]}', ha='center', fontsize=8)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_xlabel('Consecutive inactive frames (density < threshold)')
    axes[1].set_ylabel('Recall @ IoU 0.5')
    axes[1].set_title(f'{model_name}: Recall vs Inactivity Duration')
    axes[1].grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / f'{model_name}_recall_vs_density.png', dpi=120)
    plt.close()
    print(f'    A2 {model_name}: {len(density_rec)} GT instances | '
          + '  '.join(f'{l}:{r:.3f}' for l, r, _ in vd))

# ── Analysis 3: FP persistence after object exit ───────────────────────────────

def a3_fp_persistence(model_name, records, out_dir):
    # --- 3a: within-scene exit tracking ---
    fp_by_lag = defaultdict(list)   # lag (1..10) → [0/1 was-FP]

    # --- 3b: scene cold-start (hidden = None) ---
    fp_at_frame = defaultdict(list) # frame_in_scene → [fp_count]

    CONF_THRESH = 0.25   # only count confident predictions

    for sc_frames in by_scene(records).values():
        n = len(sc_frames)

        # 3b: cold-start FP count per frame index (first 15 frames)
        for fi in range(min(15, n)):
            frame = sc_frames[fi]
            fp_n  = 0
            used  = set()
            for p in frame['preds']:
                if p[4] < CONF_THRESH:
                    continue
                matched = any(
                    int(g[4]) == int(p[5]) and box_iou(p[:4], g[:4]) >= 0.5
                    for g in frame['gts']
                )
                if not matched:
                    fp_n += 1
            fp_at_frame[fi].append(fp_n)

        # 3a: object exits within scene
        for fi in range(n - 1):
            gts_curr = sc_frames[fi]['gts']
            gts_next = sc_frames[fi+1]['gts']
            matched_curr = {m[0] for m in match_gt_frames(gts_curr, gts_next)}

            for j, g in enumerate(gts_curr):
                if j in matched_curr:
                    continue
                exit_box = g[:4]; exit_cls = int(g[4])

                for lag in range(1, min(11, n - fi)):
                    fut = sc_frames[fi + lag]
                    for p in fut['preds']:
                        if p[4] < CONF_THRESH or int(p[5]) != exit_cls:
                            continue
                        if box_iou(p[:4], exit_box) < 0.10:
                            continue
                        # Is it a FP? (not explained by any current GT)
                        is_fp = not any(
                            int(fg[4]) == exit_cls and box_iou(p[:4], fg[:4]) >= 0.5
                            for fg in fut['gts']
                        )
                        fp_by_lag[lag].append(int(is_fp))

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 3a plot
    lags = sorted(fp_by_lag)
    if lags:
        fp_rates = [np.mean(fp_by_lag[l]) for l in lags]
        fp_ns    = [len(fp_by_lag[l]) for l in lags]
        bars = axes[0].bar(lags, fp_rates, color='crimson', edgecolor='k', alpha=0.85)
        for bar, lag, r, n_ in zip(bars, lags, fp_rates, fp_ns):
            axes[0].text(bar.get_x()+bar.get_width()/2, r+0.01,
                         f'n={n_}', ha='center', fontsize=7)
        axes[0].set_xticks(lags)
        axes[0].set_ylim(0, 1)
        axes[0].set_xlabel('Frames after object exit')
        axes[0].set_ylabel('FP rate (conf > 0.25, exit region)')
        axes[0].set_title(f'{model_name}: FP Persistence After Object Exit')
        axes[0].grid(True, axis='y', alpha=0.3)
    else:
        axes[0].set_title(f'{model_name}: No exit events found')

    # 3b plot
    fidxs = sorted(fp_at_frame)
    if fidxs:
        mean_fp = [np.mean(fp_at_frame[fi]) for fi in fidxs]
        axes[1].plot(fidxs, mean_fp, 'o-', color='darkorange', linewidth=2, markersize=6)
        axes[1].fill_between(fidxs, mean_fp, alpha=0.15, color='darkorange')
        axes[1].set_xlabel('Frame index within scene (0 = first frame after hidden reset)')
        axes[1].set_ylabel('Mean FP count (conf > 0.25)')
        axes[1].set_title(f'{model_name}: Scene Cold-Start FP Decay')
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / f'{model_name}_fp_persistence.png', dpi=120)
    plt.close()
    total = sum(len(v) for v in fp_by_lag.values())
    print(f'    A3 {model_name}: {total} exit-proximity preds tracked  '
          + (f'lag1 FP-rate={np.mean(fp_by_lag[1]):.3f}' if 1 in fp_by_lag else 'lag1 empty'))

# ── Analysis 4: AP by object size ─────────────────────────────────────────────

def a4_size(models_records, out_dir):
    size_results = {}

    for mname, records in models_records.items():
        all_preds = [r['preds'] for r in records]
        all_gts   = [r['gts']   for r in records]
        row = {}
        for label, lo, hi in [('small', 0, SIZE_SMALL),
                               ('medium', SIZE_SMALL, SIZE_MEDIUM),
                               ('large', SIZE_MEDIUM, 1e9)]:
            filt_gts = [[g for g in gts if lo <= box_area(g[:4]) < hi]
                        for gts in all_gts]
            _, mAP = compute_map(all_preds, filt_gts, NC, iou_thresh=0.5)
            row[label] = mAP
        size_results[mname] = row
        print(f'    A4 {mname}: small={row["small"]:.3f}  medium={row["medium"]:.3f}  large={row["large"]:.3f}')

    models_list = list(size_results)
    size_bins   = ['small', 'medium', 'large']
    x = np.arange(len(models_list)); w = 0.25
    clrs = ['#e74c3c', '#f39c12', '#2ecc71']

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (sb, c) in enumerate(zip(size_bins, clrs)):
        vals = [size_results[m][sb] for m in models_list]
        bars = ax.bar(x + i*w, vals, w, label=f'{sb} (<{[32,96][i<2]}²px)' if i < 2 else 'large (≥96²px)',
                      color=c, edgecolor='k', alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, v+0.005,
                    f'{v:.3f}', ha='center', fontsize=8, rotation=75)
    ax.set_xticks(x + w); ax.set_xticklabels(models_list)
    ax.set_ylabel('mAP50'); ax.set_ylim(0, 0.75)
    ax.set_title('AP by Object Size × Clip Length  (640×480, left, IoU=0.5)')
    ax.legend(); ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'size_vs_ap.png', dpi=120)
    plt.close()

    with open(out_dir / 'size_ap.csv', 'w', newline='') as f:
        w_ = csv.writer(f)
        w_.writerow(['model', 'small', 'medium', 'large'])
        for m in models_list:
            w_.writerow([m] + [f'{size_results[m][s]:.4f}' for s in size_bins])
    return size_results

# ── Analysis 5: Per-IoU mAP curve ─────────────────────────────────────────────

def a5_iou_sweep(models_records, out_dir):
    iou_results = {}
    for mname, records in models_records.items():
        all_preds = [r['preds'] for r in records]
        all_gts   = [r['gts']   for r in records]
        row = {}
        for t in IOU_THRESHOLDS:
            _, mAP = compute_map(all_preds, all_gts, NC, iou_thresh=t)
            row[t] = mAP
        iou_results[mname] = row
        print(f'    A5 {mname}: ' + '  '.join(f'@{t:.2f}={v:.3f}' for t, v in row.items()))

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    for mname, row in iou_results.items():
        xs = sorted(row); ys = [row[x] for x in xs]
        ax.plot(xs, ys, 'o-', label=mname, color=COLORS.get(mname),
                linewidth=2.5, markersize=8)
    ax.set_xlabel('IoU threshold'); ax.set_ylabel('mAP')
    ax.set_xlim(0.25, 0.95); ax.set_ylim(0, 0.6)
    ax.set_title('Per-IoU mAP vs Clip Length  (640×480, left)')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'per_iou_ap.png', dpi=120)
    plt.close()

    with open(out_dir / 'per_iou_ap.csv', 'w', newline='') as f:
        w_ = csv.writer(f)
        w_.writerow(['model'] + [str(t) for t in IOU_THRESHOLDS])
        for m, row in iou_results.items():
            w_.writerow([m] + [f'{row[t]:.4f}' for t in IOU_THRESHOLDS])
    return iou_results

# ── Qualitative sample frames ──────────────────────────────────────────────────

def save_qualitative(model_name, records, out_dir, n=4):
    """Save annotated event frames: high-density correct, low-density miss, scene-start FP."""
    import random
    out_dir.mkdir(parents=True, exist_ok=True)

    hd_correct = []   # (density, frame_global_idx)
    ld_miss    = []
    ss_fp      = []   # (fp_count, frame_global_idx)

    for fi, r in enumerate(records):
        vox = r['voxels']; gts = r['gts']; preds = r['preds']
        if not gts:
            continue
        frame_dens = float(np.abs(vox.astype(np.float32)).mean())
        used_p = set(); recalled = set()
        for j, g in enumerate(gts):
            for k, p in enumerate(preds):
                if k in used_p or int(p[5]) != int(g[4]):
                    continue
                if box_iou(p[:4], g[:4]) >= 0.5:
                    recalled.add(j); used_p.add(k); break
        miss_n = len(gts) - len(recalled)

        if r['frame_in_scene'] == 0:
            fp_n = sum(1 for p in preds if p[4] >= 0.25 and
                       not any(int(g[4]) == int(p[5]) and
                                box_iou(p[:4], g[:4]) >= 0.5 for g in gts))
            if fp_n > 0:
                ss_fp.append((fp_n, fi))

        if frame_dens > 0.06 and miss_n == 0 and len(gts) >= 2:
            hd_correct.append((frame_dens, fi))
        if frame_dens < 0.015 and miss_n > 0:
            ld_miss.append((frame_dens, fi))

    def draw(rec, title, path):
        img = voxels_to_gray(rec['voxels'])
        fig, ax = plt.subplots(figsize=(10, 7.5))
        ax.imshow(img, cmap='gray')
        for g in rec['gts']:
            x1,y1,x2,y2,cls = g
            ax.add_patch(patches.Rectangle((x1,y1), x2-x1, y2-y1,
                          linewidth=2, edgecolor='lime', facecolor='none'))
            ax.text(x1, max(0, y1-4), CLASS_NAMES[int(cls)],
                    color='lime', fontsize=7, fontweight='bold',
                    bbox=dict(boxstyle='square,pad=0.1', fc='black', alpha=0.4))
        for p in rec['preds']:
            if p[4] < 0.25:
                continue
            x1,y1,x2,y2,cf,cl = p
            ax.add_patch(patches.Rectangle((x1,y1), x2-x1, y2-y1,
                          linewidth=1.5, edgecolor='red', facecolor='none', linestyle='--'))
            ax.text(x2+2, y1, f'{cf:.2f}', color='red', fontsize=6)
        dens = float(np.abs(rec['voxels'].astype(np.float32)).mean())
        ax.set_title(f'{title}  [scene={rec["scene_idx"]} fi={rec["frame_in_scene"]} dens={dens:.4f}]',
                     fontsize=10)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close()

    random.shuffle(hd_correct); random.shuffle(ld_miss); random.shuffle(ss_fp)
    for i, (_, fi) in enumerate(hd_correct[:n]):
        draw(records[fi], f'{model_name} — high-density / all recalled',
             out_dir / f'{model_name}_hd_correct_{i}.png')
    for i, (_, fi) in enumerate(ld_miss[:n]):
        draw(records[fi], f'{model_name} — low-density / missed GT',
             out_dir / f'{model_name}_ld_miss_{i}.png')
    for i, (fp_n, fi) in enumerate(ss_fp[:n]):
        draw(records[fi], f'{model_name} — scene-start FP (n={fp_n})',
             out_dir / f'{model_name}_ss_fp_{i}.png')
    print(f'    Qual {model_name}: {min(n, len(hd_correct))} hd_correct '
          f'| {min(n, len(ld_miss))} ld_miss '
          f'| {min(n, len(ss_fp))} scene_start_fp')

# ── Combined cross-model overlays ──────────────────────────────────────────────

def combined_a1_overlay(a1_results, out_dir):
    blabels = ['0-2', '2-5', '5-10', '10-20', '20-40', '>40']
    x = np.arange(len(blabels)); w = 0.20
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (mname, (bl, means, counts)) in enumerate(a1_results.items()):
        vals = [m if not np.isnan(m) else 0.0 for m in means]
        ax.bar(x + i*w, vals, w, label=mname,
               color=COLORS.get(mname), edgecolor='k', alpha=0.85)
    ax.set_xticks(x + 1.5*w); ax.set_xticklabels(blabels)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel('Object displacement between frames (px)')
    ax.set_ylabel('Mean detection IoU (TP only)')
    ax.set_title('Localization Quality vs Object Motion — All Clip Lengths')
    ax.legend(); ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / 'all_models_disp_vs_iou.png', dpi=120)
    plt.close()

# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import warnings; warnings.filterwarnings('ignore')
    print('=== Diagnostic Analyses: all 4 clip models ===\n')

    for d in [OUT_DIR, CACHE_DIR, QUAL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    a1_dir = OUT_DIR / 'analysis1_localization'
    a2_dir = OUT_DIR / 'analysis2_density'
    a3_dir = OUT_DIR / 'analysis3_fp_persistence'
    a4_dir = OUT_DIR / 'analysis4_size'
    a5_dir = OUT_DIR / 'analysis5_iou_sweep'

    # ── Step 1: inference (cached after first run) ──
    print('── Inference ──')
    models_records = {}
    for mname, mpath in MODELS.items():
        models_records[mname] = run_inference(mname, mpath)

    # ── Step 2: per-model analyses ──
    a1_results = {}
    for mname, records in models_records.items():
        print(f'\n── {mname} ──')
        res = a1_localization_vs_displacement(mname, records, a1_dir)
        if res:
            a1_results[mname] = res
        a2_density(mname, records, a2_dir)
        a3_fp_persistence(mname, records, a3_dir)
        save_qualitative(mname, records, QUAL_DIR)

    # ── Step 3: cross-model analyses ──
    print('\n── Cross-model ──')
    a4_size(models_records, a4_dir)
    a5_iou_sweep(models_records, a5_dir)
    if a1_results:
        combined_a1_overlay(a1_results, a1_dir)

    print('\n=== Done. All outputs in runs/diagnostics/ ===')

#!/usr/bin/env python3
"""
Compare ReYOLOv8 / RVT / RGB YOLOv8 detections on WP1 bags.

No ground-truth needed — computes detection-rate, class-frequency,
temporal consistency, and renders a side-by-side comparison video.

Usage (from /home/loki/event/ReYOLOv8/):
    python scripts/compare_wp1_detections.py \
        --result_dirs  benchmark_results/wp1_rgb20ms_* \
        --out_dir      benchmark_results/comparison
"""
import os
import json
import argparse
import glob
from collections import Counter, defaultdict

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

CLASS_NAMES = [
    "wooden_pallet", "small_klt", "big_klt", "blue_klt",
    "amazon_luggage", "ikea_dammang_bin", "ikea_vesken_trolley",
    "ikea_sortera_bin", "ikea_drona_grey", "ikea_drona_blue",
    "ikea_knallig_box", "ikea_moppe_drawer", "ikea_labbsal_basket",
    "ikea_ivar_box", "ikea_skubb_case", "ikea_samla_box", "human",
]

MODELS = [
    ('yolov8s', 'detections_yolov8s/detections.json',   'YOLOv8s',   '#f58231'),
    ('tc',      'detections_tc_c11/detections.json',     'TC (C11)',   '#e6194b'),
    ('rvt',     'rvt_detections/rvt_detections.json',    'RVT-small',  '#3cb44b'),
    ('rgb',     'rgb_detections/rgb_detections.json',    'RGB YOLOv8', '#4363d8'),
]


# ──────────────────────────────────────────────────────────────────────────────
# Stats helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_json(path):
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def det_rate(records):
    if not records:
        return 0.0
    return sum(1 for r in records if r['boxes']) / len(records)


def mean_dets(records):
    if not records:
        return 0.0
    return sum(len(r['boxes']) for r in records) / len(records)


def class_counts(records):
    c = Counter()
    for r in records:
        for b in r['boxes']:
            c[int(b[0])] += 1
    return c


def temporal_consistency(records):
    """Mean length of consecutive runs of frames that have ≥1 detection."""
    if not records:
        return 0.0
    in_run, run_len, runs = False, 0, []
    for r in records:
        if r['boxes']:
            in_run = True
            run_len += 1
        else:
            if in_run:
                runs.append(run_len)
            in_run, run_len = False, 0
    if in_run:
        runs.append(run_len)
    return float(np.mean(runs)) if runs else 0.0


def conf_per_frame(records):
    """Mean max-confidence detection per frame (0 if no detection)."""
    vals = []
    for r in records:
        if r['boxes']:
            vals.append(max(b[1] for b in r['boxes']))
        else:
            vals.append(0.0)
    return np.array(vals)


# ──────────────────────────────────────────────────────────────────────────────
# Plot: temporal detection + confidence
# ──────────────────────────────────────────────────────────────────────────────

def plot_temporal(ax, records, color, label):
    if records is None:
        return
    n = len(records)
    x = np.arange(n)
    # binary: frame has detection or not
    has_det = np.array([1 if r['boxes'] else 0 for r in records], dtype=float)
    # smooth with 5-frame rolling window
    kernel = np.ones(15) / 15
    smooth = np.convolve(has_det, kernel, mode='same')
    ax.plot(x, smooth, color=color, lw=1.2, label=label)
    ax.fill_between(x, 0, smooth, alpha=0.15, color=color)


def plot_conf_temporal(ax, records, color, label):
    if records is None:
        return
    conf = conf_per_frame(records)
    x = np.arange(len(conf))
    kernel = np.ones(15) / 15
    smooth = np.convolve(conf, kernel, mode='same')
    ax.plot(x, smooth, color=color, lw=1.2, label=label)


# ──────────────────────────────────────────────────────────────────────────────
# Side-by-side frame renderer
# ──────────────────────────────────────────────────────────────────────────────

PALETTE_BGR = [
    (0, 255, 255), (0, 128, 255), (0, 255, 0), (255, 0, 255),
    (255, 128, 0), (0, 0, 255), (255, 255, 0), (128, 0, 255),
    (0, 255, 128), (255, 0, 128), (128, 255, 0), (0, 128, 128),
    (128, 0, 128), (255, 128, 128), (128, 128, 255), (128, 255, 128),
    (0, 200, 255),
]


def overlay_boxes(img, boxes):
    for b in boxes:
        cls_id, conf = int(b[0]), float(b[1])
        x1, y1, x2, y2 = int(b[2]), int(b[3]), int(b[4]), int(b[5])
        color = PALETTE_BGR[cls_id % len(PALETTE_BGR)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else cls_id} {conf:.2f}"
        cv2.putText(img, label, (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return img


def build_side_by_side_video(result_dir, out_video_path, fps=20.0):
    """Build a 3-panel side-by-side comparison video."""
    # Collect frame paths for each model
    frame_dirs = {
        'ReYOLOv8':   os.path.join(result_dir, 'detections', 'frames'),
        'RVT-small':  os.path.join(result_dir, 'rvt_detections', 'frames'),
        'RGB YOLOv8': None,  # use rgb_frames as background + overlay dets
    }
    rgb_frames_dir = os.path.join(result_dir, 'rgb_frames')
    rgb_det_json   = os.path.join(result_dir, 'rgb_detections', 'rgb_detections.json')

    # Check what's available
    event_frames_reyolo = sorted(glob.glob(os.path.join(frame_dirs['ReYOLOv8'], '*.png'))) \
        if frame_dirs['ReYOLOv8'] and os.path.isdir(frame_dirs['ReYOLOv8']) else []
    event_frames_rvt = sorted(glob.glob(os.path.join(frame_dirs['RVT-small'], '*.png'))) \
        if frame_dirs['RVT-small'] and os.path.isdir(frame_dirs['RVT-small']) else []
    rgb_frames = sorted(glob.glob(os.path.join(rgb_frames_dir, '*.jpg'))) \
        if os.path.isdir(rgb_frames_dir) else []
    rgb_dets = load_json(rgb_det_json)

    if not event_frames_reyolo and not event_frames_rvt and not rgb_frames:
        print(f"  [SKIP] no rendered frames found in {result_dir}")
        return

    n = max(len(event_frames_reyolo), len(event_frames_rvt), len(rgb_frames))
    if n == 0:
        return

    # Determine panel size from first available frame
    sample_path = (event_frames_reyolo or event_frames_rvt or rgb_frames)[0]
    sample = cv2.imread(sample_path)
    h, w = sample.shape[:2]
    panel_w, panel_h = w, h

    total_w = panel_w * 3
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (total_w, panel_h))

    label_colors = {
        'ReYOLOv8':   (0, 80, 230),
        'RVT-small':  (0, 180, 60),
        'RGB YOLOv8': (200, 80, 20),
    }

    for i in range(n):
        panels = []
        for model_label, frames_list, det_json_flag in [
            ('ReYOLOv8',   event_frames_reyolo, False),
            ('RVT-small',  event_frames_rvt,    False),
            ('RGB YOLOv8', rgb_frames,           True),
        ]:
            if frames_list and i < len(frames_list):
                img = cv2.imread(frames_list[i])
                if img is None:
                    img = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
                # For RGB panel: overlay detections manually
                if det_json_flag and rgb_dets and i < len(rgb_dets):
                    img = overlay_boxes(img, rgb_dets[i]['boxes'])
            else:
                img = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

            # Resize to panel size if needed
            if img.shape[:2] != (panel_h, panel_w):
                img = cv2.resize(img, (panel_w, panel_h))

            # Label overlay
            color = label_colors[model_label]
            cv2.rectangle(img, (0, 0), (panel_w, 28), (0, 0, 0), -1)
            cv2.putText(img, model_label, (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
            panels.append(img)

        row = np.concatenate(panels, axis=1)
        writer.write(row)

    writer.release()
    print(f"  side-by-side video → {out_video_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────────────────────────────────────

def print_table(rows, headers):
    col_w = [max(len(h), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    fmt = '  '.join(f'{{:<{w}}}' for w in col_w)
    sep = '  '.join('-' * w for w in col_w)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*row))


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def analyse_bag(result_dir, out_dir):
    bag_name = os.path.basename(result_dir.rstrip('/'))
    print(f"\n{'='*60}")
    print(f"Bag: {bag_name}")
    print('='*60)

    all_records = {}
    for key, rel_path, label, color in MODELS:
        full_path = os.path.join(result_dir, rel_path)
        records = load_json(full_path)
        all_records[key] = records
        if records is None:
            print(f"  [{label}] NOT FOUND: {full_path}")
        else:
            n = len(records)
            rate = det_rate(records)
            mean = mean_dets(records)
            tc   = temporal_consistency(records)
            print(f"  [{label}]  frames={n}  det_rate={rate:.1%}  "
                  f"mean_dets/frame={mean:.2f}  mean_run_len={tc:.1f}")

    # ── Figure 1: temporal detection rate ──────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=False)

    for key, _, label, color in MODELS:
        rec = all_records[key]
        if rec:
            plot_temporal(ax1, rec, color, label)
            plot_conf_temporal(ax2, rec, color, label)

    ax1.set_ylabel('Detection rate\n(15-frame rolling avg)', fontsize=9)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_title(f'Temporal detection rate — {bag_name}', fontsize=10)
    ax1.grid(alpha=0.3)

    ax2.set_ylabel('Peak confidence\n(15-frame rolling avg)', fontsize=9)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel('Frame index', fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(out_dir, f'{bag_name}_temporal.png')
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"  temporal plot → {fig_path}")

    # ── Figure 2: per-class detection counts ───────────────────────────────
    classes_seen = set()
    for key, _, _, _ in MODELS:
        rec = all_records[key]
        if rec:
            classes_seen.update(class_counts(rec).keys())
    classes_seen = sorted(classes_seen)

    if classes_seen:
        x = np.arange(len(classes_seen))
        width = 0.25
        fig, ax = plt.subplots(figsize=(max(8, len(classes_seen) * 0.8), 4))
        for idx, (key, _, label, color) in enumerate(MODELS):
            rec = all_records[key]
            if rec is None:
                continue
            cc = class_counts(rec)
            vals = [cc.get(c, 0) for c in classes_seen]
            ax.bar(x + (idx - 1) * width, vals, width, label=label, color=color, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [CLASS_NAMES[c] if c < len(CLASS_NAMES) else str(c) for c in classes_seen],
            rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Total detections')
        ax.set_title(f'Per-class detection counts — {bag_name}', fontsize=10)
        ax.legend(fontsize=8)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        fig_path2 = os.path.join(out_dir, f'{bag_name}_class_hist.png')
        plt.savefig(fig_path2, dpi=150)
        plt.close()
        print(f"  class histogram → {fig_path2}")

    # ── Side-by-side video ─────────────────────────────────────────────────
    vid_path = os.path.join(out_dir, f'{bag_name}_comparison.mp4')
    build_side_by_side_video(result_dir, vid_path, fps=20.0)

    return all_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result_dirs', nargs='+', required=True,
                    help='benchmark_results/<bag_name> directories')
    ap.add_argument('--out_dir', default='benchmark_results/comparison',
                    help='output directory for comparison plots and videos')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Expand globs
    result_dirs = []
    for pattern in args.result_dirs:
        expanded = sorted(glob.glob(pattern))
        result_dirs.extend(expanded if expanded else [pattern])

    # Per-bag analysis
    summary_rows = []
    for rd in result_dirs:
        if not os.path.isdir(rd):
            print(f"[SKIP] not a directory: {rd}")
            continue
        all_rec = analyse_bag(rd, args.out_dir)
        bag = os.path.basename(rd.rstrip('/'))
        for key, _, label, _ in MODELS:
            rec = all_rec.get(key)
            if rec:
                summary_rows.append((
                    bag[:40],
                    label,
                    len(rec),
                    f"{det_rate(rec):.1%}",
                    f"{mean_dets(rec):.2f}",
                    f"{temporal_consistency(rec):.1f}",
                ))

    # Summary table
    if summary_rows:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print('='*60)
        headers = ['Bag', 'Model', 'Frames', 'Det rate', 'Dets/frame', 'Mean run len']
        print_table(summary_rows, headers)

        # Save as CSV
        csv_path = os.path.join(args.out_dir, 'summary.csv')
        with open(csv_path, 'w') as f:
            f.write(','.join(headers) + '\n')
            for row in summary_rows:
                f.write(','.join(str(c) for c in row) + '\n')
        print(f"\nCSV → {csv_path}")

    # ── Combined bar chart across bags ────────────────────────────────────
    if len(result_dirs) > 1:
        # Group by model, one bar cluster per bag
        model_labels = [m[2] for m in MODELS]
        model_colors = [m[3] for m in MODELS]
        bags_short = [os.path.basename(rd.rstrip('/'))[:25] for rd in result_dirs
                      if os.path.isdir(rd)]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, metric, ylabel in [
            (axes[0], 'det_rate', 'Detection rate'),
            (axes[1], 'mean_dets', 'Mean dets / frame'),
        ]:
            x = np.arange(len(bags_short))
            w = 0.25
            for mi, (key, _, label, color) in enumerate(MODELS):
                vals = []
                for rd in result_dirs:
                    if not os.path.isdir(rd):
                        vals.append(0)
                        continue
                    full = os.path.join(rd, dict((m[0], m[1]) for m in MODELS)[key])
                    rec = load_json(full)
                    if rec is None:
                        vals.append(0)
                    elif metric == 'det_rate':
                        vals.append(det_rate(rec))
                    else:
                        vals.append(mean_dets(rec))
                ax.bar(x + (mi - 1) * w, vals, w, label=label, color=color, alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(bags_short, rotation=20, ha='right', fontsize=7)
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)
            ax.grid(axis='y', alpha=0.3)
            if metric == 'det_rate':
                ax.set_ylim(0, 1.0)
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))

        plt.suptitle('WP1 Benchmark: Detection comparison across bags', fontsize=11)
        plt.tight_layout()
        combined_path = os.path.join(args.out_dir, 'combined_comparison.png')
        plt.savefig(combined_path, dpi=150)
        plt.close()
        print(f"\ncombined bar chart → {combined_path}")


if __name__ == '__main__':
    main()

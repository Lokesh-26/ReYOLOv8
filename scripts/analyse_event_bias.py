#!/usr/bin/env python3
"""
Analyse DVXplorer event statistics from WP1 bags and simulate refractory
filtering to estimate target bias settings.

A higher bias_diff threshold ≈ longer per-pixel refractory period.
We sweep refractory period T_ref (µs) and find which value brings
event density and polarity ratio closest to the MTevent training target.

Run with system Python3 (has rosbag):
    python3 scripts/analyse_event_bias.py \
        --bag  /home/loki/bags/wp1_rgb20ms_bright_hall_20260526_135855.bag \
        --out_dir benchmark_results/bias_analysis
"""
import os
import sys
import argparse
import numpy as np

# ── MTevent training statistics (target) ────────────────────────────────────
# Measured from the preprocessed vtei_mtevent_640x480 dataset
TARGET_DENSITY  = 0.080   # mean fraction of nonzero pixels per 50ms voxel bin
TARGET_POL_RATIO = 1.14   # neg_count / pos_count

H, W = 480, 640           # DVXplorer resolution
BIN_MS = 50               # 50 ms voxel window
N_BINS = 5                # number of temporal bins

REFRACTORY_SWEEP_US = [0, 500, 1000, 2000, 3000, 5000, 8000, 12000, 20000]


def read_events(bag_path, topic='/dvxplorer_left/events', max_msgs=None):
    """Read all events from bag; return (t_us, x, y, p) numpy arrays."""
    import rosbag
    ts, xs, ys, ps = [], [], [], []
    bag = rosbag.Bag(bag_path)
    for i, (_, msg, _) in enumerate(bag.read_messages(topic)):
        if max_msgs and i >= max_msgs:
            break
        for e in msg.events:
            ts.append(e.ts.to_nsec() // 1000)   # ns → µs
            xs.append(e.x)
            ys.append(e.y)
            ps.append(int(e.polarity))
    bag.close()
    t = np.array(ts,  dtype=np.int64)
    x = np.array(xs,  dtype=np.int16)
    y = np.array(ys,  dtype=np.int16)
    p = np.array(ps,  dtype=np.uint8)
    # sort by time (usually already sorted within each message)
    order = np.argsort(t)
    return t[order], x[order], y[order], p[order]


def apply_refractory(t, x, y, p, ref_us):
    """Drop events fired within ref_us µs of the last event at the same pixel."""
    if ref_us == 0:
        return t, x, y, p
    last_fire = np.full((H, W), -ref_us - 1, dtype=np.int64)
    keep = np.zeros(len(t), dtype=bool)
    for i in range(len(t)):
        xi, yi, ti = int(x[i]), int(y[i]), t[i]
        if ti - last_fire[yi, xi] >= ref_us:
            keep[i] = True
            last_fire[yi, xi] = ti
    return t[keep], x[keep], y[keep], p[keep]


def compute_density_and_polarity(t, x, y, p):
    """
    Split event stream into 50 ms / 5-bin voxel grids.
    Return mean density (fraction nonzero pixels per bin) and neg/pos ratio.
    """
    if len(t) == 0:
        return 0.0, 1.0

    t0, t1 = t[0], t[-1]
    window_us = BIN_MS * 1000  # µs
    bin_us = window_us // N_BINS

    densities = []
    n_pos = n_neg = 0

    win_start = t0
    while win_start + window_us <= t1:
        win_end = win_start + window_us
        idx = np.searchsorted(t, [win_start, win_end])
        wt, wx, wy, wp = t[idx[0]:idx[1]], x[idx[0]:idx[1]], y[idx[0]:idx[1]], p[idx[0]:idx[1]]

        # per-bin density
        for b in range(N_BINS):
            b_start = win_start + b * bin_us
            b_end   = b_start + bin_us
            bidx = np.searchsorted(wt, [b_start, b_end])
            bx = wx[bidx[0]:bidx[1]]
            by = wy[bidx[0]:bidx[1]]
            nonzero = len(np.unique(np.stack([bx, by], axis=1), axis=0))
            densities.append(nonzero / (H * W))

        n_pos += int(np.sum(wp == 1))
        n_neg += int(np.sum(wp == 0))
        win_start = win_end

    mean_density = float(np.mean(densities)) if densities else 0.0
    pol_ratio = n_neg / n_pos if n_pos > 0 else float('inf')
    return mean_density, pol_ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag', required=True)
    ap.add_argument('--topic', default='/dvxplorer_left/events')
    ap.add_argument('--out_dir', default='benchmark_results/bias_analysis')
    ap.add_argument('--max_msgs', type=int, default=None,
                    help='limit number of event messages read (faster testing)')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    bag_name = os.path.basename(args.bag).replace('.bag', '')

    print(f'Reading events from {args.bag} ...')
    t, x, y, p = read_events(args.bag, args.topic, args.max_msgs)
    duration_s = (t[-1] - t[0]) / 1e6
    total_events = len(t)
    print(f'  {total_events:,} events over {duration_s:.1f}s  '
          f'({total_events/duration_s/1e6:.2f} Mev/s)')

    print(f'\n{"T_ref(µs)":>10}  {"kept%":>7}  {"density":>9}  '
          f'{"pol_ratio":>10}  {"vs_target_density":>18}  note')
    print('-' * 75)

    results = []
    for ref_us in REFRACTORY_SWEEP_US:
        tf, xf, yf, pf = apply_refractory(t, x, y, p, ref_us)
        kept_pct = 100.0 * len(tf) / len(t)
        density, pol_ratio = compute_density_and_polarity(tf, xf, yf, pf)
        gap = density - TARGET_DENSITY
        note = ''
        if abs(density - TARGET_DENSITY) < 0.015:
            note = '  <<< CLOSE TO TARGET'
        elif density < TARGET_DENSITY:
            note = '  <<< BELOW TARGET'
        results.append((ref_us, kept_pct, density, pol_ratio))
        print(f'{ref_us:>10,}  {kept_pct:>6.1f}%  {density:>9.4f}  '
              f'{pol_ratio:>10.3f}  {gap:>+18.4f}{note}')

    print(f'\nTarget: density={TARGET_DENSITY}  pol_ratio={TARGET_POL_RATIO}')

    # Find best refractory
    best = min(results, key=lambda r: abs(r[2] - TARGET_DENSITY))
    print(f'\nBest refractory: T_ref={best[0]:,} µs  '
          f'(density={best[2]:.4f}, pol_ratio={best[3]:.3f})')

    # Translate refractory period to rough bias guidance
    print('\n── Bias tuning guidance ─────────────────────────────────────────')
    print(f'Current density {results[0][2]:.4f} is '
          f'{results[0][2]/TARGET_DENSITY:.1f}x the training target.')
    print(f'Refractory period of ~{best[0]:,} µs achieves target density.')
    print()
    print('DVXplorer bias parameters to change (in record_wp1.launch or dyn_reconf):')
    print('  bias_diff_on  : raise by ~10–20 per 2x density reduction')
    print('  bias_diff_off : raise more (current bag has excess neg events)')
    print()
    # Rough empirical mapping: each +10 on bias ≈ doubles refractory period from ~500µs baseline
    ratio = results[0][2] / TARGET_DENSITY
    if ratio > 1:
        bump_on  = int(round(10 * np.log2(ratio)))
        bump_off = int(round(bump_on * (results[0][3] / TARGET_POL_RATIO)))
        print(f'  Estimated starting point:')
        print(f'    bias_diff_on  += {bump_on}   (raise sensitivity threshold)')
        print(f'    bias_diff_off += {bump_off}  (raise more to fix polarity imbalance)')
        print()
        print('  These are starting values. Fine-tune on-camera with:')
        print('    rosrun rqt_reconfigure rqt_reconfigure')
        print('  and watch /dvxplorer_left/events rate drop to ~%.1f Mev/s' %
              (total_events / duration_s / 1e6 / ratio))

    # Save results CSV
    csv_path = os.path.join(args.out_dir, f'{bag_name}_refractory_sweep.csv')
    with open(csv_path, 'w') as f:
        f.write('T_ref_us,kept_pct,density,pol_ratio\n')
        for r in results:
            f.write('%d,%.2f,%.5f,%.4f\n' % r)
    print(f'\nCSV → {csv_path}')

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        refs = [r[0] for r in results]
        dens = [r[2] for r in results]
        pols = [r[3] for r in results]

        ax1.plot(refs, dens, 'o-', color='#e6194b', lw=2, label='bag density')
        ax1.axhline(TARGET_DENSITY, color='#3cb44b', ls='--', lw=1.5,
                    label=f'training target ({TARGET_DENSITY})')
        ax1.set_ylabel('Mean event density\n(nonzero pixels / total pixels)')
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)
        ax1.set_title(f'Refractory period sweep — {bag_name}')

        ax2.plot(refs, pols, 'o-', color='#4363d8', lw=2, label='bag neg/pos ratio')
        ax2.axhline(TARGET_POL_RATIO, color='#3cb44b', ls='--', lw=1.5,
                    label=f'training target ({TARGET_POL_RATIO})')
        ax2.set_ylabel('Polarity ratio (neg/pos)')
        ax2.set_xlabel('Refractory period (µs)')
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)
        ax2.set_xscale('symlog', linthresh=100)

        plt.tight_layout()
        plot_path = os.path.join(args.out_dir, f'{bag_name}_refractory_sweep.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f'plot   → {plot_path}')
    except ImportError:
        pass


if __name__ == '__main__':
    main()

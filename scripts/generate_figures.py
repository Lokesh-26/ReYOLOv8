#!/usr/bin/env python3
"""Generate paper figures: performance curve, dataset overview, qualitative sequential."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import h5py, cv2

OUT_DIR = '/home/loki/event/paper/journal_draft/figures'
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9,
    'axes.labelsize': 9, 'axes.titlesize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'lines.linewidth': 1.6,
    'lines.markersize': 6, 'figure.dpi': 300,
})

# ─── Figure 1: Performance curve ──────────────────────────────────────────────
def fig_performance_curve():
    clip_lengths = [1, 5, 11, 21]
    seq_17  = [0.1655, 0.2114, 0.2176, 0.2247]
    clip_ev = [0.2511, 0.2179, 0.2138, 0.2147]
    seq_15  = [0.2038, 0.2536, 0.2639, 0.2647]

    tc_clip_len = 11
    tc_seq_17 = 0.2389; tc_seq_15 = 0.2894; tc_clip_v = 0.2388

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))

    # Left: sequential vs clip-batch
    ax = axes[0]
    ax.plot(clip_lengths, seq_17, 'o-', color='#1f77b4', label='Sequential', zorder=3)
    ax.plot(clip_lengths, clip_ev, 's--', color='#ff7f0e', label='Clip-batch', zorder=3)
    ax.scatter([tc_clip_len], [tc_seq_17], marker='*', s=140, color='#1f77b4',
               zorder=5, label='TC (seq.)')
    ax.scatter([tc_clip_len], [tc_clip_v], marker='*', s=140, color='#ff7f0e',
               zorder=5, label='TC (clip)')
    ax.set_xlabel('Clip length'); ax.set_ylabel(r'AP$_{50}$ (clean val, 17-cls)')
    ax.set_title('Sequential vs.\ clip-batch evaluation')
    ax.set_xticks(clip_lengths); ax.set_ylim(0.14, 0.27)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    ax.annotate('C1: highest clip-batch,\nlowest sequential',
                xy=(1, seq_17[0]), xytext=(5, 0.155),
                arrowprops=dict(arrowstyle='->', color='#555', lw=0.8),
                fontsize=7, color='#555')

    # Right: 17cls vs 15cls sequential
    ax = axes[1]
    ax.plot(clip_lengths, seq_17, 'o-', color='#1f77b4', label='17-class', zorder=3)
    ax.plot(clip_lengths, seq_15, 'D-', color='#2ca02c', label='15-class (KLT merged)', zorder=3)
    ax.scatter([tc_clip_len], [tc_seq_17], marker='*', s=140, color='#1f77b4', zorder=5)
    ax.scatter([tc_clip_len], [tc_seq_15], marker='*', s=140, color='#2ca02c', zorder=5,
               label='TC C11')
    ax.set_xlabel('Clip length'); ax.set_ylabel(r'Sequential AP$_{50}$ (clean val)')
    ax.set_title('17-class vs.\ 15-class (KLT merged)')
    ax.set_xticks(clip_lengths)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))

    fig.tight_layout(pad=1.2)
    fig.savefig(f'{OUT_DIR}/fig_performance_curve.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT_DIR}/fig_performance_curve.png', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print('Saved fig_performance_curve')


# ─── Shared H5 helpers ────────────────────────────────────────────────────────
H5_VAL  = 'preprocessed_datasets/vtei_mtevent_50ms_5bin_640x480/images/val/mtevent_val.h5'
LBL_VAL = 'preprocessed_datasets/vtei_mtevent_50ms_5bin_640x480/labels/val'

CLASS_NAMES = [
    'wooden_pallet','small_klt','big_klt','blue_klt','amazon_luggage',
    'ikea_dammang_bin','ikea_vesken_trolley','ikea_sortera_bin',
    'ikea_drona_grey','ikea_drona_blue','ikea_knallig_box','ikea_moppe_drawer',
    'ikea_labbsal_basket','ikea_ivar_box','ikea_skubb_case','ikea_samla_box','human',
]
COLORS = [
    (220,50,50),(50,150,220),(50,200,50),(180,50,220),(220,140,50),(50,50,200),
    (200,200,50),(130,50,220),(50,200,130),(200,50,130),(130,200,50),(50,130,130),
    (130,50,130),(200,130,130),(130,130,200),(130,200,130),(50,180,220),
]

def build_scene_index():
    """Returns dict: scene_id (int) -> (h5_start, h5_end, label_array)."""
    lbls = sorted(os.listdir(LBL_VAL))
    idx = {}
    offset = 0
    for lf in lbls:
        sid = int(lf.replace('scene_', '').replace('.npy', ''))
        arr = np.load(os.path.join(LBL_VAL, lf), allow_pickle=True)
        n = arr.shape[0]
        idx[sid] = (offset, offset + n, arr)
        offset += n
    return idx

def render_frame(voxel_5hw, scale=1.0):
    """Sum signed channels -> normalised grey BGR."""
    s = voxel_5hw.astype(np.float32).sum(axis=0)
    H, W = s.shape
    nz = s[s != 0]
    vmax = float(np.percentile(np.abs(nz), 95)) if len(nz) else 1.0
    vmax = max(vmax, 1.0)
    norm = np.clip(s / vmax, -1, 1)
    gray = ((norm + 1) / 2 * 255).astype(np.uint8)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if scale != 1.0:
        bgr = cv2.resize(bgr, (int(W*scale), int(H*scale)), interpolation=cv2.INTER_AREA)
    return bgr

def draw_boxes(bgr, label_row, H, W, scale=1.0):
    """label_row: (max_boxes, 5) array of [cls, cx, cy, w, h], zeros = padding."""
    img = bgr.copy()
    sH, sW = int(H*scale), int(W*scale)
    for box in label_row:
        if box.sum() == 0:
            continue
        cls, cx, cy, bw, bh = int(box[0]), *box[1:]
        x1 = int((cx - bw/2)*sW); y1 = int((cy - bh/2)*sH)
        x2 = int((cx + bw/2)*sW); y2 = int((cy + bh/2)*sH)
        c = COLORS[cls % len(COLORS)]
        cv2.rectangle(img, (x1,y1), (x2,y2), c, 1)
        short = CLASS_NAMES[cls][:12] if cls < len(CLASS_NAMES) else str(cls)
        cv2.putText(img, short, (x1, max(y1-2, 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, c, 1, cv2.LINE_AA)
    return img

# ─── Figure 2: Dataset overview ────────────────────────────────────────────────
def fig_dataset_overview():
    if not os.path.exists(H5_VAL):
        print('H5 not found, skipping dataset overview.'); return
    scene_idx = build_scene_index()

    # Clean-val scenes with interesting content: 3, 14, 26
    target = [3, 14, 26]
    target = [s for s in target if s in scene_idx][:3]
    if not target:
        print('No target scenes found.'); return

    with h5py.File(H5_VAL) as f:
        frames = f['1mp']  # (4238, 5, 480, 640)
        panels = []
        for sid in target:
            start, end, labels = scene_idx[sid]
            n = end - start
            fi = n // 2   # mid-scene
            voxel = frames[start + fi]  # (5, H, W)
            H_, W_ = voxel.shape[1], voxel.shape[2]
            bgr = render_frame(voxel, scale=0.5)
            bgr = draw_boxes(bgr, labels[fi, :, :], H_, W_, scale=0.5)
            panels.append((bgr, f'Scene {sid} (frame {fi+1}/{n})'))

    ncols = len(panels)
    fig, axes = plt.subplots(1, ncols, figsize=(6.8, 2.2))
    if ncols == 1: axes = [axes]
    for ax, (bgr, title) in zip(axes, panels):
        ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=7.5); ax.axis('off')
    fig.suptitle('MTEvent — signed temporal voxel representation with GT boxes (640×480→320×240 display)',
                 fontsize=7.5)
    fig.tight_layout(pad=0.5)
    fig.savefig(f'{OUT_DIR}/fig_dataset_overview.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT_DIR}/fig_dataset_overview.png', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print('Saved fig_dataset_overview')


# ─── Figure 3: Qualitative sequential (GT-only, model preds added later) ───────
def fig_qualitative_sequential():
    if not os.path.exists(H5_VAL):
        print('H5 not found, skipping qualitative figure.'); return
    scene_idx = build_scene_index()

    sid = 14 if 14 in scene_idx else list(scene_idx.keys())[0]
    start, end, labels = scene_idx[sid]
    n = end - start

    # 4 columns: frames 0, 1, n//2, n//2+1
    picks = [0, 1, n//2, n//2+1]
    pick_labels = [f'Cold f.{i+1}' if i < 2 else f'Mid f.{i+1}' for i in picks]

    with h5py.File(H5_VAL) as f:
        frames_arr = f['1mp']
        panels = []
        for fi, lbl in zip(picks, pick_labels):
            voxel = frames_arr[start + fi]
            H_, W_ = voxel.shape[1], voxel.shape[2]
            bgr = render_frame(voxel, scale=0.5)
            bgr = draw_boxes(bgr, labels[fi, :, :], H_, W_, scale=0.5)
            panels.append((bgr, lbl))

    fig, axes = plt.subplots(1, 4, figsize=(6.8, 1.9))
    for ax, (bgr, title) in zip(axes, panels):
        ax.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=7.5); ax.axis('off')
    fig.suptitle(f'Scene {sid} — GT annotations. Model predictions (C1 / C21 / TC) to be overlaid.',
                 fontsize=7.5)
    fig.tight_layout(pad=0.3)
    fig.savefig(f'{OUT_DIR}/fig_qualitative_gt_only.pdf', bbox_inches='tight')
    fig.savefig(f'{OUT_DIR}/fig_qualitative_gt_only.png', bbox_inches='tight', dpi=200)
    plt.close(fig)
    print('Saved fig_qualitative_gt_only')


if __name__ == '__main__':
    os.chdir('/home/loki/event/ReYOLOv8')
    fig_performance_curve()
    fig_dataset_overview()
    fig_qualitative_sequential()
    print('Done.')

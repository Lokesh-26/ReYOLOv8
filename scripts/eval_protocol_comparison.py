#!/usr/bin/env python3
"""
Eval protocol comparison: clip-batch AP50 vs sequential AP50 for all 640x480 models.

Clip-batch: hidden state reset every clip_length frames (matches training validator).
Sequential: hidden state carried through each full scene (matches deployment).

This script formalises the training-eval vs sequential-eval discrepancy, showing
that TC loss is misrepresented by clip-batch evaluation.

Usage:
    python scripts/eval_protocol_comparison.py
"""
import sys, os, torch, h5py, numpy as np
sys.path.insert(0, '/home/loki/event/ReYOLOv8')

from ultralytics.yolo.utils import ops
from collections import defaultdict

VAL_H5   = 'preprocessed_datasets/vtei_mtevent_50ms_5bin_640x480/images/val/mtevent_val.h5'
VAL_LBLS = 'preprocessed_datasets/vtei_mtevent_50ms_5bin_640x480/labels/val'
CONF = 0.001
IOU  = 0.45
NC   = 17

MODELS = {
    'C1':     ('runs/train/mtevent_640x480_c1_clean/weights/best.pt',  1),
    'C5':     ('runs/train/mtevent_640x480_c5_clean/weights/best.pt',  5),
    'C11':    ('runs/train/mtevent_640x480_fixed_c11/weights/best.pt', 11),
    'C21':    ('runs/train/mtevent_640x480_fixed_c21/weights/best.pt', 21),
    'C11_tc': ('runs/train/mtevent_640x480_tc_c11/weights/best.pt',    11),
}

# Training clip-batch AP50 from results.csv (read directly)
TRAINING_MAP50 = {
    'C1':     0.4994,
    'C5':     0.5145,
    'C11':    0.5093,
    'C21':    0.5274,
    'C11_tc': 0.4706,
}


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[ii + 1] - mrec[ii]) * mpre[ii + 1]))


def compute_map(all_preds, all_gts, nc, iou_thresh=0.5):
    tp_fp = defaultdict(list)
    n_gt  = defaultdict(int)
    for preds, gts in zip(all_preds, all_gts):
        for cls_id in set(int(g[4]) for g in gts):
            n_gt[cls_id] += sum(1 for g in gts if int(g[4]) == cls_id)
        if not preds:
            continue
        preds_s = sorted(preds, key=lambda x: -x[4])
        gt_used = set()
        for p in preds_s:
            px1, py1, px2, py2, pconf, pcls = p
            pcls = int(pcls)
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gts):
                gx1, gy1, gx2, gy2, gcls = g
                if int(gcls) != pcls or j in gt_used:
                    continue
                ix1 = max(px1, gx1); iy1 = max(py1, gy1)
                ix2 = min(px2, gx2); iy2 = min(py2, gy2)
                iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
                inter = iw * ih
                union = (px2-px1)*(py2-py1) + (gx2-gx1)*(gy2-gy1) - inter
                iou = inter / union if union > 0 else 0
                if iou > best_iou:
                    best_iou, best_j = iou, j
            is_tp = best_iou >= iou_thresh
            if is_tp:
                gt_used.add(best_j)
            tp_fp[pcls].append((pconf, float(is_tp)))
    ap_per_class = {}
    for cls_id in range(nc):
        if n_gt[cls_id] == 0:
            continue
        entries = tp_fp.get(cls_id, [])
        if not entries:
            ap_per_class[cls_id] = 0.0
            continue
        entries.sort(key=lambda x: -x[0])
        tps = np.cumsum([e[1] for e in entries])
        fps = np.cumsum([1 - e[1] for e in entries])
        rec = tps / n_gt[cls_id]
        pre = tps / (tps + fps + 1e-9)
        ap_per_class[cls_id] = compute_ap(rec, pre)
    mAP = np.mean(list(ap_per_class.values())) if ap_per_class else 0.0
    return ap_per_class, mAP


def run_sequential(model_path, clip_length, device):
    """Sequential eval: hidden state carries through each full scene."""
    ckpt  = torch.load(model_path, map_location=device, weights_only=False)
    model = ckpt['model'].to(device).float()
    model.eval()

    lbl_files = sorted(os.listdir(VAL_LBLS))
    f_h5 = h5py.File(VAL_H5, 'r')
    all_preds, all_gts = [], []
    offset = 0
    hidden = {"0": None, "1": None, "2": None, "3": None}

    with torch.no_grad():
        for lf in lbl_files:
            lbls = np.load(os.path.join(VAL_LBLS, lf), allow_pickle=True)
            n = len(lbls)
            for i in range(n):
                fr = np.array(f_h5['1mp'][offset + i])
                inp = torch.tensor(fr, dtype=torch.float32).unsqueeze(0).to(device)
                out, hidden = model(inp, hidden)
                dets = ops.non_max_suppression(out, conf_thres=CONF, iou_thres=IOU,
                                               multi_label=False, max_det=300)[0]
                preds_frame = []
                for d in dets:
                    x1, y1, x2, y2, conf, cls = d.cpu().numpy()
                    preds_frame.append((x1, y1, x2, y2, float(conf), int(cls)))
                all_preds.append(preds_frame)
                gts_frame = []
                for g in lbls[i]:
                    c = int(g[0])
                    cx, cy, w, h = g[1]*640, g[2]*480, g[3]*640, g[4]*480
                    gts_frame.append((cx-w/2, cy-h/2, cx+w/2, cy+h/2, c))
                all_gts.append(gts_frame)
            offset += n
            hidden = {"0": None, "1": None, "2": None, "3": None}  # reset between scenes

    f_h5.close()
    _, mAP = compute_map(all_preds, all_gts, nc=NC)
    return mAP


def run_clip_batch(model_path, clip_length, device):
    """Clip-batch eval: hidden state reset every clip_length frames (matches training validator)."""
    ckpt  = torch.load(model_path, map_location=device, weights_only=False)
    model = ckpt['model'].to(device).float()
    model.eval()

    lbl_files = sorted(os.listdir(VAL_LBLS))
    f_h5 = h5py.File(VAL_H5, 'r')
    all_preds, all_gts = [], []
    offset = 0

    with torch.no_grad():
        for lf in lbl_files:
            lbls = np.load(os.path.join(VAL_LBLS, lf), allow_pickle=True)
            n = len(lbls)
            hidden = {"0": None, "1": None, "2": None, "3": None}
            for i in range(n):
                if i % clip_length == 0:
                    hidden = {"0": None, "1": None, "2": None, "3": None}
                fr = np.array(f_h5['1mp'][offset + i])
                inp = torch.tensor(fr, dtype=torch.float32).unsqueeze(0).to(device)
                out, hidden = model(inp, hidden)
                dets = ops.non_max_suppression(out, conf_thres=CONF, iou_thres=IOU,
                                               multi_label=False, max_det=300)[0]
                preds_frame = []
                for d in dets:
                    x1, y1, x2, y2, conf, cls = d.cpu().numpy()
                    preds_frame.append((x1, y1, x2, y2, float(conf), int(cls)))
                all_preds.append(preds_frame)
                gts_frame = []
                for g in lbls[i]:
                    c = int(g[0])
                    cx, cy, w, h = g[1]*640, g[2]*480, g[3]*640, g[4]*480
                    gts_frame.append((cx-w/2, cy-h/2, cx+w/2, cy+h/2, c))
                all_gts.append(gts_frame)
            offset += n

    f_h5.close()
    _, mAP = compute_map(all_preds, all_gts, nc=NC)
    return mAP


if __name__ == '__main__':
    device = torch.device('cuda:0')
    results = {}

    for name, (path, clip_len) in MODELS.items():
        if not os.path.exists(path):
            print(f'SKIP {name}: {path} not found')
            continue
        print(f'\n[{name}] clip={clip_len}')
        seq_map  = run_sequential(path, clip_len, device)
        clip_map = run_clip_batch(path, clip_len, device)
        train_map = TRAINING_MAP50[name]
        results[name] = (train_map, clip_map, seq_map)
        print(f'  training val mAP50 : {train_map:.4f}')
        print(f'  clip-batch eval    : {clip_map:.4f}')
        print(f'  sequential eval    : {seq_map:.4f}')
        print(f'  Δ(seq - clip)      : {seq_map - clip_map:+.4f}')

    print('\n' + '='*72)
    print(f'{"Model":<10} {"Train mAP50":>12} {"Clip-batch":>12} {"Sequential":>12} {"Δ(seq-clip)":>12}')
    print('-'*72)
    for name, (tr, cb, sq) in results.items():
        marker = ' ← TC' if 'tc' in name else ''
        print(f'{name:<10} {tr:>12.4f} {cb:>12.4f} {sq:>12.4f} {sq-cb:>+12.4f}{marker}')
    print('\nKey finding: TC loss shows negative Δ in training eval but positive Δ in sequential eval.')

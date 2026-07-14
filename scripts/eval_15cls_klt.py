#!/usr/bin/env python3
"""
Sequential KLT-eval for models trained on 15-class merged-KLT labels.
Evaluates models whose output heads already use 15 classes (klt merged at training time).

Usage:
    python scripts/eval_15cls_klt.py [--checkpoint path/to/best.pt]
"""
import sys, os, argparse, torch, h5py, numpy as np
sys.path.insert(0, '/home/loki/event/ReYOLOv8')

from ultralytics.yolo.utils import ops

MODELS = {
    '15cls_C21_v2': 'runs/train/mtevent_640x480_15cls_c21_v2/weights/best.pt',
}

VAL_H5   = 'preprocessed_datasets/vtei_mtevent_50ms_5bin_640x480_15cls/images/val/mtevent_val.h5'
VAL_LBLS = 'preprocessed_datasets/vtei_mtevent_50ms_5bin_640x480_15cls/labels/val'
CONF = 0.001
IOU  = 0.45

CLASS_NAMES_15 = [
    'wooden_pallet', 'klt', 'amazon_luggage', 'ikea_dammang_bin', 'ikea_vesken_trolley',
    'ikea_sortera_bin', 'ikea_drona_grey', 'ikea_drona_blue', 'ikea_knallig_box',
    'ikea_moppe_drawer', 'ikea_labbsal_basket', 'ikea_ivar_box', 'ikea_skubb_case',
    'ikea_samla_box', 'human',
]


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[ii + 1] - mrec[ii]) * mpre[ii + 1]))


def compute_map(all_preds, all_gts, nc=15, iou_thresh=0.5):
    from collections import defaultdict
    tp_fp = defaultdict(list)
    n_gt  = defaultdict(int)

    for preds, gts in zip(all_preds, all_gts):
        for cls_id in set(int(g[4]) for g in gts):
            n_gt[cls_id] += sum(1 for g in gts if int(g[4]) == cls_id)

        if len(preds) == 0:
            continue
        preds_s = sorted(preds, key=lambda x: -x[4])

        matched_gt = set()
        for p in preds_s:
            px1,py1,px2,py2,pconf,pcls = p
            best_iou, best_idx = 0.0, -1
            for gi, g in enumerate(gts):
                if int(g[4]) != int(pcls) or gi in matched_gt:
                    continue
                ix1 = max(px1, g[0]); iy1 = max(py1, g[1])
                ix2 = min(px2, g[2]); iy2 = min(py2, g[3])
                inter = max(0, ix2-ix1) * max(0, iy2-iy1)
                union = (px2-px1)*(py2-py1) + (g[2]-g[0])*(g[3]-g[1]) - inter
                iou = inter/union if union > 0 else 0.0
                if iou > best_iou:
                    best_iou, best_idx = iou, gi
            is_tp = (best_iou >= iou_thresh and best_idx >= 0)
            if is_tp:
                matched_gt.add(best_idx)
            tp_fp[int(pcls)].append((pconf, 1 if is_tp else 0))

    ap_per_class = {}
    for cls_id in range(nc):
        entries = sorted(tp_fp.get(cls_id, []), key=lambda x: -x[0])
        total_gt = n_gt.get(cls_id, 0)
        if total_gt == 0:
            continue
        tps = np.cumsum([e[1] for e in entries], dtype=float)
        fps = np.cumsum([1-e[1] for e in entries], dtype=float)
        rec = tps / total_gt
        pre = tps / (tps + fps + 1e-9)
        ap_per_class[cls_id] = compute_ap(rec, pre)

    mAP = np.mean(list(ap_per_class.values())) if ap_per_class else 0.0
    return ap_per_class, mAP


def run_eval(model_name, model_path):
    device = torch.device('cuda:0')
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
            hidden = {"0": None, "1": None, "2": None, "3": None}

    f_h5.close()

    ap_dict, mAP = compute_map(all_preds, all_gts, nc=15)

    print(f'\n{"="*60}')
    print(f'Model: {model_name}  [15cls native]  mAP50={mAP:.4f}')
    print(f'{"Class":<22} {"AP50":>6}')
    print('-'*30)
    for cls_id, name in enumerate(CLASS_NAMES_15):
        ap = ap_dict.get(cls_id, float('nan'))
        print(f'  {name:<20} {ap:>6.3f}')
    print(f'{"  mAP50":<22} {mAP:>6.4f}')
    return mAP


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None,
                        help='Override checkpoint path (default: use MODELS dict)')
    args = parser.parse_args()

    if args.checkpoint:
        models_to_eval = {'15cls_C21_v2_custom': args.checkpoint}
    else:
        models_to_eval = MODELS

    results = {}
    for name, path in models_to_eval.items():
        if not os.path.exists(path):
            print(f'SKIP {name}: {path} not found')
            continue
        print(f'\nEvaluating {name}...')
        results[name] = run_eval(name, path)

    print('\n' + '='*60)
    print('SUMMARY — 15cls native eval')
    print(f'{"Model":<30} {"mAP50":>7}')
    print('-'*40)
    for name, m15 in results.items():
        print(f'  {name:<28} {m15:>7.4f}')

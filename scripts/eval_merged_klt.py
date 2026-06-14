#!/usr/bin/env python3
"""
Ablation: evaluate models with KLT sub-classes merged into one 'klt' class.
small_klt(1), big_klt(2), blue_klt(3) → klt(1). Classes 4-16 shift down by 2.

Usage:
    python scripts/eval_merged_klt.py

Outputs per-class AP for both 17-class and 15-class (merged) evaluations,
so the KLT-confusion penalty is quantified.
"""
import sys, os, torch, h5py, numpy as np
sys.path.insert(0, '/home/loki/event/ReYOLOv8')

from ultralytics.yolo.utils import ops
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
MODELS = {
    'left_C1_best':  'runs/train/mtevent_640x480_c1_clean/weights/best.pt',
    'left_C5_best':  'runs/train/mtevent_640x480_c5_clean/weights/best.pt',
    'left_C11_best': 'runs/train/mtevent_640x480_fixed_c11/weights/best.pt',
    'left_C21_best': 'runs/train/mtevent_640x480_fixed_c21/weights/best.pt',
    'left_C11_tc':   'runs/train/mtevent_640x480_tc_c11/weights/best.pt',
}
VAL_H5   = 'preprocessed_datasets/vtei_mtevent_50ms_5bin_640x480/images/val/mtevent_val.h5'
VAL_LBLS = 'preprocessed_datasets/vtei_mtevent_50ms_5bin_640x480/labels/val'
CONF = 0.001
IOU  = 0.45

CLASS_NAMES_17 = [
    'wooden_pallet','small_klt','big_klt','blue_klt','amazon_luggage',
    'ikea_dammang_bin','ikea_vesken_trolley','ikea_sortera_bin',
    'ikea_drona_grey','ikea_drona_blue','ikea_knallig_box','ikea_moppe_drawer',
    'ikea_labbsal_basket','ikea_ivar_box','ikea_skubb_case','ikea_samla_box','human',
]
CLASS_NAMES_15 = [
    'wooden_pallet','klt','amazon_luggage','ikea_dammang_bin','ikea_vesken_trolley',
    'ikea_sortera_bin','ikea_drona_grey','ikea_drona_blue','ikea_knallig_box',
    'ikea_moppe_drawer','ikea_labbsal_basket','ikea_ivar_box','ikea_skubb_case',
    'ikea_samla_box','human',
]

def remap_17_to_15(cls_id):
    """Map 17-class id → 15-class id. Classes 1,2,3 → 1 (klt). 4-16 → 2-14."""
    if cls_id in (1, 2, 3):
        return 1
    elif cls_id >= 4:
        return cls_id - 2
    return cls_id  # 0 stays 0

# ── Simple AP computation ─────────────────────────────────────────────────────
def compute_ap(recall, precision):
    """Compute AP using all-point interpolation."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[ii + 1] - mrec[ii]) * mpre[ii + 1]))

def compute_map(all_preds, all_gts, nc, iou_thresh=0.5):
    """
    all_preds: list of (x1,y1,x2,y2,conf,cls) arrays per frame
    all_gts:   list of (x1,y1,x2,y2,cls) arrays per frame
    Returns: per-class AP dict, mAP
    """
    from collections import defaultdict
    tp_fp = defaultdict(list)   # cls → list of (conf, is_tp)
    n_gt  = defaultdict(int)

    for preds, gts in zip(all_preds, all_gts):
        gt_used = set()
        for cls_id in set(int(g[4]) for g in gts):
            n_gt[cls_id] += sum(1 for g in gts if int(g[4]) == cls_id)

        # Sort preds by conf descending
        if len(preds) == 0:
            continue
        preds_s = sorted(preds, key=lambda x: -x[4])
        for p in preds_s:
            px1,py1,px2,py2,pconf,pcls = p
            pcls = int(pcls)
            # Find best matching GT
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gts):
                gx1,gy1,gx2,gy2,gcls = g
                if int(gcls) != pcls or j in gt_used:
                    continue
                ix1 = max(px1,gx1); iy1 = max(py1,gy1)
                ix2 = min(px2,gx2); iy2 = min(py2,gy2)
                iw = max(0, ix2-ix1); ih = max(0, iy2-iy1)
                inter = iw*ih
                union = (px2-px1)*(py2-py1) + (gx2-gx1)*(gy2-gy1) - inter
                iou = inter/union if union > 0 else 0
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
        fps = np.cumsum([1-e[1] for e in entries])
        rec = tps / n_gt[cls_id]
        pre = tps / (tps + fps + 1e-9)
        ap_per_class[cls_id] = compute_ap(rec, pre)

    mAP = np.mean(list(ap_per_class.values())) if ap_per_class else 0.0
    return ap_per_class, mAP


def run_eval(model_name, model_path, remap=False):
    device = torch.device('cuda:0')
    ckpt  = torch.load(model_path, map_location=device, weights_only=False)
    model = ckpt['model'].to(device).float(); model.eval()
    nc = 15 if remap else 17
    names = CLASS_NAMES_15 if remap else CLASS_NAMES_17

    # Load val scenes in order
    lbl_files = sorted(os.listdir(VAL_LBLS))
    f_h5 = h5py.File(VAL_H5, 'r')

    all_preds, all_gts = [], []
    offset = 0
    hidden = {"0":None,"1":None,"2":None,"3":None}

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
                # Build pred list (pixel coords, conf, cls)
                preds_frame = []
                for d in dets:
                    x1,y1,x2,y2,conf,cls = d.cpu().numpy()
                    c = int(cls)
                    if remap: c = remap_17_to_15(c)
                    preds_frame.append((x1,y1,x2,y2,float(conf),c))
                all_preds.append(preds_frame)

                # Build GT list (pixel coords, cls)
                gts_frame = []
                for g in lbls[i]:
                    c = int(g[0])
                    if remap: c = remap_17_to_15(c)
                    cx,cy,w,h = g[1]*640, g[2]*480, g[3]*640, g[4]*480
                    gts_frame.append((cx-w/2, cy-h/2, cx+w/2, cy+h/2, c))
                all_gts.append(gts_frame)

            offset += n
            hidden = {"0":None,"1":None,"2":None,"3":None}  # reset between scenes

    f_h5.close()

    ap_dict, mAP = compute_map(all_preds, all_gts, nc=nc)

    tag = '15cls (KLT merged)' if remap else '17cls'
    print(f'\n{"="*60}')
    print(f'Model: {model_name}  [{tag}]  mAP50={mAP:.4f}')
    print(f'{"Class":<22} {"AP50":>6}')
    print('-'*30)
    for cls_id, name in enumerate(names):
        ap = ap_dict.get(cls_id, float('nan'))
        marker = ' ← merged KLT' if (remap and cls_id == 1) else ''
        print(f'  {name:<20} {ap:>6.3f}{marker}')
    print(f'{"  mAP50":<22} {mAP:>6.4f}')
    return mAP


if __name__ == '__main__':
    results = {}
    for name, path in MODELS.items():
        if not os.path.exists(path):
            print(f'SKIP {name}: {path} not found')
            continue
        print(f'\nEvaluating {name}...')
        map17 = run_eval(name, path, remap=False)
        map15 = run_eval(name, path, remap=True)
        results[name] = (map17, map15)

    print('\n' + '='*60)
    print('SUMMARY')
    print(f'{"Model":<25} {"17cls":>7} {"15cls (KLT merged)":>20} {"gain":>6}')
    print('-'*60)
    for name, (m17, m15) in results.items():
        gain = m15 - m17
        print(f'  {name:<23} {m17:>7.4f} {m15:>20.4f} {gain:>+6.4f}')

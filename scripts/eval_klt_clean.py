#!/usr/bin/env python3
"""
KLT-merged evaluation on clean val (no leaked scenes).

Evaluates each model under continuous_scene policy with:
  - 17-class AP50 (standard)
  - 15-class AP50 (KLT classes 1,2,3 merged into single 'klt' class)

Clean val scenes: 3,4,5,6,7,8,14,26,33,35  (excludes leaked 10,21,23)

Usage:
  WANDB_MODE=disabled /home/loki/anaconda3/envs/reyolov8/bin/python \
    scripts/eval_klt_clean.py
"""
import sys, os, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import h5py
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from ultralytics.yolo.utils import ops

LEAKED_SCENES = {10, 21, 23}

MODELS = {
    'C1':      'runs/train/mtevent_640x480_c1_clean/weights/best.pt',
    'C5':      'runs/train/mtevent_640x480_c5_clean/weights/best.pt',
    'C11':     'runs/train/mtevent_640x480_fixed_c11/weights/best.pt',
    'C21':     'runs/train/mtevent_640x480_fixed_c21/weights/best.pt',
    'TC':      'runs/train/mtevent_640x480_tc_c11/weights/best.pt',
    'YOLOv8s': 'runs/train/mtevent_640x480_yolov8s_5ch_clean3/weights/best.pt',
}
CLIP_LENGTHS = {'C1': 1, 'C5': 5, 'C11': 11, 'C21': 21, 'TC': 11, 'YOLOv8s': 1}

VAL_H5   = 'preprocessed_datasets/vtei_mtevent_50ms_5bin_640x480/images/val/mtevent_val.h5'
VAL_LBLS = 'preprocessed_datasets/vtei_mtevent_50ms_5bin_640x480/labels/val'
CONF = 0.001
IOU  = 0.45
AP_IOU = 0.50
IMGSZ = (480, 640)   # H, W

def remap_17_to_15(cls_id):
    if cls_id in (1, 2, 3): return 1
    elif cls_id >= 4: return cls_id - 2
    return cls_id

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
    for preds_frame, gts_frame in zip(all_preds, all_gts):
        for g in gts_frame: n_gt[int(g[4])] += 1
        matched = set()
        preds_s = sorted(preds_frame, key=lambda x: -x[4])
        for p in preds_s:
            px1,py1,px2,py2,pconf,pcls = p
            best_iou, best_j = 0.0, -1
            for j, g in enumerate(gts_frame):
                if int(g[4]) != int(pcls) or j in matched: continue
                ix1=max(px1,g[0]); iy1=max(py1,g[1])
                ix2=min(px2,g[2]); iy2=min(py2,g[3])
                inter=max(0,ix2-ix1)*max(0,iy2-iy1)
                union=(px2-px1)*(py2-py1)+(g[2]-g[0])*(g[3]-g[1])-inter
                iou=inter/union if union>0 else 0
                if iou>best_iou: best_iou,best_j=iou,j
            if best_iou>=iou_thresh and best_j>=0:
                matched.add(best_j); tp_fp[int(pcls)].append((1,pconf))
            else:
                tp_fp[int(pcls)].append((0,pconf))
    ap_per_class = {}
    aps = []
    for cls_id in range(nc):
        if cls_id not in n_gt: continue
        if cls_id not in tp_fp:
            ap_per_class[cls_id]=0.0; aps.append(0.0); continue
        entries=sorted(tp_fp[cls_id], key=lambda x:-x[1])
        tp_arr=np.array([e[0] for e in entries])
        cum_tp=np.cumsum(tp_arr); cum_fp=np.cumsum(1-tp_arr)
        rec=cum_tp/max(n_gt[cls_id],1)
        pre=cum_tp/np.maximum(cum_tp+cum_fp,1e-10)
        ap=compute_ap(rec,pre); ap_per_class[cls_id]=ap; aps.append(ap)
    return ap_per_class, float(np.mean(aps)) if aps else 0.0


def eval_model_klt(model_name, model_path, clip_length):
    device = torch.device('cuda:0')
    print(f'\n[{model_name}] loading {model_path}')
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = ckpt['model'].to(device).float(); model.eval()

    lbl_files = sorted(Path(VAL_LBLS).glob('scene_*.npy'))
    H, W = IMGSZ
    f_h5 = h5py.File(VAL_H5, 'r')
    h5_data = f_h5['1mp']

    all_preds_17, all_gts_17 = [], []
    all_preds_15, all_gts_15 = [], []
    h5_offset = 0

    with torch.no_grad():
        for lf in lbl_files:
            sid = int(lf.stem.split('_')[1])
            lbls = np.load(str(lf), allow_pickle=True)
            n = len(lbls)
            hidden = {"0":None,"1":None,"2":None,"3":None}
            is_clean = sid not in LEAKED_SCENES

            for i in range(n):
                fr = np.array(h5_data[h5_offset + i], dtype=np.float32)
                inp = torch.tensor(fr).unsqueeze(0).to(device)
                ph = (32 - inp.shape[2] % 32) % 32
                pw = (32 - inp.shape[3] % 32) % 32
                if ph > 0 or pw > 0:
                    inp = F.pad(inp, (0, pw, 0, ph))
                out, hidden = model(inp, hidden)
                dets = ops.non_max_suppression(out, conf_thres=CONF,
                                               iou_thres=IOU,
                                               multi_label=False, max_det=300)[0]
                if is_clean:
                    scale_h = H / inp.shape[2]; scale_w = W / inp.shape[3]
                    pf17 = []
                    for d in dets:
                        x1,y1,x2,y2,conf,cls = d.cpu().numpy()
                        pf17.append((x1*scale_w, y1*scale_h,
                                     x2*scale_w, y2*scale_h,
                                     float(conf), int(cls)))
                    gf17 = []
                    for ann in lbls[i]:
                        c=int(ann[0]); cx,cy,bw,bh=ann[1]*W,ann[2]*H,ann[3]*W,ann[4]*H
                        gf17.append((cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2,c))
                    all_preds_17.append(pf17); all_gts_17.append(gf17)
                    all_preds_15.append(
                        [(x1,y1,x2,y2,c,remap_17_to_15(int(cl))) for x1,y1,x2,y2,c,cl in pf17])
                    all_gts_15.append(
                        [(x1,y1,x2,y2,remap_17_to_15(c)) for x1,y1,x2,y2,c in gf17])

            h5_offset += n
            # Reset at scene boundary (continuous_scene policy)

    f_h5.close()

    ap17, map17 = compute_map(all_preds_17, all_gts_17, nc=17, iou_thresh=AP_IOU)
    ap15, map15 = compute_map(all_preds_15, all_gts_15, nc=15, iou_thresh=AP_IOU)

    gain = map15 - map17
    print(f'  17cls mAP50={map17:.4f}  15cls mAP50={map15:.4f}  gain={gain:+.4f}')
    return map17, map15, gain, ap17, ap15


def main():
    os.makedirs('benchmark_results/clean_validation/klt', exist_ok=True)
    results = {}
    print('KLT ablation — clean val (no leaked scenes)')
    print('='*60)
    print(f'{"Model":6} | {"17cls":6} | {"15cls":6} | {"gain":7}')
    print('-'*35)
    for mname, mpath in MODELS.items():
        cl = CLIP_LENGTHS[mname]
        map17, map15, gain, ap17, ap15 = eval_model_klt(mname, mpath, cl)
        results[mname] = {'map17': map17, 'map15': map15, 'gain': gain,
                          'ap17': {str(k):v for k,v in ap17.items()},
                          'ap15': {str(k):v for k,v in ap15.items()}}
        print(f'{mname:6} | {map17:.4f} | {map15:.4f} | {gain:+.4f}')
    with open('benchmark_results/clean_validation/klt/results.json','w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved to benchmark_results/clean_validation/klt/results.json')

if __name__ == '__main__':
    main()

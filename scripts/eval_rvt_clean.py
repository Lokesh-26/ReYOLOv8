#!/usr/bin/env python3
"""
Equivalent clean-val evaluation for RVT on the same 10-scene clean val,
using the same all-points AP50 implementation as eval_clean_val.py.

Run from /home/loki/event/RVT/:
  /home/loki/venvs/rvt/bin/python \
    /home/loki/event/ReYOLOv8/scripts/eval_rvt_clean.py \
    --ckpt dummy/jk20t51u/checkpoints/epoch=008-step=77000-val_AP=0.20.ckpt \
    --data_dir /home/loki/event/ReYOLOv8/preprocessed_datasets/rvt_mtevent_10ch \
    --out_dir /home/loki/event/ReYOLOv8/benchmark_results/clean_validation/rvt_combined

Clean val scenes: 3,4,5,6,7,8,14,26,33,35  (excludes leaked 10,21,23)
"""
import sys, os, json, argparse
from pathlib import Path
import numpy as np
import torch
import h5py

# ── RVT imports ─────────────────────────────────────────────────────────────
RVT_ROOT = Path(__file__).parent.parent.parent / 'RVT'
sys.path.insert(0, str(RVT_ROOT))
from omegaconf import OmegaConf
from config.modifier import dynamically_modify_train_config
from modules.detection import Module
from models.detection.yolox.utils.boxes import postprocess

CLEAN_SCENES = [3, 4, 5, 6, 7, 8, 14, 26, 33, 35]
CONF  = 0.001
NMS   = 0.45
AP_IOU = 0.50
NC    = 17

def build_config(data_dir: Path, resolution_hw=(256, 320), dim_head=32):
    """Construct the Hydra config programmatically for MTEvent 10ch."""
    cfg = OmegaConf.create({
        'model': {
            'name': 'rnndet',
            'backbone': {
                'name': 'MaxViTRNN',
                'compile': {'enable': False, 'args': {'mode': 'reduce-overhead'}},
                'input_channels': 10,
                'enable_masking': False,
                'partition_split_32': 1,
                'embed_dim': 48,
                'dim_multiplier': [1, 2, 4, 8],
                'num_blocks': [1, 1, 1, 1],
                'T_max_chrono_init': [4, 8, 16, 32],
                'stem': {'patch_size': 4},
                'stage': {
                    'downsample': {'type': 'patch', 'overlap': True, 'norm_affine': True},
                    'attention': {
                        'use_torch_mha': False,
                        'partition_size': None,
                        'dim_head': dim_head,
                        'attention_bias': True,
                        'mlp_activation': 'gelu',
                        'mlp_gated': False,
                        'mlp_bias': True,
                        'mlp_ratio': 4,
                        'drop_mlp': 0,
                        'drop_path': 0,
                        'ls_init_value': 1e-5,
                    },
                    'lstm': {
                        'dws_conv': False,
                        'dws_conv_only_hidden': True,
                        'dws_conv_kernel_size': 3,
                        'drop_cell_update': 0,
                    },
                },
            },
            'fpn': {
                'name': 'PAFPN',
                'compile': {'enable': False, 'args': {'mode': 'reduce-overhead'}},
                'depth': 0.33,
                'in_stages': [2, 3, 4],
                'depthwise': False,
                'act': 'silu',
            },
            'head': {
                'name': 'YoloX',
                'compile': {'enable': False, 'args': {'mode': 'reduce-overhead'}},
                'depthwise': False,
                'act': 'silu',
                'num_classes': NC,
            },
            'postprocess': {'confidence_threshold': CONF, 'nms_threshold': NMS},
        },
        'dataset': {
            'name': 'mtevent',
            'path': str(data_dir),
            'ev_repr_name': 'stacked_histogram_dt=50_nbins=5_split_pol',
            'sequence_length': 11,
            'resolution_hw': list(resolution_hw),
            'downsample_by_factor_2': False,
            'only_load_end_labels': False,
            'train': {'sampling': 'mixed', 'random': {'weighted_sampling': False},
                      'mixed': {'w_stream': 1, 'w_random': 1}},
            'eval': {'sampling': 'stream'},
            'data_augmentation': {
                'random': {'prob_hflip': 0.5,
                           'rotate': {'prob': 0, 'min_angle_deg': 2, 'max_angle_deg': 6},
                           'zoom': {'prob': 0.8, 'min_factor': 0.9, 'max_factor': 1.1}},
            },
        },
        'training': {'precision': 16},
        'hardware': {'gpus': 0, 'num_workers': {'train': 4, 'eval': 2}},
        'batch_size': {'train': 8, 'eval': 2},
        'use_test_set': False,
    })
    dynamically_modify_train_config(cfg)
    return cfg


def load_scene(data_dir: Path, scene_id: int, split: str = 'val'):
    """Load event representations and labels for one scene."""
    scene_dir = data_dir / f'{split}/scene_{scene_id:06d}'
    repr_dir  = scene_dir / 'event_representations_v2' / \
                'stacked_histogram_dt=50_nbins=5_split_pol'
    lbl_path  = scene_dir / 'labels_v2' / 'labels.npz'

    with h5py.File(repr_dir / 'event_representations.h5') as f:
        frames = f['data'][:]                       # (T, C, H, W)

    # objframe_idx_2_repr_idx: maps labeled-frame index → repr frame index
    obj2repr = np.load(repr_dir / 'objframe_idx_2_repr_idx.npy')

    lbl_data   = np.load(lbl_path, allow_pickle=True)
    labels_raw = lbl_data['labels']                 # structured array
    obj2lbl    = lbl_data['objframe_idx_2_label_idx']  # cumulative: obj2lbl[i] = start idx

    # Build repr_idx → label array mapping
    H, W = frames.shape[2], frames.shape[3]
    repr2boxes = {}
    n_obj_frames = len(obj2repr)
    for obj_idx in range(n_obj_frames):
        repr_idx = int(obj2repr[obj_idx])
        lbl_start = int(obj2lbl[obj_idx])
        lbl_end   = int(obj2lbl[obj_idx + 1]) if obj_idx + 1 < n_obj_frames else len(labels_raw)
        rows = labels_raw[lbl_start:lbl_end]
        if len(rows) == 0:
            continue
        boxes = np.stack([
            rows['class_id'].astype(np.float32),
            (rows['x'] + rows['w'] / 2) / W,
            (rows['y'] + rows['h'] / 2) / H,
            rows['w'] / W,
            rows['h'] / H,
        ], axis=1)
        repr2boxes[repr_idx] = boxes

    T = len(frames)
    per_frame_labels = [repr2boxes.get(t, np.zeros((0, 5), dtype=np.float32)) for t in range(T)]
    return frames, per_frame_labels


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])


def compute_map50(all_preds, all_gts, nc, iou_thr=0.50):
    aps = []
    for c in range(nc):
        gt_c  = [(g[g[:, 0] == c, 1:5]) for g in all_gts]
        det_c = []
        for p in all_preds:
            mask = p[:, 5] == c if len(p) else np.zeros(0, bool)
            det_c.append(p[mask][:, :5] if len(p) else np.zeros((0, 5)))

        n_gt = sum(len(g) for g in gt_c)
        if n_gt == 0:
            continue

        # Gather all detections sorted by confidence
        dets = []
        for fi, d in enumerate(det_c):
            for row in d:
                dets.append((row[4], fi, row[:4]))  # conf, frame_idx, xyxy
        if not dets:
            aps.append(0.0)
            continue
        dets.sort(key=lambda x: -x[0])

        matched = [np.zeros(len(g), dtype=bool) for g in gt_c]
        tp = np.zeros(len(dets))
        fp = np.zeros(len(dets))
        for di, (conf, fi, box) in enumerate(dets):
            gt = gt_c[fi]
            if len(gt) == 0:
                fp[di] = 1
                continue
            # IoU (xyxy vs xywh GT)
            gx1 = gt[:, 0] - gt[:, 2] / 2
            gy1 = gt[:, 1] - gt[:, 3] / 2
            gx2 = gt[:, 0] + gt[:, 2] / 2
            gy2 = gt[:, 1] + gt[:, 3] / 2
            ix1 = np.maximum(box[0], gx1)
            iy1 = np.maximum(box[1], gy1)
            ix2 = np.minimum(box[2], gx2)
            iy2 = np.minimum(box[3], gy2)
            inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
            ba   = (box[2] - box[0]) * (box[3] - box[1])
            ga   = gt[:, 2] * gt[:, 3]
            iou  = inter / (ba + ga - inter + 1e-9)
            best = iou.argmax()
            if iou[best] >= iou_thr and not matched[fi][best]:
                tp[di] = 1
                matched[fi][best] = True
            else:
                fp[di] = 1

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        rec  = cum_tp / n_gt
        prec = cum_tp / (cum_tp + cum_fp + 1e-9)
        aps.append(compute_ap(rec, prec))

    return float(np.mean(aps)) if aps else 0.0


def run(ckpt_path: str, data_dir: str, out_dir: str, resolution_hw=(256, 320), dim_head=32,
        scenes=None, split='val'):
    if scenes is None:
        scenes = CLEAN_SCENES
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_dir = Path(data_dir)
    out_dir  = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build config and model
    cfg = build_config(data_dir, resolution_hw=resolution_hw, dim_head=dim_head)
    model = Module(cfg)

    print(f'Loading checkpoint: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['state_dict'])
    model.eval().to(device)
    model.half()

    all_preds = []
    all_gts   = []

    for scene_id in scenes:
        print(f'  Scene {scene_id}...', end=' ', flush=True)
        frames, per_frame_labels = load_scene(data_dir, scene_id, split=split)
        T = len(frames)

        prev_states = None
        for t in range(T):
            x = torch.from_numpy(frames[t]).unsqueeze(0).half().to(device)
            with torch.no_grad():
                pred_raw, _, prev_states = model(x, previous_states=prev_states,
                                                  retrieve_detections=True)

            # pred_raw: (1, num_anchors, 5+NC)
            preds = postprocess(pred_raw, NC, conf_thre=CONF, nms_thre=NMS)
            p = preds[0]  # (N, 7): x1,y1,x2,y2,obj_conf,cls_conf,cls_id

            H, W = frames.shape[2], frames.shape[3]
            if p is not None and len(p):
                p_np = p.cpu().float().numpy()
                # Normalise to [0,1]
                p_np[:, 0] /= W; p_np[:, 2] /= W
                p_np[:, 1] /= H; p_np[:, 3] /= H
                score = p_np[:, 4] * p_np[:, 5]
                cls   = p_np[:, 6].astype(int)
                # Format: (N, 6) — x1,y1,x2,y2,score,cls
                out = np.column_stack([p_np[:, :4], score, cls])
                all_preds.append(out)
            else:
                all_preds.append(np.zeros((0, 6), dtype=np.float32))

            all_gts.append(per_frame_labels[t])

        print(f'{T} frames')

    # Detach states from graph across scenes (already handled by fresh prev_states=None per scene)
    map50 = compute_map50(all_preds, all_gts, NC, iou_thr=AP_IOU)
    print(f'\nClean val AP50 (17-class, sequential): {map50:.4f}')

    result = {'map50_sequential': map50, 'checkpoint': str(ckpt_path),
              'scenes': scenes, 'split': split, 'nc': NC}
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f'Saved to {out_dir}/results.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',       required=True)
    parser.add_argument('--data_dir',   required=True)
    parser.add_argument('--out_dir',    required=True)
    parser.add_argument('--resolution', default='256x320',
                        help='HxW e.g. 256x320 or 480x640')
    parser.add_argument('--dim_head',   type=int, default=32)
    parser.add_argument('--split',      default='val', help='data subdir: val or test')
    parser.add_argument('--scenes',     nargs='+', type=int, default=None,
                        help='scene ids to evaluate (default: clean val scenes)')
    args = parser.parse_args()
    h, w = (int(x) for x in args.resolution.split('x'))
    run(args.ckpt, args.data_dir, args.out_dir, resolution_hw=(h, w), dim_head=args.dim_head,
        scenes=args.scenes, split=args.split)

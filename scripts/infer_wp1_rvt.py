#!/usr/bin/env python3
"""
RVT inference on WP1 event H5 files (ReYOLOv8 voxel format).

Loads the fine-tuned RVT-small 5ch model, runs frame-by-frame inference with
recurrent state propagation, renders event+detection frames, saves video + JSON.

JSON format matches infer_wp1_bags.py: list of {frame, boxes: [[cls, conf, x1, y1, x2, y2], ...]}
Coordinates are in original (256×320) pixel space.

Usage (from /home/loki/event/ReYOLOv8/):
    /home/loki/anaconda3/envs/reyolov8/bin/python scripts/infer_wp1_rvt.py \
        --h5      benchmark_results/<name>/images/test/mtevent_test.h5 \
        --weights /home/loki/event/RVT/dummy/mkts9bwe/checkpoints/best_5ch_finetune.ckpt \
        --out_dir benchmark_results/<name>/rvt_detections \
        --device  cuda:0 --conf 0.25
"""
import os
import sys
import json
import argparse

import numpy as np
import h5py
import cv2
import torch

RVT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        '..', 'RVT')
RVT_ROOT = os.path.normpath(RVT_ROOT)
sys.path.insert(0, RVT_ROOT)

from omegaconf import OmegaConf, open_dict

from models.detection.yolox_extension.models.detector import YoloXDetector
from models.detection.yolox.utils.boxes import postprocess
from config.modifier import dynamically_modify_train_config
from utils.padding import InputPadderFromShape


CLASS_NAMES = [
    "wooden_pallet", "small_klt", "big_klt", "blue_klt",
    "amazon_luggage", "ikea_dammang_bin", "ikea_vesken_trolley",
    "ikea_sortera_bin", "ikea_drona_grey", "ikea_drona_blue",
    "ikea_knallig_box", "ikea_moppe_drawer", "ikea_labbsal_basket",
    "ikea_ivar_box", "ikea_skubb_case", "ikea_samla_box", "human",
]
PALETTE = [
    (0, 255, 255), (0, 128, 255), (0, 255, 0), (255, 0, 255),
    (255, 128, 0), (0, 0, 255), (255, 255, 0), (128, 0, 255),
    (0, 255, 128), (255, 0, 128), (128, 255, 0), (0, 128, 128),
    (128, 0, 128), (255, 128, 128), (128, 128, 255), (128, 255, 128),
    (0, 200, 255),
]


def build_config_5ch_small():
    """Manually build OmegaConf config for RVT-small 5ch mtevent model."""
    cfg = OmegaConf.create({
        'dataset': {
            'name': 'mtevent',
            'path': 'dummy',
            'ev_repr_name': 'stacked_histogram_dt=50_nbins=5',
            'sequence_length': 11,
            'resolution_hw': [256, 320],
            'downsample_by_factor_2': False,
            'only_load_end_labels': False,
            'train': {'sampling': 'mixed', 'random': {'weighted_sampling': False},
                      'mixed': {'w_stream': 1, 'w_random': 1}},
            'eval': {'sampling': 'stream'},
            'data_augmentation': {
                'random': {'prob_hflip': 0.5, 'rotate': {'prob': 0, 'min_angle_deg': 2, 'max_angle_deg': 6},
                           'zoom': {'prob': 0.8, 'zoom_in': {'weight': 8, 'factor': {'min': 1, 'max': 1.5}},
                                    'zoom_out': {'weight': 2, 'factor': {'min': 1, 'max': 1.2}}}},
                'stream': {'prob_hflip': 0.5, 'rotate': {'prob': 0, 'min_angle_deg': 2, 'max_angle_deg': 6},
                           'zoom': {'prob': 0.5, 'zoom_out': {'factor': {'min': 1, 'max': 1.2}}}},
            },
        },
        'model': {
            'name': 'rnndet',
            'backbone': {
                'name': 'MaxViTRNN',
                'input_channels': 5,
                'partition_split_32': 1,
                'embed_dim': 48,
                'dim_multiplier': [1, 2, 4, 8],
                'num_blocks': [1, 1, 1, 1],
                'T_max_chrono_init': [4, 8, 16, 32],
                'compile': {'enable': False, 'args': {'mode': 'reduce-overhead'}},
                'enable_masking': False,
                'stem': {'patch_size': 4},
                'stage': {
                    'downsample': {'type': 'patch', 'overlap': True, 'norm_affine': True},
                    'attention': {
                        'use_torch_mha': False,
                        'partition_size': [8, 10],
                        'dim_head': 24,
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
                'depth': 0.33,
                'compile': {'enable': False, 'args': {'mode': 'reduce-overhead'}},
                'in_stages': [2, 3, 4],
                'depthwise': False,
                'act': 'silu',
            },
            'head': {
                'name': 'YoloX',
                'depthwise': False,
                'act': 'silu',
                'compile': {'enable': False, 'args': {'mode': 'reduce-overhead'}},
                'num_classes': 17,
            },
            'postprocess': {
                'confidence_threshold': 0.1,
                'nms_threshold': 0.45,
            },
        },
        'hardware': {'gpus': 0, 'num_workers': {'eval': 4}},
        'batch_size': {'eval': 1, 'train': 8},
        'training': {'precision': 16},
    })
    dynamically_modify_train_config(cfg)
    return cfg


def compute_global_scale(frames, clip_pct=95):
    sample = np.abs(frames[::10].astype(np.float32))
    nonzero = sample[sample > 0]
    return float(np.percentile(nonzero, clip_pct)) if len(nonzero) else 1.0


def event_frame_to_bgr(frame_chw, scale=1.0):
    acc = frame_chw.astype(np.float32).sum(axis=0)
    red  = np.clip( acc / scale * 255, 0, 255).astype(np.uint8)
    blue = np.clip(-acc / scale * 255, 0, 255).astype(np.uint8)
    bgr = np.zeros((*acc.shape, 3), dtype=np.uint8)
    bgr[:, :, 2] = red
    bgr[:, :, 0] = blue
    return bgr


def draw_detections(bgr, dets):
    if dets is None or len(dets) == 0:
        return bgr
    for det in dets:
        x1, y1, x2, y2, obj_conf, cls_conf, cls_id = det.tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls_id = int(cls_id)
        conf = obj_conf * cls_conf
        color = PALETTE[cls_id % len(PALETTE)]
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        label = f"{CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else cls_id} {conf:.2f}"
        cv2.putText(bgr, label, (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return bgr


def run(args):
    import torch.nn.functional as F

    device = torch.device(args.device)

    print(f"[INFO] loading H5: {args.h5}")
    with h5py.File(args.h5, 'r') as f:
        frames = f['1mp'][:]  # (N, C, H, W) int8
    N, C, orig_h, orig_w = frames.shape
    print(f"[INFO] frames shape: {frames.shape}  ({N} frames, {C}ch, {orig_h}×{orig_w})")

    cfg = build_config_5ch_small()
    in_res_hw = tuple(cfg.model.backbone.in_res_hw)
    assert C == cfg.model.backbone.input_channels, \
        f"H5 has {C} channels but model expects {cfg.model.backbone.input_channels}"

    mdl_h, mdl_w = in_res_hw
    need_resize = (orig_h != mdl_h or orig_w != mdl_w)
    if need_resize:
        print(f"[INFO] will resize {orig_h}×{orig_w} → {mdl_h}×{mdl_w} for model input")
        # scale factors to map model coords back to original image coords
        sx = orig_w / mdl_w
        sy = orig_h / mdl_h

    print(f"[INFO] loading weights: {args.weights}")
    mdl = YoloXDetector(cfg.model)

    ckpt = torch.load(args.weights, map_location='cpu', weights_only=False)
    sd = ckpt['state_dict']
    # Keys in checkpoint are prefixed with 'mdl.' — load into YoloXDetector directly
    sd_stripped = {k[len('mdl.'):]: v for k, v in sd.items() if k.startswith('mdl.')}
    missing, unexpected = mdl.load_state_dict(sd_stripped, strict=True)
    if missing:
        print(f"[WARN] missing keys: {missing[:5]}")
    mdl = mdl.to(device).eval()

    input_padder = InputPadderFromShape(desired_hw=in_res_hw)

    os.makedirs(args.out_dir, exist_ok=True)
    frames_dir = os.path.join(args.out_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    fps = 20.0
    video_path = os.path.join(args.out_dir, 'rvt_detection.mp4')
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (orig_w, orig_h))

    global_scale = compute_global_scale(frames)
    print(f"[INFO] contrast scale (p95 nonzero): {global_scale:.2f}")
    print(f"[INFO] running inference on {N} frames ...")

    all_detections = []
    prev_states = None

    with torch.no_grad():
        for t in range(N):
            frame = frames[t]  # (C, H, W) int8
            x = torch.from_numpy(frame.copy()).float().unsqueeze(0).to(device)  # (1, C, H, W)
            if need_resize:
                x = F.interpolate(x, size=(mdl_h, mdl_w), mode='bilinear', align_corners=False)
            x = input_padder.pad_tensor_ev_repr(x)

            backbone_features, prev_states = mdl.forward_backbone(x, previous_states=prev_states)
            outputs, _ = mdl.forward_detect(backbone_features=backbone_features)

            preds = postprocess(
                outputs,
                num_classes=cfg.model.head.num_classes,
                conf_thre=args.conf,
                nms_thre=args.iou,
            )
            dets = preds[0]  # (N_det, 7): [x1, y1, x2, y2, obj_conf, cls_conf, cls_id]

            frame_record = {'frame': t, 'boxes': []}
            dets_np = dets.cpu().numpy() if dets is not None and len(dets) > 0 else None
            if dets_np is not None:
                for det in dets_np:
                    x1, y1, x2, y2, obj_conf, cls_conf, cls_id = det.tolist()
                    conf = obj_conf * cls_conf
                    if need_resize:
                        x1, x2 = x1 * sx, x2 * sx
                        y1, y2 = y1 * sy, y2 * sy
                    frame_record['boxes'].append([
                        int(cls_id), round(conf, 4),
                        round(x1, 1), round(y1, 1),
                        round(x2, 1), round(y2, 1),
                    ])
            all_detections.append(frame_record)

            bgr = event_frame_to_bgr(frame, scale=global_scale)
            if dets_np is not None:
                dets_draw = dets_np.copy()
                if need_resize:
                    dets_draw[:, 0] *= sx; dets_draw[:, 2] *= sx
                    dets_draw[:, 1] *= sy; dets_draw[:, 3] *= sy
                bgr = draw_detections(bgr, dets_draw)

            cv2.imwrite(os.path.join(frames_dir, f'frame_{t:06d}.png'), bgr)
            writer.write(bgr)

            if (t + 1) % 100 == 0:
                n_det = len(dets) if dets is not None else 0
                print(f"  frame {t+1}/{N}  detections={n_det}")

    writer.release()

    json_path = os.path.join(args.out_dir, 'rvt_detections.json')
    with open(json_path, 'w') as f:
        json.dump(all_detections, f)

    total_dets = sum(len(r['boxes']) for r in all_detections)
    frames_with_dets = sum(1 for r in all_detections if r['boxes'])
    print(f"\n[DONE] {N} frames processed")
    print(f"       {frames_with_dets}/{N} frames with detections ({total_dets} total boxes)")
    print(f"       video  → {video_path}")
    print(f"       JSON   → {json_path}")
    print(f"       frames → {frames_dir}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--h5',      required=True, help='path to mtevent_test.h5 (key 1mp)')
    ap.add_argument('--weights', required=True, help='path to fine-tuned RVT checkpoint')
    ap.add_argument('--out_dir', required=True, help='output directory')
    ap.add_argument('--device',  default='cuda:0')
    ap.add_argument('--conf',    type=float, default=0.25)
    ap.add_argument('--iou',     type=float, default=0.45)
    args = ap.parse_args()
    run(args)


if __name__ == '__main__':
    main()

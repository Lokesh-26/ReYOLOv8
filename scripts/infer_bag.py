#!/usr/bin/env python3
"""
Run YOLOv8s / ReYOLOv8s inference on a raw ROS bag and render output video.

Events are accumulated into 50ms voxel grids (5 bins, signed polarity),
matching the MTEvent preprocessing pipeline.

Usage (from /home/loki/event/ReYOLOv8/):
  python3 scripts/infer_bag.py \
    --bag /home/loki/bags/wp1_rgb20ms_benchmark_run1_20260612_133658.bag \
    --weights runs/train/mtevent_640x480_yolov8s_5ch_clean3/weights/best.pt \
    --out /home/loki/event/ReYOLOv8/benchmark_results/bag_inference/yolov8s_run1.mp4 \
    --label YOLOv8s

  # Multiple models side-by-side:
  --weights2 runs/train/mtevent_640x480_tc_c11/weights/best.pt --label2 TC

Requires: system python3 (has rosbag), then loads model via subprocess or shared env.
Note: run with /usr/bin/python3 (has rosbag) — model loaded via reyolov8 torch.
"""
import sys, os, argparse, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2

# rosbag requires system python3
try:
    import rosbag
    from dvs_msgs.msg import EventArray
except ImportError:
    sys.exit("Run with system python3 (/usr/bin/python3) which has rosbag installed.")

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Constants matching MTEvent preprocessing ────────────────────────────────
WINDOW_US  = 50_000   # 50 ms accumulation window
N_BINS     = 5
EVENT_TOPIC = '/dvxplorer_left/events'
CONF_THR   = 0.25
NMS_THR    = 0.45
FPS        = 20       # 50ms per frame → 20 fps
IMG_H, IMG_W = 480, 640

CLASS_NAMES = [
    'wooden_pallet','small_klt','big_klt','blue_klt','amazon_luggage',
    'ikea_dammang_bin','ikea_vesken_trolley','ikea_sortera_bin',
    'ikea_drona_grey','ikea_drona_blue','ikea_knallig_box','ikea_moppe_drawer',
    'ikea_labbsal_basket','ikea_ivar_box','ikea_skubb_case','ikea_samla_box','human',
]
PALETTE = [
    (0,255,255),(0,128,255),(0,255,0),(255,0,255),(255,128,0),(0,0,255),
    (255,255,0),(128,0,255),(0,255,128),(255,0,128),(128,255,0),(0,128,128),
    (128,0,128),(255,128,128),(128,128,255),(128,255,128),(0,200,255),
]


def events_to_voxel(events_x, events_y, events_t, events_p,
                    t_start, t_end, H, W, n_bins):
    """Accumulate events into signed voxel grid (n_bins, H, W)."""
    voxel = np.zeros((n_bins, H, W), dtype=np.float32)
    if len(events_t) == 0:
        return voxel
    duration = t_end - t_start
    if duration <= 0:
        return voxel
    bin_idx = ((events_t - t_start) / duration * n_bins).astype(np.int32)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    polarity = np.where(events_p > 0, 1.0, -1.0)
    x = np.clip(events_x, 0, W - 1).astype(np.int32)
    y = np.clip(events_y, 0, H - 1).astype(np.int32)
    np.add.at(voxel, (bin_idx, y, x), polarity)
    voxel = np.clip(voxel, -127, 127)
    return voxel


def voxel_to_bgr(voxel):
    """(C,H,W) → BGR: red=positive, blue=negative, black=no event."""
    acc = voxel.sum(axis=0)
    scale = max(np.abs(acc).max(), 1.0)
    red  = np.clip( acc / scale * 255, 0, 255).astype(np.uint8)
    blue = np.clip(-acc / scale * 255, 0, 255).astype(np.uint8)
    bgr  = np.zeros((voxel.shape[1], voxel.shape[2], 3), dtype=np.uint8)
    bgr[:, :, 2] = red
    bgr[:, :, 0] = blue
    return bgr


def draw_boxes(img, boxes_xyxy, scores, classes, label_prefix=''):
    """Draw detection boxes on img (in-place)."""
    for (x1, y1, x2, y2), score, cls in zip(boxes_xyxy, scores, classes):
        cls = int(cls)
        color = PALETTE[cls % len(PALETTE)]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        text = f'{label_prefix}{CLASS_NAMES[cls]} {score:.2f}'
        cv2.putText(img, text, (int(x1), max(int(y1) - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return img


def load_model(weights_path):
    """Load ReYOLOv8s or YOLOv8s checkpoint."""
    from ultralytics.yolo.utils import ops
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    model = ckpt['model'].float().eval().to(device)
    return model, device


def run_inference(model, device, voxel, conf_thr, nms_thr, img_h, img_w):
    """Run model on voxel grid, return (boxes_xyxy_pixel, scores, classes)."""
    from ultralytics.yolo.utils import ops
    x = torch.from_numpy(voxel).unsqueeze(0).float().to(device)
    with torch.no_grad():
        out = model(x)
    # out may be tuple (predictions, ...) or just predictions
    pred = out[0] if isinstance(out, (tuple, list)) else out
    if pred.ndim == 2:
        pred = pred.unsqueeze(0)
    # NMS
    pred_nms = ops.non_max_suppression(pred, conf_thr, nms_thr, max_det=100)
    dets = pred_nms[0]  # (N, 6): x1,y1,x2,y2,conf,cls
    if dets is None or len(dets) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0)
    dets = dets.cpu().numpy()
    # Scale from model input to pixel
    boxes  = dets[:, :4]
    scores = dets[:, 4]
    classes = dets[:, 5]
    return boxes, scores, classes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag',      required=True)
    parser.add_argument('--weights',  required=True, help='Primary model weights')
    parser.add_argument('--label',    default='Model1')
    parser.add_argument('--weights2', default=None,  help='Second model (optional)')
    parser.add_argument('--label2',   default='Model2')
    parser.add_argument('--out',      required=True, help='Output MP4 path')
    parser.add_argument('--conf',     type=float, default=CONF_THR)
    parser.add_argument('--max_frames', type=int, default=0, help='0=all')
    args = parser.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print(f'Loading model: {args.weights}')
    model1, device = load_model(args.weights)
    model2 = None
    if args.weights2:
        print(f'Loading model2: {args.weights2}')
        model2, _ = load_model(args.weights2)
        model2 = model2.to(device)

    # Hidden states for recurrent models
    state1 = None
    state2 = None

    # Prepare video writer
    out_w = IMG_W * (2 if model2 else 1)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.out, fourcc, FPS, (out_w, IMG_H))

    print(f'Reading bag: {args.bag}')
    bag = rosbag.Bag(args.bag, 'r')

    # Accumulate events into windows
    ev_buf_x, ev_buf_y, ev_buf_t, ev_buf_p = [], [], [], []
    window_start_us = None
    frame_count = 0

    for topic, msg, t in bag.read_messages(topics=[EVENT_TOPIC]):
        for ev in msg.events:
            ts_us = ev.ts.to_nsec() // 1000
            if window_start_us is None:
                window_start_us = ts_us

            if ts_us - window_start_us >= WINDOW_US:
                # Process accumulated window
                if len(ev_buf_t) > 0:
                    voxel = events_to_voxel(
                        np.array(ev_buf_x), np.array(ev_buf_y),
                        np.array(ev_buf_t), np.array(ev_buf_p),
                        window_start_us, window_start_us + WINDOW_US,
                        IMG_H, IMG_W, N_BINS
                    )

                    # Run inference
                    b1, s1, c1 = run_inference(model1, device, voxel,
                                                args.conf, NMS_THR, IMG_H, IMG_W)

                    bgr1 = voxel_to_bgr(voxel)
                    draw_boxes(bgr1, b1, s1, c1, label_prefix='')
                    cv2.putText(bgr1, args.label, (8, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                    if model2:
                        b2, s2, c2 = run_inference(model2, device, voxel,
                                                    args.conf, NMS_THR, IMG_H, IMG_W)
                        bgr2 = voxel_to_bgr(voxel)
                        draw_boxes(bgr2, b2, s2, c2, label_prefix='')
                        cv2.putText(bgr2, args.label2, (8, 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                        frame = np.hstack([bgr1, bgr2])
                    else:
                        frame = bgr1

                    writer.write(frame)
                    frame_count += 1
                    if frame_count % 50 == 0:
                        print(f'  {frame_count} frames rendered...')
                    if args.max_frames > 0 and frame_count >= args.max_frames:
                        break

                # Reset window
                ev_buf_x.clear(); ev_buf_y.clear()
                ev_buf_t.clear(); ev_buf_p.clear()
                window_start_us = ts_us

            ev_buf_x.append(ev.x)
            ev_buf_y.append(ev.y)
            ev_buf_t.append(ts_us)
            ev_buf_p.append(1 if ev.polarity else 0)

        if args.max_frames > 0 and frame_count >= args.max_frames:
            break

    bag.close()
    writer.release()
    print(f'Done. {frame_count} frames → {args.out}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
RGB inference on WP1 bags using trained YOLOv8 baseline.

Run with conda python (has ultralytics + torch):
    python scripts/infer_wp1_rgb.py \
        --frames_dir  benchmark_results/<name>/rgb_frames \
        --weights     runs/detect/runs/train/rgb_baseline_proper/weights/best.pt \
        --out_dir     benchmark_results/<name>/rgb_detections \
        --fps         20
"""
import os, sys, json, argparse, glob, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # project-local ultralytics (has DetectionModel2)
import cv2
import numpy as np
import torch
import torch.nn.functional as F

CLASS_NAMES = [
    "wooden_pallet","small_klt","big_klt","blue_klt","amazon_luggage",
    "ikea_dammang_bin","ikea_vesken_trolley","ikea_sortera_bin",
    "ikea_drona_grey","ikea_drona_blue","ikea_knallig_box",
    "ikea_moppe_drawer","ikea_labbsal_basket","ikea_ivar_box",
    "ikea_skubb_case","ikea_samla_box","human",
]
PALETTE = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),
           (128,0,0),(0,128,0),(0,0,128),(128,128,0),(128,0,128),(0,128,128),
           (255,128,0),(255,0,128),(0,255,128),(128,255,0),(64,64,255)]


def load_model(weights_path, device):
    from ultralytics.yolo.utils import ops
    import ultralytics.nn.modules as _flat_modules
    # The RGB checkpoint was pickled with standard ultralytics where nn.modules is a
    # package (nn.modules.conv.Conv etc.).  Project-local ultralytics has it as a flat
    # file that does define Conv.  Register aliases so torch.load's unpickler can find them.
    for _sub in ('conv', 'block', 'head', 'transformer', 'utils'):
        sys.modules.setdefault(f'ultralytics.nn.modules.{_sub}', _flat_modules)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    model = ckpt['model'].float().eval().to(device)
    return model


def infer_frame(model, img_bgr, device, conf_thr, iou_thr):
    """Run YOLOv8 on a BGR image; returns (boxes_xyxy, scores, class_ids) as numpy arrays."""
    from ultralytics.yolo.utils import ops
    H, W = img_bgr.shape[:2]
    # Resize to 640×640 (standard YOLOv8 input size), keep aspect ratio via letterbox-style pad
    scale = 640 / max(H, W)
    new_h, new_w = int(round(H * scale)), int(round(W * scale))
    resized = cv2.resize(img_bgr, (new_w, new_h))
    # Pad to 640×640
    pad_h = math.ceil(new_h / 32) * 32
    pad_w = math.ceil(new_w / 32) * 32
    canvas = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
    canvas[:new_h, :new_w] = resized
    x = torch.from_numpy(canvas).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    x = x.to(device)
    with torch.no_grad():
        out = model(x)
    # YOLOv8 returns (predictions, ...) tuple; predictions shape (B, num_anchors, 4+nc)
    pred = out[0] if isinstance(out, (tuple, list)) else out
    while isinstance(pred, (tuple, list)):
        pred = pred[0]
    if pred.ndim == 2:
        pred = pred.unsqueeze(0)
    preds_nms = ops.non_max_suppression(pred, conf_thr, iou_thr, max_det=100)
    dets = preds_nms[0]
    if dets is None or len(dets) == 0:
        return np.zeros((0,4)), np.zeros(0), np.zeros(0, dtype=int)
    dets = dets.cpu().numpy()
    # Scale back from padded/resized coords to original pixel coords
    boxes  = dets[:, :4] / scale
    boxes[:, [0,2]] = np.clip(boxes[:, [0,2]], 0, W-1)
    boxes[:, [1,3]] = np.clip(boxes[:, [1,3]], 0, H-1)
    return boxes, dets[:, 4], dets[:, 5].astype(int)


def run(args):
    frames = sorted(glob.glob(os.path.join(args.frames_dir, "*.jpg")) +
                    glob.glob(os.path.join(args.frames_dir, "*.png")))
    if not frames:
        raise RuntimeError(f"No frames found in {args.frames_dir}")
    print(f"[INFO] {len(frames)} frames  weights={args.weights}")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = load_model(args.weights, device)

    sample = cv2.imread(frames[0])
    H, W = sample.shape[:2]

    os.makedirs(args.out_dir, exist_ok=True)
    video_path = os.path.join(args.out_dir, "rgb_detection.mp4")
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))

    all_detections = []
    for t, fpath in enumerate(frames):
        img = cv2.imread(fpath)
        boxes, scores, cls_ids = infer_frame(model, img, device, args.conf, args.iou)

        frame_record = {"frame": t, "boxes": []}
        for (x1,y1,x2,y2), conf, cls_id in zip(boxes, scores, cls_ids):
            frame_record["boxes"].append([int(cls_id), round(float(conf), 4),
                                          round(float(x1),1), round(float(y1),1),
                                          round(float(x2),1), round(float(y2),1)])
            color = PALETTE[int(cls_id) % len(PALETTE)]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"{CLASS_NAMES[int(cls_id)] if int(cls_id) < len(CLASS_NAMES) else cls_id} {conf:.2f}"
            cv2.putText(img, label, (int(x1), max(0, int(y1)-4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        all_detections.append(frame_record)
        writer.write(img)

        if (t + 1) % 100 == 0:
            print(f"  frame {t+1}/{len(frames)}  dets={len(frame_record['boxes'])}")

    writer.release()
    json_path = os.path.join(args.out_dir, "rgb_detections.json")
    with open(json_path, "w") as f:
        json.dump(all_detections, f)

    total = sum(len(r["boxes"]) for r in all_detections)
    with_dets = sum(1 for r in all_detections if r["boxes"])
    print(f"\n[DONE] {with_dets}/{len(frames)} frames with detections ({total} total boxes)")
    print(f"       video → {video_path}")
    print(f"       JSON  → {json_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--weights",    required=True)
    ap.add_argument("--out_dir",    required=True)
    ap.add_argument("--conf",  type=float, default=0.25)
    ap.add_argument("--iou",   type=float, default=0.45)
    ap.add_argument("--fps",   type=float, default=20.0)
    ap.add_argument("--device", default="cuda:0")
    run(ap.parse_args())

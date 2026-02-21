import os
import numpy as np
import h5py
import cv2

H5_PATH   = "preprocessed_datasets/vtei_mtevent_50ms_5bin/images/val/mtevent_val.h5"
NPZ_LABEL = "preprocessed_datasets/vtei_mtevent_50ms_5bin/labels/val/scene70.npz"
OUT_DIR   = "mtevent/mtevent_baseline5_valsave/gt_images_scene70"
MAX_FRAMES = None  # set e.g. 200 to limit

os.makedirs(OUT_DIR, exist_ok=True)

# --- load labels (npz format: counts + boxes [cls,cx,cy,w,h]) ---
d = np.load(NPZ_LABEL)
counts = d["counts"].astype(int)
boxes_all = d["boxes"].astype(np.float32)

# --- load event frames ---
with h5py.File(H5_PATH, "r") as f:
    X = f["1mp"]  # (T, C, H, W), int8
    T, C, H, W = X.shape
    n = min(T, len(counts))
    if MAX_FRAMES is not None:
        n = min(n, int(MAX_FRAMES))

    print("H5:", (T, C, H, W), "labels frames:", len(counts), "rendering:", n)

    off = 0
    for t in range(n):
        c = int(counts[t])
        b = boxes_all[off:off+c] if c > 0 else np.zeros((0,5), np.float32)
        off += c

        frame = X[t].astype(np.float32)  # (C,H,W)

        # --- make a viewable 2D image ---
        # Option 1: sum over channels
        img = frame.sum(axis=0)  # (H,W)

        # normalize to 0..255 for visualization
        mn, mx = float(img.min()), float(img.max())
        if mx - mn < 1e-6:
            vis = np.zeros((H, W), dtype=np.uint8)
        else:
            vis = ((img - mn) / (mx - mn) * 255.0).astype(np.uint8)

        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

        # --- draw GT boxes (YOLO normalized cx,cy,w,h) ---
        for row in b:
            cls, cx, cy, bw, bh = row.tolist()
            x1 = int((cx - bw/2) * W)
            y1 = int((cy - bh/2) * H)
            x2 = int((cx + bw/2) * W)
            y2 = int((cy + bh/2) * H)

            # clip
            x1 = max(0, min(W-1, x1))
            y1 = max(0, min(H-1, y1))
            x2 = max(0, min(W-1, x2))
            y2 = max(0, min(H-1, y2))

            cv2.rectangle(vis_bgr, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(vis_bgr, f"{int(cls)}", (x1, max(0,y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

        out = os.path.join(OUT_DIR, f"scene70_{t:06d}.png")
        cv2.imwrite(out, vis_bgr)

print("[OK] wrote images to:", OUT_DIR)

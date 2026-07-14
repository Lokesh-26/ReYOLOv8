#!/usr/bin/env python3
"""
Hand-eye calibration: Vicon rig body → RGB camera optical frame.

Inputs:
  - ROS bag with /vicon/eventrig/pose and /rgb/image_raw
  - Kalibr results-cam.yaml  (intrinsics for RGB camera)
  - Checkerboard dimensions

Output:
  H_viconbody_2_rgb  (4×4 rigid transform)
  H_viconbody_2_event = inv(H_event_2_rgb) @ H_viconbody_2_rgb

Run with system python3 (has rosbag):
  /usr/bin/python3 scripts/handeye_calibration.py \
      --bag      /home/loki/bags/calib_handeye.bag \
      --kalibr   /home/loki/calib/results-cam.yaml \
      --rows 6 --cols 8 --square 0.05 \
      --vicon_topic /vicon/eventrig/pose \
      --rgb_topic   /rgb/image_raw
"""
import sys, os, argparse, json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

try:
    import rosbag
    import yaml
except ImportError:
    sys.exit("Run with /usr/bin/python3 (has rosbag)")


# ── Helpers ──────────────────────────────────────────────────────────────────

def pose_stamped_to_mat(msg):
    """geometry_msgs/PoseStamped → 4×4 numpy matrix."""
    p = msg.pose.position
    q = msg.pose.orientation
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3,  3] = [p.x, p.y, p.z]
    return T


def load_kalibr_yaml(path, cam_index=0):
    """
    Parse Kalibr results-cam.yaml.
    Returns (K 3×3, dist 1×4-or-5, H_event_2_rgb 4×4 or None).

    Kalibr names cameras cam0, cam1, ...
    cam0 = first topic you passed (event), cam1 = second (RGB) — or vice versa.
    Check your yaml to confirm which is which.
    T_cn_cnm1 in cam1 block = T_cam1_cam0 = H_cam0_2_cam1.
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    # Find the RGB camera block (look for 'pinhole-radtan' or check topic name)
    cam_keys = sorted(k for k in data if k.startswith('cam'))
    print(f"[kalibr] cameras found: {cam_keys}")
    for k in cam_keys:
        print(f"  {k}: topic={data[k].get('rostopic','?')}  model={data[k].get('camera_model','?')}")

    rgb_key = cam_keys[cam_index]
    cam_data = data[rgb_key]

    intr = cam_data['intrinsics']   # [fx, fy, cx, cy]
    dist = cam_data['distortion_coeffs']
    K = np.array([[intr[0], 0, intr[2]],
                  [0, intr[1], intr[3]],
                  [0,       0,       1]], dtype=np.float64)
    D = np.array(dist, dtype=np.float64)

    # Stereo extrinsic: T_cn_cnm1 lives in cam1 block
    # It is T_cam1_from_cam0, i.e. a point in cam0 frame → cam1 frame
    H_event_2_rgb = None
    if 'T_cn_cnm1' in cam_data:
        rows = cam_data['T_cn_cnm1']
        H_event_2_rgb = np.array(rows, dtype=np.float64)  # 4×4
        print(f"[kalibr] loaded T_cn_cnm1 (event→rgb) from {rgb_key}")

    return K, D, H_event_2_rgb


def detect_board(img_bgr, board_size, square_size, K, D):
    """
    Detect checkerboard and solve PnP.
    Returns (R_vec, t_vec) of board in camera frame, or None.
    board_size: (cols-1, rows-1) inner corners.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray, board_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not found:
        return None, None

    corners = cv2.cornerSubPix(
        gray, corners, (11, 11), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    )

    # 3D object points (board frame, z=0)
    obj_pts = np.zeros((board_size[0] * board_size[1], 3), dtype=np.float32)
    obj_pts[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    obj_pts *= square_size

    ok, rvec, tvec = cv2.solvePnP(obj_pts, corners, K, D)
    if not ok:
        return None, None
    return rvec, tvec


def rvec_tvec_to_mat(rvec, tvec):
    """OpenCV rvec, tvec → 4×4 matrix (board in camera frame)."""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = tvec.ravel()
    return T


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bag',         required=True)
    ap.add_argument('--kalibr',      required=True, help='results-cam.yaml from Kalibr')
    ap.add_argument('--rows',        type=int, default=6,    help='checkerboard inner rows')
    ap.add_argument('--cols',        type=int, default=8,    help='checkerboard inner cols')
    ap.add_argument('--square',      type=float, default=0.05, help='square size in metres')
    ap.add_argument('--vicon_topic', default='/vicon/eventrig/pose')
    ap.add_argument('--rgb_topic',   default='/rgb/image_raw')
    ap.add_argument('--rgb_cam_idx', type=int, default=1,
                    help='which cam block in Kalibr yaml is RGB (0=cam0, 1=cam1)')
    ap.add_argument('--sync_tol_ms', type=float, default=20.0,
                    help='max timestamp difference for matching (ms)')
    ap.add_argument('--out',         default='handeye_result.json')
    ap.add_argument('--vis',         action='store_true', help='show detection images')
    args = ap.parse_args()

    board_size = (args.cols, args.rows)   # inner corners (cols, rows)
    sync_tol_ns = int(args.sync_tol_ms * 1e6)

    # Load Kalibr intrinsics
    K, D, H_event_2_rgb = load_kalibr_yaml(args.kalibr, cam_index=args.rgb_cam_idx)
    print(f"[kalibr] K=\n{K}")
    print(f"[kalibr] D={D}")
    if H_event_2_rgb is not None:
        print(f"[kalibr] H_event_2_rgb=\n{H_event_2_rgb}")

    # Read bag: collect all Vicon and RGB messages with timestamps
    print(f"\n[bag] reading {args.bag} ...")
    bag = rosbag.Bag(args.bag)

    vicon_msgs = []  # (ts_ns, msg)
    rgb_msgs   = []  # (ts_ns, msg)

    for topic, msg, t in bag.read_messages(topics=[args.vicon_topic, args.rgb_topic]):
        ts_ns = t.to_nsec()
        if topic == args.vicon_topic:
            vicon_msgs.append((ts_ns, msg))
        else:
            rgb_msgs.append((ts_ns, msg))
    bag.close()

    print(f"[bag] vicon={len(vicon_msgs)} msgs  rgb={len(rgb_msgs)} msgs")

    # Synchronize: for each RGB frame, find nearest Vicon msg
    import numpy as np
    vicon_ts = np.array([m[0] for m in vicon_msgs])

    pairs = []   # (T_world_rig 4×4, T_cam_board 4×4)
    detected = 0

    for rgb_ts, rgb_msg in rgb_msgs:
        # Find nearest Vicon message
        idx = np.argmin(np.abs(vicon_ts - rgb_ts))
        dt_ns = abs(vicon_ts[idx] - rgb_ts)
        if dt_ns > sync_tol_ns:
            continue

        # Decode RGB image
        h, w = rgb_msg.height, rgb_msg.width
        enc  = rgb_msg.encoding
        data = np.frombuffer(rgb_msg.data, dtype=np.uint8).reshape(h, w, -1)
        if enc == 'rgb8':
            img = data[:, :, ::-1].copy()   # RGB→BGR
        else:
            img = data.copy()

        # Detect checkerboard
        rvec, tvec = detect_board(img, board_size, args.square, K, D)
        if rvec is None:
            continue
        detected += 1

        T_cam_board = rvec_tvec_to_mat(rvec, tvec)
        T_world_rig = pose_stamped_to_mat(vicon_msgs[idx][1])
        pairs.append((T_world_rig, T_cam_board))

        if args.vis:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, board_size,
                                      cv2.findChessboardCorners(
                                          cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                                          board_size)[1], True)
            cv2.imshow('detection', cv2.resize(vis, (800, 600)))
            cv2.waitKey(200)

    print(f"\n[detect] {detected}/{len(rgb_msgs)} frames had valid board detections")
    print(f"[detect] {len(pairs)} synchronized pairs collected")

    if len(pairs) < 5:
        sys.exit("[ERROR] need at least 5 pairs — move the rig to more poses")

    # Build rotation/translation arrays for calibrateHandEye
    # Convention: "gripper2base" = rig in world = T_world_rig
    # "target2cam" = board in camera = T_cam_board
    R_g2b, t_g2b = [], []
    R_t2c, t_t2c = [], []

    for T_world_rig, T_cam_board in pairs:
        R_g2b.append(T_world_rig[:3, :3])
        t_g2b.append(T_world_rig[:3,  3].reshape(3, 1))
        R_t2c.append(T_cam_board[:3, :3])
        t_t2c.append(T_cam_board[:3,  3].reshape(3, 1))

    # Run four methods and compare — they should agree within ~1cm / ~1deg
    results = {}
    for method_name, method in [
        ('TSAI',     cv2.CALIB_HAND_EYE_TSAI),
        ('HORAUD',   cv2.CALIB_HAND_EYE_HORAUD),
        ('ANDREFF',  cv2.CALIB_HAND_EYE_ANDREFF),
        ('DANIILIDIS', cv2.CALIB_HAND_EYE_DANIILIDIS),
    ]:
        R, t = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=method)
        H = np.eye(4)
        H[:3, :3] = R
        H[:3,  3] = t.ravel()
        results[method_name] = H
        euler = Rotation.from_matrix(R).as_euler('xyz', degrees=True)
        print(f"[{method_name}]  t={t.ravel().round(4)}  euler_deg={euler.round(2)}")

    # Use TSAI as primary result
    H_viconbody_2_rgb = results['TSAI']
    print(f"\nH_viconbody_2_rgb (TSAI):\n{H_viconbody_2_rgb.round(6)}")

    # Derive event camera transform using Kalibr stereo extrinsic
    if H_event_2_rgb is not None:
        # H_event_2_rgb: point in event frame → rgb frame
        # H_viconbody_2_rgb: point in vicon body frame → rgb frame
        # We want: point in vicon body frame → event frame
        # H_viconbody_2_event = inv(H_event_2_rgb) @ H_viconbody_2_rgb
        H_rgb_2_event = np.linalg.inv(H_event_2_rgb)
        H_viconbody_2_event = H_rgb_2_event @ H_viconbody_2_rgb
        print(f"\nH_viconbody_2_event:\n{H_viconbody_2_event.round(6)}")
    else:
        H_viconbody_2_event = None
        print("\n[WARN] no stereo extrinsic in yaml — H_viconbody_2_event not computed")

    # Save result
    out = {
        'H_viconbody_2_rgb':   H_viconbody_2_rgb.tolist(),
        'H_viconbody_2_event': H_viconbody_2_event.tolist() if H_viconbody_2_event is not None else None,
        'H_event_2_rgb':       H_event_2_rgb.tolist() if H_event_2_rgb is not None else None,
        'K_rgb':   K.tolist(),
        'D_rgb':   D.tolist(),
        'n_pairs': len(pairs),
        'methods': {k: v.tolist() for k, v in results.items()},
    }
    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] saved → {args.out}")

    if args.vis:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

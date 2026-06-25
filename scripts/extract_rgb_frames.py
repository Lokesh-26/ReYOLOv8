#!/usr/bin/env python3
"""
Extract RGB frames from a WP1 ROS bag.

Requirements: rosbag, numpy, opencv-python
Run with system python3 (has rosbag): /usr/bin/python3

Usage:
    /usr/bin/python3 scripts/extract_rgb_frames.py <bag> <out_dir> [topic]

Output:
    <out_dir>/frame_000000.jpg   sequential frames
    <out_dir>/timestamps.csv     frame_index, bag_timestamp_ns, header_timestamp_ns
"""
import os
import sys
import csv
import numpy as np
import cv2
import rosbag

bag_path = sys.argv[1]
out_dir  = sys.argv[2]
topic    = sys.argv[3] if len(sys.argv) > 3 else '/rgb/image_raw'

os.makedirs(out_dir, exist_ok=True)
bag  = rosbag.Bag(bag_path, 'r')
idx  = 0
rows = []

for _, msg, bag_t in bag.read_messages(topics=[topic]):
    h, w = msg.height, msg.width
    enc  = msg.encoding
    data = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, -1)
    if enc == 'rgb8':
        data = data[:, :, ::-1]   # RGB → BGR for cv2
    cv2.imwrite(os.path.join(out_dir, f'frame_{idx:06d}.jpg'), data)
    hdr_ns = msg.header.stamp.to_nsec() if hasattr(msg.header.stamp, 'to_nsec') else 0
    rows.append((idx, bag_t.to_nsec(), hdr_ns))
    idx += 1

bag.close()

with open(os.path.join(out_dir, 'timestamps.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['frame_index', 'bag_timestamp_ns', 'header_timestamp_ns'])
    w.writerows(rows)

print(f'Extracted {idx} RGB frames ({h}x{w} {enc}) → {out_dir}')

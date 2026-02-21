#!/usr/bin/env python3
import rosbag

bag_path = "left.bag"
topic = "/dvxplorer_left/events"

max_x = 0
max_y = 0
min_x = 10**9
min_y = 10**9

bag = rosbag.Bag(bag_path)
for _, msg, _ in bag.read_messages(topics=[topic]):
    for e in msg.events:
        x = int(e.x); y = int(e.y)
        if x > max_x: max_x = x
        if y > max_y: max_y = y
        if x < min_x: min_x = x
        if y < min_y: min_y = y
bag.close()

print("min_x,min_y:", min_x, min_y)
print("max_x,max_y:", max_x, max_y)
print("Suggested W,H:", max_x + 1, max_y + 1)


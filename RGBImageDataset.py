"""
RGB image dataset compatible with the ReYOLOv8 training loop.

Reads standard YOLO-format directories:
  images/{split}/*.jpg
  labels/{split}/*.txt   (cls cx cy w h, normalised)

Returns the same dict keys as EventVideoDetectionDataset so that the
existing collate_fn, training loop, and loss function work unchanged.
With clip_length=1 the img tensor is (1, 3, H, W) float32 in [0, 1].
"""
import os
import glob
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class RGBImageDataset(Dataset):

    def __init__(self, images_dir, clip_length, clip_stride, channels,
                 aug_param, load_type="train", mode="batched"):
        self.images_dir = images_dir
        self.load_type = load_type

        labels_dir = images_dir.replace("images", "labels")
        jpg_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))

        self.samples = []
        for jpg in jpg_paths:
            stem = os.path.splitext(os.path.basename(jpg))[0]
            txt = os.path.join(labels_dir, stem + ".txt")
            if not os.path.isfile(txt):
                continue
            rows = []
            with open(txt) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        rows.append([float(p) for p in parts])
            if not rows:
                continue
            self.samples.append((jpg, np.array(rows, dtype=np.float32)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        jpg_path, labels = self.samples[idx]

        img = np.array(Image.open(jpg_path).convert("RGB"), dtype=np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

        cls = torch.from_numpy(labels[:, 0]).float()
        bboxes = torch.from_numpy(labels[:, 1:]).float()

        return {
            "img": img,
            "bboxes": bboxes,
            "cls": cls,
            "batch_idx": torch.zeros(cls.shape),
            "sequence": [idx],
            "vid_file": [jpg_path],
            "vid_pos": torch.zeros(len(cls), dtype=torch.long),
            "clip_pos": torch.zeros(1, dtype=torch.long),
        }

    @staticmethod
    def collate_fn(batch):
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))

        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)
            if k in ["bboxes", "cls", "clip_pos", "vid_pos"]:
                value = torch.cat(value, 0)
            new_batch[k] = value

        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch

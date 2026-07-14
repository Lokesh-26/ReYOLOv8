import torch
import numpy as np
import os
import random
import sys 
import time
from ultralytics.yolo.utils import LOGGER, colorstr
from ultralytics.yolo.data.utils import  PIN_MEMORY, RANK
from EventVideoDataset import EventVideoDetectionDataset
from torch.utils.data import DataLoader, dataloader, distributed
from ultralytics.yolo.utils.torch_utils import torch_distributed_zero_first



def seed_worker(worker_id):
    # Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class StreamBatchSampler(torch.utils.data.Sampler):
    """Batch sampler for stream-aware (RVT-style) recurrent training.

    Groups the dataset's non-overlapping windows by scene, assigns whole scenes
    to `batch_size` parallel streams (shuffled each epoch), and yields batches
    where slot b at step k+1 temporally continues slot b at step k. Shorter
    streams wrap around to their own start; the wrapped window is a scene start,
    so the trainer resets that slot's hidden state via batch['scene_start'].

    Requires the dataset to be built with clip_stride == clip_length.
    """

    def __init__(self, dataset, batch_size, seed=0):
        assert dataset.clip_stride == dataset.clip_length, \
            "stream-aware training needs non-overlapping windows (clip_stride == clip_length)"
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        # scene file -> ordered window indices (dataset builds them in temporal order)
        self.scene_to_indices = {}
        for idx, f in enumerate(dataset.sample_scene_file):
            self.scene_to_indices.setdefault(f, []).append(idx)
        self.scenes = sorted(self.scene_to_indices.keys())

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _build_batches(self):
        rng = random.Random(self.seed + self.epoch)
        order = self.scenes[:]
        rng.shuffle(order)
        streams = [[] for _ in range(self.batch_size)]
        for sc in order:  # greedy: next scene to the currently shortest stream
            tgt = min(range(self.batch_size), key=lambda b: len(streams[b]))
            streams[tgt].extend(self.scene_to_indices[sc])
        n_steps = max(len(s) for s in streams)
        return [[streams[b][k % len(streams[b])] for b in range(self.batch_size)]
                for k in range(n_steps)]

    def __iter__(self):
        return iter(self._build_batches())

    def __len__(self):
        return len(self._build_batches())


def build_video_stream_dataloader(cfg, video_config, batch_size, video_path, aug_param, rank=-1, seed=0):
    """Training dataloader with stream-aware batch composition (train mode only)."""
    assert rank == -1, "stream-aware training does not support DDP yet"
    stream_config = dict(video_config)
    stream_config["clip_stride"] = stream_config["clip_length"]
    dataset = EventVideoDetectionDataset(video_path, stream_config["clip_length"],
                                         stream_config["clip_stride"], stream_config["channels"],
                                         aug_param, "train", "batched")
    batch_sampler = StreamBatchSampler(dataset, batch_size, seed=seed)
    nd = torch.cuda.device_count()
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, cfg.workers])
    loader = DataLoader(dataset=dataset,
                        batch_sampler=batch_sampler,
                        num_workers=nw,
                        pin_memory=PIN_MEMORY,
                        collate_fn=getattr(dataset, "collate_fn", None),
                        worker_init_fn=seed_worker)
    return loader, dataset


def build_video_dataloader(cfg, video_config, batch_size, video_path, aug_param, mode, rank=-1, load = "batched", random_seed = False):

    shuffle = (mode == "train")
    #print("video path", video_path)
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = EventVideoDetectionDataset(video_path,video_config["clip_length"], video_config["clip_stride"], video_config["channels"], aug_param,mode, load)

    batch_size = min(batch_size, len(dataset))
  
    nd = torch.cuda.device_count()  # number of CUDA devices
    workers = cfg.workers if mode == "train" else cfg.workers * 2
    #workers = cfg
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    #nw = workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader # allow attribute updates
    generator = torch.Generator()
    if not random_seed:
     generator.manual_seed(6148914691236517205  + RANK)
    
    return loader(dataset=dataset,
                  batch_size=batch_size,
                  shuffle=shuffle and sampler is None,
                  num_workers=nw,
                  sampler=sampler,
                  pin_memory=PIN_MEMORY,
                  collate_fn=getattr(dataset, "collate_fn", None),
                  worker_init_fn=seed_worker,
                  generator=generator), dataset


def build_video_val_standalone_dataloader(cfg, video_config, batch_size, video_path, rank=-1, mode = "sequential", speed = False, zero_hidden = False):

    shuffle = False 
    batch_size = 1    

    if mode != "sequential":
       batch_size = batch_size

    if zero_hidden:  

       mode = "batched"
    
    if speed:  

       video_config["clip_length"] = 1
       video_config["clip_stride"] = 1
       mode = "batched"  

    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        
        dataset = EventVideoDetectionDataset(video_path,video_config["clip_length"], video_config["clip_stride"], video_config["channels"], [None],"val", mode)

  
    nd = torch.cuda.device_count()  # number of CUDA devices
    workers = cfg.workers if mode == "train" else cfg.workers * 2
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader # allow attribute updates
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + RANK)
    
    return loader(dataset=dataset,
                  batch_size=batch_size,
                  shuffle=shuffle and sampler is None,
                  num_workers=nw,
                  sampler=sampler,
                  pin_memory=PIN_MEMORY,
                  collate_fn=getattr(dataset, "collate_fn_val", None),
                  worker_init_fn=seed_worker,
                  generator=generator), dataset



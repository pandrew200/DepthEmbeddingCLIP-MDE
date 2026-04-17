# src/data/nyu_wds.py
from __future__ import annotations

import glob
import io
import os
from dataclasses import dataclass
from typing import Tuple

import h5py
import numpy as np
import torch
import webdataset as wds


def _expand(pattern: str):
    files = sorted(glob.glob(pattern))
    if len(files) == 0:
        raise FileNotFoundError(f"No shards matched pattern: {pattern}")
    return files


def take_first(x):
    """(h5_bytes,) -> h5_bytes"""
    return x[0]


def decode_h5_bytes_to_tensors(h5_bytes: bytes, min_depth: float = 0.1, max_depth: float = 10.0):
    """h5_bytes -> (rgb, depth, valid)"""
    with h5py.File(io.BytesIO(h5_bytes), "r") as f:
        rgb = np.array(f["rgb"])      # (3,480,640) uint8
        depth = np.array(f["depth"])  # (480,640) float32

    rgb_t = torch.from_numpy(rgb).float() / 255.0
    depth_t = torch.from_numpy(depth).float()

    # Clamp for stability / consistent masking
    depth_t = torch.clamp(depth_t, min=min_depth, max=max_depth)
    valid = torch.isfinite(depth_t) & (depth_t >= min_depth) & (depth_t <= max_depth)
    return rgb_t, depth_t, valid


@dataclass(frozen=True)
class RandomCrop:
    out_h: int = 416
    out_w: int = 544

    def __call__(self, rgb, depth, valid):
        _, H, W = rgb.shape
        if H < self.out_h or W < self.out_w:
            raise ValueError(f"Crop {self.out_h}x{self.out_w} bigger than input {H}x{W}")
        top = torch.randint(0, H - self.out_h + 1, (1,)).item()
        left = torch.randint(0, W - self.out_w + 1, (1,)).item()
        return (
            rgb[:, top:top+self.out_h, left:left+self.out_w],
            depth[top:top+self.out_h, left:left+self.out_w],
            valid[top:top+self.out_h, left:left+self.out_w],
        )


@dataclass(frozen=True)
class CenterCrop:
    out_h: int = 416
    out_w: int = 544

    def __call__(self, rgb, depth, valid):
        _, H, W = rgb.shape
        if H < self.out_h or W < self.out_w:
            raise ValueError(f"Crop {self.out_h}x{self.out_w} bigger than input {H}x{W}")
        top = (H - self.out_h) // 2
        left = (W - self.out_w) // 2
        return (
            rgb[:, top:top+self.out_h, left:left+self.out_w],
            depth[top:top+self.out_h, left:left+self.out_w],
            valid[top:top+self.out_h, left:left+self.out_w],
        )


@dataclass(frozen=True)
class ApplyTriplet:
    """Apply a (rgb,depth,valid)->(rgb,depth,valid) transform to a sample tuple."""
    tfm: object

    def __call__(self, sample):
        rgb, depth, valid = sample
        return self.tfm(rgb, depth, valid)


def make_nyu_wds_loaders(
    root: str,
    batch_size: int,
    num_workers: int,
    train_crop: Tuple[int, int] = (416, 544),
    val_crop: Tuple[int, int] = (416, 544),
    shardshuffle: int = 100,
    sample_shuffle: int = 1000,
):
    train_shards = _expand(os.path.join(root, "train-*.tar"))
    val_shards   = _expand(os.path.join(root, "val-*.tar"))

    train_ds = (
        wds.WebDataset(train_shards, shardshuffle=shardshuffle)
        .to_tuple("h5")                 # -> (h5_bytes,)
        .map(take_first)                # -> h5_bytes
        .map(decode_h5_bytes_to_tensors)  # -> (rgb, depth, valid)
        .map(ApplyTriplet(RandomCrop(*train_crop)))
        .shuffle(sample_shuffle)
    )

    val_ds = (
        wds.WebDataset(val_shards, shardshuffle=False)
        .to_tuple("h5")
        .map(take_first)
        .map(decode_h5_bytes_to_tensors)
        .map(ApplyTriplet(CenterCrop(*val_crop)))
    )

    # pin_memory and persistent_workers reduce per-epoch overhead; num_workers > 1 overlaps I/O with training.
    train_loader = wds.WebLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )
    val_loader = wds.WebLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader


# if __name__ == "__main__":
#     root = os.environ.get("NYU_WDS_ROOT", "/Users/andrewpan/Documents/datasets/nyu_depth_v2/data")
#     print("Using NYU_WDS_ROOT =", root)

#     # If anything is still weird, set num_workers=0 to debug deterministically.
#     train_loader, _ = make_nyu_wds_loaders(
#         root=root,
#         batch_size=4,
#         num_workers=2,
#     )

#     rgb, depth, valid = next(iter(train_loader))
#     print("rgb:", rgb.shape, rgb.dtype, float(rgb.min()), float(rgb.max()))
#     dv = depth[valid]
#     print("depth valid min/max:", float(dv.min()), float(dv.max()))
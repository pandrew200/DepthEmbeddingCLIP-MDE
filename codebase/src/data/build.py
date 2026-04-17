from __future__ import annotations

import os
from typing import Any, Dict, Iterator, Tuple

import torch


def _cfg_get(cfg: Any, path: str, default: Any = None) -> Any:
    """
    Safe nested getter for either:
      - OmegaConf objects (cfg.foo.bar)
      - plain dicts (cfg["foo"]["bar"])
    path example: "model.image_size"
    """
    cur = cfg
    for key in path.split("."):
        if cur is None:
            return default
        if hasattr(cur, key):  # OmegaConf / SimpleNamespace
            cur = getattr(cur, key)
        elif isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return default
    return cur


class DictBatchLoader:
    """
    Wraps a loader that yields (rgb, depth, valid) into a loader that yields:
      {
        "rgb":   (B,3,H,W) float32 in [0,1]
        "depth": (B,1,H,W) float32 meters
        "valid": (B,1,H,W) bool
        "meta":  {}
      }
    """
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        for rgb, depth, valid in self.loader:
            if depth.ndim == 3:
                depth = depth.unsqueeze(1)
            if valid.ndim == 3:
                valid = valid.unsqueeze(1)
            yield {"rgb": rgb, "depth": depth, "valid": valid, "meta": {}}

    def __len__(self) -> int:
        if hasattr(self.loader, "__len__"):
            return len(self.loader)  # type: ignore[arg-type]
        raise TypeError("Underlying loader does not define __len__")

    @property
    def dataset(self):
        return getattr(self.loader, "dataset", None)


def build_dataloaders(cfg: Any):
    """
    Build (train_loader, val_loader) for the *current* repo state:
      - NYU Depth v2 via WebDataset tar shards containing per-sample .h5 files.

    Uses (env overrides config):
      - crop size: cfg.model.image_size (square crop)
      - root: NYU_WDS_ROOT or cfg.data.root
      - batch_size: BATCH_SIZE or cfg.data.batch_size (default 4)
      - train workers: NUM_WORKERS or cfg.data.num_workers (default 2)
      - val workers: VAL_NUM_WORKERS (default 0)  <-- fixes WebDataset "No samples found" errors
    """
    # 1) Infer crop size from model.image_size (you currently have 352)
    image_size = int(_cfg_get(cfg, "model.image_size", 352))
    train_crop = (image_size, image_size)
    val_crop = (image_size, image_size)

    # 2) Read runtime knobs from env or config
    root = os.environ.get("NYU_WDS_ROOT", None) or _cfg_get(cfg, "data.root", None)
    if not root:
        root = "/Users/andrewpan/Documents/datasets/nyu_depth_v2/data"  # machine-specific fallback

    batch_size = int(os.environ.get("BATCH_SIZE", str(_cfg_get(cfg, "data.batch_size", 4))))

    # Train workers (fast)
    num_workers_train = int(os.environ.get("NUM_WORKERS", str(_cfg_get(cfg, "data.num_workers", 2))))
    # Val workers (safe). Default 0 to avoid "fewer shards than workers" for small val sets.
    num_workers_val = int(os.environ.get("VAL_NUM_WORKERS", "0"))

    # WebDataset shuffle knobs (env overridable)
    shardshuffle = int(os.environ.get("WDS_SHARDSHUFFLE", "100"))
    sample_shuffle = int(os.environ.get("WDS_SAMPLE_SHUFFLE", "1000"))

    from .nyu_wds import make_nyu_wds_loaders

    # 3a) Build TRAIN loader with requested workers + shuffling
    train_loader, _ = make_nyu_wds_loaders(
        root=root,
        batch_size=batch_size,
        num_workers=num_workers_train,
        train_crop=train_crop,
        val_crop=val_crop,
        shardshuffle=shardshuffle,
        sample_shuffle=sample_shuffle,
    )

    # 3b) Build VAL loader with safe worker count (default 0).
    # For val, shuffling isn't necessary; set to 0 for determinism & speed.
    _, val_loader = make_nyu_wds_loaders(
        root=root,
        batch_size=batch_size,
        num_workers=num_workers_val,
        train_crop=train_crop,
        val_crop=val_crop,
        shardshuffle=0,
        sample_shuffle=0,
    )

    # 4) Standardize output format for the trainer
    return DictBatchLoader(train_loader), DictBatchLoader(val_loader)


if __name__ == "__main__":
    # Smoke test (no YAML needed)
    cfg = {"model": {"image_size": 352}}
    train_loader, val_loader = build_dataloaders(cfg)

    batch = next(iter(train_loader))
    print("train rgb:", batch["rgb"].shape, batch["rgb"].dtype)

    batch = next(iter(val_loader))
    print("val rgb:", batch["rgb"].shape, batch["rgb"].dtype)
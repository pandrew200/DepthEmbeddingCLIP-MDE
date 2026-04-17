# tools/debug_forward_nyu.py
"""
End-to-end single forward debug on one NYU RGB image.

Run from repo root:
  python -m tools.debug_forward_nyu \
    --config configs/model_base.yaml \
    --image /path/to/nyu_rgb.jpg \
    --out outputs/debug_forward.png

This checks:
  - CLIP loads
  - preprocess runs (352x352)
  - intermediate features extracted (L3/L6/L9)
  - mirror -> q produced
  - dense predictor returns [1,1,352,352]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
from torchvision import transforms

from src.models.model import DepthModel


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_clip_preprocess_352(open_clip_preprocess, image_size: int = 352):
    """
    open_clip gives a preprocess transform (usually resize/crop to 224).
    We want 352 while keeping the SAME normalization (mean/std) as CLIP expects.

    Strategy:
      - Extract the Normalize(mean,std) from open_clip_preprocess.
      - Build our own transform: Resize(image_size,image_size) -> ToTensor -> Normalize(mean,std)
    """
    mean, std = None, None
    # open_clip_preprocess is usually torchvision.transforms.Compose
    if hasattr(open_clip_preprocess, "transforms"):
        for t in open_clip_preprocess.transforms:
            if isinstance(t, transforms.Normalize):
                mean, std = t.mean, t.std
                break

    if mean is None or std is None:
        raise RuntimeError(
            "Could not find Normalize(mean,std) inside open_clip preprocess. "
            "Print backbone.get_preprocess() and adjust this function."
        )

    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def save_debug_viz(pil_rgb: Image.Image, depth: torch.Tensor, out_path: Path):
    """
    Save side-by-side visualization:
      - original RGB
      - predicted depth (normalized colormap)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    depth_np = depth.squeeze().detach().cpu().numpy()  # [H,W]
    # Normalize for visualization (robust percentiles)
    lo, hi = np.percentile(depth_np, 2), np.percentile(depth_np, 98)
    depth_vis = np.clip((depth_np - lo) / (hi - lo + 1e-8), 0, 1)

    fig = plt.figure(figsize=(10, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(pil_rgb)
    ax1.set_title("RGB")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(depth_vis, cmap="magma")
    ax2.set_title("Predicted depth (normalized)")
    ax2.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/model_base.yaml")
    ap.add_argument("--image", type=str, required=True, help="Path to NYU RGB image (jpg/png).")
    ap.add_argument("--out", type=str, default="outputs/debug_forward.png")
    ap.add_argument("--device", type=str, default=None, help="Override device from config (cpu/cuda).")
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    # ----------------------------
    # Pull sub-configs
    # ----------------------------
    clip_cfg = cfg.get("clip", {})
    depth_embedding_cfg = cfg.get("depth_embedding", {})
    dense_cfg = cfg.get("dense_predictor", {})
    model_cfg = cfg.get("model", {"image_size": 352, "layers": [3, 6, 9]})

    if args.device is not None:
        clip_cfg["device"] = args.device

    device = clip_cfg.get("device", "cpu")

    # ----------------------------
    # Build model
    # ----------------------------
    model = DepthModel(
        clip_cfg=clip_cfg,
        depth_embedding_cfg=depth_embedding_cfg,
        dense_predictor_cfg=dense_cfg,
        model_cfg=model_cfg,
    ).to(device)

    model.eval()

    # Build a 352x352 preprocess with CLIP normalization
    open_clip_preprocess = model.backbone.get_preprocess()
    preprocess_352 = build_clip_preprocess_352(open_clip_preprocess, image_size=model.model_cfg.image_size)

    # ----------------------------
    # Load image
    # ----------------------------
    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    pil_img = Image.open(img_path).convert("RGB")
    print(f"Loaded image: {img_path}")
    print(f"Original size: {pil_img.size} (W,H)")

    # Preprocess -> [1,3,352,352]
    x = preprocess_352(pil_img).unsqueeze(0).to(device)
    print(f"Input tensor: {tuple(x.shape)} dtype={x.dtype} device={x.device}")

    # ----------------------------
    # Forward pass
    # ----------------------------
    with torch.no_grad():
        depth = model(x, out_hw=[model.model_cfg.image_size, model.model_cfg.image_size])

    print(f"Depth output: {tuple(depth.shape)}")
    print(f"Depth stats: min={depth.min().item():.4f}, max={depth.max().item():.4f}, mean={depth.mean().item():.4f}")

    # Save visualization
    out_path = Path(args.out)
    save_debug_viz(pil_img.resize((model.model_cfg.image_size, model.model_cfg.image_size)), depth[0, 0], out_path)
    print(f"Saved debug visualization to: {out_path}")


if __name__ == "__main__":
    main()
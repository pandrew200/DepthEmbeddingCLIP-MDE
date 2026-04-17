from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml
import numpy as np
from PIL import Image

import torch

from src.models.clip_backbone import CLIPBackbone


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/model_base.yaml")
    ap.add_argument("--image", type=str, default=None, help="Path to an RGB image.")
    ap.add_argument("--device", type=str, default=None, help="Override device from config.")
    ap.add_argument("--no_patch_grid", action="store_true", help="Return full token seq instead of patch grid.")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    clip_cfg = cfg.get("clip", {})
    if args.device is not None:
        clip_cfg["device"] = args.device

    backbone = CLIPBackbone(clip_cfg)
    preprocess = backbone.get_preprocess()

    device = clip_cfg.get("device", "cuda")
    layers = clip_cfg.get("layers", [3, 6, 9])
    return_patch = not args.no_patch_grid

    if args.image is None:
        # Dummy image (useful for shape sanity)
        img = Image.fromarray(np.zeros((352, 352, 3), dtype=np.uint8))
        print("No --image provided; using a dummy black image.")
    else:
        img_path = Path(args.image)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = Image.open(img_path).convert("RGB")
        print(f"Loaded image: {img_path}")

    # preprocess returns a tensor [3,H,W]
    x = preprocess(img).unsqueeze(0).to(device)  # [1,3,H,W]
    print(f"Input tensor: {tuple(x.shape)} dtype={x.dtype} device={x.device}")

    # Extract intermediates
    feats = backbone.extract_vit_intermediates(
        x,
        layers=layers,
        return_patch_tokens=return_patch,
    )

    print("\nIntermediate features:")
    for l in sorted(feats.keys()):
        t = feats[l]
        print(f"  L{l}: shape={tuple(t.shape)} dtype={t.dtype}")

    # Also print final image embedding
    emb = backbone.encode_image(x)
    print(f"\nFinal image embedding: shape={tuple(emb.shape)} dtype={emb.dtype}")


if __name__ == "__main__":
    main()
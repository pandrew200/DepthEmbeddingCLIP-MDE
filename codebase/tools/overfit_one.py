# tools/overfit_one.py
"""
Overfit a single NYU image as a debugging step.

This script:
- Loads ONE RGB and ONE depth map from file paths
- Builds the depth model (frozen CLIP + trainable depth embedding + dense predictor)
- Optimizes the model to fit this single example
- Saves visualizations every N steps

Run (from repo root):
python -m tools.overfit_one \
  --config configs/model_base.yaml \
  --rgb /path/to/rgb_00045.jpg \
  --depth /path/to/depth_00045.png \
  --out_dir outputs/overfit_one \
  --steps 800 \
  --lr 0.003

Depth file expectations:
- Prefer a 16-bit PNG depth map in *meters* or a known scale.
- If it's in millimeters, pass --depth_scale 1000 to convert to meters.

If you only have the .mat NYU labeled file, use your NYU extraction pipeline to generate depth pngs,
or tell me your current depth storage format and I'll adapt the loader.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt

from src.models.model import DepthModel
from src.losses.si_loss import ScaleInvariantLogLoss, SILossConfig
from src.utils.depth_vis import normalize_depth, depth_to_colormap


# -----------------------------
# Utility: YAML loading
# -----------------------------
def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def tensor_stats(name: str, t: torch.Tensor) -> None:
    t = t.detach()
    print(
        f"{name}: "
        f"mean={t.mean().item():.6e} "
        f"std={t.std().item():.6e} "
        f"min={t.min().item():.6e} "
        f"max={t.max().item():.6e} "
        f"abs_mean={t.abs().mean().item():.6e} "
        f"abs_max={t.abs().max().item():.6e}"
    )


# -----------------------------
# Utility: build 352 preprocess with CLIP normalization
# -----------------------------
def build_clip_preprocess_352(open_clip_preprocess, image_size: int = 352):
    """
    open_clip provides a preprocess transform (commonly for 224). We need:
      Resize(image_size,image_size) + ToTensor + Normalize(mean,std)

    We extract the Normalize(mean,std) from the open_clip preprocess.
    """
    mean, std = None, None
    if hasattr(open_clip_preprocess, "transforms"):
        for t in open_clip_preprocess.transforms:
            if isinstance(t, transforms.Normalize):
                mean, std = t.mean, t.std
                break
    if mean is None or std is None:
        raise RuntimeError("Could not find Normalize(mean,std) in open_clip preprocess.")

    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# -----------------------------
# Depth loading
# -----------------------------
def load_depth_png(path: Path) -> np.ndarray:
    """
    Load depth as numpy array.

    Supports:
    - 16-bit PNG (common for depth): mode 'I;16'
    - 8-bit PNG (rare): mode 'L' (not ideal)
    - 32-bit float TIFF/PNG if present (PIL may load as 'F' or 'I')

    Returns:
      depth: float32 array [H,W] in raw units (caller scales to meters).
    """
    img = Image.open(path)
    depth = np.array(img)

    # Convert to float32
    depth = depth.astype(np.float32)
    return depth


def make_valid_mask(depth_m: torch.Tensor, min_depth: float = 1e-3, max_depth: float = 80.0) -> torch.Tensor:
    """
    Valid pixels: depth in (min_depth, max_depth)
    depth_m: [B,1,H,W] in meters
    """
    return (depth_m > min_depth) & (depth_m < max_depth)


# -----------------------------
# Visualization
# -----------------------------
def save_viz(
    rgb_vis: Image.Image,
    gt: torch.Tensor,
    pred: torch.Tensor,
    out_path: Path,
    *,
    mode: str = "percentile_each",  # "percentile_each" matches visualize_depth feel
    fixed_min_m: float = 0.1,
    fixed_max_m: float = 10.0,
    cmap: str = "plasma",
    invalid_gray: int = 128,        # set invalid GT pixels to gray
):
    """
    Save side-by-side visualization:
      RGB | GT depth (colormap) | Pred depth (colormap)

    mode:
      - "percentile_each": normalize GT and Pred independently using their own 2/98 percentiles
                            (this matches the contrast you'd see from visualize_depth on each map)
      - "fixed": normalize both using fixed_min_m/fixed_max_m (consistent scale across steps)
      - "percentile_shared": normalize both using shared bounds (GT+Pred combined) for comparable colors
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gt_np = gt.detach().float().cpu().numpy()
    pr_np = pred.detach().float().cpu().numpy()

    def bounds_percentile(x: np.ndarray):
        v = x[x > 0]
        if v.size == 0:
            return fixed_min_m, fixed_max_m
        lo, hi = np.percentile(v, 2), np.percentile(v, 98)
        lo = float(max(lo, 1e-6))
        hi = float(max(hi, lo + 1e-6))
        return lo, hi

    if mode == "fixed":
        gt_min, gt_max = fixed_min_m, fixed_max_m
        pr_min, pr_max = fixed_min_m, fixed_max_m

    elif mode == "percentile_each":
        gt_min, gt_max = bounds_percentile(gt_np)
        pr_min, pr_max = bounds_percentile(pr_np)

    elif mode == "percentile_shared":
        valid = np.concatenate([gt_np[gt_np > 0], pr_np[pr_np > 0]], axis=0)
        if valid.size == 0:
            shared_min, shared_max = fixed_min_m, fixed_max_m
        else:
            shared_min, shared_max = np.percentile(valid, 2), np.percentile(valid, 98)
            shared_min = float(max(shared_min, 1e-6))
            shared_max = float(max(shared_max, shared_min + 1e-6))
        gt_min, gt_max = shared_min, shared_max
        pr_min, pr_max = shared_min, shared_max

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Normalize to [0,1]
    gt_norm = normalize_depth(gt_np, min_depth=gt_min, max_depth=gt_max, clip=True, ignore_invalid=True)
    pr_norm = normalize_depth(pr_np, min_depth=pr_min, max_depth=pr_max, clip=True, ignore_invalid=True)

    # Colormap to uint8 RGB
    gt_rgb = depth_to_colormap(gt_norm, cmap=cmap)
    pr_rgb = depth_to_colormap(pr_norm, cmap=cmap)

    # Make invalid GT pixels gray (so they don’t look like “near” depth)
    gt_valid = gt_np > 0
    gt_rgb[~gt_valid] = invalid_gray

    gt_pil = Image.fromarray(gt_rgb)
    pr_pil = Image.fromarray(pr_rgb)

    # Ensure same size as rgb_vis
    W, H = rgb_vis.size
    if gt_pil.size != (W, H):
        gt_pil = gt_pil.resize((W, H), Image.NEAREST)
    if pr_pil.size != (W, H):
        pr_pil = pr_pil.resize((W, H), Image.NEAREST)

    canvas = Image.new("RGB", (W * 3, H))
    canvas.paste(rgb_vis, (0, 0))
    canvas.paste(gt_pil, (W, 0))
    canvas.paste(pr_pil, (W * 2, 0))
    canvas.save(out_path)


# -----------------------------
# Parameter selection
# -----------------------------
def get_trainable_params(model: torch.nn.Module):
    """
    Only optimize parameters that require gradients.
    CLIP encoders should be frozen already; mirror tokens + dense predictor are trainable.
    """
    return [p for p in model.parameters() if p.requires_grad]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/model_base.yaml")
    ap.add_argument("--rgb", type=str, required=True, help="Path to RGB image (NYU jpg/png).")
    ap.add_argument("--depth", type=str, required=True, help="Path to depth image (PNG).")
    ap.add_argument(
        "--depth_scale",
        type=float,
        default=1000.0,
        help="Divide raw depth by this to get meters. Use 1000 if depth PNG is in millimeters.",
    )
    ap.add_argument("--out_dir", type=str, default="outputs/overfit_one")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--device", type=str, default=None, help="Override device from config (cpu/cuda).")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    clip_cfg = cfg.get("clip", {})
    depth_embedding_cfg = cfg.get("depth_embedding", {})
    dense_cfg = cfg.get("dense_predictor", {})
    model_cfg = cfg.get("model", {"image_size": 352, "layers": [3, 6, 9]})

    # Force mean pooling for depth embedding tokens
    depth_embedding_cfg["pooling"] = "mean"

    if args.device is not None:
        clip_cfg["device"] = args.device
    device = clip_cfg.get("device", "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Build model
    # ----------------------------
    model = DepthModel(
        clip_cfg=clip_cfg,
        depth_embedding_cfg=depth_embedding_cfg,
        dense_predictor_cfg=dense_cfg,
        model_cfg=model_cfg,
    ).to(device)
    model.train()

    # ----------------------------
    # Preprocess RGB using CLIP normalization at 352
    # ----------------------------
    open_clip_preprocess = model.backbone.get_preprocess()
    preprocess_rgb = build_clip_preprocess_352(open_clip_preprocess, image_size=model.model_cfg.image_size)

    rgb_path = Path(args.rgb)
    depth_path = Path(args.depth)
    if not rgb_path.exists():
        raise FileNotFoundError(rgb_path)
    if not depth_path.exists():
        raise FileNotFoundError(depth_path)

    pil_rgb = Image.open(rgb_path).convert("RGB")
    rgb_vis = pil_rgb.resize((model.model_cfg.image_size, model.model_cfg.image_size), Image.BICUBIC)

    # RGB tensor: [1,3,352,352]
    rgb = preprocess_rgb(pil_rgb).unsqueeze(0).to(device)

    # Depth tensor:
    depth_raw = load_depth_png(depth_path)  # [H,W] raw units
    depth_raw_pil = Image.fromarray(depth_raw)
    depth_raw_resized = depth_raw_pil.resize((model.model_cfg.image_size, model.model_cfg.image_size), Image.NEAREST)
    depth_raw_resized = np.array(depth_raw_resized).astype(np.float32)

    # Convert to meters
    depth_m = depth_raw_resized / float(args.depth_scale)
    depth = torch.from_numpy(depth_m).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,352,352]

    # Valid mask
    mask = make_valid_mask(depth)  # [1,1,352,352]
    print(f"Valid ratio: {mask.float().mean().item():.3f}")

    # ----------------------------
    # Optimizer + Loss
    # ----------------------------
    trainable = get_trainable_params(model)
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    # Verify mirror tokens are in the optimizer
    mirror_param = model.mirror.mirror_tokens
    in_opt = any(id(p) == id(mirror_param) for g in optimizer.param_groups for p in g["params"])
    print(f"Mirror tokens requires_grad? {mirror_param.requires_grad}")
    print(f"Mirror tokens in optimizer? {in_opt}")
    if not in_opt:
        raise RuntimeError("Mirror tokens are NOT in optimizer param groups!")

    si_loss = ScaleInvariantLogLoss(SILossConfig(alpha=10.0, lam=0.85)).to(device)

    # ----------------------------
    # Mirror sanity checks (safe: no double-backward)
    # ----------------------------
    print("\n[Mirror sanity checks]")

    # (A) Does q depend on mirror tokens?
    q = model.mirror()  # [D]
    gq = torch.autograd.grad(q.sum(), mirror_param, allow_unused=True)[0]
    print("grad(q.sum -> mirror_tokens) is None?", gq is None)
    if gq is not None:
        print("grad(q.sum) abs max:", float(gq.abs().max().item()))

    # (C) Single optimizer step and measure delta (especially meaningful with --weight_decay 0)
    with torch.no_grad():
        mirror_before = mirror_param.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    pred2 = model(rgb, out_hw=[model.model_cfg.image_size, model.model_cfg.image_size])
    loss2 = si_loss(pred2, depth, mask)
    loss2.backward()

    # Look at the actual gradient that will be used for the step
    g2 = mirror_param.grad
    print("mirror_param.grad is None?", g2 is None)
    if g2 is not None:
        print("mirror_param.grad abs max:", float(g2.abs().max().item()))
        print("mirror_param.grad abs mean:", float(g2.abs().mean().item()))

    optimizer.step()

    with torch.no_grad():
        mirror_after = mirror_param.detach().clone()
        update = mirror_after - mirror_before
        print("mirror update abs max:", float(update.abs().max().item()))
        print("mirror update abs mean:", float(update.abs().mean().item()))

    # Reset baseline after the one-step sanity update
    mirror_init = mirror_param.detach().clone()

    # ----------------------------
    # Training loop: overfit one example
    # ----------------------------
    losses = []
    for step in range(args.steps + 1):
        pred = model(rgb, out_hw=[model.model_cfg.image_size, model.model_cfg.image_size])  # [1,1,352,352]
        loss = si_loss(pred, depth, mask)

        losses.append(loss.item())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if step % 50 == 0:
            g = mirror_param.grad
            if g is None:
                print(f"[{step:04d}] mirror grad: None")
            else:
                tensor_stats(f"[{step:04d}] mirror grad", g)

        optimizer.step()

        if step % 50 == 0:
            delta = (mirror_param.detach() - mirror_init)
            tensor_stats(f"[{step:04d}] mirror Δfrom_init", delta)

        if step % 25 == 0:
            with torch.no_grad():
                pmin, pmax, pmean = pred.min().item(), pred.max().item(), pred.mean().item()
            print(f"[{step:04d}] loss={loss.item():.6f} pred(min/mean/max)=({pmin:.4f},{pmean:.4f},{pmax:.4f})")

        if step % args.save_every == 0:
            save_path = out_dir / f"step_{step:04d}.png"
            save_viz(rgb_vis, depth[0, 0], pred[0, 0], save_path, mode="percentile_each")

    # ----------------------------
    # Plot loss curve
    # ----------------------------
    plt.figure(figsize=(10, 6))
    plt.semilogy(losses)
    plt.xlabel("Step")
    plt.ylabel("Scale-Invariant Log Loss")
    plt.title("Overfit One Image - Loss Curve")
    plt.grid(True)

    loss_plot_path = out_dir / "loss_curve.png"
    plt.savefig(loss_plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Saved loss curve to: {loss_plot_path}")

    print(f"Done. Saved frames to: {out_dir}")


if __name__ == "__main__":
    main()
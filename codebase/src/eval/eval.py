"""
Evaluation utilities for depth model.

Provides:
  - evaluate_checkpoint: load a checkpoint and run full evaluation on the val set
  - evaluate_model: run evaluation on an already-loaded model
  - save_qualitative: save side-by-side prediction visualizations
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
import numpy as np

from src.eval.metrics import evaluate_batch


@dataclass
class EvalResult:
    metrics: Dict[str, float]
    num_samples: int
    seconds: float
    per_image_metrics: List[Dict[str, float]] = field(default_factory=list)


def evaluate_model(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    loss_fn=None,
    collect_per_image: bool = False,
) -> EvalResult:
    """
    Run full evaluation on the validation set.

    Args:
        model: trained model (will be set to eval mode)
        val_loader: yields dict batches with rgb, depth, valid
        device: torch device
        loss_fn: optional loss function to compute val_loss
        collect_per_image: if True, store per-image metrics (slower, more memory)

    Returns:
        EvalResult with aggregated metrics and timing
    """
    model.eval()
    t0 = time.time()

    total_samples = 0
    metric_sums: Dict[str, float] = {}
    loss_sum = 0.0
    per_image: List[Dict[str, float]] = []

    with torch.no_grad():
        for batch in val_loader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            gt = batch["depth"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)

            pred = model(rgb)

            bs = int(rgb.shape[0])
            total_samples += bs

            if loss_fn is not None:
                vloss = loss_fn(pred, gt, valid).detach().float().item()
                loss_sum += vloss * bs

            if collect_per_image:
                for i in range(bs):
                    m = evaluate_batch(
                        pred[i : i + 1], gt[i : i + 1], valid[i : i + 1]
                    )
                    per_image.append(m)

            batch_metrics = evaluate_batch(pred, gt, valid)
            for k, v in batch_metrics.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + float(v) * bs

    out = {k: v / max(total_samples, 1) for k, v in metric_sums.items()}
    if loss_fn is not None:
        out["val_loss"] = loss_sum / max(total_samples, 1)

    return EvalResult(
        metrics=out,
        num_samples=total_samples,
        seconds=time.time() - t0,
        per_image_metrics=per_image,
    )


def evaluate_checkpoint(
    checkpoint_path: str,
    cfg: Any,
    device: torch.device,
    collect_per_image: bool = False,
) -> EvalResult:
    """
    Load a checkpoint and evaluate on the validation set.

    Args:
        checkpoint_path: path to .pt file with "model" and "cfg" keys
        cfg: config dict (used for building model + data)
        device: torch device
        collect_per_image: if True, store per-image metrics

    Returns:
        EvalResult
    """
    from src.models.build import build_model
    from src.data.build import build_dataloaders

    # Build model
    model = build_model(cfg).to(device)
    for p in model.backbone.parameters():
        p.requires_grad = False

    # Load weights
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    print(f"[eval] loaded checkpoint: {checkpoint_path}")
    if "epoch" in ckpt:
        print(f"[eval] checkpoint epoch: {ckpt['epoch']}")
    if "best_metric" in ckpt:
        print(f"[eval] checkpoint best_metric: {ckpt['best_metric']:.6f}")

    # Build val loader only
    _, val_loader = build_dataloaders(cfg)

    # Build loss (optional, for val_loss)
    loss_fn = None
    try:
        from src.losses.si_loss import build
        loss_fn = build(cfg)
    except Exception:
        pass

    return evaluate_model(
        model=model,
        val_loader=val_loader,
        device=device,
        loss_fn=loss_fn,
        collect_per_image=collect_per_image,
    )


def save_qualitative(
    model: torch.nn.Module,
    val_loader,
    device: torch.device,
    out_dir: str,
    num_samples: int = 16,
    cmap: str = "magma",
):
    """
    Save side-by-side visualizations: RGB | GT depth | Predicted depth | Error map.

    Args:
        model: trained model
        val_loader: validation data loader
        device: torch device
        out_dir: directory to save images
        num_samples: number of samples to visualize
        cmap: matplotlib colormap for depth
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    count = 0
    with torch.no_grad():
        for batch in val_loader:
            if count >= num_samples:
                break

            rgb = batch["rgb"].to(device, non_blocking=True)
            gt = batch["depth"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)
            pred = model(rgb)

            bs = int(rgb.shape[0])
            for i in range(bs):
                if count >= num_samples:
                    break

                rgb_np = rgb[i].cpu().permute(1, 2, 0).numpy()
                gt_np = gt[i, 0].cpu().numpy()
                pred_np = pred[i, 0].cpu().numpy()
                valid_np = valid[i, 0].cpu().numpy().astype(bool)

                # Depth range for consistent colormap
                vmin = gt_np[valid_np].min() if valid_np.any() else 0
                vmax = gt_np[valid_np].max() if valid_np.any() else 10

                # Error map
                error = np.abs(pred_np - gt_np)
                error[~valid_np] = 0

                fig, axes = plt.subplots(1, 4, figsize=(20, 5))

                axes[0].imshow(rgb_np)
                axes[0].set_title("RGB Input")
                axes[0].axis("off")

                im1 = axes[1].imshow(gt_np, cmap=cmap, vmin=vmin, vmax=vmax)
                axes[1].set_title("GT Depth")
                axes[1].axis("off")
                plt.colorbar(im1, ax=axes[1], fraction=0.046)

                im2 = axes[2].imshow(pred_np, cmap=cmap, vmin=vmin, vmax=vmax)
                axes[2].set_title("Predicted Depth")
                axes[2].axis("off")
                plt.colorbar(im2, ax=axes[2], fraction=0.046)

                im3 = axes[3].imshow(error, cmap="hot", vmin=0, vmax=vmax * 0.3)
                axes[3].set_title("Abs Error")
                axes[3].axis("off")
                plt.colorbar(im3, ax=axes[3], fraction=0.046)

                # Per-image metrics
                m = evaluate_batch(
                    pred[i : i + 1], gt[i : i + 1], valid[i : i + 1]
                )
                fig.suptitle(
                    f"Sample {count}  |  abs_rel={m['abs_rel']:.4f}  "
                    f"rmse={m['rmse']:.4f}  δ1={m['delta1']:.4f}",
                    fontsize=12,
                )

                fig.savefig(
                    os.path.join(out_dir, f"sample_{count:04d}.png"),
                    dpi=120, bbox_inches="tight",
                )
                plt.close()
                count += 1

    print(f"[eval] saved {count} qualitative samples to {out_dir}/")
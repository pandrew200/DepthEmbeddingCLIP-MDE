"""
Scale-invariant depth loss (AdaBins-style).

Form:
  g_i = log(d_hat_i) - log(d_i)
  L = alpha * sqrt( mean(g^2) - lam * (mean(g))^2 )

We compute this only on valid pixels (mask).

Notes:
- Clamp pred/gt to eps before log to avoid -inf.
- Use a small relu clamp inside sqrt to prevent NaNs due to numerical noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class SILossConfig:
    alpha: float = 10.0
    lam: float = 0.85
    eps: float = 1e-6
    min_depth: Optional[float] = None
    max_depth: Optional[float] = None

    def __post_init__(self):
        self.alpha = float(self.alpha)
        self.lam = float(self.lam)
        self.eps = float(self.eps)
        if self.min_depth is not None:
            self.min_depth = float(self.min_depth)
        if self.max_depth is not None:
            self.max_depth = float(self.max_depth)


class ScaleInvariantLogLoss(nn.Module):
    """
    AdaBins-style scale-invariant log loss with masking.

    Inputs:
      pred: [B,1,H,W] predicted depth (positive)
      gt:   [B,1,H,W] ground-truth depth (positive)
      mask: [B,1,H,W] boolean mask of valid gt pixels (True = valid)

    Returns:
      scalar loss
    """
    def __init__(self, cfg: Optional[SILossConfig] = None):
        super().__init__()
        self.cfg = cfg if cfg is not None else SILossConfig()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if pred.shape != gt.shape:
            raise ValueError(f"pred shape {pred.shape} != gt shape {gt.shape}")
        if mask.shape != gt.shape:
            raise ValueError(f"mask shape {mask.shape} != gt shape {gt.shape}")
        if mask.dtype != torch.bool:
            mask = mask.bool()

        pred = pred.clamp(min=self.cfg.eps)
        gt = gt.clamp(min=self.cfg.eps)

        # Optional clamping to dataset depth range (helpful if GT includes outliers)
        if self.cfg.min_depth is not None:
            gt = gt.clamp(min=self.cfg.min_depth)
            pred = pred.clamp(min=self.cfg.min_depth)
        if self.cfg.max_depth is not None:
            gt = gt.clamp(max=self.cfg.max_depth)
            pred = pred.clamp(max=self.cfg.max_depth)

        g = pred.log() - gt.log()  # [B,1,H,W]
        g = g[mask]                # [T] flattened valid entries

        if g.numel() == 0:
            # If mask is empty, return 0 but warn via NaN-safe scalar
            return pred.new_tensor(0.0)

        mean_g2 = (g * g).mean()
        mean_g = g.mean()
        inside = mean_g2 - self.cfg.lam * (mean_g * mean_g)

        # Avoid sqrt of negative due to float error
        inside = torch.relu(inside)

        loss = self.cfg.alpha * torch.sqrt(inside + 1e-12)
        return loss


def build(cfg) -> ScaleInvariantLogLoss:
    """
    Build from a top-level YAML cfg dict.

    Expected:
      cfg["loss"] = {"name": "si_loss", "alpha": ..., "lam": ..., "eps": ..., ...}
    """
    loss_cfg = cfg.get("loss", {}) if isinstance(cfg, dict) else {}
    # drop "name" and pass the rest into SILossConfig
    kwargs = {k: v for k, v in loss_cfg.items() if k != "name"}
    return ScaleInvariantLogLoss(SILossConfig(**kwargs))
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


EPS = 1e-8


def _prepare_tensors(pred: torch.Tensor, gt: torch.Tensor, min_depth: Optional[float] = None, max_depth: Optional[float] = None):
    """
    Flattens tensors and applies validity mask (gt > 0 and optional depth range).

    Args:
        pred: [B, 1, H, W] or [B, H, W]
        gt:   same shape as pred
        min_depth: optional minimum valid depth
        max_depth: optional maximum valid depth

    Returns:
        pred_valid, gt_valid (1D tensors of valid pixels only)
    """
    if pred.ndim == 4:
        pred = pred.squeeze(1)
    if gt.ndim == 4:
        gt = gt.squeeze(1)

    assert pred.shape == gt.shape, "Prediction and GT must have same shape"

    mask = gt > 0  # remove invalid depth

    if min_depth is not None:
        mask = mask & (gt >= min_depth)
    if max_depth is not None:
        mask = mask & (gt <= max_depth)

    pred = pred[mask]
    gt = gt[mask]

    return pred, gt


def _prepare_tensors_with_mask(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor):
    """
    Flattens tensors using an explicit validity mask (True = valid).
    Used by evaluate_batch to match the dataloader's batch["valid"].

    Args:
        pred: [B, 1, H, W] or [B, H, W]
        gt:   same shape as pred
        valid: same shape as pred, bool (True = valid pixel)

    Returns:
        pred_valid, gt_valid (1D tensors of valid pixels only)
    """
    if pred.ndim == 4:
        pred = pred.squeeze(1)
    if gt.ndim == 4:
        gt = gt.squeeze(1)
    if valid.ndim == 4:
        valid = valid.squeeze(1)

    assert pred.shape == gt.shape == valid.shape, "pred, gt, valid must have same shape"
    mask = valid.bool()

    pred = pred[mask]
    gt = gt[mask]
    return pred, gt


# --------------------------------------------------------
# Abs Rel
# --------------------------------------------------------

def abs_rel(pred, gt, min_depth=None, max_depth=None):
    """
    Absolute Relative Error:
        mean( |pred - gt| / gt )
    """
    pred, gt = _prepare_tensors(pred, gt, min_depth, max_depth)
    return torch.mean(torch.abs(pred - gt) / (gt + EPS))


# --------------------------------------------------------
# RMSE
# --------------------------------------------------------

def rmse(pred, gt, min_depth=None, max_depth=None):
    """
    Root Mean Squared Error:
        sqrt( mean( (pred - gt)^2 ) )
    """
    pred, gt = _prepare_tensors(pred, gt, min_depth, max_depth)
    return torch.sqrt(torch.mean((pred - gt) ** 2))


# --------------------------------------------------------
# Delta Threshold Metrics
# --------------------------------------------------------

def delta_metrics(pred, gt, min_depth=None, max_depth=None):
    """
    Computes δ1, δ2, δ3 accuracy metrics.

    δn = percentage of pixels where:
        max(pred/gt, gt/pred) < 1.25^n
    """
    pred, gt = _prepare_tensors(pred, gt, min_depth, max_depth)

    ratio = torch.max(pred / (gt + EPS), gt / (pred + EPS))

    delta1 = torch.mean((ratio < 1.25).float())
    delta2 = torch.mean((ratio < 1.25 ** 2).float())
    delta3 = torch.mean((ratio < 1.25 ** 3).float())

    return delta1, delta2, delta3


# --------------------------------------------------------
# Combined evaluator (mask-based: used by eval_one_epoch)
# --------------------------------------------------------

def evaluate_batch(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> Dict[str, float]:
    """
    Compute depth metrics over a batch using the dataloader's validity mask.
    Called by eval_one_epoch with batch["valid"] (True = valid pixel).

    Returns:
        Dict with abs_rel, rmse, delta1, delta2, delta3 (scalar floats).
    """
    pred, gt = _prepare_tensors_with_mask(pred, gt, valid)
    if pred.numel() == 0:
        return {
            "abs_rel": 0.0,
            "rmse": 0.0,
            "delta1": 0.0,
            "delta2": 0.0,
            "delta3": 0.0,
        }

    abs_rel_val = torch.mean(torch.abs(pred - gt) / (gt + EPS)).item()
    rmse_val = torch.sqrt(torch.mean((pred - gt) ** 2)).item()
    ratio = torch.max(pred / (gt + EPS), gt / (pred + EPS))
    d1 = (ratio < 1.25).float().mean().item()
    d2 = (ratio < 1.25 ** 2).float().mean().item()
    d3 = (ratio < 1.25 ** 3).float().mean().item()

    return {
        "abs_rel": abs_rel_val,
        "rmse": rmse_val,
        "delta1": d1,
        "delta2": d2,
        "delta3": d3,
    }


def compute_depth_metrics(pred, gt, min_depth=None, max_depth=None):
    """
    Convenience function returning all metrics in a dict (uses gt > 0 mask).
    """
    metrics = {}

    metrics["abs_rel"] = abs_rel(pred, gt, min_depth, max_depth)
    metrics["rmse"] = rmse(pred, gt, min_depth, max_depth)

    d1, d2, d3 = delta_metrics(pred, gt, min_depth, max_depth)
    metrics["delta1"] = d1
    metrics["delta2"] = d2
    metrics["delta3"] = d3

    return metrics
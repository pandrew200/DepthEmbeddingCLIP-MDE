from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass
class EvalEpochResult:
    metrics: Dict[str, float]
    num_samples: int
    seconds: float


def _masked_metrics_fallback(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> Dict[str, float]:
    """
    Fallback metrics (masked) if src.eval.metrics isn't available / doesn't match expected API.
    Computes: abs_rel, rmse, delta1, delta2, delta3
    """
    if pred.ndim == 3:
        pred = pred.unsqueeze(1)
    if gt.ndim == 3:
        gt = gt.unsqueeze(1)
    if valid.ndim == 3:
        valid = valid.unsqueeze(1)

    pred = pred.float()
    gt = gt.float()
    valid = valid.bool()

    # avoid div-by-zero
    eps = 1e-6
    gt_safe = torch.clamp(gt, min=eps)

    e = (pred - gt).abs()
    abs_rel = (e / gt_safe)[valid].mean()

    rmse = torch.sqrt(((pred - gt) ** 2)[valid].mean())

    ratio = torch.max(pred / gt_safe, gt_safe / torch.clamp(pred, min=eps))
    d1 = (ratio < 1.25)[valid].float().mean()
    d2 = (ratio < 1.25 ** 2)[valid].float().mean()
    d3 = (ratio < 1.25 ** 3)[valid].float().mean()

    return {
        "abs_rel": float(abs_rel.item()),
        "rmse": float(rmse.item()),
        "delta1": float(d1.item()),
        "delta2": float(d2.item()),
        "delta3": float(d3.item()),
    }


def _compute_metrics(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> Dict[str, float]:
    """
    Try to use src.eval.metrics if present, otherwise fallback.
    Supports two common styles:
      - functions: abs_rel(pred, gt, valid), rmse(...), delta1/2/3(...)
      - or a function evaluate_batch(...) returning dict
    """
    try:
        from src.eval import metrics as m  # your existing module

        # Style A: m.evaluate_batch(...)
        if hasattr(m, "evaluate_batch"):
            out = m.evaluate_batch(pred, gt, valid)
            return {k: float(v) for k, v in out.items()}

        # Style B: individual functions
        out = {}
        if hasattr(m, "abs_rel"):
            out["abs_rel"] = float(m.abs_rel(pred, gt, valid))
        if hasattr(m, "rmse"):
            out["rmse"] = float(m.rmse(pred, gt, valid))
        # deltas may be provided as delta1/delta2/delta3
        for name in ("delta1", "delta2", "delta3"):
            if hasattr(m, name):
                out[name] = float(getattr(m, name)(pred, gt, valid))

        if len(out) > 0:
            return out

    except Exception:
        # fall back below
        pass

    return _masked_metrics_fallback(pred, gt, valid)


def eval_one_epoch(
    *,
    model: torch.nn.Module,
    val_loader: Any,
    device: torch.device,
    loss_fn: Optional[Any] = None,
    amp: bool = True,
) -> EvalEpochResult:
    """
    One validation epoch. Returns aggregated metrics (averaged over samples).
    """
    model.eval()

    t0 = time.time()
    total_samples = 0

    # We average metrics over samples (not over batches), to be stable with last batch size.
    sums: Dict[str, float] = {}
    loss_sum = 0.0

    use_amp = amp and (device.type in ("cuda", "mps"))
    autocast_dtype = torch.float16

    with torch.no_grad():
        for batch in val_loader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            gt = batch["depth"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)

            if use_amp:
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    pred = model(rgb)
            else:
                pred = model(rgb)

            bs = int(rgb.shape[0])
            total_samples += bs

            # optional val loss
            if loss_fn is not None:
                vloss = loss_fn(pred, gt, valid).detach().float().item()
                loss_sum += float(vloss) * bs

            metrics = _compute_metrics(pred, gt, valid)
            for k, v in metrics.items():
                sums[k] = sums.get(k, 0.0) + float(v) * bs

    out = {k: v / max(total_samples, 1) for k, v in sums.items()}
    if loss_fn is not None:
        out["val_loss"] = loss_sum / max(total_samples, 1)

    seconds = time.time() - t0
    return EvalEpochResult(metrics=out, num_samples=total_samples, seconds=seconds)
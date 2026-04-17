from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass
class TrainEpochResult:
    loss: float
    num_samples: int
    seconds: float
    first_k_loss: float
    last_k_loss: float


def _default_loss(pred: torch.Tensor, gt: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """
    Safe default: masked L1 on depth (meters).
    pred:  (B,1,H,W) or (B,H,W)
    gt:    (B,1,H,W) or (B,H,W)
    valid: (B,1,H,W) bool
    """
    if pred.ndim == 3:
        pred = pred.unsqueeze(1)
    if gt.ndim == 3:
        gt = gt.unsqueeze(1)
    if valid.ndim == 3:
        valid = valid.unsqueeze(1)

    # Ensure boolean mask
    valid = valid.bool()
    denom = valid.sum().clamp_min(1)
    return (pred.sub(gt).abs() * valid).sum() / denom


def train_one_epoch(
    *,
    model: torch.nn.Module,
    train_loader: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: Optional[Any] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    amp: bool = True,
    grad_clip_norm: Optional[float] = None,
    log_every: int = 50,
    max_steps: Optional[int] = None, 
    k_report: int = 20
) -> TrainEpochResult:
    """
    One training epoch.

    Expects train_loader to yield dict batches:
      batch["rgb"]:   (B,3,H,W) float32
      batch["depth"]: (B,1,H,W) float32
      batch["valid"]: (B,1,H,W) bool
    """
    model.train()
    loss_fn = loss_fn or _default_loss

    t0 = time.time()
    total_loss = 0.0
    total_samples = 0

    use_amp = amp and (device.type in ("cuda", "mps"))
    autocast_dtype = torch.float16  # conservative default

    # Bounded storage for first_k and last_k loss (avoid unbounded list over long epochs)
    first_k_losses: list = []
    last_k_losses: deque = deque(maxlen=k_report)

    for step, batch in enumerate(train_loader):
        rgb = batch["rgb"].to(device, non_blocking=True)
        gt = batch["depth"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                pred = model(rgb)
                loss = loss_fn(pred, gt, valid)
            if scaler is not None:
                scaler.scale(loss).backward()
                if grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
        else:
            pred = model(rgb)
            loss = loss_fn(pred, gt, valid)
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        bs = int(rgb.shape[0])
        total_samples += bs
        loss_val = float(loss.detach().item())
        total_loss += loss_val * bs

        if len(first_k_losses) < k_report:
            first_k_losses.append(loss_val)
        last_k_losses.append(loss_val)

        if max_steps is not None and (step + 1) >= max_steps:
            break

        if log_every > 0 and (step % log_every == 0):
            # keep logging minimal; loop.py handles nicer formatting
            pass

    seconds = time.time() - t0
    avg_loss = total_loss / max(total_samples, 1)

    n_first = len(first_k_losses)
    n_last = len(last_k_losses)
    first_k = sum(first_k_losses) / max(n_first, 1)
    last_k = sum(last_k_losses) / max(n_last, 1)

    return TrainEpochResult(
        loss=avg_loss,
        num_samples=total_samples,
        seconds=seconds,
        first_k_loss=first_k,
        last_k_loss=last_k,
    )
    # return TrainEpochResult(loss=avg_loss, num_samples=total_samples, seconds=seconds)
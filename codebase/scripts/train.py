#!/usr/bin/env python3
"""
Full training script for depth model.

Trains for N epochs (default 10), evaluates after every epoch,
saves best.pt / last.pt checkpoints, and produces:
  - loss_curve.png        (per-step training loss)
  - metrics_table.png     (epoch-by-epoch eval summary)
  - train_log.json        (machine-readable log of all results)

Usage:
    # Basic (10 epochs, config defaults):
    caffeinate -s python scripts/train.py

    # Override epochs / output dir:
    EPOCHS=25 python scripts/train.py --out runs/full_25ep

    # Resume from checkpoint:
    python scripts/train.py --resume runs/v1/last.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import yaml
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.build import build_dataloaders
from src.models.build import build_model
from src.train.eval_one_epoch import eval_one_epoch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg_get(cfg: Any, path: str, default: Any = None) -> Any:
    cur = cfg
    for key in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        elif hasattr(cur, key):
            cur = getattr(cur, key)
        else:
            return default
    return cur


def _build_loss(cfg: Any):
    from src.losses.si_loss import build
    return build(cfg)


def _save_checkpoint(path: str, state: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def _count_params(module: torch.nn.Module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


# ---------------------------------------------------------------------------
# Train one epoch (with per-step loss recording)
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: torch.nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn,
    grad_clip: Optional[float] = 1.0,
    log_every: int = 100,
    epoch: int = 0,
    total_epochs: int = 0,
) -> tuple[float, List[float], float]:
    """
    Returns (avg_loss, per_step_losses, elapsed_seconds).
    """
    model.train()
    step_losses: List[float] = []
    total_loss = 0.0
    total_samples = 0
    t0 = time.time()

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    for step, batch in enumerate(train_loader):
        rgb   = batch["rgb"].to(device, non_blocking=True)
        gt    = batch["depth"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(rgb)
        loss = loss_fn(pred, gt, valid)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
        optimizer.step()

        bs = int(rgb.shape[0])
        lv = loss.detach().item()
        step_losses.append(lv)
        total_loss += lv * bs
        total_samples += bs

        if log_every > 0 and (step % log_every == 0):
            elapsed = time.time() - t0
            steps_done = step + 1
            try:
                steps_total = len(train_loader)
            except (TypeError, AttributeError):
                steps_total = "?"
            eta = (elapsed / steps_done) * (int(steps_total) - steps_done) if isinstance(steps_total, int) else 0
            print(
                f"  [epoch {epoch}/{total_epochs}] "
                f"step {step}/{steps_total}  "
                f"loss={lv:.5f}  "
                f"elapsed={elapsed:.0f}s  eta={eta:.0f}s"
            )

    elapsed = time.time() - t0
    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss, step_losses, elapsed


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

METRIC_KEYS = ["abs_rel", "rmse", "delta1", "delta2", "delta3", "val_loss"]


def plot_loss_curve(
    all_step_losses: List[float],
    epoch_boundaries: List[int],
    out_path: str,
    total_epochs: int,
):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        range(len(all_step_losses)), all_step_losses,
        linewidth=0.3, alpha=0.35, color="steelblue", label="per-step",
    )

    # Smoothed curve
    window = max(1, min(200, len(all_step_losses) // 20))
    smoothed = []
    for i in range(len(all_step_losses)):
        lo = max(0, i - window // 2)
        hi = min(len(all_step_losses), i + window // 2 + 1)
        smoothed.append(sum(all_step_losses[lo:hi]) / (hi - lo))
    ax.plot(
        range(len(smoothed)), smoothed,
        linewidth=2, color="crimson", label=f"smoothed (w={window})",
    )

    # Epoch boundaries
    for i, b in enumerate(epoch_boundaries):
        label = "epoch boundary" if i == 0 else None
        ax.axvline(x=b, color="gray", linestyle="--", linewidth=0.7, alpha=0.6, label=label)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (SI log)")
    ax.set_title(f"Training Loss — {total_epochs} epochs on NYU Depth v2")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_metrics_table(
    epoch_metrics: List[Dict[str, Any]],
    out_path: str,
):
    """Save a summary table as an image and print to stdout."""
    avail = [k for k in METRIC_KEYS if k in epoch_metrics[0].get("metrics", {})]
    if not avail:
        return

    # Print to stdout
    hdr = f"  {'epoch':>6s}  {'train_loss':>11s}" + "".join(f"  {k:>10s}" for k in avail) + f"  {'time':>7s}"
    sep = "-" * len(hdr)
    print(f"\n{sep}")
    print("  EPOCH-BY-EPOCH RESULTS")
    print(sep)
    print(hdr)
    print(sep)
    for row in epoch_metrics:
        ep = row["epoch"]
        tl = row.get("train_loss", float("nan"))
        vals = "".join(f"  {row['metrics'].get(k, float('nan')):10.6f}" for k in avail)
        secs = row.get("train_seconds", 0)
        best_marker = " *" if row.get("is_best", False) else ""
        print(f"  {ep:>6s}  {tl:11.5f}{vals}  {secs:6.0f}s{best_marker}")
    print(sep)
    print("  * = new best abs_rel\n")

    # Matplotlib table figure
    fig, ax = plt.subplots(figsize=(14, 0.4 * (len(epoch_metrics) + 2)))
    ax.axis("off")

    col_labels = ["Epoch", "Train Loss"] + avail + ["Time"]
    cell_data = []
    colors = []
    for row in epoch_metrics:
        ep = row["epoch"]
        tl = row.get("train_loss", float("nan"))
        vals = [f"{row['metrics'].get(k, float('nan')):.4f}" for k in avail]
        secs = row.get("train_seconds", 0)
        cell_data.append([ep, f"{tl:.4f}"] + vals + [f"{secs:.0f}s"])
        colors.append(
            ["#d4edda"] * len(col_labels) if row.get("is_best", False)
            else ["white"] * len(col_labels)
        )

    table = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellColours=colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Bold header
    for j in range(len(col_labels)):
        table[0, j].set_text_props(fontweight="bold")
        table[0, j].set_facecolor("#343a40")
        table[0, j].set_text_props(color="white", fontweight="bold")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train depth model")
    parser.add_argument("--config", default=os.path.join(ROOT, "configs", "model_base.yaml"))
    parser.add_argument("--out", default=None, help="Output directory (default: runs/train_<timestamp>)")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--warm-restart", action="store_true",
                        help="Resume weights+optimizer but reset epoch counter and LR schedule (cosine warm restart)")
    args = parser.parse_args()

    # ---- Config ----
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ---- Output dir ----
    if args.out:
        out_dir = os.path.join(ROOT, args.out) if not os.path.isabs(args.out) else args.out
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(ROOT, "runs", f"train_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[out] {out_dir}")

    # Save config snapshot
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # ---- Device ----
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[device] {device}")

    # ---- Seed ----
    seed = int(_cfg_get(cfg, "train.seed", 42) or 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ---- Model ----
    print("[build] loading model...")
    t0 = time.time()
    model = build_model(cfg).to(device)
    for p in model.backbone.parameters():
        p.requires_grad = False
    print(f"[build] model ready ({time.time()-t0:.1f}s)")

    tot, tr = _count_params(model)
    print(f"[params] total={tot/1e6:.2f}M  trainable={tr/1e6:.2f}M")

    # ---- Data ----
    print("[build] loading data...")
    train_loader, val_loader = build_dataloaders(cfg)
    try:
        steps_per_epoch = len(train_loader)
    except TypeError:
        steps_per_epoch = 3000
    print(f"[data] ~{steps_per_epoch} steps/epoch")

    def _rebuild_train_loader():
        """Rebuild train loader to re-randomize shard ordering each epoch."""
        tl, _ = build_dataloaders(cfg)
        return tl

    # ---- Loss ----
    loss_fn = _build_loss(cfg)
    print(f"[loss] {type(loss_fn).__name__}")

    # ---- Optimizer ----
    lr = float(os.environ.get("LR", _cfg_get(cfg, "train.lr", 3e-3) or 3e-3))
    embedding_lr = float(os.environ.get("EMBEDDING_LR", _cfg_get(cfg, "train.embedding_lr", None) or 0))
    wd = float(_cfg_get(cfg, "train.weight_decay", 0.01) or 0.01)

    if embedding_lr > 0:
        # Separate LR for depth embedding tokens vs decoder
        embedding_params = [p for p in model.depth_embedding.parameters() if p.requires_grad]
        other_params = [p for n, p in model.named_parameters()
                        if p.requires_grad and not n.startswith("depth_embedding.")]
        param_groups = [
            {"params": embedding_params, "lr": embedding_lr},
            {"params": other_params, "lr": lr},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=wd)
        print(f"[optim] AdamW decoder_lr={lr} embedding_lr={embedding_lr} wd={wd}")
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=wd)
        print(f"[optim] AdamW lr={lr} wd={wd}")

    # ---- Scheduler ----
    total_epochs = int(os.environ.get("EPOCHS", _cfg_get(cfg, "train.epochs", 10) or 10))
    sched_name = _cfg_get(cfg, "train.scheduler", None)
    scheduler = None
    if sched_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
        print(f"[sched] CosineAnnealingLR T_max={total_epochs}")

    grad_clip = float(_cfg_get(cfg, "train.grad_clip_norm", 1.0) or 1.0)
    log_every = int(os.environ.get("LOG_EVERY", "100"))

    # ---- Resume ----
    start_epoch = 1
    best_metric = float("inf")
    best_epoch = -1
    all_step_losses: List[float] = []
    epoch_boundaries: List[int] = []
    epoch_metrics: List[Dict[str, Any]] = []

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])

        if args.warm_restart:
            # Warm restart: load weights + optimizer momentum/variance, but reset
            # epoch counter and LR back to initial value for a fresh cosine schedule
            for pg in optimizer.param_groups:
                pg["lr"] = lr
                pg["initial_lr"] = lr
            # Re-create scheduler so it picks up the reset base_lrs
            if sched_name == "cosine":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
            print(f"[warm-restart] loaded weights from {args.resume} (epoch {ckpt.get('epoch', '?')})")
            print(f"[warm-restart] resetting schedule: epochs 1-{total_epochs}, lr={lr}")
        else:
            # Standard resume: continue from where we left off
            start_epoch = int(ckpt.get("epoch", 0)) + 1
            best_metric = float(ckpt.get("best_metric", best_metric))
            best_epoch = int(ckpt.get("best_epoch", best_epoch))
            if scheduler is not None and start_epoch > 1:
                for _ in range(1, start_epoch):
                    scheduler.step()
            print(f"[resume] loaded {args.resume} (start_epoch={start_epoch}, best_abs_rel={best_metric:.6f})")

        # ---- Restore history from output dir (continuous across pauses) ----
        prev_log = os.path.join(out_dir, "train_log.json")
        prev_losses = os.path.join(out_dir, "step_losses.npy")
        if os.path.isfile(prev_log):
            with open(prev_log) as f:
                epoch_metrics = json.load(f)
            # Reconstruct epoch boundaries from step counts
            # Each epoch's step count can be inferred from the per-step losses
            print(f"[resume] restored {len(epoch_metrics)} entries from train_log.json")
        if os.path.isfile(prev_losses):
            all_step_losses = np.load(prev_losses).tolist()
            # Reconstruct epoch boundaries: divide evenly by completed training epochs
            n_trained = sum(1 for r in epoch_metrics if r["epoch"] not in ("init",) and not str(r["epoch"]).startswith("resume"))
            if n_trained > 0:
                steps_per = len(all_step_losses) // n_trained
                epoch_boundaries = [steps_per * i for i in range(1, n_trained + 1)]
            print(f"[resume] restored {len(all_step_losses)} per-step losses from step_losses.npy")

    if not epoch_metrics:
        # ---- Eval @ init (only if starting fresh) ----
        print(f"\n{'='*55}")
        print(f"  Eval @ init")
        print(f"{'='*55}")
        ev_init = eval_one_epoch(model=model, val_loader=val_loader, device=device, loss_fn=loss_fn, amp=False)
        for k, v in sorted(ev_init.metrics.items()):
            print(f"  {k:12s} = {v:.6f}")
        print(f"  ({ev_init.num_samples} samples, {ev_init.seconds:.1f}s)")

        epoch_metrics.append({
            "epoch": "init",
            "train_loss": float("nan"),
            "metrics": ev_init.metrics,
            "train_seconds": 0,
            "is_best": False,
        })
    else:
        print(f"\n[resume] skipping init eval — history already contains {len(epoch_metrics)} entries")

    # ---- Training loop ----
    wall0 = time.time()
    print(f"\n[train] {total_epochs} epochs, starting from epoch {start_epoch}")
    print(f"[train] estimated wall time: ~{total_epochs * steps_per_epoch * 0.95 / 60:.0f} min on CPU\n")

    for epoch in range(start_epoch, total_epochs + 1):
        # Rebuild train loader each epoch to re-randomize shard ordering
        train_loader = _rebuild_train_loader()

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n{'─'*55}")
        print(f"  Epoch {epoch}/{total_epochs}  (lr={current_lr:.6f})")
        print(f"{'─'*55}")

        avg_loss, step_losses, train_secs = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            loss_fn=loss_fn,
            grad_clip=grad_clip,
            log_every=log_every,
            epoch=epoch,
            total_epochs=total_epochs,
        )

        # Record step losses
        epoch_boundaries.append(len(all_step_losses))
        all_step_losses.extend(step_losses)

        if scheduler is not None:
            scheduler.step()

        print(f"  train_loss={avg_loss:.5f}  ({train_secs:.1f}s, {len(step_losses)} steps)")

        # ---- Eval ----
        ev = eval_one_epoch(model=model, val_loader=val_loader, device=device, loss_fn=loss_fn, amp=False)
        for k, v in sorted(ev.metrics.items()):
            print(f"  {k:12s} = {v:.6f}")
        print(f"  ({ev.num_samples} samples, {ev.seconds:.1f}s)")

        # ---- Best tracking ----
        is_best = False
        abs_rel = ev.metrics.get("abs_rel", float("inf"))
        if abs_rel < best_metric:
            best_metric = abs_rel
            best_epoch = epoch
            is_best = True
            _save_checkpoint(
                os.path.join(out_dir, "best.pt"),
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_metric": best_metric,
                    "best_epoch": best_epoch,
                    "metrics": ev.metrics,
                    "cfg": cfg,
                },
            )
            print(f"  >>> NEW BEST abs_rel={best_metric:.6f} @ epoch {epoch} — saved best.pt")

        # ---- Save last.pt ----
        _save_checkpoint(
            os.path.join(out_dir, "last.pt"),
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "metrics": ev.metrics,
                "cfg": cfg,
            },
        )

        # ---- Record epoch ----
        epoch_metrics.append({
            "epoch": str(epoch),
            "train_loss": avg_loss,
            "metrics": ev.metrics,
            "train_seconds": train_secs,
            "is_best": is_best,
        })

        # ---- Save incremental log (so progress survives crashes) ----
        log_path = os.path.join(out_dir, "train_log.json")
        with open(log_path, "w") as f:
            json.dump(epoch_metrics, f, indent=2)

        # ---- Save per-step losses (compact binary, fast to write) ----
        np.save(os.path.join(out_dir, "step_losses.npy"), np.array(all_step_losses, dtype=np.float32))

        # ---- Incremental loss curve (update every epoch) ----
        plot_loss_curve(all_step_losses, epoch_boundaries,
                        os.path.join(out_dir, "loss_curve.png"), total_epochs)

    # ---- Final summary ----
    wall_total = time.time() - wall0
    print(f"\n[done] best abs_rel={best_metric:.6f} @ epoch {best_epoch}")
    print(f"[done] total wall time: {wall_total/60:.1f} min ({wall_total/3600:.2f} hr)")
    print(f"[done] checkpoints in {out_dir}/")

    # ---- Final plots ----
    plot_loss_curve(all_step_losses, epoch_boundaries,
                    os.path.join(out_dir, "loss_curve.png"), total_epochs)
    plot_metrics_table(epoch_metrics, os.path.join(out_dir, "metrics_table.png"))

    print(f"[plot] loss_curve.png saved")
    print(f"[plot] metrics_table.png saved")


if __name__ == "__main__":
    main()

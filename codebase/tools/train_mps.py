#!/usr/bin/env python3
"""
2 full epochs on MPS with evaluation after each epoch.
Plots a per-step loss curve to runs/train_mps_test/loss_curve.png.

Usage:
    python tools/train_mps.py
"""
import os, sys, time

import yaml
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.build import build_dataloaders
from src.models.build import build_model
from src.train.eval_one_epoch import eval_one_epoch


def build_loss(cfg):
    from src.losses.si_loss import build
    return build(cfg)


def evaluate(model, val_loader, device, loss_fn, label):
    ev = eval_one_epoch(
        model=model, val_loader=val_loader, device=device,
        loss_fn=loss_fn, amp=False,
    )
    print(f"\n{'='*55}")
    print(f"  Eval @ {label}")
    print(f"{'='*55}")
    for k, v in sorted(ev.metrics.items()):
        print(f"  {k:12s} = {v:.6f}")
    print(f"  ({ev.num_samples} samples, {ev.seconds:.1f}s)")
    return ev.metrics


def train_epoch(model, train_loader, optimizer, device, loss_fn,
                epoch_num, num_epochs, max_steps=None, grad_clip=1.0):
    """Train for one full epoch (or max_steps if set). Returns list of per-step losses."""
    model.train()
    step_losses = []
    t0 = time.time()

    for step, batch in enumerate(train_loader):
        if max_steps is not None and step >= max_steps:
            break

        rgb   = batch["rgb"].to(device, non_blocking=True)
        gt    = batch["depth"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(rgb)
        loss = loss_fn(pred, gt, valid)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], grad_clip
            )
        optimizer.step()

        lv = loss.detach().item()
        step_losses.append(lv)

        if step % 100 == 0:
            elapsed = time.time() - t0
            steps_done = step + 1
            s_per_step = elapsed / steps_done
            total_steps = max_steps if max_steps else "?"
            eta = s_per_step * (max_steps - steps_done) if max_steps else 0
            print(f"  [epoch {epoch_num}/{num_epochs}] step {step}/{total_steps}  "
                  f"loss={lv:.5f}  {s_per_step:.2f}s/step  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    elapsed = time.time() - t0
    avg = sum(step_losses) / len(step_losses) if step_losses else 0
    print(f"  epoch {epoch_num} done: {elapsed:.1f}s ({elapsed/60:.1f}m), "
          f"avg_loss={avg:.5f}, {len(step_losses)} steps")
    return step_losses


def main():
    # ---- config ----
    cfg_path = os.path.join(ROOT, "configs", "model_base.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # ---- device ----
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    torch.manual_seed(42)
    print(f"[device] {device}")

    # ---- model ----
    print("[build] loading model...")
    t0 = time.time()
    model = build_model(cfg).to(device)
    for p in model.backbone.parameters():
        p.requires_grad = False
    print(f"[build] model ready ({time.time()-t0:.1f}s)")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[params] total={total/1e6:.2f}M  trainable={trainable/1e6:.2f}M")

    # ---- data ----
    print("[build] loading data...")
    train_loader, val_loader = build_dataloaders(cfg)

    try:
        steps_per_epoch = len(train_loader)
    except TypeError:
        steps_per_epoch = 3000
    print(f"[steps] steps_per_epoch={steps_per_epoch}")

    # ---- loss + optimizer ----
    loss_fn = build_loss(cfg)
    print(f"[loss] {type(loss_fn).__name__}")

    lr = float(cfg.get("train", {}).get("lr", 3e-3))
    wd = float(cfg.get("train", {}).get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=wd,
    )
    print(f"[optim] AdamW lr={lr} wd={wd}")

    num_epochs = 2
    out_dir = os.path.join(ROOT, "runs", "train_mps_test")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Eval @ init ----
    metrics_init = evaluate(model, val_loader, device, loss_fn, "init (random weights)")

    # ---- Training loop ----
    all_losses = []
    epoch_boundaries = []
    all_metrics = [("init", metrics_init)]

    for epoch in range(1, num_epochs + 1):
        print(f"\n--- Epoch {epoch}/{num_epochs} ({steps_per_epoch} steps) ---")
        losses = train_epoch(
            model, train_loader, optimizer, device, loss_fn,
            epoch_num=epoch, num_epochs=num_epochs, max_steps=steps_per_epoch,
        )
        all_losses.extend(losses)
        epoch_boundaries.append(len(all_losses))

        metrics = evaluate(model, val_loader, device, loss_fn, f"epoch {epoch}")
        all_metrics.append((f"epoch {epoch}", metrics))

    # ---- Summary table ----
    print(f"\n{'='*75}")
    print("  SUMMARY")
    print(f"{'='*75}")
    header_keys = ["abs_rel", "rmse", "delta1", "delta2", "delta3"]
    avail_keys = [k for k in header_keys if k in metrics_init]
    hdr = "  {:12s}" + "".join(f" {k:>10s}" for k in avail_keys)
    print(hdr.format(""))
    for label, m in all_metrics:
        vals = "".join(f" {m.get(k, float('nan')):10.6f}" for k in avail_keys)
        print(f"  {label:12s}{vals}")

    # ---- Loss curve ----
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(len(all_losses)), all_losses,
            linewidth=0.3, alpha=0.35, color="steelblue", label="per-step")

    # Smoothed (moving average)
    window = max(1, min(100, len(all_losses) // 10))
    smoothed = []
    for i in range(len(all_losses)):
        lo = max(0, i - window // 2)
        hi = min(len(all_losses), i + window // 2 + 1)
        smoothed.append(sum(all_losses[lo:hi]) / (hi - lo))
    ax.plot(range(len(smoothed)), smoothed,
            linewidth=2, color="crimson", label=f"smoothed (w={window})")

    # Epoch boundaries
    for i, b in enumerate(epoch_boundaries[:-1]):
        ax.axvline(x=b, color="gray", linestyle="--", linewidth=1,
                   label="epoch boundary" if i == 0 else None)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (SI log)")
    ax.set_title(f"MPS Smoke Test — 2 full epochs on NYU Depth v2 ({device})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(out_dir, "loss_curve.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[plot] saved to {plot_path}")
    print("[done]")


if __name__ == "__main__":
    main()
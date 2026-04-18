#!/usr/bin/env python3
"""
Evaluate a trained depth model checkpoint on the NYU Depth v2 validation set.

Produces:
  - Metric summary table (printed + saved as metrics.json)
  - Qualitative visualizations (RGB | GT | Pred | Error)
  - Per-image metric distribution histograms
  - Optional: per-image CSV for further analysis

Usage:
    # Evaluate best checkpoint:
    python scripts/evaluate.py --checkpoint runs/full_25ep/best.pt

    # Evaluate with qualitative outputs:
    python scripts/evaluate.py --checkpoint runs/full_25ep/best.pt --qualitative --num-vis 24

    # Evaluate last checkpoint, save to specific dir:
    python scripts/evaluate.py --checkpoint runs/full_25ep/last.pt --out runs/full_25ep/eval_last
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import yaml
import torch
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.data.build import build_dataloaders
from src.models.build import build_model
from src.eval.eval import evaluate_model, save_qualitative
from src.eval.metrics import evaluate_batch


def plot_metric_histograms(per_image_metrics, out_dir):
    """Plot histograms of per-image metrics."""
    if not per_image_metrics:
        return

    keys = ["abs_rel", "rmse", "delta1"]
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4))

    for ax, key in zip(axes, keys):
        vals = [m[key] for m in per_image_metrics if key in m]
        if not vals:
            continue
        ax.hist(vals, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(np.mean(vals), color="crimson", linewidth=2, linestyle="--",
                   label=f"mean={np.mean(vals):.4f}")
        ax.axvline(np.median(vals), color="orange", linewidth=2, linestyle="--",
                   label=f"median={np.median(vals):.4f}")
        ax.set_xlabel(key)
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of {key}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "metric_histograms.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[eval] saved metric histograms to {path}")


def plot_metrics_comparison(metrics, paper_targets, out_dir):
    """Bar chart comparing model metrics vs paper targets."""
    keys = ["abs_rel", "rmse", "delta1", "delta2", "delta3"]
    avail = [k for k in keys if k in metrics and k in paper_targets]
    if not avail:
        return

    x = np.arange(len(avail))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ours = [metrics[k] for k in avail]
    paper = [paper_targets[k] for k in avail]

    bars1 = ax.bar(x - width / 2, ours, width, label="Ours", color="steelblue")
    bars2 = ax.bar(x + width / 2, paper, width, label="CLIP2Depth (paper)", color="coral")

    ax.set_xticks(x)
    ax.set_xticklabels(avail)
    ax.set_ylabel("Value")
    ax.set_title("Model vs Paper (CLIP2Depth)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

    path = os.path.join(out_dir, "comparison_vs_paper.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[eval] saved comparison chart to {path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate depth model checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--config", default=None, help="Config YAML (default: from checkpoint or model_base.yaml)")
    parser.add_argument("--out", default=None, help="Output directory (default: <checkpoint_dir>/eval/)")
    parser.add_argument("--qualitative", action="store_true", help="Save qualitative visualizations")
    parser.add_argument("--num-vis", type=int, default=16, help="Number of qualitative samples")
    parser.add_argument("--per-image", action="store_true", help="Collect per-image metrics")
    parser.add_argument("--csv", action="store_true", help="Save per-image metrics as CSV")
    args = parser.parse_args()

    # ---- Config ----
    if args.config:
        cfg_path = args.config if os.path.isabs(args.config) else os.path.join(ROOT, args.config)
    else:
        # Try loading config from checkpoint, fall back to model_base.yaml
        ckpt_peek = torch.load(args.checkpoint, map_location="cpu")
        if "cfg" in ckpt_peek:
            cfg = ckpt_peek["cfg"]
            print("[eval] using config from checkpoint")
        else:
            cfg_path = os.path.join(ROOT, "configs", "model_base.yaml")
            print(f"[eval] using config from {cfg_path}")
        if "cfg" not in ckpt_peek:
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = ckpt_peek["cfg"]
        del ckpt_peek

    if args.config:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

    # ---- Output dir ----
    if args.out:
        out_dir = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    else:
        ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
        ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        out_dir = os.path.join(ckpt_dir, f"eval_{ckpt_name}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[eval] output dir: {out_dir}")

    # ---- Device ----
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[eval] device: {device}")

    # ---- Build model ----
    print("[eval] building model...")
    model = build_model(cfg).to(device)
    for p in model.backbone.parameters():
        p.requires_grad = False

    # ---- Load weights ----
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    epoch_info = ckpt.get("epoch", "?")
    best_info = ckpt.get("best_metric", "?")
    print(f"[eval] loaded checkpoint: {args.checkpoint}")
    print(f"[eval]   epoch={epoch_info}, best_metric={best_info}")

    # ---- Build val loader ----
    print("[eval] building data loader...")
    _, val_loader = build_dataloaders(cfg)

    # ---- Build loss ----
    loss_fn = None
    try:
        from src.losses.si_loss import build
        loss_fn = build(cfg)
    except Exception:
        pass

    # ---- Run evaluation ----
    collect_per_image = args.per_image or args.qualitative or args.csv
    print("[eval] running evaluation...")
    result = evaluate_model(
        model=model,
        val_loader=val_loader,
        device=device,
        loss_fn=loss_fn,
        collect_per_image=collect_per_image,
    )

    # ---- Print results ----
    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*60}")
    for k, v in sorted(result.metrics.items()):
        print(f"  {k:12s} = {v:.6f}")
    print(f"  {'samples':12s} = {result.num_samples}")
    print(f"  {'time':12s} = {result.seconds:.1f}s")
    print(f"{'='*60}")

    # ---- Save metrics.json ----
    metrics_out = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "epoch": epoch_info,
        "metrics": result.metrics,
        "num_samples": result.num_samples,
        "eval_seconds": result.seconds,
        "device": str(device),
    }
    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"[eval] saved metrics to {metrics_path}")

    # ---- Per-image histograms ----
    if collect_per_image and result.per_image_metrics:
        plot_metric_histograms(result.per_image_metrics, out_dir)

        if args.csv:
            csv_path = os.path.join(out_dir, "per_image_metrics.csv")
            keys = list(result.per_image_metrics[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["index"] + keys)
                writer.writeheader()
                for i, m in enumerate(result.per_image_metrics):
                    writer.writerow({"index": i, **m})
            print(f"[eval] saved per-image CSV to {csv_path}")

    # ---- Qualitative visualizations ----
    if args.qualitative:
        print(f"[eval] generating {args.num_vis} qualitative samples...")
        vis_dir = os.path.join(out_dir, "visualizations")
        save_qualitative(
            model=model,
            val_loader=val_loader,
            device=device,
            out_dir=vis_dir,
            num_samples=args.num_vis,
        )

    print(f"\n[eval] done! Results in {out_dir}/")


if __name__ == "__main__":
    main()
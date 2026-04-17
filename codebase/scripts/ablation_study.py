#!/usr/bin/env python3
"""
Ablation study: 6 experiments + baseline comparison.
Each modifies one variable from the v3_base config.
Runs sequentially, 10 epochs each. Can be paused/resumed.

Usage:
    caffeinate -s python scripts/ablation_search.py
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.path.join(ROOT, "..", ".venv", "bin", "python3.11")
TRAIN_SCRIPT = os.path.join(ROOT, "scripts", "train.py")

# Load base config
with open(os.path.join(ROOT, "configs", "model_base.yaml")) as f:
    BASE_CFG = yaml.safe_load(f)

ABLATIONS = [
    {
        "name": "embedding_tokens_16",
        "desc": "Embedding tokens: 16 (vs baseline 64)",
        "changes": {"depth_embedding.num_tokens": 16},
    },
    {
        "name": "embedding_tokens_128",
        "desc": "Embedding tokens: 128 (vs baseline 64)",
        "changes": {"depth_embedding.num_tokens": 128},
    },
    {
        "name": "pooling_mean",
        "desc": "Pooling: mean (vs baseline eos)",
        "changes": {"depth_embedding.pooling": "mean"},
    },
    {
        "name": "random_fixed_q",
        "desc": "Random fixed q (embedding frozen, FiLM active)",
        "changes": {"depth_embedding.freeze_embedding": True},
    },
    {
        "name": "no_embedding_no_film",
        "desc": "No embedding/FiLM (decoder on raw features)",
        "changes": {"depth_embedding.use_embedding": False, "dense_predictor.use_film": False},
    },
    {
        "name": "single_layer_9",
        "desc": "Feature layers: [9] only (vs baseline [3,6,9])",
        "changes": {"model.layers": [9], "dense_predictor.layers": [9]},
    },
]

EPOCHS = 10
LR = 0.001
BATCH_SIZE = 16
NUM_WORKERS = 4
SHUFFLE = 1500


def apply_changes(cfg, changes):
    """Apply dot-separated key changes to a nested config dict."""
    cfg = copy.deepcopy(cfg)
    for key, value in changes.items():
        parts = key.split(".")
        d = cfg
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return cfg


def run_ablation(abl):
    name = abl["name"]
    out_dir = os.path.join(ROOT, "runs", "ablations", name)
    cfg_path = os.path.join(out_dir, "config.yaml")
    log_path = os.path.join(out_dir, "train_log.json")
    os.makedirs(out_dir, exist_ok=True)

    # Check if already completed
    if os.path.isfile(log_path):
        with open(log_path) as f:
            data = json.load(f)
        completed = sum(1 for r in data if r["epoch"] not in ("init",))
        if completed >= EPOCHS:
            print(f"  [skip] {name} already has {completed} epochs — skipping")
            return _read_result(name, out_dir, log_path)
        elif completed > 0:
            print(f"  [resume] {name} has {completed}/{EPOCHS} epochs — resuming")

    # Build config and save to both locations
    cfg = apply_changes(BASE_CFG, abl["changes"])
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    # Also save to configs/ablations/ for discoverability
    ablation_cfg_dir = os.path.join(ROOT, "configs", "ablations")
    os.makedirs(ablation_cfg_dir, exist_ok=True)
    with open(os.path.join(ablation_cfg_dir, f"{name}.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # Check for existing checkpoint to resume
    last_pt = os.path.join(out_dir, "last.pt")
    cmd = [PYTHON, "-u", TRAIN_SCRIPT, "--config", cfg_path, "--out", out_dir]
    if os.path.isfile(last_pt):
        cmd += ["--resume", last_pt]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["EPOCHS"] = str(EPOCHS)
    env["LR"] = str(LR)
    env["BATCH_SIZE"] = str(BATCH_SIZE)
    env["NUM_WORKERS"] = str(NUM_WORKERS)
    env["WDS_SAMPLE_SHUFFLE"] = str(SHUFFLE)
    env["LOG_EVERY"] = "500"

    print(f"\n{'#'*60}")
    print(f"  ABLATION: {name}")
    print(f"  {abl['desc']}")
    print(f"  Output: {out_dir}")
    print(f"{'#'*60}\n")

    t0 = time.time()
    proc = subprocess.Popen(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)
    proc.wait()
    elapsed = time.time() - t0

    print(f"\n  [{name}] finished in {elapsed/60:.1f} min (exit code {proc.returncode})")

    return _read_result(name, out_dir, log_path)


def _read_result(name, out_dir, log_path):
    result = {"name": name}
    if os.path.isfile(log_path):
        with open(log_path) as f:
            data = json.load(f)
        trained = [r for r in data if r["epoch"] not in ("init",)]
        if trained:
            best = min(trained, key=lambda r: r["metrics"]["abs_rel"])
            result["best_epoch"] = best["epoch"]
            result["best_abs_rel"] = best["metrics"]["abs_rel"]
            result["best_rmse"] = best["metrics"]["rmse"]
            result["best_delta1"] = best["metrics"]["delta1"]
            result["last_abs_rel"] = trained[-1]["metrics"]["abs_rel"]
            result["epochs"] = len(trained)
    return result


def main():
    print(f"{'='*60}")
    print(f"  ABLATION STUDY — {len(ABLATIONS)} experiments, {EPOCHS} epochs each")
    print(f"  lr={LR}, bs={BATCH_SIZE}, cosine T_max={EPOCHS}")
    print(f"{'='*60}")

    wall0 = time.time()
    results = []

    for i, abl in enumerate(ABLATIONS):
        print(f"\n>>> Ablation {i+1}/{len(ABLATIONS)}: {abl['name']}")
        result = run_ablation(abl)
        results.append(result)

    # Final summary
    wall_total = time.time() - wall0
    print(f"\n{'='*60}")
    print(f"  ABLATION RESULTS (sorted by abs_rel)")
    print(f"  Total time: {wall_total/60:.1f} min ({wall_total/3600:.1f} hr)")
    print(f"{'='*60}")

    results_sorted = sorted(results, key=lambda r: r.get("best_abs_rel", float("inf")))

    hdr = "{:>22s}  {:>10s}  {:>10s}  {:>10s}  {:>8s}".format(
        "Name", "abs_rel", "rmse", "delta1", "BestEp")
    print(hdr)
    print("-" * 65)
    for r in results_sorted:
        print("{:>22s}  {:10.6f}  {:10.6f}  {:10.6f}  {:>8s}".format(
            r["name"],
            r.get("best_abs_rel", float("nan")),
            r.get("best_rmse", float("nan")),
            r.get("best_delta1", float("nan")),
            str(r.get("best_epoch", "?"))))

    # Save
    out_path = os.path.join(ROOT, "runs", "ablations", "summary.json")
    with open(out_path, "w") as f:
        json.dump(results_sorted, f, indent=2)
    print(f"\n[saved] {out_path}")
    print("[done]")


if __name__ == "__main__":
    main()

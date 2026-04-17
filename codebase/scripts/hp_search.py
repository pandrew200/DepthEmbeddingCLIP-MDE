#!/usr/bin/env python3
"""
Hyperparameter search: sequential runs with different lr/bs combos.
Each run uses a 10-epoch cosine scheduler but is killed after 5 epochs.
Results are compared in a summary table at the end.

Usage:
    caffeinate -s python scripts/hp_search.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIGS = [
    {"name": "lr3e-3_bs8",  "lr": 3e-3, "bs": 8},
    {"name": "lr2e-3_bs8",  "lr": 2e-3, "bs": 8},
    {"name": "lr1e-3_bs8",  "lr": 1e-3, "bs": 8},
    {"name": "lr3e-3_bs16", "lr": 3e-3, "bs": 16},
    {"name": "lr2e-3_bs16", "lr": 2e-3, "bs": 16},
    {"name": "lr1e-3_bs16", "lr": 1e-3, "bs": 16},
]

EPOCHS_SCHEDULER = 10   # T_max for cosine schedule
EPOCHS_RUN = 5          # actually train this many
SHUFFLE = 3000
NUM_WORKERS = 2
PYTHON = os.path.join(ROOT, "..", ".venv", "bin", "python3.11")
TRAIN_SCRIPT = os.path.join(ROOT, "scripts", "train.py")


def run_one(cfg: dict) -> dict:
    name = cfg["name"]
    out_dir = os.path.join(ROOT, "runs", "hp_search", name)
    os.makedirs(out_dir, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["EPOCHS"] = str(EPOCHS_SCHEDULER)
    env["LR"] = str(cfg["lr"])
    env["BATCH_SIZE"] = str(cfg["bs"])
    env["WDS_SAMPLE_SHUFFLE"] = str(SHUFFLE)
    env["NUM_WORKERS"] = str(NUM_WORKERS)
    env["LOG_EVERY"] = "200"

    cmd = [PYTHON, "-u", TRAIN_SCRIPT, "--out", out_dir]

    print(f"\n{'#'*65}")
    print(f"  HP SEARCH: {name}  (lr={cfg['lr']}, bs={cfg['bs']})")
    print(f"  Output: {out_dir}")
    print(f"  Scheduler: cosine T_max={EPOCHS_SCHEDULER}, running {EPOCHS_RUN} epochs")
    print(f"{'#'*65}\n")

    t0 = time.time()
    proc = subprocess.Popen(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)

    # Monitor train_log.json and kill after EPOCHS_RUN epochs
    log_path = os.path.join(out_dir, "train_log.json")
    while proc.poll() is None:
        time.sleep(30)
        if os.path.isfile(log_path):
            try:
                with open(log_path) as f:
                    data = json.load(f)
                completed = sum(1 for r in data if r["epoch"] not in ("init",))
                if completed >= EPOCHS_RUN:
                    print(f"\n  [{name}] {EPOCHS_RUN} epochs complete — stopping.")
                    proc.terminate()
                    proc.wait(timeout=30)
                    break
            except (json.JSONDecodeError, KeyError):
                pass

    elapsed = time.time() - t0

    # Read final metrics
    result = {"name": name, "lr": cfg["lr"], "bs": cfg["bs"], "elapsed_min": elapsed / 60}
    if os.path.isfile(log_path):
        try:
            with open(log_path) as f:
                data = json.load(f)
            # Find best epoch by abs_rel
            best = None
            for row in data:
                if row["epoch"] in ("init",):
                    continue
                ar = row.get("metrics", {}).get("abs_rel", float("inf"))
                if best is None or ar < best.get("metrics", {}).get("abs_rel", float("inf")):
                    best = row
            if best:
                result["best_epoch"] = best["epoch"]
                result["best_abs_rel"] = best["metrics"]["abs_rel"]
                result["best_rmse"] = best["metrics"]["rmse"]
                result["best_delta1"] = best["metrics"]["delta1"]
            # Last epoch metrics
            last = [r for r in data if r["epoch"] not in ("init",)][-1]
            result["last_epoch"] = last["epoch"]
            result["last_abs_rel"] = last["metrics"]["abs_rel"]
        except Exception:
            pass

    return result


def main():
    print(f"{'='*65}")
    print(f"  HYPERPARAMETER SEARCH — {len(CONFIGS)} configurations")
    print(f"  Schedule: cosine T_max={EPOCHS_SCHEDULER}, run {EPOCHS_RUN} epochs each")
    print(f"  Shuffle: {SHUFFLE}, Workers: {NUM_WORKERS}")
    print(f"  Estimated total time: ~{len(CONFIGS) * EPOCHS_RUN * 36 / 60:.0f} hours")
    print(f"{'='*65}")

    wall0 = time.time()
    results = []
    for i, cfg in enumerate(CONFIGS):
        print(f"\n>>> Run {i+1}/{len(CONFIGS)}")
        result = run_one(cfg)
        results.append(result)

        # Print running summary after each run
        print(f"\n{'─'*65}")
        print(f"  RESULTS SO FAR ({i+1}/{len(CONFIGS)} complete)")
        print(f"{'─'*65}")
        hdr = "{:>18s}  {:>6s}  {:>4s}  {:>10s}  {:>10s}  {:>10s}  {:>6s}".format(
            "Name", "LR", "BS", "abs_rel", "rmse", "delta1", "Time")
        print(hdr)
        for r in results:
            print("{:>18s}  {:>6s}  {:>4d}  {:10.6f}  {:10.6f}  {:10.6f}  {:5.1f}m".format(
                r["name"], f"{r['lr']:.0e}", r["bs"],
                r.get("best_abs_rel", float("nan")),
                r.get("best_rmse", float("nan")),
                r.get("best_delta1", float("nan")),
                r.get("elapsed_min", 0)))
        print(f"{'─'*65}")

    # Final summary
    wall_total = time.time() - wall0
    print(f"\n{'='*65}")
    print(f"  FINAL HYPERPARAMETER SEARCH RESULTS")
    print(f"  Total time: {wall_total/60:.1f} min ({wall_total/3600:.1f} hr)")
    print(f"{'='*65}")
    hdr = "{:>18s}  {:>6s}  {:>4s}  {:>10s}  {:>10s}  {:>10s}  {:>8s}  {:>6s}".format(
        "Name", "LR", "BS", "abs_rel", "rmse", "delta1", "BestEp", "Time")
    print(hdr)
    print("-" * 80)

    # Sort by best abs_rel
    results_sorted = sorted(results, key=lambda r: r.get("best_abs_rel", float("inf")))
    for i, r in enumerate(results_sorted):
        marker = " <<<" if i == 0 else ""
        print("{:>18s}  {:>6s}  {:>4d}  {:10.6f}  {:10.6f}  {:10.6f}  {:>8s}  {:5.1f}m{}".format(
            r["name"], f"{r['lr']:.0e}", r["bs"],
            r.get("best_abs_rel", float("nan")),
            r.get("best_rmse", float("nan")),
            r.get("best_delta1", float("nan")),
            str(r.get("best_epoch", "?")),
            r.get("elapsed_min", 0),
            marker))

    # Save results
    out_path = os.path.join(ROOT, "runs", "hp_search", "summary.json")
    with open(out_path, "w") as f:
        json.dump(results_sorted, f, indent=2)
    print(f"\n[saved] {out_path}")
    print("[done]")


if __name__ == "__main__":
    main()

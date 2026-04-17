# Learning a Continuous Depth Embedding from CLIP for Monocular Depth Estimation

Yale CPSC 4900 Senior Thesis — Andrew Pan, advised by Alex Wong

## Overview

This project investigates whether CLIP's frozen vision-language representations can support dense monocular depth estimation. We learn a **depth embedding**—a set of continuous learnable tokens processed by the frozen CLIP text encoder—that conditions multi-layer visual features through FiLM modulation and a lightweight transformer-based decoder.

Key features:
- **Frozen CLIP backbone** (ViT-B/16) — no fine-tuning of the pretrained model
- **Only 1.16M trainable parameters** — depth embedding tokens, FiLM layers, and decoder
- **Continuous depth regression** — unlike prior CLIP-based methods that discretize depth into bins
- **Scale-invariant log loss** (AdaBins-style) for training

## Results

Trained and evaluated on NYU Depth v2:

| Metric | Value |
|--------|-------|
| Abs Rel | 0.249 |
| RMSE | 0.784 |
| δ₁ (< 1.25) | 0.547 |
| δ₂ (< 1.25²) | 0.844 |
| δ₃ (< 1.25³) | 0.952 |

## Architecture

```
Input Image (352x352)
    │
    ▼
┌─────────────────────┐     ┌───────────────────────┐
│ Frozen CLIP Image   │     │ Depth Embedding       │
│ Encoder (ViT-B/16)  │     │ N learnable tokens    │
│                     │     │ + BOS/EOS framing     │
│ Extract layers      │     │         │             │
│ 3, 6, 9             │     │         ▼             │
└────────┬────────────┘     │ Frozen CLIP Text      │
         │                  │ Encoder (causal mask) │
         │                  │         │             │
         │                  │ EOS-pooling           │
         │                  │ + text projection     │
         │                  └─────────┬─────────────┘
         │                            │
         │              q ∈ R^512     │
         │                            │
         ▼                            ▼
┌─────────────────────────────────────────────┐
│ Dense Predictor                             │
│  • Projection: 768 → 64 (per layer)         │
│  • FiLM: q → γ, β to modulate features      │
│  • 3 Transformer blocks (w=64, h=4)         │
│  • Deconv: 22×22 → 88×88 → 352×352          │
│  • Softplus output                          │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
          Depth Map (352x352)
```

## Setup

```bash
# Create environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch torchvision open_clip_torch webdataset h5py matplotlib pyyaml numpy
pip install transformers  # only needed for pretrained decoder experiments
```

## Training

```bash
# Train the model (25 epochs on MPS/CUDA/CPU)
EPOCHS=25 LR=0.001 BATCH_SIZE=16 NUM_WORKERS=4 WDS_SAMPLE_SHUFFLE=1500 \
  python scripts/train.py --out runs/my_run

# Resume from checkpoint
EPOCHS=25 LR=0.001 BATCH_SIZE=16 NUM_WORKERS=4 WDS_SAMPLE_SHUFFLE=1500 \
  python scripts/train.py --resume runs/my_run/last.pt --out runs/my_run
```

## Evaluation

```bash
# Evaluate best checkpoint
python scripts/evaluate.py --checkpoint runs/my_run/best.pt

# Full evaluation with visualizations
python scripts/evaluate.py --checkpoint runs/my_run/best.pt \
  --qualitative --num-vis 24 --per-image --csv
```

## Ablation Study

```bash
# Run all 6 ablations (10 epochs each)
python scripts/ablation_search.py
```

## Project Structure

```
src/
  models/
    model.py              # End-to-end model
    depth_embedding.py    # Learnable depth embedding
    decoder_base.py       # Dense predictor (random init)
    decoder_pretrained.py # CLIPSeg pretrained decoder
    clip_backbone.py      # Frozen CLIP feature extractor
    build.py              # Model factory
  train/
    train_one_epoch.py    # Training step
    eval_one_epoch.py     # Validation step
  eval/
    eval.py               # Evaluation utilities
    metrics.py            # Depth metrics
  losses/
    si_loss.py            # Scale-invariant log loss
  data/
    build.py              # Data loader factory
    nyu_wds.py            # NYU Depth v2 WebDataset
scripts/
  train.py                # Main training script
  evaluate.py             # Evaluation script
  hp_search.py            # Hyperparameter search
  ablation_search.py      # Ablation study
configs/
  model_base.yaml         # Base configuration
  model_pretrained.yaml   # Pretrained decoder config
  ablations/              # Ablation configs
```

## Dataset

We use the [NYU Depth v2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) dataset ([`sayakpaul/nyu_depth_v2`](https://huggingface.co/datasets/sayakpaul/nyu_depth_v2) on HuggingFace), stored as WebDataset tar shards with H5-packed samples.

Download the dataset:
```bash
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='sayakpaul/nyu_depth_v2',
    repo_type='dataset',
    local_dir='nyu_depth_v2',
    local_dir_use_symlinks=False,
)
"
```

Then set the data path:
```bash
export NYU_WDS_ROOT=/path/to/nyu_depth_v2/data
```

<!-- ## Acknowledgements

This work builds on ideas from [Kim et al. (2024)](https://arxiv.org/abs/2402.03251), [CLIPSeg (Luddecke & Ecker, 2022)](https://arxiv.org/abs/2112.10003), and [CLIP (Radford et al., 2021)](https://arxiv.org/abs/2103.00020). -->

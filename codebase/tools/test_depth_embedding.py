"""
Quick test to verify DepthEmbedding works.

Run from repo root:
python -m tools.test_depth_embedding
"""

import torch
from src.models.clip_backbone import CLIPBackbone
from src.models.depth_embedding import DepthEmbedding
import yaml


def load_cfg():
    with open("configs/model_base.yaml") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_cfg()
    device = cfg["clip"]["device"]

    # load CLIP backbone
    backbone = CLIPBackbone(cfg["clip"])

    # create depth embedding tokens
    depth_emb = DepthEmbedding(
        clip_model=backbone.model,
        num_tokens=64,
        pooling="mean",
    ).to(device)

    # forward pass
    depth_query = depth_emb()

    print("\nDepth query vector:")
    print(depth_query.shape)
    print(depth_query[:10])  # show first 10 values


    # EXTRA
    loss = depth_query.sum()
    loss.backward()
    print(depth_emb.embedding_tokens.grad is not None)

    print(sum(p.requires_grad for p in backbone.model.parameters()))


if __name__ == "__main__":
    main()
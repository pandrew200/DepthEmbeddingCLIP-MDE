# src/models/build.py
from __future__ import annotations

from typing import Any, Dict

from src.models.model import DepthModel


def _cfg_get(cfg: Any, key: str, default=None):
    # cfg is currently a dict loaded by yaml.safe_load
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def build_model(cfg: Dict[str, Any]) -> DepthModel:
    """
    Build the end-to-end model from configs/model_base.yaml (dict).

    Expects top-level keys:
      - model
      - clip
      - depth_embedding
      - dense_predictor
    """
    model_cfg = _cfg_get(cfg, "model", {})
    clip_cfg = _cfg_get(cfg, "clip", {})
    depth_embedding_cfg = _cfg_get(cfg, "depth_embedding", {})
    dense_cfg = _cfg_get(cfg, "dense_predictor", {})
    decoder_cfg = _cfg_get(cfg, "decoder", None)

    return DepthModel(
        clip_cfg=clip_cfg,
        depth_embedding_cfg=depth_embedding_cfg,
        dense_predictor_cfg=dense_cfg,
        model_cfg=model_cfg,
        decoder_cfg=decoder_cfg,
    )
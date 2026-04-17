# src/models/model.py
"""
End-to-end depth model wiring module.

This module wires together:
  - CLIPBackbone: frozen CLIP image encoder + intermediate patch-grid features
  - DepthEmbedding: learnable tokens through frozen CLIP text encoder -> depth query q
  - DensePredictor: dense predictor (proj 768->64, FiLM@64,
    3 transformer blocks width 64 head 4, deconvs to 352)

Key outputs:
  - depth map: [B, 1, 352, 352] (or [B, 1, H, W] if out_hw override is used)

Important assumptions:
  - You are using a ViT-based CLIP (ViT-B/16)
  - You feed 352x352 inputs to the vision encoder OR you explicitly interpolate output size.
  - Your CLIP feature extractor correctly handles positional embedding resizing for 352x352.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.clip_backbone import CLIPBackbone
from src.models.depth_embedding import DepthEmbedding
from src.models.decoder_base import DensePredictor
from src.models.decoder_pretrained import CLIPSegDecoderForDepth


@dataclass
class DepthModelConfig:
    """
    Top-level wiring config.

    image_size:
        Uses 352x352 inputs. We default to 352.

    layers:
        Which CLIP intermediate layers to extract (must match dense predictor config).
    """
    image_size: int = 352
    layers: List[int] = None  # default [3,6,9]

    def __post_init__(self):
        if self.layers is None:
            self.layers = [3, 6, 9]


class DepthModel(nn.Module):
    """
    End-to-end depth model (frozen CLIP encoders + learnable depth embedding + dense predictor).
    """

    def __init__(
        self,
        clip_cfg: dict,
        depth_embedding_cfg: dict,
        dense_predictor_cfg: dict,
        model_cfg: Optional[Union[DepthModelConfig, dict]] = None,
        decoder_cfg: Optional[dict] = None,
    ):
        super().__init__()

        # Allow passing dict for model_cfg
        if model_cfg is None:
            model_cfg = DepthModelConfig()
        elif isinstance(model_cfg, dict):
            model_cfg = DepthModelConfig(**model_cfg)
        self.model_cfg = model_cfg

        # -------------------------
        # Backbone: frozen CLIP
        # -------------------------
        self.backbone = CLIPBackbone(clip_cfg)

        # -------------------------
        # Depth embedding: learnable tokens through frozen CLIP text encoder
        # NOTE: we pass the same open_clip model object used by backbone.
        # -------------------------
        self.use_embedding = depth_embedding_cfg.pop("use_embedding", True)
        freeze_embedding = depth_embedding_cfg.pop("freeze_embedding", False)

        self.depth_embedding = DepthEmbedding(
            clip_model=self.backbone.model,
            **depth_embedding_cfg,
        )

        if freeze_embedding:
            for p in self.depth_embedding.parameters():
                p.requires_grad_(False)
            print("[model] depth embedding tokens frozen (random fixed q)")

        # -------------------------
        # Decoder: either pretrained CLIPSeg or custom dense predictor
        # -------------------------
        decoder_type = (decoder_cfg or {}).get("type", "dense_predictor")
        self.decoder_type = decoder_type

        if decoder_type == "clipseg":
            self.dense_predictor = CLIPSegDecoderForDepth(decoder_cfg or {})
        else:
            self.dense_predictor = DensePredictor(dense_predictor_cfg)

        # Convenience: store device string if present
        self.device_str = clip_cfg.get("device", "cpu")

        self.clip_normalize = clip_cfg.get("normalize", False)

        # OpenAI CLIP normalization
        self.register_buffer(
            "clip_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "clip_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
        )

    def _ensure_batch_q(self, q: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        DepthEmbedding.forward() returns [D] (no batch dim).
        The dense predictor expects q: [B, D].

        This function converts:
          - [D] -> [B, D] by expanding
          - [1, D] -> [B, D] by expanding
          - [B, D] stays as-is
        """
        if q.dim() == 1:
            q = q.unsqueeze(0)  # [1, D]
        if q.shape[0] == 1 and batch_size > 1:
            q = q.expand(batch_size, -1).contiguous()
        return q

    def forward(
        self,
        image: torch.Tensor,
        out_hw: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            image:
                [B,3,H,W] float tensor already normalized for CLIP
                and ideally resized to 352x352.

            out_hw:
                Optionally force output to a specific spatial size [H, W].

        Returns:
            depth:
                [B,1,H,W] (typically [B,1,352,352])
        """
        B = image.shape[0]
        raw_image = image  # preserve [0,1] range for CLIPSeg path

        if self.clip_normalize:
            image = (image - self.clip_mean) / self.clip_std

        # 1) Extract multi-layer features from frozen CLIP image encoder.
        #    For the base decoder: use open_clip backbone features (CLIP-normalized).
        #    For CLIPSeg decoder: backbone is skipped — CLIPSeg uses its own
        #    frozen HF vision encoder with raw [0,1] input for feature compatibility.
        if self.decoder_type != "clipseg":
            feats: Dict[int, torch.Tensor] = self.backbone.extract_vit_intermediates(
                image,
                layers=self.model_cfg.layers,
                return_patch_tokens=True,
            )
        else:
            feats = {}  # CLIPSeg decoder extracts its own features

        # 2) Compute depth query embedding q from depth embedding tokens via frozen CLIP text encoder.
        #    If use_embedding is False, q is a zero vector (no conditioning).
        if self.use_embedding:
            q = self.depth_embedding()        # [D]
            q = self._ensure_batch_q(q, B)    # [B, D]
        else:
            q = torch.zeros(B, 512, device=image.device)

        # 3) Decoder: base decoder uses backbone feats; CLIPSeg uses its own vision encoder
        #    with raw [0,1] images (CLIPSeg expects unnormalized input).
        if self.decoder_type == "clipseg":
            depth = self.dense_predictor(feats, q, out_hw=out_hw, image=raw_image)
        else:
            depth = self.dense_predictor(feats, q, out_hw=out_hw)

        return depth

    @torch.no_grad()
    def forward_from_pil(
        self,
        pil_image,
        preprocess_fn,
        out_hw: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Convenience helper:
          - preprocess PIL -> tensor [1,3,H,W]
          - forward pass
        """
        x = preprocess_fn(pil_image).unsqueeze(0).to(self.device_str)
        return self.forward(x, out_hw=out_hw)
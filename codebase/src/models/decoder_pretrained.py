"""
CLIPSeg pretrained decoder adapted for depth estimation.

Uses HuggingFace's full CLIPSeg model (vision encoder + decoder) to ensure
feature-level compatibility. The open_clip backbone produces different
intermediate features due to implementation differences (attention, numerical
precision), so we use CLIPSeg's own frozen vision encoder for the decoder path.

The model still uses open_clip's CLIP for the depth embedding (text encoder path),
which is fine since the depth embedding produces a 512-dim embedding independent of
the vision encoder implementation.

Architecture (3 decoder blocks, processed in reverse order: layer 9 -> 6 -> 3):
  - Reduce: Linear(768 -> 64) per layer
  - Skip connections between blocks
  - FiLM conditioning at block 0 only (layer 9 features)
  - Post-norm transformer blocks (width=64, heads=4, MLP=2048, ReLU)
  - Deconv head: Conv2d + ConvTranspose2d x2 -> [B, 1, 352, 352]
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPSegDecoderForDepth(nn.Module):
    """
    Wraps the full HuggingFace CLIPSeg model (vision encoder + decoder),
    using its own vision encoder for feature extraction to ensure compatibility
    with the pretrained decoder weights.

    Config keys (from YAML):
        clipseg_model_id: str  (default "CIDAS/clipseg-rd64-refined")
        freeze_decoder: bool   (default False — fine-tune the decoder)
        freeze_vision: bool    (default True — keep vision encoder frozen)
        out_activation: str    (default "softplus" — ensures positive depth)
    """

    def __init__(self, cfg: dict):
        super().__init__()

        model_id = cfg.get("clipseg_model_id", "CIDAS/clipseg-rd64-refined")
        freeze_decoder = cfg.get("freeze_decoder", False)
        freeze_vision = cfg.get("freeze_vision", True)
        self.out_activation = cfg.get("out_activation", "softplus")

        # Load the full CLIPSeg model (vision encoder + decoder)
        from transformers import CLIPSegForImageSegmentation
        self.clipseg = CLIPSegForImageSegmentation.from_pretrained(model_id)

        # Store extract_layers config for reference
        self.extract_layers = self.clipseg.config.extract_layers  # [3, 6, 9]

        # Freeze vision encoder (always — it should match the pretrained decoder)
        if freeze_vision:
            for p in self.clipseg.clip.vision_model.parameters():
                p.requires_grad_(False)

        # Freeze text encoder (not used, but included in the model)
        for p in self.clipseg.clip.text_model.parameters():
            p.requires_grad_(False)

        # Optionally freeze the decoder too
        if freeze_decoder:
            for p in self.clipseg.decoder.parameters():
                p.requires_grad_(False)

        # Freeze FiLM layers (paper: "we also freeze the two feed-forward
        # networks within the FiLM blocks, which slightly improves the
        # convergence of the depth embedding")
        for name in ("film_mul", "film_add"):
            film = getattr(self.clipseg.decoder, name, None)
            if film is not None:
                for p in film.parameters():
                    p.requires_grad_(False)
                print(f"[CLIPSegDecoderForDepth] froze {name}")

        # Learnable depth shift and scale to bridge the gap between
        # CLIPSeg's segmentation logits (deeply negative) and depth values (positive).
        # Applied before softplus: depth = softplus(logits * scale + shift)
        self.depth_shift = nn.Parameter(torch.tensor(2.0))
        self.depth_scale = nn.Parameter(torch.tensor(1.0))

        # Count params
        dec_params = sum(p.numel() for p in self.clipseg.decoder.parameters())
        dec_trainable = sum(p.numel() for p in self.clipseg.decoder.parameters() if p.requires_grad)
        vis_params = sum(p.numel() for p in self.clipseg.clip.vision_model.parameters())
        print(f"[CLIPSegDecoderForDepth] loaded from {model_id}")
        print(f"[CLIPSegDecoderForDepth] decoder: {dec_params/1e6:.2f}M params, {dec_trainable/1e6:.2f}M trainable")
        print(f"[CLIPSegDecoderForDepth] vision encoder: {vis_params/1e6:.2f}M params (frozen)")
        print(f"[CLIPSegDecoderForDepth] depth_shift={self.depth_shift.item():.1f}, depth_scale={self.depth_scale.item():.1f}")

    def forward(
        self,
        feats: Dict[int, torch.Tensor],
        q: torch.Tensor,
        out_hw: Optional[List[int]] = None,
        image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            feats: ignored — we use CLIPSeg's own vision encoder instead
            q: [B, 512] depth query from depth embedding
            out_hw: optional output size override [H, W]
            image: [B, 3, H, W] input image (passed through for CLIPSeg's vision encoder)

        Returns:
            depth: [B, 1, H, W] positive depth map
        """
        if image is None:
            raise ValueError(
                "CLIPSegDecoderForDepth requires the raw image tensor. "
                "Make sure model.py passes image= to the decoder."
            )

        # Run CLIPSeg's own vision encoder + decoder in one pass
        outputs = self.clipseg(
            pixel_values=image,
            conditional_embeddings=q,
            output_hidden_states=True,
        )

        # outputs.logits: [B, 352, 352]
        logits = outputs.logits

        # Add channel dim and apply learnable shift/scale to bridge
        # segmentation logits (negative) to depth range (positive)
        depth = logits.unsqueeze(1)  # [B, 1, H, W]
        depth = depth * self.depth_scale + self.depth_shift

        # Apply activation for positive depth
        if self.out_activation == "softplus":
            depth = F.softplus(depth)
        elif self.out_activation == "relu":
            depth = F.relu(depth)
        elif self.out_activation == "sigmoid":
            depth = torch.sigmoid(depth) * 10.0
        # else: raw logits

        # Optional resize
        if out_hw is not None:
            depth = F.interpolate(depth, size=out_hw, mode="bilinear", align_corners=False)

        return depth
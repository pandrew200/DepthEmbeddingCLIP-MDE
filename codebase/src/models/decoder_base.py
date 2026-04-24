"""
Dense Predictor.

Inputs:
- multi-layer CLIP visual patch-grid features:
    feats[L] : [B, 768, Gh, Gw]   (e.g., L in {3,6,9}, Gh=Gw=22 for 352 with patch16)
- depth query embedding from depth embedding through CLIP text encoder:
    q : [B, 512]

Pipeline (Table 1 + paper forward equation):
1) Projection: 768 -> 64 on each selected layer feature map.
2) FiLM: q (512) -> gamma (64), beta (64), apply to projected features.
3) Dense predictor transformer blocks: 3 blocks, width 64, heads 4, MLP 2048.
   Uses skip connections from the three CLIP layers (L3/L6/L9) into each block.
4) Deconvolution head: upsample from Gh x Gw to 352 x 352 using two stride-4 transposed convs.
5) Softplus output for positive depth.

Important note about the table:
- The table lists a "conv2d k=3, s=3, p=1" inside deconvolution.
  If interpreted literally at Gh=22, stride=3 would reduce spatial size and cannot reach 352
  with two stride-4 transpose convs (22 -> 8 -> 32 -> 128).
  To produce 352x352, we implement the conventional refinement conv with stride=1.
  This is almost certainly what was intended (or the stride belongs to a different step).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Transformer block: width=64, heads=4, MLP hidden=2048
# -----------------------------------------------------------------------------

class PreNormTransformerBlock(nn.Module):
    """
    A minimal pre-norm Transformer block for token sequences.

    Token shape:
        x: [B, N, D] where D=64 and N=Gh*Gw (e.g., 484)

    Block:
        x = x + MHA(LN(x))
        x = x + MLP(LN(x))   with hidden=2048, GELU

    This matches the table "64 -> 2048 -> 64" for the FC/MLP.
    """
    def __init__(self, d_model: int = 64, n_heads: int = 4, mlp_hidden: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)

        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-head attention (batch_first=True => x is [B,N,D])
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out

        # MLP
        h = self.ln2(x)
        x = x + self.mlp(h)
        return x


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class DensePredictorConfig:
    # Which CLIP layers are fed in as skip connections to the 3 transformer blocks
    layers: List[int] = None           # default [3, 6, 9]

    # Dimensions from the table
    clip_feat_dim: int = 768           # CLIP ViT-B/16 width
    q_dim: int = 512                   # CLIP text embedding dim (ViT-B/16 text width)
    d_model: int = 64                  # Dense predictor width (table: 64)
    n_heads: int = 4                   # Dense predictor heads (table: 4)
    n_blocks: int = 3                  # Dense predictor transformer blocks (table: 3)
    mlp_hidden: int = 2048             # Dense predictor MLP hidden (table: 2048)
    use_film: bool = True              # Whether to apply FiLM conditioning
    reverse_layer_order: bool = False  # If True: process L9 -> L6 -> L3 (CLIPSeg-style)

    # Deconv head: 22 -> 88 -> 352 when stride=4 twice
    deconv_mid_ch: int = 32            # table: 32
    out_activation: str = "softplus"   # table uses Softplus

    # Refinement conv at patch-grid resolution (table shows conv2d k=3)
    refine_kernel: int = 3
    refine_stride: int = 1             # see note above; stride=3 would break 352 output
    refine_padding: int = 1

    def __post_init__(self):
        if self.layers is None:
            self.layers = [3, 6, 9]
        if len(self.layers) not in (1, 3):
            raise ValueError("Dense predictor expects 1 or 3 layers")


# -----------------------------------------------------------------------------
# Dense Predictor
# -----------------------------------------------------------------------------

class DensePredictor(nn.Module):
    """
    Dense predictor module for depth estimation.

    Forward:
      depth = D(E_visual_feats, q)

    Where:
      E_visual_feats provides multi-layer spatial patch features, and q is the depth query
      from the depth embedding + frozen CLIP text encoder.
    """

    def __init__(self, cfg: Union[DensePredictorConfig, dict]):
        super().__init__()
        if isinstance(cfg, dict):
            cfg = DensePredictorConfig(**cfg)
        self.cfg = cfg

        # 1) Projection: 768 -> 64 per layer (1x1 conv over [B,C,Gh,Gw])
        self.proj = nn.ModuleDict({
            str(l): nn.Conv2d(cfg.clip_feat_dim, cfg.d_model, kernel_size=1, bias=False)
            for l in cfg.layers
        })

        # 2) FiLM: q (512) -> gamma (64), beta (64)
        # The figure/table shows separate FiLM Add and FiLM Mul paths.
        self.use_film = cfg.use_film
        if cfg.use_film:
            self.film_mul = nn.Linear(cfg.q_dim, cfg.d_model, bias=True)  # gamma
            self.film_add = nn.Linear(cfg.q_dim, cfg.d_model, bias=True)  # beta

            # Optional identity-ish init: start near no-op (gamma≈1, beta≈0)
            nn.init.normal_(self.film_mul.weight, mean=0.0, std=1e-3)
            nn.init.ones_(self.film_mul.bias)
            nn.init.normal_(self.film_add.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.film_add.bias)

        # 3) Three transformer blocks at width 64, heads 4, MLP hidden 2048
        self.blocks = nn.ModuleList([
            PreNormTransformerBlock(
                d_model=cfg.d_model,
                n_heads=cfg.n_heads,
                mlp_hidden=cfg.mlp_hidden,
                dropout=0.0,
            )
            for _ in range(cfg.n_blocks)
        ])

        # 4) Deconvolution head to get dense map at 352x352
        # Refinement conv at patch-grid scale (k=3, p=1).
        self.refine = nn.Conv2d(
            cfg.d_model, cfg.d_model,
            kernel_size=cfg.refine_kernel,
            stride=cfg.refine_stride,
            padding=cfg.refine_padding,
            bias=True
        )

        # Two stride-4 transpose convs: 22->88->352
        self.deconv1 = nn.ConvTranspose2d(cfg.d_model, cfg.deconv_mid_ch, kernel_size=4, stride=4, padding=0, bias=True)
        self.deconv2 = nn.ConvTranspose2d(cfg.deconv_mid_ch, 1, kernel_size=4, stride=4, padding=0, bias=True)

        # Output activation
        if cfg.out_activation == "softplus":
            self.out_act = nn.Softplus()
        elif cfg.out_activation == "relu":
            self.out_act = nn.ReLU()
        elif cfg.out_activation == "none":
            self.out_act = nn.Identity()
        else:
            raise ValueError(f"Unknown out_activation: {cfg.out_activation}")

    def _apply_film(self, F: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM to a feature map.

        F: [B, 64, Gh, Gw]
        q: [B, 512]
        """
        gamma = self.film_mul(q).unsqueeze(-1).unsqueeze(-1)  # [B,64,1,1]
        beta = self.film_add(q).unsqueeze(-1).unsqueeze(-1)   # [B,64,1,1]
        return gamma * F + beta

    def forward(
        self,
        feats: Dict[int, torch.Tensor],
        q: torch.Tensor,
        out_hw: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """
        Args:
            feats: dict layer -> [B, 768, Gh, Gw], expected keys are cfg.layers (e.g. 3,6,9)
            q: [B, 512] depth query embedding from depth embedding/text encoder
            out_hw: optional [H,W] final interpolation target.
                    If None, output size is determined by deconvs (typically 352x352 if Gh=22).

        Returns:
            depth: [B, 1, H, W]
        """
        # Project and optionally FiLM each selected layer feature map to width 64
        layer_feats_64 = []
        layer_order = list(reversed(self.cfg.layers)) if self.cfg.reverse_layer_order else self.cfg.layers
        for l in layer_order:
            if l not in feats:
                raise KeyError(f"Missing layer {l} in feats keys={list(feats.keys())}")
            F_l = self.proj[str(l)](feats[l])         # [B,64,Gh,Gw]
            if self.use_film:
                F_l = self._apply_film(F_l, q)        # [B,64,Gh,Gw]
            layer_feats_64.append(F_l)

        # Sequential processing with skip connections
        x = layer_feats_64[0]
        B, C, Gh, Gw = x.shape

        x_tok = x.flatten(2).transpose(1, 2)  # [B, Gh*Gw, 64]
        x_tok = self.blocks[0](x_tok)

        if len(layer_feats_64) == 3:
            # 3-layer mode: skip connections from layers 2 and 3
            x1 = layer_feats_64[1].flatten(2).transpose(1, 2)
            x_tok = x_tok + x1
            x_tok = self.blocks[1](x_tok)

            x2 = layer_feats_64[2].flatten(2).transpose(1, 2)
            x_tok = x_tok + x2
            x_tok = self.blocks[2](x_tok)
        else:
            # 1-layer mode: run all blocks on the same features
            x_tok = self.blocks[1](x_tok)
            x_tok = self.blocks[2](x_tok)

        # Tokens back to spatial map: [B,64,Gh,Gw]
        x = x_tok.transpose(1, 2).reshape(B, C, Gh, Gw)

        # Refinement conv at patch grid resolution
        x = self.refine(x)

        # Deconvolution head: Gh->4Gh->16Gh (22->88->352)
        x = self.deconv1(x)
        x = F.gelu(x)
        x = self.deconv2(x)

        # Ensure positive depth
        x = self.out_act(x)

        # Optionally force exact output size (useful if input size differs from 352)
        if out_hw is not None:
            x = F.interpolate(x, size=tuple(out_hw), mode="bilinear", align_corners=False)

        return x
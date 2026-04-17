from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import open_clip


@dataclass
class CLIPConfig:
    model_name: str = "ViT-B-16"
    pretrained: str = "openai"
    layers: List[int] = None  # e.g. [3, 6, 9] meaning after blocks 3/6/9 (1-indexed)
    device: str = "cuda"
    precision: str = "fp32"   # "fp32" or "amp"
    return_patch_tokens: bool = True  # True: return patch grid; False: return full token seq
    normalize: bool = False   # whether to L2-normalize embeddings

    def __post_init__(self):
        if self.layers is None:
            self.layers = [3, 6, 9]


class CLIPBackbone(nn.Module):
    """
    Wrapper around open_clip for:
      - Loading CLIP model + preprocess
      - Extracting intermediate ViT block outputs (e.g. L3/L6/L9)

    Note: This is written for ViT-based CLIP (VisualTransformer).
    """
    def __init__(self, cfg: Union[CLIPConfig, dict]):
        super().__init__()
        if isinstance(cfg, dict):
            cfg = CLIPConfig(**cfg)
        self.cfg = cfg

        # Load CLIP model and preprocess transforms
        model, _, preprocess = open_clip.create_model_and_transforms(
            cfg.model_name,
            pretrained=cfg.pretrained,
            device=cfg.device,
        )
        self.model = model.eval()
        self.preprocess = preprocess  # torchvision transform pipeline

        # Freeze everything (you can unfreeze later if desired)
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Validate visual backbone is ViT
        visual = getattr(self.model, "visual", None)
        if visual is None or not hasattr(visual, "transformer"):
            raise ValueError(
                "This CLIPBackbone implementation expects a ViT-based CLIP model "
                "(model.visual.transformer.resblocks). You may be using a ResNet CLIP."
            )

        self.visual = visual  # VisualTransformer

        # Convert requested layers (1-indexed) -> resblock indices (0-indexed)
        self._layer_idxs = [int(l) - 1 for l in cfg.layers]
        n_blocks = len(self.visual.transformer.resblocks)
        for i in self._layer_idxs:
            if i < 0 or i >= n_blocks:
                raise ValueError(f"Requested layer {i+1} but ViT has {n_blocks} blocks.")

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        image: [B,3,H,W] float, preprocessed for CLIP
        returns: [B, D] image embedding
        """
        emb = self.model.encode_image(image)
        if self.cfg.normalize:
            emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
        return emb

    @torch.no_grad()
    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        text_tokens: tokenized text tensor (from open_clip tokenizer)
        returns: [B, D] text embedding
        """
        emb = self.model.encode_text(text_tokens)
        if self.cfg.normalize:
            emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-8)
        return emb


    def _resize_pos_embed(self, pos_embed: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
        """
        Resize CLIP ViT positional embeddings to match a new patch grid (gh x gw).

        Args:
            pos_embed: [T, C] where T = 1 + H0*W0 (CLS + patch positions)
            gh, gw: new grid size

        Returns:
            new_pos_embed: [1 + gh*gw, C]
        """
        # pos_embed is [T, C]
        cls_pos = pos_embed[:1]          # [1, C]
        patch_pos = pos_embed[1:]        # [H0*W0, C]

        # Infer original grid size (H0 == W0 for CLIP ViT)
        n = patch_pos.shape[0]
        h0 = w0 = int(n ** 0.5)
        if h0 * w0 != n:
            raise ValueError(f"Pos embed patch length {n} is not a perfect square.")

        # Reshape to [1, C, H0, W0] for interpolation
        patch_pos = patch_pos.reshape(h0, w0, -1).permute(2, 0, 1).unsqueeze(0)

        # Bicubic interpolate to [1, C, gh, gw]
        patch_pos = F.interpolate(patch_pos, size=(gh, gw), mode="bicubic", align_corners=False)

        # Back to [gh*gw, C]
        patch_pos = patch_pos.squeeze(0).permute(1, 2, 0).reshape(gh * gw, -1)

        # Concatenate CLS back
        return torch.cat([cls_pos, patch_pos], dim=0)

    @torch.no_grad()
    def extract_vit_intermediates(
        self,
        image: torch.Tensor,
        layers: Optional[List[int]] = None,
        return_patch_tokens: Optional[bool] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Extract intermediate features from ViT blocks.

        Args:
          image: [B,3,H,W] preprocessed CLIP input
          layers: optional list like [3,6,9] (1-indexed)
          return_patch_tokens:
            - True: returns patch grid features [B, C, H_p, W_p]
            - False: returns full token sequence [B, T, C] (includes CLS token)

        Returns:
          dict mapping layer_number (1-indexed) -> feature tensor
        """
        if layers is None:
            layer_idxs = self._layer_idxs
            layer_nums = [i + 1 for i in layer_idxs]
        else:
            layer_idxs = [int(l) - 1 for l in layers]
            layer_nums = [i + 1 for i in layer_idxs]

        if return_patch_tokens is None:
            return_patch_tokens = self.cfg.return_patch_tokens

        v = self.visual  # VisualTransformer
        x = image

        # ---- This replicates open_clip VisualTransformer forward() up to the transformer blocks ----
        # Patchify
        x = v.conv1(x)  # [B, width, grid, grid]
        B, C, Gh, Gw = x.shape
        x = x.reshape(B, C, Gh * Gw).permute(0, 2, 1)  # [B, N, C]

        # Add CLS token
        cls = v.class_embedding.to(x.dtype)
        cls_tokens = cls.expand(B, 1, -1)  # [B,1,C]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 1+N, C]

        # Positional embedding
        pos = v.positional_embedding.to(x.dtype)  # [T0, C]
        if pos.shape[0] != x.shape[1]:
            # x is [B, T, C] at this point
            pos = self._resize_pos_embed(pos, Gh, Gw).to(x.dtype)
        x = x + pos
        x = v.ln_pre(x)

        # Transformer expects [T,B,C] in open_clip
        x = x.permute(1, 0, 2)  # [T,B,C]

        feats: Dict[int, torch.Tensor] = {}

        # Iterate blocks
        for bi, blk in enumerate(v.transformer.resblocks):
            x = blk(x)
            if bi in layer_idxs:
                # x is [T,B,C] -> [B,T,C]
                xb = x.permute(1, 0, 2).contiguous()
                if return_patch_tokens:
                    # remove CLS token and reshape back to grid
                    patch = xb[:, 1:, :]  # [B, N, C]
                    patch = patch.permute(0, 2, 1).contiguous().reshape(B, C, Gh, Gw)  # [B,C,Gh,Gw]
                    feats[bi + 1] = patch
                else:
                    feats[bi + 1] = xb  # [B,T,C]

        return feats

    def get_preprocess(self):
        return self.preprocess

    def get_tokenizer(self):
        return open_clip.get_tokenizer(self.cfg.model_name)
"""
Depth embedding module for CLIP-based depth conditioning.

This module learns a set of tokens that are passed through the frozen
CLIP text encoder to produce a single "depth query" embedding.

Why this works:
---------------
CLIP aligns image and text embeddings in a shared space.
By learning tokens that pass through the text encoder,
we constrain the depth embedding to live inside CLIP's
joint language-vision representation space.

This is similar to prompt tuning in language models.

Output:
-------
q ∈ R^D  (single depth query vector)
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn


class DepthEmbedding(nn.Module):
    """
    Learnable depth embedding tokens that produce a depth query vector.

    Steps:
    1. Initialize learnable token embeddings
    2. Insert BOS/EOS tokens
    3. Pass through frozen CLIP text encoder
    4. Pool output → single depth query vector
    """

    def __init__(
        self,
        clip_model,
        num_tokens: int = 64,
        embed_dim: Optional[int] = None,
        pooling: str = "eos",
        normalize: bool = False,
    ):
        """
        Args:
            clip_model:
                CLIP model from open_clip (backbone.model)

            num_tokens:
                Number of learnable tokens (depth embedding tokens)

            embed_dim:
                Dimension of token embeddings.
                If None, inferred from CLIP text embedding dimension.

            pooling:
                How to convert token outputs into a single vector.
                Options:
                    "mean"
                    "cls"
                    "eos" (default)

            normalize:
                If True, L2 normalize the output embedding.
        """
        super().__init__()

        self.clip_model = clip_model
        self.pooling = pooling
        self.normalize = normalize

        # Access CLIP text transformer
        self.text = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

        # Causal attention mask (required by CLIP text transformer)
        self.register_buffer(
            "attn_mask", clip_model.attn_mask.clone() if hasattr(clip_model, "attn_mask") and clip_model.attn_mask is not None else None
        )

        if embed_dim is None:
            embed_dim = self.token_embedding.weight.shape[1]

        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

        # ---------------------------------------------------------
        # BOS / EOS token IDs (SOT / EOT in CLIP vocabulary)
        # For OpenAI CLIP: SOT = vocab_size - 2, EOT = vocab_size - 1
        # ---------------------------------------------------------
        vocab_size = self.token_embedding.weight.shape[0]
        self.sot_index = vocab_size - 2   # 49406 for ViT-B/16
        self.eot_index = vocab_size - 1   # 49407 for ViT-B/16

        # ---------------------------------------------------------
        # Learnable depth embedding tokens
        # ---------------------------------------------------------
        self.embedding_tokens = nn.Parameter(
            torch.randn(num_tokens, embed_dim) * 0.02
        )

        # Freeze CLIP text encoder parameters
        for p in clip_model.parameters():
            p.requires_grad_(False)

    def forward(self) -> torch.Tensor:
        """
        Returns:
            depth_query: [D] vector
        """

        device = self.embedding_tokens.device

        # ---------------------------------------------------------
        # Build token sequence:
        # [BOS] + embedding tokens + [EOS]
        # ---------------------------------------------------------

        bos = self.token_embedding.weight[self.sot_index : self.sot_index + 1]
        eos = self.token_embedding.weight[self.eot_index : self.eot_index + 1]

        tokens = torch.cat([bos, self.embedding_tokens, eos], dim=0)
        tokens = tokens.unsqueeze(0)  # shape: [1, T, D] (batch_first)

        # ---------------------------------------------------------
        # Add positional embeddings
        # ---------------------------------------------------------

        T = tokens.shape[1]
        max_pos = self.positional_embedding.shape[0]  # 77 for CLIP
        if T <= max_pos:
            tokens = tokens + self.positional_embedding[:T].unsqueeze(0)
        else:
            # Interpolate positional embeddings for longer sequences
            pos = self.positional_embedding.unsqueeze(0).transpose(1, 2)  # [1, D, 77]
            pos = torch.nn.functional.interpolate(pos, size=T, mode="linear", align_corners=False)
            tokens = tokens + pos.transpose(1, 2)  # [1, T, D]

        # ---------------------------------------------------------
        # Pass through frozen text transformer (with causal attention mask)
        # Transformer is batch_first=True, expects [B, T, D]
        # ---------------------------------------------------------
        if self.attn_mask is not None and T <= self.attn_mask.shape[0]:
            attn_mask = self.attn_mask[:T, :T]
        else:
            # Build causal mask for longer sequences
            attn_mask = torch.triu(torch.full((T, T), float("-inf"), device=tokens.device), diagonal=1)
        x = self.text(tokens, attn_mask=attn_mask)

        # Final layer norm
        x = self.ln_final(x)

        # ---------------------------------------------------------
        # Pool tokens → single embedding
        # x is [1, T, D] (batch_first)
        # ---------------------------------------------------------
        if self.pooling == "mean":
            pooled = x[0].mean(dim=0)  # average across tokens → [D]
        elif self.pooling == "cls":
            pooled = x[0, 0]  # CLS/BOS token → [D]
        elif self.pooling == "eos":
            pooled = x[0, -1]  # EOS token → [D]
        else:
            raise ValueError("Unknown pooling method")

        # Project into CLIP embedding space
        pooled = pooled @ self.text_projection

        if self.normalize:
            pooled = pooled / pooled.norm(dim=-1, keepdim=True)

        return pooled.squeeze(0)  # [D]
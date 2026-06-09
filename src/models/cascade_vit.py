"""Cascade ViT classifiers for acoustic leak detection."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.vit_blocks import PatchEmbedding, TransformerEncoder


class BinaryViT(nn.Module):
    """Binary classifier: normal (0) vs leak (1)."""

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        emb_dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(1, patch_size, emb_dim, img_size)
        self.encoder = nn.Sequential(
            *[TransformerEncoder(emb_dim, n_heads) for _ in range(n_layers)]
        )
        self.head = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(self.patch_embed(x))
        return self.head(x[:, 0])


class LeakRegressorViT(nn.Module):
    """Regresses leak distance as a scalar in [0, 1] (normalised metres)."""

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        emb_dim: int = 128,
        n_layers: int = 8,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(1, patch_size, emb_dim, img_size)
        self.encoder = nn.Sequential(
            *[TransformerEncoder(emb_dim, n_heads) for _ in range(n_layers)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(self.patch_embed(x))
        return self.head(x[:, 0]).squeeze(1)


class LeakPosViT(nn.Module):
    """Classifies leak into one of n_pos discrete position classes."""

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        emb_dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        n_pos: int = 11,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbedding(1, patch_size, emb_dim, img_size)
        self.encoder = nn.Sequential(
            *[TransformerEncoder(emb_dim, n_heads) for _ in range(n_layers)]
        )
        self.head = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, n_pos))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(self.patch_embed(x))
        return self.head(x[:, 0])

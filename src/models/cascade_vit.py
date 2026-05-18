import torch.nn as nn
from src.models.vit_blocks import PatchEmbedding, TransformerEncoder


class BinaryViT(nn.Module):
    """Binary classifier: normalidade (0) vs vazamento (1)."""

    def __init__(self, img_size=64, patch_size=8, emb_dim=128, n_layers=4, n_heads=4):
        super().__init__()
        self.patch_embed = PatchEmbedding(1, patch_size, emb_dim, img_size)
        self.encoder = nn.Sequential(*[TransformerEncoder(emb_dim, n_heads) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, 2))

    def forward(self, x):
        x = self.encoder(self.patch_embed(x))
        return self.head(x[:, 0])


class LeakPosViT(nn.Module):
    """Classifies leak into one of n_pos discrete position classes."""

    def __init__(self, img_size=64, patch_size=8, emb_dim=128, n_layers=4, n_heads=4, n_pos=11):
        super().__init__()
        self.patch_embed = PatchEmbedding(1, patch_size, emb_dim, img_size)
        self.encoder = nn.Sequential(*[TransformerEncoder(emb_dim, n_heads) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, n_pos))

    def forward(self, x):
        x = self.encoder(self.patch_embed(x))
        return self.head(x[:, 0])


class LeakRegressorViT(nn.Module):
    """Regresses leak distance to [0, 1] (normalized linear position)."""

    def __init__(self, img_size=64, patch_size=8, emb_dim=128, n_layers=8, n_heads=4):
        super().__init__()
        self.patch_embed = PatchEmbedding(1, patch_size, emb_dim, img_size)
        self.encoder = nn.Sequential(*[TransformerEncoder(emb_dim, n_heads) for _ in range(n_layers)])
        self.head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(self.patch_embed(x))
        return self.head(x[:, 0]).squeeze(1)

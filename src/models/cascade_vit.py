"""
Arquivo: src/models/cascade_vit.py
Descrição: Montagem dos modelos ViT finais para a cascata (BinaryViT, LeakPosViT e LeakRegressorViT).
"""

import torch.nn as nn
from src.models.vit_blocks import PatchEmbedding, TransformerEncoder


class BinaryViT(nn.Module):
    """Classifica entre Normalidade (0) e Vazamento (1)"""

    def __init__(self, img_size=64, patch_size=8, emb_dim=128, n_layers=4, n_heads=4):
        super().__init__()
        self.patch_embed = PatchEmbedding(1, patch_size, emb_dim, img_size)
        self.encoder = nn.Sequential(
            *[TransformerEncoder(emb_dim, n_heads) for _ in range(n_layers)]
        )
        self.classifier = nn.Sequential(nn.LayerNorm(emb_dim), nn.Linear(emb_dim, 2))

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder(x)
        return self.classifier(x[:, 0])  # Pega só o token [CLS]


class LeakRegressorViT(nn.Module):
    """Estima a distância do vazamento (Regressão [0, 1])"""

    def __init__(self, img_size=64, patch_size=8, emb_dim=128, n_layers=8, n_heads=4):
        super().__init__()
        self.patch_embed = PatchEmbedding(1, patch_size, emb_dim, img_size)
        self.encoder = nn.Sequential(
            *[TransformerEncoder(emb_dim, n_heads) for _ in range(n_layers)]
        )
        self.regressor = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder(x)
        return self.regressor(x[:, 0]).squeeze(1)

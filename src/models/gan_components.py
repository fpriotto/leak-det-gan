"""
Arquivo: src/models/gan_components.py
Descrição: Arquitetura das redes neurais da WGAN (Gen1D, Critic1D e CriticSpec).
"""

import torch
import torch.nn as nn


class Gen1D(nn.Module):
    """Gerador Condicional 1D para sinais temporais."""

    def __init__(self, z_dim=100, cond_dim=1, seq_len=2500):
        super().__init__()
        self.fc = nn.Linear(z_dim + cond_dim, 256 * 20)
        self.up = nn.Upsample(size=seq_len, mode="linear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv1d(256, 128, 7, padding=3),
            nn.ReLU(True),
            nn.Conv1d(128, 64, 7, padding=3),
            nn.ReLU(True),
            nn.Conv1d(64, 1, 7, padding=3),
            nn.Tanh(),
        )

    def forward(self, z, c):
        x = torch.cat([z, c], dim=1)
        x = self.fc(x).view(-1, 256, 20)
        x = self.up(x)
        return self.conv(x)


class Critic1D(nn.Module):
    """Crítico Condicional para avaliar a fidelidade do sinal no Domínio do Tempo."""

    def __init__(self, cond_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1 + cond_dim, 64, 25, stride=4, padding=11),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(64, 128, 25, stride=4, padding=11),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(128, 256, 25, stride=4, padding=11),
            nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(256, 1)

    def forward(self, x, c):
        # Expande a condição (distância) para o tamanho do sinal
        c_feat = c.unsqueeze(-1).repeat(1, 1, x.size(-1))
        x_in = torch.cat([x, c_feat], dim=1)
        h = self.net(x_in).squeeze(-1)
        return self.fc(h)


class CriticSpec(nn.Module):
    """Crítico para avaliar a fidelidade no Domínio da Frequência (Espectrograma)."""

    def __init__(self, input_shape=(129, 20)):
        super().__init__()
        in_dim = input_shape[0] * input_shape[1]
        self.model = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 1),
        )

    def forward(self, spec):
        flat = spec.flatten(start_dim=1)
        return self.model(flat)

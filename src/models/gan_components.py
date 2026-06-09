"""Generator and critic architectures for the conditional WGAN-GP."""
from __future__ import annotations

import torch
import torch.nn as nn


class Gen1D(nn.Module):
    """Conditional 1-D generator: maps (z, condition) → time-series signal.

    Architecture: FC projection → upsample → three Conv1d layers with Tanh output.
    """

    def __init__(self, z_dim: int = 100, cond_dim: int = 1, seq_len: int = 2500) -> None:
        super().__init__()
        self.fc = nn.Linear(z_dim + cond_dim, 256 * 20)
        self.up = nn.Upsample(size=seq_len, mode="linear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv1d(256, 128, 7, padding=3), nn.ReLU(True),
            nn.Conv1d(128,  64, 7, padding=3), nn.ReLU(True),
            nn.Conv1d( 64,   1, 7, padding=3), nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = self.fc(torch.cat([z, c], dim=1)).view(-1, 256, 20)
        return self.conv(self.up(x))


class Critic1D(nn.Module):
    """Conditional critic operating in the time domain.

    Concatenates the condition as a repeated feature channel, then applies
    strided convolutions followed by a linear score head.
    """

    def __init__(self, cond_dim: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1 + cond_dim,  64, 25, stride=4, padding=11), nn.LeakyReLU(0.2, True),
            nn.Conv1d( 64,          128, 25, stride=4, padding=11), nn.LeakyReLU(0.2, True),
            nn.Conv1d(128,          256, 25, stride=4, padding=11), nn.LeakyReLU(0.2, True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        c_feat = c.unsqueeze(-1).repeat(1, 1, x.size(-1))
        return self.fc(self.net(torch.cat([x, c_feat], dim=1)).squeeze(-1))


class CriticSpec(nn.Module):
    """Unconditional critic operating on flattened log-spectrograms.

    Expects a spectrogram tensor of shape matching ``input_shape``.
    """

    def __init__(self, input_shape: tuple[int, int] = (129, 20)) -> None:
        super().__init__()
        in_dim = input_shape[0] * input_shape[1]
        self.model = nn.Sequential(
            nn.Linear(in_dim, 512), nn.LeakyReLU(0.2, True),
            nn.Linear(512,    256), nn.LeakyReLU(0.2, True),
            nn.Linear(256,      1),
        )

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        return self.model(spec.flatten(start_dim=1))

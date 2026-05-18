import torch
import torch.nn as nn


class Gen1D(nn.Module):
    """Conditional 1-D generator for time-series signals."""

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
        x = self.fc(torch.cat([z, c], dim=1)).view(-1, 256, 20)
        return self.conv(self.up(x))


class Critic1D(nn.Module):
    """Conditional critic operating in the time domain."""

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
        c_feat = c.unsqueeze(-1).repeat(1, 1, x.size(-1))
        return self.fc(self.net(torch.cat([x, c_feat], dim=1)).squeeze(-1))


class CriticSpec(nn.Module):
    """Unconditional critic operating on spectrograms."""

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
        return self.model(spec.flatten(start_dim=1))

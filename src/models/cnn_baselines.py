"""CNN baseline models for comparison with the cascade ViT."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as models


class SimpleCNN(nn.Module):
    """Lightweight 3-layer CNN baseline.

    ``n_classes >= 2`` → multi-class logits (cross-entropy loss).
    ``n_classes == 1`` → scalar regression output (Sigmoid activation).
    """

    def __init__(self, n_classes: int = 2) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_stack(x).flatten(1)
        out  = self.fc(feat)
        if self.n_classes == 1:
            return torch.sigmoid(out).squeeze(1)
        return out


class ResNetBaseline(nn.Module):
    """ResNet-18 trained from scratch on single-channel spectrograms.

    The first conv layer is replaced to accept 1-channel input.
    ``n_classes == 1`` → regression output (Sigmoid activation).
    """

    def __init__(self, n_classes: int = 2) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.resnet    = models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc    = nn.Linear(self.resnet.fc.in_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.resnet(x)
        if self.n_classes == 1:
            return torch.sigmoid(out).squeeze(1)
        return out

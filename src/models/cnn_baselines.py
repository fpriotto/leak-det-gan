import torch
import torch.nn as nn
import torchvision.models as models


class SimpleCNN(nn.Module):
    """Lightweight CNN. n_classes=1 → regressor (Sigmoid), n_classes≥2 → classifier."""

    def __init__(self, n_classes=2):
        super().__init__()
        self.n_classes = n_classes
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        feat = self.conv_stack(x).flatten(1)
        out = self.fc(feat)
        if self.n_classes == 1:
            return torch.sigmoid(out).squeeze(1)
        return out


class ResNetBaseline(nn.Module):
    """ResNet-18 from scratch on single-channel spectrograms. n_classes=1 → regressor."""

    def __init__(self, n_classes=2):
        super().__init__()
        self.n_classes = n_classes
        self.resnet = models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, n_classes)

    def forward(self, x):
        out = self.resnet(x)
        if self.n_classes == 1:
            return torch.sigmoid(out).squeeze(1)
        return out

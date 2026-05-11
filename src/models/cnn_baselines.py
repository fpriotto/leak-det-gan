import torch
import torch.nn as nn
import torchvision.models as models


class SimpleCNN(nn.Module):
    """CNN básica para benchmark de classificação e regressão."""

    def __init__(self, num_outputs=1, is_classifier=False):
        super().__init__()
        self.is_classifier = is_classifier
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, 2 if is_classifier else 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_stack(x).flatten(1)
        x = self.fc(x)
        return x if self.is_classifier else self.sigmoid(x).squeeze(1)


class ResNetBaseline(nn.Module):
    """ResNet-18 para benchmark acadêmico (Ablation Study)."""

    def __init__(self, is_classifier=False):
        super().__init__()
        self.is_classifier = is_classifier

        # Carrega a arquitetura ResNet-18 sem pesos pré-treinados
        self.resnet = models.resnet18(weights=None)

        # Modifica a primeira camada para aceitar 1 canal (Grayscale) em vez de 3 (RGB)
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Modifica a última camada (Fully Connected) para 2 classes (Binário) ou 1 (Regressão)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 2 if is_classifier else 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        return x if self.is_classifier else self.sigmoid(x).squeeze(1)

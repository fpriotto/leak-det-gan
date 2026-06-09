"""Dataset and transform utilities for loading spectrogram PNGs for ViT training."""
from __future__ import annotations

import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SpecMask:
    """SpecAugment-style frequency and time masking for spectrograms.

    Randomly zeroes a band of frequency rows and a band of time columns,
    each drawn uniformly up to ``img_size // 8`` wide.
    """

    def __init__(self, img_size: int = 64) -> None:
        self.freq = img_size // 8
        self.time = img_size // 8

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        _, h, w = img.shape
        f = random.randint(0, self.freq)
        f0 = random.randint(0, h - f)
        img[:, f0 : f0 + f, :] = 0
        t = random.randint(0, self.time)
        t0 = random.randint(0, w - t)
        img[:, :, t0 : t0 + t] = 0
        return img


def get_transforms(img_size: int = 64) -> tuple[transforms.Compose, transforms.Compose]:
    """Build train and validation transform pipelines.

    Train augmentation: SpecMask + RandomErasing (50% probability).
    Validation: resize, grayscale, normalise — no augmentation.

    Args:
        img_size: Target square image size in pixels.

    Returns:
        Tuple of (train_transform, val_transform).
    """
    basic = transforms.Compose([
        transforms.Resize(img_size),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    train_tf = transforms.Compose([
        basic,
        SpecMask(img_size),
        transforms.RandomApply([transforms.RandomErasing(p=1.0, scale=(0.02, 0.2))], p=0.5),
        transforms.Normalize([0.5], [0.5]),
    ])
    val_tf = transforms.Compose([basic, transforms.Normalize([0.5], [0.5])])
    return train_tf, val_tf


class SpectroDataset(Dataset):
    """Dataset that loads grayscale spectrogram PNGs from disk.

    Args:
        samples: List of (file_path, label) pairs.
        transform: Transform pipeline applied to each loaded image.
    """

    def __init__(self, samples: list[tuple[str, int]], transform: transforms.Compose) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, lab = self.samples[idx]
        return (
            self.transform(Image.open(path).convert("L")),
            torch.tensor(lab, dtype=torch.long),
        )

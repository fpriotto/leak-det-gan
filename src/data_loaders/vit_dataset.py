import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SpecMask:
    def __init__(self, img_size=64):
        self.freq = img_size // 8
        self.time = img_size // 8

    def __call__(self, img):
        _, h, w = img.shape
        f = random.randint(0, self.freq)
        f0 = random.randint(0, h - f)
        img[:, f0 : f0 + f, :] = 0
        t = random.randint(0, self.time)
        t0 = random.randint(0, w - t)
        img[:, :, t0 : t0 + t] = 0
        return img


def get_transforms(img_size=64):
    basic = transforms.Compose(
        [transforms.Resize(img_size), transforms.Grayscale(), transforms.ToTensor()]
    )
    train_tf = transforms.Compose(
        [
            basic,
            SpecMask(img_size),
            transforms.RandomApply([transforms.RandomErasing(p=1.0, scale=(0.02, 0.2))], p=0.5),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    val_tf = transforms.Compose([basic, transforms.Normalize([0.5], [0.5])])
    return train_tf, val_tf


class SpectroDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, lab = self.samples[idx]
        return self.transform(Image.open(path).convert("L")), torch.tensor(lab, dtype=torch.long)

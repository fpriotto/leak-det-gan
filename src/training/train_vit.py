"""
Arquivo: src/training/train_vit.py
Descrição: Loop de treinamento das redes ViT utilizando Autocast e GradScaler para performance em GPU.
"""

import torch
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()


def train_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train(train)
    total_loss, total_correct, total = 0, 0, 0
    loop = tqdm(loader, desc="Train" if train else "Val", leave=False)

    for X, y in loop:
        X, y = X.to(device), y.to(device)

        if train:
            optimizer.zero_grad()
            with autocast():
                out = model(X)
                loss = criterion(out, y) if out.ndim == 1 else criterion(out, y.long())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.no_grad():
                out = model(X)
                loss = criterion(out, y) if out.ndim == 1 else criterion(out, y.long())

        total_loss += loss.item() * X.size(0)
        if out.ndim > 1:  # Se for classificação, calcula acurácia
            total_correct += (out.argmax(1) == y.long()).sum().item()
        total += X.size(0)

    acc = (total_correct / total) if total_correct > 0 else None
    return total_loss / total, acc

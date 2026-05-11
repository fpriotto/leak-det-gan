"""
Arquivo: src/data_loaders/gan_dataset.py
Descrição: Criação do TensorDataset e DataLoaders do PyTorch específicos para treinar a GAN com dados temporais.
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchaudio.transforms as T
from pathlib import Path


def load_and_prepare_gan_data(treino_dir: str, seq_len: int = 2500):
    """Lê os dados .pkl, corta no seq_len, normaliza o tempo e as posições."""
    map_pos = {1: 10, 2: 25, 3: 60}
    X, y, specs = [], [], []

    # Opcional: Se for rodar na GPU, podemos mandar o T.Spectrogram direto pro device depois
    spec_transform = T.Spectrogram(n_fft=256, hop_length=128, power=2)

    for pos in range(1, 4):
        file_path = Path(treino_dir) / f"vallen_{pos}_v1.pkl"
        if not file_path.exists():
            print(f"Aviso: {file_path} não encontrado. Pulando...")
            continue

        with open(file_path, "rb") as f:
            series_list = pickle.load(f)

        for s in series_list:
            # Se for pd.Series usa .values, se for Numpy usa direto
            sig = (
                s.values.astype(np.float32)
                if hasattr(s, "values")
                else np.asarray(s, dtype=np.float32)
            )

            # Preenche com zeros se for menor que SEQ_LEN, ou corta se for maior
            if len(sig) < seq_len:
                sig = np.pad(sig, (0, seq_len - len(sig)), "constant")
            else:
                sig = sig[:seq_len]

            X.append(sig)
            y.append(pos)

            # Já calcula o espectrograma base para usar no TensorDataset
            with torch.no_grad():
                tsig = torch.from_numpy(sig).unsqueeze(0)
                spec = spec_transform(tsig).log1p()
                specs.append(spec.numpy())

    if not X:
        raise ValueError("Nenhum dado válido encontrado para treinar a GAN!")

    X = np.stack(X)
    y = np.array(y, dtype=np.int32).reshape(-1, 1)
    specs = np.stack(specs)

    # Normaliza sinais no tempo para a escala [-1, 1]
    time_min, time_max = X.min(), X.max()
    X_scaled = 2 * (X - time_min) / (time_max - time_min) - 1

    # Normaliza posição como log10 na escala [0, 1]
    log_pos = np.log10(np.array(list(map_pos.values())))
    log_pos_scaled = (log_pos - log_pos.min()) / (log_pos.max() - log_pos.min())
    log_map_pos = {idx + 1: log_pos_scaled[idx] for idx in range(3)}

    y_scaled = np.array([log_map_pos[int(p)] for p in y.flatten()]).reshape(-1, 1)

    # Converte tudo para tensores do PyTorch
    X_tensor = torch.from_numpy(X_scaled).unsqueeze(1)
    y_tensor = torch.from_numpy(y_scaled).float()
    spec_tensor = torch.from_numpy(specs).float()

    dataset = TensorDataset(y_tensor, X_tensor, spec_tensor)

    # Dicionário de metadados para salvar no final (fundamental para a inferência/ViT)
    meta = {
        "GLOBAL_MIN": float(time_min),
        "GLOBAL_MAX": float(time_max),
        "pos_min": float(min(map_pos.values())),
        "pos_max": float(max(map_pos.values())),
        "SEQ_LEN": seq_len,
    }

    return dataset, meta


def get_gan_dataloader(treino_dir: str, batch_size: int, seq_len: int = 2500):
    """Retorna o DataLoader e os metadados de normalização."""
    dataset, meta = load_and_prepare_gan_data(treino_dir, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader, meta

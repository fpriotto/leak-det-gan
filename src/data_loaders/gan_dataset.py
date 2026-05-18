import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchaudio.transforms as T
from pathlib import Path


def denorm_signal(sig_scaled: np.ndarray, meta: dict) -> np.ndarray:
    """Inverts the [-1, 1] normalization applied during GAN training."""
    return (sig_scaled + 1) / 2 * (meta["GLOBAL_MAX"] - meta["GLOBAL_MIN"]) + meta["GLOBAL_MIN"]


def load_and_prepare_gan_data(train_dir: str, seq_len: int = 2500):
    pos_map = {1: 10, 2: 25, 3: 60}
    spec_transform = T.Spectrogram(n_fft=256, hop_length=128, power=2)
    X, y, specs = [], [], []

    for pos in range(1, 4):
        file_path = Path(train_dir) / f"vallen_{pos}_v1.pkl"
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping.")
            continue
        with open(file_path, "rb") as f:
            series_list = pickle.load(f)

        for s in series_list:
            sig = s.values.astype(np.float32) if hasattr(s, "values") else np.asarray(s, dtype=np.float32)
            if len(sig) < seq_len:
                sig = np.pad(sig, (0, seq_len - len(sig)), "constant")
            else:
                sig = sig[:seq_len]
            X.append(sig)
            y.append(pos)
            with torch.no_grad():
                spec = spec_transform(torch.from_numpy(sig).unsqueeze(0)).log1p()
                specs.append(spec.numpy())

    if not X:
        raise ValueError("No valid training data found.")

    X = np.stack(X)
    y = np.array(y, dtype=np.int32).reshape(-1, 1)
    specs = np.stack(specs)

    # 5th/95th percentile clipping prevents rare amplitude spikes from compressing
    # the useful signal range to a narrow band in the [-1, 1] normalized space.
    p5, p95 = np.percentile(X, 5), np.percentile(X, 95)
    time_min, time_max = float(p5), float(p95)
    X_scaled = np.clip(2 * (X - time_min) / (time_max - time_min) - 1, -1.0, 1.0)

    log_pos = np.log10(np.array(list(pos_map.values())))
    log_pos_scaled = (log_pos - log_pos.min()) / (log_pos.max() - log_pos.min())
    log_pos_map = {idx + 1: log_pos_scaled[idx] for idx in range(3)}
    y_scaled = np.array([log_pos_map[int(p)] for p in y.flatten()]).reshape(-1, 1)

    dataset = TensorDataset(
        torch.from_numpy(y_scaled).float(),
        torch.from_numpy(X_scaled).unsqueeze(1),
        torch.from_numpy(specs).float(),
    )
    meta = {
        "GLOBAL_MIN": time_min,
        "GLOBAL_MAX": time_max,
        "pos_min": float(min(pos_map.values())),
        "pos_max": float(max(pos_map.values())),
        "SEQ_LEN": seq_len,
    }
    return dataset, meta


def get_gan_dataloader(train_dir: str, batch_size: int, seq_len: int = 2500):
    dataset, meta = load_and_prepare_gan_data(train_dir, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader, meta

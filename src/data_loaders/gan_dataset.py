"""GAN dataset loader: loads raw leak signals and prepares tensors for WGAN-GP training."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio.transforms as T
from torch.utils.data import DataLoader, TensorDataset


_DEFAULT_POS_MAP: dict[int, int] = {1: 10, 2: 25, 3: 60}


def denorm_signal(sig_scaled: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    """Invert the [-1, 1] min-max normalisation applied during GAN training.

    Args:
        sig_scaled: Normalised signal in [-1, 1].
        meta: Metadata dict containing ``GLOBAL_MIN`` and ``GLOBAL_MAX``.

    Returns:
        Signal restored to its original amplitude range.
    """
    return (sig_scaled + 1) / 2 * (meta["GLOBAL_MAX"] - meta["GLOBAL_MIN"]) + meta[
        "GLOBAL_MIN"
    ]


def load_and_prepare_gan_data(
    train_dir: str,
    seq_len: int = 2500,
    pos_map: dict[int, int] | None = None,
) -> tuple[TensorDataset, dict[str, Any]]:
    """Load pickle files, build normalised tensors, and return a TensorDataset.

    Each pkl file contains a list of pd.Series (one per recording window).
    Signals are clipped to the 5th/95th percentile range and rescaled to [-1, 1].
    Position labels are log-scaled and normalised to [0, 1].

    Args:
        train_dir: Directory containing ``vallen_{pos}_v1.pkl`` files.
        seq_len: Fixed signal length in samples; shorter signals are zero-padded,
            longer ones are truncated.
        pos_map: Mapping from sensor index (1-based) to distance in metres.
            Defaults to ``{1: 10, 2: 25, 3: 60}`` if not provided.

    Returns:
        Tuple of (dataset, meta) where dataset yields (cond, signal, spectrogram)
        tensors and meta carries normalisation statistics.

    Raises:
        ValueError: If no valid pkl files are found in ``train_dir``.
    """
    if pos_map is None:
        pos_map = _DEFAULT_POS_MAP

    spec_transform = T.Spectrogram(n_fft=256, hop_length=128, power=2)
    X: list[np.ndarray] = []
    y: list[int] = []
    specs: list[np.ndarray] = []

    for pos in sorted(pos_map.keys()):
        file_path = Path(train_dir) / f"vallen_{pos}_v1.pkl"
        if not file_path.exists():
            print(f"Warning: {file_path} not found, skipping.")
            continue
        with open(file_path, "rb") as f:
            series_list = pickle.load(f)

        for s in series_list:
            sig = (
                s.values.astype(np.float32)
                if hasattr(s, "values")
                else np.asarray(s, dtype=np.float32)
            )
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
        raise ValueError(f"No valid training data found in {train_dir!r}.")

    X_arr = np.stack(X)
    y_arr = np.array(y, dtype=np.int32).reshape(-1, 1)
    specs_arr = np.stack(specs)

    # 5th/95th percentile clipping prevents rare amplitude spikes from compressing
    # the useful signal range to a narrow band in the [-1, 1] normalised space.
    p5, p95 = np.percentile(X_arr, 5), np.percentile(X_arr, 95)
    time_min, time_max = float(p5), float(p95)
    X_scaled = np.clip(2 * (X_arr - time_min) / (time_max - time_min) - 1, -1.0, 1.0)

    distances = np.array(list(pos_map.values()), dtype=np.float64)
    log_pos = np.log10(distances)
    log_pos_scaled = (log_pos - log_pos.min()) / (log_pos.max() - log_pos.min())
    log_pos_map = {
        idx: log_pos_scaled[i] for i, idx in enumerate(sorted(pos_map.keys()))
    }
    y_scaled = np.array([log_pos_map[int(p)] for p in y_arr.flatten()]).reshape(-1, 1)

    dataset = TensorDataset(
        torch.from_numpy(y_scaled).float(),
        torch.from_numpy(X_scaled).unsqueeze(1),
        torch.from_numpy(specs_arr).float(),
    )
    meta: dict[str, Any] = {
        "GLOBAL_MIN": time_min,
        "GLOBAL_MAX": time_max,
        "pos_min": float(min(pos_map.values())),
        "pos_max": float(max(pos_map.values())),
        "SEQ_LEN": seq_len,
    }
    return dataset, meta


def get_gan_dataloader(
    train_dir: str,
    batch_size: int,
    seq_len: int = 2500,
    pos_map: dict[int, int] | None = None,
) -> tuple[DataLoader, dict[str, Any]]:
    """Build a shuffled DataLoader from the GAN training directory.

    Args:
        train_dir: Directory containing ``vallen_{pos}_v1.pkl`` files.
        batch_size: Number of samples per batch; incomplete last batch is dropped.
        seq_len: Fixed signal length in samples.
        pos_map: Sensor index → distance mapping (see ``load_and_prepare_gan_data``).

    Returns:
        Tuple of (loader, meta) ready for use in the WGAN training loop.
    """
    dataset, meta = load_and_prepare_gan_data(train_dir, seq_len, pos_map)
    loader: DataLoader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    return loader, meta

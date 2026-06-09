"""Train/test split and persistence utilities for processed sensor signals."""

from __future__ import annotations

import os
import pickle
import random
from pathlib import Path
from typing import Any


def load_pkl_folder(folder_path: str) -> dict[str, list[Any]]:
    """Load all .pkl files in a directory into a dict keyed by file stem.

    Args:
        folder_path: Path to the directory containing .pkl files.

    Returns:
        Mapping of file stem → unpickled object. Returns an empty dict if
        the directory does not exist.
    """
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Warning: folder not found -> {folder_path}")
        return {}
    result: dict[str, Any] = {}
    for f in os.listdir(folder):
        if f.endswith(".pkl"):
            with open(folder / f, "rb") as fh:
                result[f.replace(".pkl", "")] = pickle.load(fh)
    return result


def extract_channel_2(
    leak_data: dict[str, list[Any]],
    normal_data: dict[str, list[Any]] | None = None,
) -> dict[str, list[Any]]:
    """Extract the ``channel_2`` column from raw sensor DataFrames.

    Sensor indices 1–3 are leak signals (``_v1``); indices 1–3 are also
    used for normal signals (``_v0``).

    Args:
        leak_data: Raw leak recordings keyed by sensor bundle name.
        normal_data: Raw normal recordings; may be None.

    Returns:
        Dict mapping ``vallen_{i}_v{tag}`` → list of channel_2 Series.
    """
    sensor_data: dict[str, list[Any]] = {}
    if leak_data:
        for i in range(1, 5):
            key = f"c1_qingcheng_{i}_c2_vallen_{i}"
            if key in leak_data:
                sensor_data[f"vallen_{i}_v1"] = [df["Canal 2"] for df in leak_data[key]]
    if normal_data:
        for i in range(1, 4):
            key = f"c1_qingcheng_{i}_c2_vallen_{i}"
            if key in normal_data:
                sensor_data[f"vallen_{i}_v0"] = [
                    df["Canal 2"] for df in normal_data[key]
                ]
    return sensor_data


def split_and_save(
    sensor_data: dict[str, list[Any]],
    train_dir: str,
    test_dir: str,
    split_ratio: float = 0.8,
    seed: int = 42,
) -> None:
    """Randomly split each signal list and persist both halves as .pkl files.

    Args:
        sensor_data: Mapping of key → list of signal series to split.
        train_dir: Output directory for training-split files.
        test_dir: Output directory for test-split files.
        split_ratio: Fraction of data assigned to train (0 < ratio < 1).
        seed: Random seed for reproducibility.
    """
    Path(train_dir).mkdir(parents=True, exist_ok=True)
    Path(test_dir).mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    for key, series_list in sensor_data.items():
        indices = list(range(len(series_list)))
        random.shuffle(indices)
        split = int(len(series_list) * split_ratio)
        for idxs, out_dir in [
            (indices[:split], train_dir),
            (indices[split:], test_dir),
        ]:
            with open(Path(out_dir) / f"{key}.pkl", "wb") as f:
                pickle.dump([series_list[i] for i in idxs], f)
    print(f"Split {int(split_ratio * 100)}/{100 - int(split_ratio * 100)} done.")

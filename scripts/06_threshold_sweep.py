"""Sweep binary classification thresholds using held-out test pkl files.

Reads test pkl directly (bypassing saved spectrograms) to avoid any
contamination between the real_test evaluation set and the sweep itself.
Prints recall on pos_60m vs false-positive rate on normal signals for
each threshold; marks combinations that meet ≥70% recall and ≤50% FP.

Usage:
    python scripts/06_threshold_sweep.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from matplotlib import cm
from PIL import Image
from scipy.signal import spectrogram as scipy_spg
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.core.config_parser import load_config
from src.data_prep.spectro_maker import get_freq_mask
from src.models.cascade_vit import BinaryViT


class InMemorySpectroDataset(Dataset):
    """In-memory dataset that wraps pre-computed PIL images for inference."""

    def __init__(self, images: list[Image.Image], transform: transforms.Compose) -> None:
        self.images = images
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.transform(self.images[idx]), torch.tensor(0)


def signals_to_images(
    signals: list,
    fs: int,
    nperseg: int,
    noverlap: int,
    mask: np.ndarray,
    img_size: int,
    vmin: float,
    vmax: float,
) -> list[Image.Image]:
    """Convert raw signals to resized grayscale PIL images via log-spectrogram."""
    images: list[Image.Image] = []
    for sig in signals:
        arr = sig.values if hasattr(sig, "values") else np.asarray(sig)
        _, _, Sxx = scipy_spg(arr, fs=fs, nperseg=nperseg, noverlap=noverlap)
        S = Sxx[mask, :]
        if np.isnan(S).any():
            continue
        S_log = np.flipud(10 * np.log10(S + 1e-12))
        norm = np.clip((S_log - vmin) / (vmax - vmin), 0, 1)
        rgb = (cm.magma(norm)[..., :3] * 255).astype(np.uint8)
        img = (
            Image.fromarray(rgb)
            .resize((img_size, img_size), Image.LANCZOS)
            .convert("L")
        )
        images.append(img)
    return images


def run_sweep(pct: int) -> None:
    """Run threshold sweep for a given train/test split percentage."""
    cfg = load_config(f"configs/exp_train_{pct}.yaml")
    spc = cfg["spectro"]
    ev = cfg["eval_model"]
    vp = ev["vit_params"]

    scale_file = Path(f"data/spectrograms/scale_{pct}pct.json")
    if not scale_file.exists():
        print(f"  [{pct}%] scale file missing — run 03_generate_specs.py first")
        return

    scale = json.loads(scale_file.read_text())
    vmin, vmax = scale["vmin"], scale["vmax"]

    mask = get_freq_mask(spc["fs"], spc["nperseg"], spc["f_min"], spc["f_max"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bin_model = BinaryViT(
        ev["img_size"],
        vp["patch_size"],
        vp["emb_dim"],
        vp["n_layers_bin"],
        vp["n_heads"],
    ).to(device)
    model_path = Path(f"outputs/evaluation/{pct}pct/cascade/bin_model.pth")
    if not model_path.exists():
        print(f"  [{pct}%] model not found")
        return
    bin_model.load_state_dict(torch.load(model_path, map_location=device))
    bin_model.eval()

    val_tf = transforms.Compose(
        [
            transforms.Resize(ev["img_size"]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    pos_map: dict[int, int] = cfg["data"]["pos_map"]
    idx_60m = next(k for k, v in pos_map.items() if v == 60)
    test_pkl = Path(cfg["data"]["test_dir"]) / f"vallen_{idx_60m}_v1.pkl"
    with open(test_pkl, "rb") as f:
        test_60m = pickle.load(f)

    train_pkl_norm = Path(cfg["data"]["train_dir"]) / "vallen_1_v0.pkl"
    with open(train_pkl_norm, "rb") as f:
        all_norm = pickle.load(f)
    norm_pool = all_norm[: spc["n_norm_samples"]]

    print(f"\n{'=' * 55}")
    print(
        f"{pct}% split — {len(test_60m)} test-60m signals | scale [{vmin:.1f},{vmax:.1f}] dB"
    )

    def collect_probs(signals: list) -> list[float]:
        imgs = signals_to_images(
            signals,
            spc["fs"],
            spc["nperseg"],
            spc["noverlap"],
            mask,
            ev["img_size"],
            vmin,
            vmax,
        )
        if not imgs:
            return []
        loader = DataLoader(
            InMemorySpectroDataset(imgs, val_tf), ev["batch_size"], num_workers=0
        )
        probs: list[float] = []
        with torch.no_grad():
            for X, _ in loader:
                probs.extend(
                    torch.softmax(bin_model(X.to(device)), dim=1)[:, 1].cpu().tolist()
                )
        return probs

    p60 = collect_probs(test_60m)
    p_norm = collect_probs(norm_pool)

    print(f"{'t':>5}  {'60m recall':>10}  {'norm FP':>10}")
    for t in np.arange(0.05, 0.96, 0.01):
        r = 100 * sum(1 for p in p60 if p > t) / len(p60)
        f = 100 * sum(1 for p in p_norm if p > t) / len(p_norm)
        flag = " [OK]" if r >= 70 and f <= 50 else ""
        if flag or (t <= 0.55 and (f <= 60 or r >= 50)):
            print(f"  t={t:.2f}  {r:9.1f}%  {f:9.1f}%{flag}")


if __name__ == "__main__":
    for pct in [80, 65, 50, 35, 20]:
        run_sweep(pct)

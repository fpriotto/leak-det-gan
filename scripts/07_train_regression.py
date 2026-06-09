"""Train and evaluate the cascade ViT regressor for all data splits.

Two-stage inference: BinaryViT (normal vs leak) → LeakRegressorViT (distance in metres).
The regressor output is normalised to [0, 1] with POS_MIN=10 and POS_MAX=60.

Usage:
    python scripts/07_train_regression.py
    python scripts/07_train_regression.py --splits 50 80
    python scripts/07_train_regression.py --eval-only --splits 80
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.core.config_parser import load_config
from src.data_loaders.vit_dataset import get_transforms
from src.models.cascade_vit import BinaryViT, LeakRegressorViT
from src.training.losses import FocalLoss

_SEED = 42
_WANTED_SORTED = ["pos_10m", "pos_20m", "pos_30m", "pos_40m", "pos_50m", "pos_60m"]
_DIST_M: dict[str, float] = {
    "pos_10m": 10.0,
    "pos_20m": 20.0,
    "pos_30m": 30.0,
    "pos_40m": 40.0,
    "pos_50m": 50.0,
    "pos_60m": 60.0,
}
_POS_MIN = 10.0
_POS_MAX = 60.0
_ALL_SPLITS = [20, 35, 50, 65, 80]


def _norm(m: float) -> float:
    return (m - _POS_MIN) / (_POS_MAX - _POS_MIN)


def _denorm(n: float) -> float:
    return n * (_POS_MAX - _POS_MIN) + _POS_MIN


class _RegressionDataset(Dataset):
    """Loads spectrogram PNGs; returns (image, float_label) pairs.

    Args:
        samples: List of (file_path, float_label) pairs.
        transform: Torchvision transform pipeline.
    """

    def __init__(
        self, samples: list[tuple[str, float]], transform: transforms.Compose
    ) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, lab = self.samples[idx]
        img = self.transform(Image.open(path).convert("L"))
        return img, torch.tensor(lab, dtype=torch.float32)


class _Splits(NamedTuple):
    train_bin: list[tuple[str, int]]
    val_bin: list[tuple[str, int]]
    train_reg: list[tuple[str, float]]
    val_reg: list[tuple[str, float]]
    test_reg: list[tuple[str, float]]
    test_bin: list[tuple[str, int]]


def _build_splits(cfg: dict, pct: int) -> _Splits:
    """Load generated + normal spectrograms and build all sample lists."""
    eval_cfg = cfg["eval_model"]
    n_test_norm = eval_cfg.get("n_test_norm", 150)
    spec_root = cfg["data"].get("spec_root", "data/spectrograms")

    gen_root = Path(f"{spec_root}/generated_{pct}pct")
    normal_root = Path(f"{spec_root}/real/normal")

    # Leak samples with normalised float distance label
    leak_samples: list[tuple[str, float]] = []
    for cls in _WANTED_SORTED:
        folder = gen_root / cls
        if not folder.exists():
            continue
        for p in sorted(folder.glob("*.png")):
            leak_samples.append((str(p), _norm(_DIST_M[cls])))

    print(f"[{pct}%] Generated leak samples: {len(leak_samples)}")
    print(f"  Distribution: {Counter([round(_denorm(lab)) for _, lab in leak_samples])}")

    labels = [lab for _, lab in leak_samples]
    # Stratify on rounded distance to preserve class balance
    strat = [round(_denorm(lab)) for lab in labels]
    X_tr, X_tmp, _, s_tmp = train_test_split(
        leak_samples, strat, test_size=0.30, stratify=strat, random_state=_SEED
    )
    X_val, X_te_leak, _, _ = train_test_split(
        X_tmp, s_tmp, test_size=0.50, stratify=s_tmp, random_state=_SEED
    )

    normal_samples = [(str(p), -1.0) for p in sorted(normal_root.glob("*.png"))]
    rng = np.random.RandomState(_SEED)
    rng.shuffle(normal_samples)
    n = len(normal_samples) // 3
    tr_norm = normal_samples[:n]
    va_norm = normal_samples[n : n * 2]
    te_norm = normal_samples[n * 2 : n * 2 + n_test_norm]

    # Binary lists: 1=leak, 0=normal
    def to_bin(lst: list[tuple[str, float]]) -> list[tuple[str, int]]:
        return [(p, 1 if lab >= 0 else 0) for p, lab in lst]

    train_bin = to_bin(X_tr + tr_norm)
    val_bin = to_bin(X_val + va_norm)
    test_bin = to_bin(X_te_leak + te_norm)

    # Regression lists: only leak samples
    train_reg = X_tr
    val_reg = X_val
    test_reg = X_te_leak

    return _Splits(train_bin, val_bin, train_reg, val_reg, test_reg, test_bin)


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScaler | None = None,
    *,
    training: bool = True,
    regression: bool = False,
) -> tuple[float, float | None]:
    """One epoch of training or validation.

    Returns:
        (mean_loss, accuracy_or_None) — accuracy is None for regression.
    """
    model.train(training)
    total_loss = total_correct = total = 0

    for X, y in tqdm(loader, desc="train" if training else "val", leave=False):
        X, y = X.to(device), y.to(device)
        if training and scaler is not None:
            optimizer.zero_grad()
            with autocast(device_type=device.type):
                out = model(X)
                loss = criterion(out, y) if regression else criterion(out, y.long())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.no_grad():
                out = model(X)
                loss = criterion(out, y) if regression else criterion(out, y.long())

        total_loss += loss.item() * X.size(0)
        if not regression:
            total_correct += (out.argmax(1) == y.long()).sum().item()
        total += X.size(0)

    acc = (total_correct / total) if not regression else None
    return total_loss / total, acc


def _evaluate_regression_generated(
    model_bin: nn.Module,
    model_reg: nn.Module,
    test_reg: list[tuple[str, float]],
    test_bin: list[tuple[str, int]],
    val_tf: transforms.Compose,
    cfg: dict,
    pct: int,
    out_dir: Path,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate cascade on generated test split; save boxplot + metrics JSON."""
    batch_size = cfg["eval_model"]["batch_size"]
    num_workers = cfg["eval_model"].get("num_workers", 4)
    threshold = cfg["eval_model"].get("bin_threshold", 0.5)

    model_bin.eval()
    model_reg.eval()

    # Collect all samples: leak (has float label) + normal
    combined = [(p, lab) for p, lab in test_reg] + [
        (p, -1.0) for p, _ in test_bin if _ == 0
    ]
    loader = DataLoader(
        _RegressionDataset(combined, val_tf),
        batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    y_true_m: list[float] = []
    y_pred_m: list[float] = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            probs_bin = torch.softmax(model_bin(X), dim=1)[:, 1]
            pred_leak = (probs_bin > threshold).cpu()
            y_np = y.numpy()
            for i in range(len(y_np)):
                true_raw = float(y_np[i])
                if true_raw < 0:
                    continue  # skip normal in regression eval
                true_m = _denorm(true_raw)
                if pred_leak[i]:
                    pred_norm = model_reg(X[i : i + 1]).item()
                    pred_m = _denorm(pred_norm)
                else:
                    pred_m = float("nan")
                y_true_m.append(true_m)
                y_pred_m.append(pred_m)

    # Filter out NaN (false negatives) for metric computation
    valid = [(t, p) for t, p in zip(y_true_m, y_pred_m) if not np.isnan(p)]
    if not valid:
        print(f"[{pct}%] No valid regression predictions for generated test set.")
        return {}

    y_t = np.array([v[0] for v in valid])
    y_p = np.array([v[1] for v in valid])

    mae_global = float(np.mean(np.abs(y_t - y_p)))
    rmse_global = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
    print(f"\n[{pct}%] Generated test — MAE: {mae_global:.2f} m | RMSE: {rmse_global:.2f} m")

    metrics: dict[str, float] = {"mae_global": mae_global, "rmse_global": rmse_global}
    print(f"  {'Distance':>10}  {'MAE':>8}  {'RMSE':>8}  {'N':>6}")
    for dist in sorted(set(y_t)):
        mask = y_t == dist
        mae_d = float(np.mean(np.abs(y_t[mask] - y_p[mask])))
        rmse_d = float(np.sqrt(np.mean((y_t[mask] - y_p[mask]) ** 2)))
        n_d = int(mask.sum())
        print(f"  {dist:>9.0f}m  {mae_d:>8.2f}  {rmse_d:>8.2f}  {n_d:>6}")
        metrics[f"mae_{int(dist)}m"] = mae_d
        metrics[f"rmse_{int(dist)}m"] = rmse_d

    (out_dir / "regression_metrics_gen.json").write_text(json.dumps(metrics, indent=2))

    return metrics


def _evaluate_regression_real(
    model_bin: nn.Module,
    model_reg: nn.Module,
    val_tf: transforms.Compose,
    cfg: dict,
    pct: int,
    out_dir: Path,
    device: torch.device,
) -> None:
    """Evaluate cascade on real test spectrograms; save boxplot."""
    batch_size = cfg["eval_model"]["batch_size"]
    num_workers = cfg["eval_model"].get("num_workers", 4)
    threshold = cfg["eval_model"].get("bin_threshold", 0.5)
    real_root = Path("data/spectrograms/real_test")

    model_bin.eval()
    model_reg.eval()

    if not real_root.exists():
        print(f"[{pct}%] real_test directory not found, skipping real evaluation.")
        return

    y_true_m: list[float] = []
    y_pred_m: list[float] = []
    row_names: list[str] = []

    for pos_dir in sorted(p for p in real_root.iterdir() if p.is_dir()):
        imgs = sorted(pos_dir.glob("*.png"))
        if not imgs:
            continue
        dist_str = pos_dir.name  # e.g. "pos_10m"
        true_m = float(dist_str.split("_")[1][:-1])
        row_names.append(dist_str)

        samples = [(str(p), 0.0) for p in imgs]
        loader = DataLoader(
            _RegressionDataset(samples, val_tf),
            batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        preds_m: list[float] = []
        with torch.no_grad():
            for X, _ in loader:
                X = X.to(device)
                probs_bin = torch.softmax(model_bin(X), dim=1)[:, 1]
                pred_leak = probs_bin > threshold
                for i in range(X.size(0)):
                    if pred_leak[i]:
                        pred_norm = model_reg(X[i : i + 1]).item()
                        preds_m.append(_denorm(pred_norm))
                    else:
                        preds_m.append(float("nan"))

        n_leak = sum(1 for p in preds_m if not np.isnan(p))
        mae = float(np.nanmean(np.abs(np.array(preds_m) - true_m)))
        print(
            f"  [{dist_str}] {n_leak}/{len(imgs)} classified as leak | "
            f"MAE (leak only): {mae:.2f} m"
        )
        y_true_m.extend([true_m] * len(preds_m))
        y_pred_m.extend(preds_m)

    valid = [(t, p) for t, p in zip(y_true_m, y_pred_m) if not np.isnan(p)]
    if not valid:
        return


def _run_split(pct: int, *, config: str | None = None, eval_only: bool = False) -> None:
    """Full training + evaluation pipeline for one split percentage."""
    random.seed(_SEED)
    np.random.seed(_SEED)
    torch.manual_seed(_SEED)

    cfg = load_config(config or f"configs/exp_train_{pct}.yaml")
    eval_cfg = cfg["eval_model"]
    vp = eval_cfg["vit_params"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = eval_cfg["batch_size"]
    num_workers = eval_cfg.get("num_workers", 4)

    print(f"\n{'=' * 60}")
    print(f"  Regression | split {pct}% | device: {device}")
    print(f"{'=' * 60}")

    _, val_tf = get_transforms(eval_cfg["img_size"])

    _base = eval_cfg.get("eval_out_dir", f"outputs/evaluation/{pct}pct")
    out_dir = Path(_base) / "regression"
    out_dir.mkdir(parents=True, exist_ok=True)

    bin_model = BinaryViT(
        eval_cfg["img_size"],
        vp["patch_size"],
        vp["emb_dim"],
        vp["n_layers_bin"],
        vp["n_heads"],
    ).to(device)
    reg_model = LeakRegressorViT(
        eval_cfg["img_size"],
        vp["patch_size"],
        vp["emb_dim"],
        vp["n_layers_reg"],
        vp["n_heads"],
    ).to(device)

    if eval_only:
        bin_model.load_state_dict(
            torch.load(out_dir / "bin_model.pth", map_location=device)
        )
        reg_model.load_state_dict(
            torch.load(out_dir / "reg_model.pth", map_location=device)
        )
        print(f"[{pct}%] Loaded models from {out_dir}")
    else:
        splits = _build_splits(cfg, pct)
        train_tf, _ = get_transforms(eval_cfg["img_size"])

        dl_bin_tr = DataLoader(
            _RegressionDataset(splits.train_bin, train_tf),  # type: ignore[arg-type]
            batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        dl_bin_va = DataLoader(
            _RegressionDataset(splits.val_bin, val_tf),  # type: ignore[arg-type]
            batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        dl_reg_tr = DataLoader(
            _RegressionDataset(splits.train_reg, train_tf),
            batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        dl_reg_va = DataLoader(
            _RegressionDataset(splits.val_reg, val_tf),
            batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        gamma = eval_cfg.get("focal_gamma", 2.0)
        crit_bin = FocalLoss(gamma=gamma)
        crit_reg = nn.SmoothL1Loss(beta=0.1)  # beta=0.1 in [0,1] normalised ≈ 5 m
        opt_bin = optim.AdamW(bin_model.parameters(), lr=eval_cfg["lr_bin"])
        opt_reg = optim.AdamW(
            reg_model.parameters(), lr=eval_cfg["lr_reg"], weight_decay=1e-5
        )
        scaler = GradScaler(device=device.type)
        sched_bin = optim.lr_scheduler.CosineAnnealingLR(
            opt_bin, T_max=eval_cfg["epochs"], eta_min=1e-5
        )
        sched_reg = optim.lr_scheduler.CosineAnnealingLR(
            opt_reg, T_max=eval_cfg["epochs"], eta_min=1e-5
        )

        for ep in range(1, eval_cfg["epochs"] + 1):
            print(f"\n  Epoch {ep}/{eval_cfg['epochs']}")
            loss_bt, acc_bt = _train_epoch(
                bin_model, dl_bin_tr, opt_bin, crit_bin, device, scaler, training=True
            )
            loss_bv, acc_bv = _train_epoch(
                bin_model, dl_bin_va, opt_bin, crit_bin, device, training=False
            )
            print(
                f"  [BIN] train loss={loss_bt:.4f} acc={acc_bt:.3f} | "
                f"val loss={loss_bv:.4f} acc={acc_bv:.3f}"
            )
            loss_rt, _ = _train_epoch(
                reg_model,
                dl_reg_tr,
                opt_reg,
                crit_reg,
                device,
                scaler,
                training=True,
                regression=True,
            )
            loss_rv, _ = _train_epoch(
                reg_model,
                dl_reg_va,
                opt_reg,
                crit_reg,
                device,
                training=False,
                regression=True,
            )
            # Scale L1 loss back to metres for readability
            print(
                f"  [REG] train L1={loss_rt * 50:.2f} m | val L1={loss_rv * 50:.2f} m"
            )
            sched_bin.step()
            sched_reg.step()

        torch.save(bin_model.state_dict(), out_dir / "bin_model.pth")
        torch.save(reg_model.state_dict(), out_dir / "reg_model.pth")
        print(f"[{pct}%] Models saved to {out_dir}")

        # Grab test splits before losing scope
        test_reg = splits.test_reg
        test_bin = splits.test_bin

    if eval_only:
        # Rebuild test splits for evaluation
        splits = _build_splits(cfg, pct)
        test_reg = splits.test_reg
        test_bin = splits.test_bin

    _evaluate_regression_generated(
        bin_model, reg_model, test_reg, test_bin, val_tf, cfg, pct, out_dir, device
    )
    _evaluate_regression_real(bin_model, reg_model, val_tf, cfg, pct, out_dir, device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits",
        nargs="+",
        type=int,
        default=_ALL_SPLITS,
        help="Which split percentages to run (default: all 5)",
    )
    parser.add_argument("--config", default=None, help="Path to a single config file")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    for pct in args.splits:
        if pct not in _ALL_SPLITS:
            print(f"[WARN] {pct}% not a valid split — skipping.")
            continue
        _run_split(pct, config=args.config, eval_only=args.eval_only)


if __name__ == "__main__":
    main()

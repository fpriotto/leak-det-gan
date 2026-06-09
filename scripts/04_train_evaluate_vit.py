"""Train and evaluate the cascade ViT classifier for a given data split.

Two-stage inference: BinaryViT (normal vs leak) → LeakPosViT (6 positions).
Saves confusion matrices and model weights under outputs/evaluation/{pct}pct/.

Usage:
    python scripts/04_train_evaluate_vit.py --config configs/exp_train_50.yaml
    python scripts/04_train_evaluate_vit.py --config configs/exp_train_50.yaml --eval-only
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.core.config_parser import load_config
from src.data_loaders.vit_dataset import SpectroDataset, get_transforms
from src.models.cascade_vit import BinaryViT, LeakPosViT
from src.training.losses import FocalLoss

_SEED = 42
_WANTED_POSITIONS = frozenset(
    {"pos_10m", "pos_20m", "pos_30m", "pos_40m", "pos_50m", "pos_60m"}
)
_WANTED_SORTED = sorted(_WANTED_POSITIONS, key=lambda x: int(x.split("_")[1][:-1]))


class _DataSplits(NamedTuple):
    train_samples: list[tuple[str, int]]
    val_samples: list[tuple[str, int]]
    test_samples: list[tuple[str, int]]
    wanted2new: dict[str, int]


def _build_samples(cfg: dict, pct: int) -> _DataSplits:
    """Build train/val/test sample lists from generated and real spectrograms.

    Returns:
        _DataSplits named tuple with train_samples, val_samples, test_samples,
        and wanted2new label mapping.
    """
    eval_cfg = cfg["eval_model"]
    wanted2new = {cls: i for i, cls in enumerate(_WANTED_SORTED)}
    n_pos = len(_WANTED_POSITIONS)
    n_test_norm = eval_cfg.get("n_test_norm", 150)
    spec_root = cfg["data"].get("spec_root", "data/spectrograms")

    gen_root = Path(f"{spec_root}/generated_{pct}pct")
    normal_root = Path(f"{spec_root}/real/normal")

    samples_gen: list[tuple[str, int]] = []
    for cls in _WANTED_SORTED:
        folder = gen_root / cls
        if folder.exists():
            for p in folder.glob("*.png"):
                samples_gen.append((str(p), wanted2new[cls]))

    print(f"Generated samples (6 positions): {len(samples_gen)}")
    print("Distribution:", Counter([lbl for _, lbl in samples_gen]))

    labels_gen = [lbl for _, lbl in samples_gen]
    X_tr, X_tmp, _, y_tmp = train_test_split(
        samples_gen, labels_gen, test_size=0.30, stratify=labels_gen, random_state=_SEED
    )
    X_val, X_te_gen, _, _ = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=_SEED
    )

    samples_norm = [(str(p), n_pos) for p in sorted(normal_root.glob("*.png"))]
    rng = np.random.RandomState(_SEED)
    rng.shuffle(samples_norm)
    n = len(samples_norm) // 3
    X_tr_norm = samples_norm[:n]
    X_val_norm = samples_norm[n : n * 2]
    X_te_norm = samples_norm[n * 2 : n * 2 + n_test_norm]

    train_samples = X_tr + X_tr_norm
    val_samples = X_val + X_val_norm
    test_samples = X_te_gen + X_te_norm

    if eval_cfg.get("inject_real_60m", False):
        real_train_root = Path(f"{spec_root}/real_train_{pct}pct")
        if real_train_root.exists():
            for cls, idx in wanted2new.items():
                real_dir = real_train_root / cls
                if not real_dir.exists():
                    continue
                imgs = sorted(real_dir.glob("*.png"))
                n_tr = int(0.70 * len(imgs))
                n_val = int(0.15 * len(imgs))
                train_samples += [(str(p), idx) for p in imgs[:n_tr]]
                val_samples += [(str(p), idx) for p in imgs[n_tr : n_tr + n_val]]
                print(f"  [inject] {cls}: +{n_tr} train, +{n_val} val")
            print(
                f"After injection — train: {len(train_samples)}, val: {len(val_samples)}"
            )

    return _DataSplits(train_samples, val_samples, test_samples, wanted2new)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: GradScaler | None = None,
    *,
    training: bool = True,
) -> tuple[float, float]:
    """Run one epoch of training or validation.

    Returns:
        Mean loss and accuracy over the epoch.
    """
    model.train(training)
    total_loss = total_correct = total = 0

    for X, y in tqdm(loader, desc="train" if training else "val", leave=False):
        X, y = X.to(device), y.to(device)
        if training and scaler is not None:
            optimizer.zero_grad()
            with autocast(device_type=device.type):
                out = model(X)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.no_grad():
                out = model(X)
                loss = criterion(out, y)

        total_loss += loss.item() * X.size(0)
        total_correct += (out.argmax(1) == y).sum().item()
        total += X.size(0)

    return total_loss / total, total_correct / total


def predict_cascade(
    x: torch.Tensor,
    model_bin: nn.Module,
    model_pos: nn.Module,
    n_pos: int,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Two-stage cascade: binary gate then positional classifier."""
    with torch.no_grad():
        probs_bin = torch.softmax(model_bin(x), dim=1)
        pred_bin = (probs_bin[:, 1] > threshold).long()
        pred_final = torch.full_like(pred_bin, fill_value=n_pos)
        if (pred_bin == 1).any():
            pred_final[pred_bin == 1] = model_pos(x[pred_bin == 1]).argmax(1)
    return pred_final


def evaluate_generated(
    samples: list[tuple[str, int]],
    label_names: list[str],
    bin_model: nn.Module,
    pos_model: nn.Module,
    n_pos: int,
    val_tf: transforms.Compose,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    threshold: float,
    pct: int,
    out_dir: Path | None = None,
) -> None:
    """Evaluate cascade on generated test split and save confusion matrix."""
    bin_model.eval()
    pos_model.eval()
    loader = DataLoader(
        SpectroDataset(samples, val_tf),
        batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for X, y in loader:
            pred = predict_cascade(
                X.to(device), bin_model, pos_model, n_pos, threshold=threshold
            )
            y_true.extend(y.tolist())
            y_pred.extend(pred.cpu().tolist())

    print("\n=== Cascade evaluation — generated test split ===")
    print(classification_report(y_true, y_pred, target_names=label_names, digits=3))

    cm = confusion_matrix(y_true, y_pred, labels=range(len(label_names)))
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
    )
    plt.title(f"Confusion matrix — generated data | {pct}% train")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    base = out_dir if out_dir is not None else Path(f"outputs/evaluation/{pct}pct")
    out_path = base / "plots" / "confusion_generated.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved: {out_path}")


def evaluate_real(
    base_path: Path,
    label_names: list[str],
    wanted2new: dict[str, int],
    bin_model: nn.Module,
    pos_model: nn.Module,
    n_pos: int,
    val_tf: transforms.Compose,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    threshold: float,
    pct: int,
    out_dir: Path | None = None,
) -> None:
    """Evaluate cascade on real test spectrograms and save confusion matrix."""
    bin_model.eval()
    pos_model.eval()

    pos_dirs = sorted(p for p in base_path.iterdir() if p.is_dir())
    row_names: list[str] = []
    cm_rows: list[list[int]] = []
    y_true_trained: list[int] = []
    y_pred_trained: list[int] = []

    for pos_dir in pos_dirs:
        imgs = sorted(pos_dir.glob("*.png"))
        if not imgs:
            continue
        loader = DataLoader(
            SpectroDataset([(str(p), 0) for p in imgs], val_tf),
            batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        preds: list[int] = []
        with torch.no_grad():
            for X, _ in loader:
                preds.extend(
                    predict_cascade(
                        X.to(device), bin_model, pos_model, n_pos, threshold=threshold
                    )
                    .cpu()
                    .tolist()
                )

        counts = [preds.count(c) for c in range(len(label_names))]
        n_leak = sum(1 for p in preds if p != n_pos)
        n_normal = sum(1 for p in preds if p == n_pos)
        print(
            f"  [{pos_dir.name}] binary: {n_leak}/{len(imgs)} LEAK, {n_normal}/{len(imgs)} NORMAL"
        )

        is_trained = pos_dir.name in wanted2new
        row_names.append(pos_dir.name)
        cm_rows.append(counts)

        if is_trained:
            lbl = wanted2new[pos_dir.name]
            y_true_trained.extend([lbl] * len(imgs))
            y_pred_trained.extend(preds)

    print("\n=== Cascade evaluation — real test data (trained positions) ===")
    trained_labels = sorted(set(y_true_trained))
    trained_names = [label_names[lbl] for lbl in trained_labels]
    print(
        classification_report(
            y_true_trained,
            y_pred_trained,
            labels=trained_labels,
            target_names=trained_names,
            digits=3,
            zero_division=0,
        )
    )

    cm_array = np.array(cm_rows)
    plt.figure(figsize=(12, max(3, len(row_names) * 1.5)))
    sns.heatmap(
        cm_array,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=row_names,
    )
    plt.title(f"Confusion matrix — real data | {pct}% train")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    base = out_dir if out_dir is not None else Path(f"outputs/evaluation/{pct}pct")
    out_path = base / "plots" / "confusion_real.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved: {out_path}")


def _collect_leak_probs(
    folder: Path,
    bin_model: nn.Module,
    val_tf: transforms.Compose,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[list[float], int]:
    """Collect binary leak probabilities for all PNGs in a folder."""
    imgs = sorted(folder.glob("*.png"))
    if not imgs:
        return [], 0
    loader = DataLoader(
        SpectroDataset([(str(p), 0) for p in imgs], val_tf),
        batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    probs: list[float] = []
    with torch.no_grad():
        for X, _ in loader:
            p = torch.softmax(bin_model(X.to(device)), dim=1)[:, 1]
            probs.extend(p.cpu().tolist())
    return probs, len(imgs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp_train_50.yaml")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    random.seed(_SEED)
    np.random.seed(_SEED)
    torch.manual_seed(_SEED)

    cfg = load_config(args.config)
    eval_cfg = cfg["eval_model"]
    pct = int(round(cfg["data"]["split_ratio"] * 100))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = eval_cfg["batch_size"]
    num_workers = eval_cfg.get("num_workers", 4)
    threshold = eval_cfg.get("bin_threshold", 0.5)
    n_pos = len(_WANTED_POSITIONS)

    print(f"\nArchitecture: ViT | device: {device} | split: {pct}%")

    train_tf, val_tf = get_transforms(eval_cfg["img_size"])
    train_samples, val_samples, test_samples, wanted2new = _build_samples(cfg, pct)

    def _bin(lbl: int) -> int:
        return 1 if lbl < n_pos else 0  # leak=1, normal=0

    train_bin = [(p, _bin(lbl)) for p, lbl in train_samples]
    val_bin = [(p, _bin(lbl)) for p, lbl in val_samples]
    train_pos = [(p, lbl) for p, lbl in train_samples if lbl != n_pos]
    val_pos = [(p, lbl) for p, lbl in val_samples if lbl != n_pos]

    print(f"\nTrain — bin: {len(train_bin)} | pos: {len(train_pos)}")
    print(f"Val   — bin: {len(val_bin)}   | pos: {len(val_pos)}")
    print("Train pos distribution:", Counter([lbl for _, lbl in train_pos]))

    dl_bin_train = DataLoader(
        SpectroDataset(train_bin, train_tf),
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    dl_bin_val = DataLoader(
        SpectroDataset(val_bin, val_tf),
        batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    dl_pos_train = DataLoader(
        SpectroDataset(train_pos, train_tf),
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    dl_pos_val = DataLoader(
        SpectroDataset(val_pos, val_tf),
        batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    vp = eval_cfg["vit_params"]
    bin_model = BinaryViT(
        eval_cfg["img_size"],
        vp["patch_size"],
        vp["emb_dim"],
        vp["n_layers_bin"],
        vp["n_heads"],
    ).to(device)
    pos_model = LeakPosViT(
        eval_cfg["img_size"],
        vp["patch_size"],
        vp["emb_dim"],
        vp["n_layers_bin"],
        vp["n_heads"],
        n_pos=n_pos,
    ).to(device)

    _out = eval_cfg.get("eval_out_dir")
    model_dir = (Path(_out) / "cascade") if _out else Path(f"outputs/evaluation/{pct}pct/cascade")

    if args.eval_only:
        bin_model.load_state_dict(
            torch.load(model_dir / "bin_model.pth", map_location=device)
        )
        pos_model.load_state_dict(
            torch.load(model_dir / "pos_model.pth", map_location=device)
        )
        print(f"[eval-only] Loaded models from {model_dir}")
    else:
        gamma = eval_cfg.get("focal_gamma", 2.0)
        crit_bin = FocalLoss(gamma=gamma)
        crit_pos = FocalLoss(gamma=gamma)
        opt_bin = optim.AdamW(bin_model.parameters(), lr=eval_cfg["lr_bin"])
        opt_pos = optim.AdamW(
            pos_model.parameters(), lr=eval_cfg["lr_reg"], weight_decay=1e-5
        )
        scaler = GradScaler(device=device.type)
        sched_bin = optim.lr_scheduler.CosineAnnealingLR(
            opt_bin, T_max=eval_cfg["epochs"], eta_min=1e-5
        )
        sched_pos = optim.lr_scheduler.CosineAnnealingLR(
            opt_pos, T_max=eval_cfg["epochs"], eta_min=1e-5
        )

        for ep in range(1, eval_cfg["epochs"] + 1):
            print(f"\n=== Epoch {ep}/{eval_cfg['epochs']} ===")
            loss_bt, acc_bt = train_epoch(
                bin_model,
                dl_bin_train,
                opt_bin,
                crit_bin,
                device,
                scaler,
                training=True,
            )
            loss_bv, acc_bv = train_epoch(
                bin_model, dl_bin_val, opt_bin, crit_bin, device, training=False
            )
            print(
                f"[BIN] train loss={loss_bt:.4f} acc={acc_bt:.3f} | val loss={loss_bv:.4f} acc={acc_bv:.3f}"
            )

            loss_pt, acc_pt = train_epoch(
                pos_model,
                dl_pos_train,
                opt_pos,
                crit_pos,
                device,
                scaler,
                training=True,
            )
            loss_pv, acc_pv = train_epoch(
                pos_model, dl_pos_val, opt_pos, crit_pos, device, training=False
            )
            print(
                f"[POS] train loss={loss_pt:.4f} acc={acc_pt:.3f} | val loss={loss_pv:.4f} acc={acc_pv:.3f}"
            )

            sched_bin.step()
            sched_pos.step()

        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(bin_model.state_dict(), model_dir / "bin_model.pth")
        torch.save(pos_model.state_dict(), model_dir / "pos_model.pth")
        print(f"Models saved to {model_dir}")

    # ── Evaluation ────────────────────────────────────────────────────────────
    final_classes = _WANTED_SORTED + ["normal"]
    spec_root = cfg["data"].get("spec_root", "data/spectrograms")
    real_root = Path(f"{spec_root}/real_test")
    eval_out_dir = (
        Path(eval_cfg["eval_out_dir"])
        if "eval_out_dir" in eval_cfg
        else Path(f"outputs/evaluation/{pct}pct")
    )

    evaluate_generated(
        test_samples,
        final_classes,
        bin_model,
        pos_model,
        n_pos,
        val_tf,
        batch_size,
        num_workers,
        device,
        threshold,
        pct,
        out_dir=eval_out_dir,
    )
    evaluate_real(
        real_root,
        final_classes,
        wanted2new,
        bin_model,
        pos_model,
        n_pos,
        val_tf,
        batch_size,
        num_workers,
        device,
        threshold,
        pct,
        out_dir=eval_out_dir,
    )

    # ── Threshold sweep ───────────────────────────────────────────────────────
    bin_model.eval()
    norm_dir = Path(f"{spec_root}/real/normal")
    pos60_probs, n60 = _collect_leak_probs(
        real_root / "pos_60m", bin_model, val_tf, batch_size, num_workers, device
    )
    norm_probs, n_norm = _collect_leak_probs(
        norm_dir, bin_model, val_tf, batch_size, num_workers, device
    )

    if pos60_probs:
        print(
            "\n=== Threshold sweep — pos_60m recall vs normal false-positive rate ==="
        )
        print(f"  {'t':>5}  {'60m LEAK%':>10}  {'norm FP%':>10}")
        for t in np.arange(0.05, 0.55, 0.05):
            r60 = 100 * sum(1 for p in pos60_probs if p > t) / n60
            fp_n = (
                100 * sum(1 for p in norm_probs if p > t) / n_norm
                if n_norm
                else float("nan")
            )
            print(f"  t={t:.2f}  {r60:9.1f}%  {fp_n:9.1f}%")


if __name__ == "__main__":
    main()

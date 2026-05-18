import sys
import random
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.core.config_parser import load_config
from src.models.cascade_vit import BinaryViT, LeakPosViT
from src.data_loaders.vit_dataset import get_transforms, SpectroDataset

# ── Config ────────────────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

cfg = load_config("configs/exp_treino_50.yaml")
eval_cfg = cfg["eval_model"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = eval_cfg["img_size"]
BATCH = eval_cfg["batch_size"]

SNR_TRAIN = 50
GEN_ROOT = Path(f"data/spectrograms/snr_{SNR_TRAIN}/gerados")
NORMAL_ROOT = Path(f"data/spectrograms/snr_{SNR_TRAIN}/reais/normalidade")
REAL_ROOT = Path(f"data/spectrograms/snr_{SNR_TRAIN}/reais_teste")

WANTED = {"pos_10m", "pos_20m", "pos_30m", "pos_40m", "pos_50m", "pos_60m"}
wanted_sorted = sorted(WANTED, key=lambda x: int(x.split("_")[1][:-1]))
wanted2new = {cls: i for i, cls in enumerate(wanted_sorted)}
N_POS = len(WANTED)  # normalidade label = N_POS = 6

# ── Data preparation ──────────────────────────────────────────────────────────
train_tf, val_tf = get_transforms(IMG_SIZE)

samples_gen = []
for cls in wanted_sorted:
    folder = GEN_ROOT / cls
    if folder.exists():
        for p in folder.glob("*.png"):
            samples_gen.append((str(p), wanted2new[cls]))

print(f"Generated samples (6 positions): {len(samples_gen)}")
print("Distribution:", Counter([lbl for _, lbl in samples_gen]))

# 70/15/15 stratified split of generated data (matches dissertation)
labels_gen = [lbl for _, lbl in samples_gen]
X_train_gen, X_temp, y_train_gen, y_temp = train_test_split(
    samples_gen, labels_gen, test_size=0.30, stratify=labels_gen, random_state=SEED
)
X_val_gen, X_test_gen, _, _ = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
)

# 1/3 | 1/3 | 1/3 split for normal class (matches dissertation)
samples_norm = [(str(p), N_POS) for p in sorted(NORMAL_ROOT.glob("*.png"))]
rng = np.random.RandomState(SEED)
rng.shuffle(samples_norm)
n = len(samples_norm) // 3
X_train_norm = samples_norm[:n]
X_val_norm = samples_norm[n : n * 2]
X_test_norm = samples_norm[n * 2 :]

train_samples = X_train_gen + X_train_norm
val_samples = X_val_gen + X_val_norm
test_samples = X_test_gen + X_test_norm


def bin_label(lbl):
    return 1 if lbl < N_POS else 0  # leak=1, normal=0


train_bin = [(p, bin_label(lbl)) for p, lbl in train_samples]
val_bin = [(p, bin_label(lbl)) for p, lbl in val_samples]
test_bin = [(p, bin_label(lbl)) for p, lbl in test_samples]

train_pos = [(p, lbl) for p, lbl in train_samples if lbl != N_POS]
val_pos = [(p, lbl) for p, lbl in val_samples if lbl != N_POS]
test_pos = [(p, lbl) for p, lbl in test_samples if lbl != N_POS]

print(f"\nTrain — bin: {len(train_bin)} | pos: {len(train_pos)}")
print(f"Val   — bin: {len(val_bin)}   | pos: {len(val_pos)}")
print(f"Test  — bin: {len(test_bin)}  | pos: {len(test_pos)}")
print("Train pos distribution:", Counter([lbl for _, lbl in train_pos]))

dl_bin_train = DataLoader(SpectroDataset(train_bin, train_tf), BATCH, shuffle=True, num_workers=4, pin_memory=True)
dl_bin_val = DataLoader(SpectroDataset(val_bin, val_tf), BATCH, shuffle=False, num_workers=4, pin_memory=True)
dl_pos_train = DataLoader(SpectroDataset(train_pos, train_tf), BATCH, shuffle=True, num_workers=4, pin_memory=True)
dl_pos_val = DataLoader(SpectroDataset(val_pos, val_tf), BATCH, shuffle=False, num_workers=4, pin_memory=True)

# ── Models ────────────────────────────────────────────────────────────────────
vp = eval_cfg["vit_params"]
bin_model = BinaryViT(
    IMG_SIZE, vp["patch_size"], vp["emb_dim"], vp["n_layers_bin"], vp["n_heads"]
).to(DEVICE)
pos_model = LeakPosViT(
    IMG_SIZE, vp["patch_size"], vp["emb_dim"], vp["n_layers_bin"], vp["n_heads"], n_pos=N_POS
).to(DEVICE)

opt_bin = optim.AdamW(bin_model.parameters(), lr=eval_cfg["lr_bin"])
opt_pos = optim.AdamW(pos_model.parameters(), lr=eval_cfg["lr_reg"], weight_decay=1e-5)
crit = nn.CrossEntropyLoss()
scaler = GradScaler(device=DEVICE.type)


# ── Training ──────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, train=True):
    model.train(train)
    total_loss = total_correct = total = 0

    for X, y in tqdm(loader, desc="train" if train else "val", leave=False):
        X, y = X.to(DEVICE), y.to(DEVICE)
        if train:
            optimizer.zero_grad()
            with autocast(device_type=DEVICE.type):
                out = model(X)
                loss = crit(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.no_grad():
                out = model(X)
                loss = crit(out, y)

        total_loss += loss.item() * X.size(0)
        total_correct += (out.argmax(1) == y).sum().item()
        total += X.size(0)

    return total_loss / total, total_correct / total


print(f"\nArchitecture: ViT | device: {DEVICE}")
for ep in range(1, eval_cfg["epochs"] + 1):
    print(f"\n=== Epoch {ep}/{eval_cfg['epochs']} ===")
    loss_bt, acc_bt = train_epoch(bin_model, dl_bin_train, opt_bin, train=True)
    loss_bv, acc_bv = train_epoch(bin_model, dl_bin_val, opt_bin, train=False)
    print(f"[BIN] train loss={loss_bt:.4f} acc={acc_bt:.3f} | val loss={loss_bv:.4f} acc={acc_bv:.3f}")

    loss_pt, acc_pt = train_epoch(pos_model, dl_pos_train, opt_pos, train=True)
    loss_pv, acc_pv = train_epoch(pos_model, dl_pos_val, opt_pos, train=False)
    print(f"[POS] train loss={loss_pt:.4f} acc={acc_pt:.3f} | val loss={loss_pv:.4f} acc={acc_pv:.3f}")


# ── Cascade inference ─────────────────────────────────────────────────────────
def predict_cascade(x, model_bin, model_pos):
    with torch.no_grad():
        pred_bin = model_bin(x).argmax(1)
        pred_final = torch.full_like(pred_bin, fill_value=N_POS)  # default: normal
        if (pred_bin == 1).any():
            pred_final[pred_bin == 1] = model_pos(x[pred_bin == 1]).argmax(1)
    return pred_final


# ── Evaluation on generated test split ───────────────────────────────────────
def evaluate_generated(samples, label_names):
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    bin_model.eval()
    pos_model.eval()
    loader = DataLoader(SpectroDataset(samples, val_tf), BATCH, shuffle=False, num_workers=4)

    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in loader:
            pred = predict_cascade(X.to(DEVICE), bin_model, pos_model)
            y_true.extend(y.tolist())
            y_pred.extend(pred.cpu().tolist())

    print("\n=== Cascade evaluation — generated test split ===")
    print(classification_report(y_true, y_pred, target_names=label_names, digits=3))

    cm = confusion_matrix(y_true, y_pred, labels=range(len(label_names)))
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names)
    plt.title("Confusion Matrix — generated test")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    out_path = Path("outputs/evaluation/confusion_generated.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved: {out_path}")


# ── Evaluation on real test data ──────────────────────────────────────────────
def evaluate_real(base_path, label_names):
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    bin_model.eval()
    pos_model.eval()

    pos_dirs = sorted(p for p in Path(base_path).iterdir() if p.is_dir())
    real_samples, real_label_names = [], []
    for i, pos_dir in enumerate(pos_dirs):
        real_samples.extend([(str(p), i) for p in sorted(pos_dir.glob("*.png"))])
        real_label_names.append(pos_dir.name)

    loader = DataLoader(SpectroDataset(real_samples, val_tf), BATCH, shuffle=False, num_workers=4)

    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in loader:
            pred = predict_cascade(X.to(DEVICE), bin_model, pos_model)
            y_true.extend(y.tolist())
            y_pred.extend(pred.cpu().tolist())

    print("\n=== Cascade evaluation — real test data ===")
    print(classification_report(
        y_true, y_pred,
        labels=range(len(label_names)),
        target_names=label_names,
        digits=3,
    ))

    cm = confusion_matrix(y_true, y_pred, labels=range(len(label_names)))
    cm = cm[: len(real_label_names), :]  # only rows that have real data
    plt.figure(figsize=(12, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=real_label_names)
    plt.title("Confusion Matrix — real test data")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    out_path = Path("outputs/evaluation/confusion_real.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Saved: {out_path}")


# ── Run evaluations ───────────────────────────────────────────────────────────
final_classes = wanted_sorted + ["normal"]

evaluate_generated(test_samples, final_classes)
evaluate_real(REAL_ROOT, final_classes)

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.core.config_parser import load_config
from src.models.cascade_vit import BinaryViT, LeakRegressorViT
from src.models.cnn_baselines import SimpleCNN, ResNetBaseline
from src.data_loaders.vit_dataset import get_transforms, SpectroDataset
from src.training.train_vit import train_epoch


def main():
    cfg = load_config("configs/exp_treino_50.yaml")
    eval_cfg = cfg["eval_model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    snr_treino = 50
    gen_root = Path(f"data/spectrograms/snr_{snr_treino}/gerados")
    real_root = Path(f"data/spectrograms/snr_{snr_treino}/reais")

    samples_gen = []
    for p in gen_root.rglob("*.png"):
        try:
            pos = int(p.parent.name.split("_")[1].replace("m", ""))
            samples_gen.append((str(p), pos))
        except:
            continue

    samples_norm = [(str(p), -1.0) for p in (real_root / "normalidade").glob("*.png")]

    pos_labels = [l for _, l in samples_gen]
    POS_MIN, POS_MAX = min(pos_labels), max(pos_labels)

    def normalize_pos(p):
        return (p - POS_MIN) / (POS_MAX - POS_MIN)

    train_bin = [(p, 0 if l == -1 else 1) for p, l in samples_gen + samples_norm[:800]]
    val_bin = [(p, 0 if l == -1 else 1) for p, l in samples_norm[800:]]

    train_reg = [(p, normalize_pos(l)) for p, l in samples_gen]

    train_tf, val_tf = get_transforms(eval_cfg["img_size"])
    dl_bin_train = DataLoader(
        SpectroDataset(train_bin, train_tf), eval_cfg["batch_size"], shuffle=True
    )
    dl_bin_val = DataLoader(SpectroDataset(val_bin, val_tf), eval_cfg["batch_size"])
    dl_reg_train = DataLoader(
        SpectroDataset(train_reg, train_tf), eval_cfg["batch_size"], shuffle=True
    )

    model_type = eval_cfg["type"].lower()
    print(
        f"Iniciando treinamento com arquitetura: {model_type.upper()} em SNR de {snr_treino}dB"
    )

    if model_type == "vit":
        vp = eval_cfg["vit_params"]
        model_bin = BinaryViT(
            img_size=eval_cfg["img_size"],
            patch_size=vp["patch_size"],
            emb_dim=vp["emb_dim"],
            n_layers=vp["n_layers_bin"],
            n_heads=vp["n_heads"],
        ).to(device)
        model_reg = LeakRegressorViT(
            img_size=eval_cfg["img_size"],
            patch_size=vp["patch_size"],
            emb_dim=vp["emb_dim"],
            n_layers=vp["n_layers_reg"],
            n_heads=vp["n_heads"],
        ).to(device)
    elif model_type == "cnn":
        model_bin = SimpleCNN(is_classifier=True).to(device)
        model_reg = SimpleCNN(is_classifier=False).to(device)
    elif model_type == "resnet":
        model_bin = ResNetBaseline(is_classifier=True).to(device)
        model_reg = ResNetBaseline(is_classifier=False).to(device)
    else:
        raise ValueError("Arquitetura desconhecida! Use 'vit', 'cnn' ou 'resnet'.")

    opt_bin = optim.AdamW(model_bin.parameters(), lr=eval_cfg["lr_bin"])
    opt_reg = optim.AdamW(
        model_reg.parameters(), lr=eval_cfg["lr_reg"], weight_decay=5e-4
    )

    crit_bin = nn.CrossEntropyLoss()
    crit_reg = nn.L1Loss()

    for ep in range(1, eval_cfg["epochs"] + 1):
        print(f"\n=== Época {ep}/{eval_cfg['epochs']} ===")
        loss_bin, acc_bin = train_epoch(
            model_bin, dl_bin_train, opt_bin, crit_bin, device, train=True
        )
        print(f"[BIN] Train Loss: {loss_bin:.4f} | Train Acc: {acc_bin:.3f}")

        loss_reg, _ = train_epoch(
            model_reg, dl_reg_train, opt_reg, crit_reg, device, train=True
        )
        print(f"[REG] Train Loss: {loss_reg:.4f}")

    print(
        "\n✅ Treinamento concluído. Iniciando Stress Test (Avaliação de Robustez AWGN)..."
    )

    snrs_teste = [50, 40, 30, 20, 10]

    for snr in snrs_teste:
        real_root_snr = Path(f"data/spectrograms/snr_{snr}/reais")
        samples_norm_snr = [
            (str(p), -1.0) for p in (real_root_snr / "normalidade").glob("*.png")
        ]
        samples_vaz_snr = []
        for p in real_root_snr.rglob("*.png"):
            if "normalidade" not in p.parent.name:
                try:
                    pos = int(p.parent.name.split("_")[1].replace("m", ""))
                    samples_vaz_snr.append((str(p), pos))
                except:
                    continue

        test_bin_snr = [
            (p, 0 if l == -1 else 1) for p, l in samples_vaz_snr + samples_norm_snr
        ]
        test_reg_snr = [(p, normalize_pos(l)) for p, l in samples_vaz_snr]

        dl_bin_test = DataLoader(
            SpectroDataset(test_bin_snr, val_tf), eval_cfg["batch_size"]
        )
        dl_reg_test = DataLoader(
            SpectroDataset(test_reg_snr, val_tf), eval_cfg["batch_size"]
        )

        print(f"\n--- Testando em Ambiente com SNR de {snr}dB ---")
        _, acc_bin_test = train_epoch(
            model_bin, dl_bin_test, opt_bin, crit_bin, device, train=False
        )
        loss_reg_test, _ = train_epoch(
            model_reg, dl_reg_test, opt_reg, crit_reg, device, train=False
        )

        mae_metros = loss_reg_test * (POS_MAX - POS_MIN)

        print(f"👉 Acurácia de Detecção (Binário): {acc_bin_test:.3f}")
        print(f"👉 Erro Médio de Localização (MAE): {mae_metros:.2f} metros")

    print("\n🚀 Bateria de experimentos finalizada!")


if __name__ == "__main__":
    main()

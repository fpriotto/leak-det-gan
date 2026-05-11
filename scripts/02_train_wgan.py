"""
Script: scripts/02_train_wgan.py
Descrição: Treina o modelo de aumento de dados (WGAN-GP ou Vanilla CGAN dependendo da config).
Uso: python scripts/02_train_wgan.py
"""

import sys
import torch
import torch.optim as optim
import joblib
from pathlib import Path

# Garante que o python ache a pasta 'src'
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.core.config_parser import load_config
from src.models.gan_components import Gen1D, Critic1D, CriticSpec
from src.training.train_gan import WGANTrainer, CGANTrainer
from src.data_loaders.gan_dataset import get_gan_dataloader


def main():
    print("Iniciando pipeline de treinamento da GAN...")

    config_path = "configs/exp_treino_50.yaml"
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    gan_cfg = cfg["gan"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de treino: {device}")

    print("Carregando e normalizando dados para a GAN...")
    loader, meta = get_gan_dataloader(
        treino_dir=data_cfg["treino_dir"],
        batch_size=gan_cfg["batch_size"],
        seq_len=gan_cfg["seq_len"],
    )

    meta["z_dim"] = gan_cfg["z_dim"]

    print("Inicializando Modelos...")
    G = Gen1D(
        z_dim=gan_cfg["z_dim"], cond_dim=gan_cfg["cond_dim"], seq_len=gan_cfg["seq_len"]
    ).to(device)
    D_time = Critic1D(cond_dim=gan_cfg["cond_dim"]).to(device)

    # Otimizadores Clássicos da GAN (Adam, mas note que a literatura de Vanilla GAN geralmente sugere betas=(0.5, 0.999))
    opt_G = optim.Adam(G.parameters(), lr=gan_cfg["lr"], betas=(0.5, 0.999))
    opt_Dt = optim.Adam(D_time.parameters(), lr=gan_cfg["lr"], betas=(0.5, 0.999))

    # Verifica qual arquitetura rodar
    gan_type = gan_cfg.get("type", "wgan").lower()

    if gan_type == "wgan":
        print("🔧 Modo Arquitetura: WGAN-GP")
        D_spec = CriticSpec().to(device)
        opt_Ds = optim.Adam(D_spec.parameters(), lr=gan_cfg["lr"], betas=(0.0, 0.99))

        # Ajusta os betas da WGAN-GP de volta ao padrão
        opt_G.param_groups[0]["betas"] = (0.0, 0.99)
        opt_Dt.param_groups[0]["betas"] = (0.0, 0.99)

        trainer = WGANTrainer(G, D_time, D_spec, opt_G, opt_Dt, opt_Ds, gan_cfg, device)
    elif gan_type == "cgan":
        print("🔧 Modo Arquitetura: Vanilla CGAN (Baseline Ablação)")
        trainer = CGANTrainer(G, D_time, opt_G, opt_Dt, gan_cfg, device)
    else:
        raise ValueError("Tipo de GAN desconhecido. Use 'wgan' ou 'cgan'.")

    epochs = gan_cfg["epochs"]
    print(f"Iniciando treino por {epochs} épocas...")

    for ep in range(epochs):
        loss_Dt_ep, loss_Ds_ep, loss_G_ep = 0.0, 0.0, 0.0
        batches = 0

        for cond, real, _ in loader:
            cond, real = cond.to(device), real.to(device)

            l_dt, l_ds, l_g = trainer.train_step(real, cond)

            loss_Dt_ep += l_dt
            loss_Ds_ep += l_ds
            loss_G_ep += l_g
            batches += 1

        if ep == 0 or (ep + 1) % 10 == 0:
            if gan_type == "wgan":
                print(
                    f"[Ep {ep + 1:04d}/{epochs}] D_time: {loss_Dt_ep / batches:+.3f} | D_spec: {loss_Ds_ep / batches:+.3f} | G: {loss_G_ep / batches:+.3f}"
                )
            else:
                print(
                    f"[Ep {ep + 1:04d}/{epochs}] Discriminador: {loss_Dt_ep / batches:+.3f} | Gerador: {loss_G_ep / batches:+.3f}"
                )

    save_dir = Path("data/modelo_gan")
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(G.state_dict(), save_dir / "generator_wgan_80_pct.pth")
    joblib.dump(meta, save_dir / "meta_normalizacao_80_pct.save")

    print(f"\n✅ Modelo Gerador e Metadados salvos em {save_dir.resolve()}")


if __name__ == "__main__":
    main()

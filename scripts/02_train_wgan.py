"""Train the conditional WGAN-GP generator for a given data split.

Usage:
    python scripts/02_train_wgan.py --config configs/exp_train_50.yaml
"""

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
import torch
import torch.optim as optim

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.core.config_parser import load_config
from src.data_loaders.gan_dataset import get_gan_dataloader
from src.models.gan_components import Critic1D, CriticSpec, Gen1D
from src.training.train_gan import CGANTrainer, WGANTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp_train_50.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    gan_cfg = cfg["gan"]
    pct = int(round(data_cfg["split_ratio"] * 100))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | split: {pct}%")

    loader, meta = get_gan_dataloader(
        train_dir=data_cfg["train_dir"],
        batch_size=gan_cfg["batch_size"],
        seq_len=gan_cfg["seq_len"],
        pos_map=data_cfg["pos_map"],
    )
    meta["z_dim"] = gan_cfg["z_dim"]

    G = Gen1D(
        z_dim=gan_cfg["z_dim"], cond_dim=gan_cfg["cond_dim"], seq_len=gan_cfg["seq_len"]
    ).to(device)
    D_time = Critic1D(cond_dim=gan_cfg["cond_dim"]).to(device)
    opt_G = optim.Adam(G.parameters(), lr=gan_cfg["lr"], betas=(0.0, 0.99))
    opt_Dt = optim.Adam(D_time.parameters(), lr=gan_cfg["lr"], betas=(0.0, 0.99))

    gan_type = gan_cfg.get("type", "wgan").lower()
    if gan_type == "wgan":
        print("Architecture: WGAN-GP")
        D_spec = CriticSpec().to(device)
        opt_Ds = optim.Adam(D_spec.parameters(), lr=gan_cfg["lr"], betas=(0.0, 0.99))
        trainer = WGANTrainer(G, D_time, D_spec, opt_G, opt_Dt, opt_Ds, gan_cfg, device)
    elif gan_type == "cgan":
        print("Architecture: Vanilla CGAN")
        trainer = CGANTrainer(G, D_time, opt_G, opt_Dt, gan_cfg, device)
    else:
        raise ValueError(f"Unknown GAN type '{gan_type}'. Use 'wgan' or 'cgan'.")

    epochs = gan_cfg["epochs"]
    print(f"Training for {epochs} epochs...")
    history = []
    save_dir = Path(data_cfg["gan_model_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(epochs):
        loss_dt_ep = loss_ds_ep = loss_g_ep = 0.0
        batches = 0
        for cond, real, _ in loader:
            cond, real = cond.to(device), real.to(device)
            l_dt, l_ds, l_g = trainer.train_step(real, cond)
            loss_dt_ep += l_dt
            loss_ds_ep += l_ds
            loss_g_ep += l_g
            batches += 1

        avg_dt = loss_dt_ep / batches
        avg_ds = loss_ds_ep / batches
        avg_g = loss_g_ep / batches

        history.append(
            {"epoch": ep + 1, "D_time": avg_dt, "D_spec": avg_ds, "G": avg_g}
        )
        pd.DataFrame(history).to_csv(
            save_dir / f"loss_history_{pct}pct.csv", index=False
        )

        if ep == 0 or (ep + 1) % 10 == 0:
            print(
                f"[{ep + 1:04d}/{epochs}] D_time: {avg_dt:+.3f} | D_spec: {avg_ds:+.3f} | G: {avg_g:+.3f}"
            )

    torch.save(G.state_dict(), save_dir / f"generator_wgan_{pct}_pct.pth")
    joblib.dump(meta, save_dir / f"meta_norm_{pct}_pct.save")
    print(f"Model and metadata saved to {save_dir.resolve()}")


if __name__ == "__main__":
    main()

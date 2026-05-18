import sys
import torch
import torch.optim as optim
import joblib
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.core.config_parser import load_config
from src.models.gan_components import Gen1D, Critic1D, CriticSpec
from src.training.train_gan import WGANTrainer, CGANTrainer
from src.data_loaders.gan_dataset import get_gan_dataloader


def main():
    cfg = load_config("configs/exp_treino_50.yaml")
    data_cfg = cfg["data"]
    gan_cfg = cfg["gan"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    loader, meta = get_gan_dataloader(
        train_dir=data_cfg["train_dir"],
        batch_size=gan_cfg["batch_size"],
        seq_len=gan_cfg["seq_len"],
    )
    meta["z_dim"] = gan_cfg["z_dim"]

    G = Gen1D(z_dim=gan_cfg["z_dim"], cond_dim=gan_cfg["cond_dim"], seq_len=gan_cfg["seq_len"]).to(device)
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
    save_dir = Path("data/modelo_gan")
    save_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(epochs):
        loss_Dt_ep = loss_Ds_ep = loss_G_ep = 0.0
        batches = 0
        for cond, real, _ in loader:
            cond, real = cond.to(device), real.to(device)
            l_dt, l_ds, l_g = trainer.train_step(real, cond)
            loss_Dt_ep += l_dt
            loss_Ds_ep += l_ds
            loss_G_ep += l_g
            batches += 1

        avg_dt = loss_Dt_ep / batches
        avg_ds = loss_Ds_ep / batches
        avg_g = loss_G_ep / batches

        history.append({"epoch": ep + 1, "D_time": avg_dt, "D_spec": avg_ds, "G": avg_g})
        pd.DataFrame(history).to_csv(save_dir / "loss_history.csv", index=False)

        if ep == 0 or (ep + 1) % 10 == 0:
            print(f"[{ep + 1:04d}/{epochs}] D_time: {avg_dt:+.3f} | D_spec: {avg_ds:+.3f} | G: {avg_g:+.3f}")

    torch.save(G.state_dict(), save_dir / "generator_wgan_50_pct.pth")
    joblib.dump(meta, save_dir / "meta_normalizacao_50_pct.save")
    print(f"Model and metadata saved to {save_dir.resolve()}")


if __name__ == "__main__":
    main()

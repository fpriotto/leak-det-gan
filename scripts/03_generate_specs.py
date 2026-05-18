import sys
import torch
import joblib
import pickle
import random
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.core.config_parser import load_config
from src.models.gan_components import Gen1D
from src.data_loaders.gan_dataset import denorm_signal
from src.data_prep.spectro_maker import get_freq_mask, calc_global_scale, save_spectrograms_as_png


def load_folder(folder: str) -> dict:
    data = {}
    for pkl_file in Path(folder).glob("*.pkl"):
        with open(pkl_file, "rb") as f:
            data[pkl_file.stem] = pickle.load(f)
    return data


def main():
    cfg = load_config("configs/exp_treino_50.yaml")
    spc_cfg = cfg["spectro"]
    data_cfg = cfg["data"]
    gan_cfg = cfg["gan"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mask = get_freq_mask(spc_cfg["fs"], spc_cfg["nperseg"], spc_cfg["f_min"], spc_cfg["f_max"])

    # Combine train + test splits → 1000 samples/position for real spectrograms.
    # GAN only trained on train_dir; ViT never sees real leak data in training,
    # so using both halves for evaluation is valid and matches dissertation (Tabela 5).
    train_data = load_folder(data_cfg["train_dir"])
    test_data  = load_folder(data_cfg["test_dir"])
    all_keys = set(train_data) | set(test_data)
    real_data = {k: train_data.get(k, []) + test_data.get(k, []) for k in all_keys}

    # Global color scale from combined data.
    vmin, vmax = calc_global_scale(real_data, spc_cfg["fs"], spc_cfg["nperseg"], spc_cfg["noverlap"], mask)
    print(f"Color scale: {vmin:.2f} → {vmax:.2f} dB")

    snrs = [50, 40, 30, 20, 10]
    pos_map = {1: 10, 2: 25, 3: 60}

    print("[1/2] Generating real spectrograms at multiple SNRs...")
    for snr in snrs:
        out_dir_real = Path(f"data/spectrograms/snr_{snr}/reais")
        normal_pool = []

        for key, series_list in real_data.items():
            if "_v0" in key:
                normal_pool.extend(series_list)
            else:
                idx = int(key.split("_")[1])
                if idx not in pos_map:
                    continue
                save_spectrograms_as_png(
                    series_list,
                    out_dir_real / f"pos_{pos_map[idx]}m",
                    vmin, vmax,
                    spc_cfg["fs"], spc_cfg["nperseg"], spc_cfg["noverlap"],
                    mask, spc_cfg["img_size"],
                    snr_db=snr,
                )

        random.seed(data_cfg.get("seed", 42))
        random.shuffle(normal_pool)
        save_spectrograms_as_png(
            normal_pool[: spc_cfg["n_norm_samples"]],
            out_dir_real / "normalidade",
            vmin, vmax,
            spc_cfg["fs"], spc_cfg["nperseg"], spc_cfg["noverlap"],
            mask, spc_cfg["img_size"],
            snr_db=snr,
        )

    print("[2/2] Generating synthetic spectrograms with GAN (SNR 50 dB only)...")
    meta = joblib.load("data/modelo_gan/meta_normalizacao_50_pct.save")

    G = Gen1D(z_dim=meta["z_dim"], cond_dim=gan_cfg["cond_dim"], seq_len=meta["SEQ_LEN"]).to(device)
    G.load_state_dict(torch.load("data/modelo_gan/generator_wgan_50_pct.pth", map_location=device))
    G.eval()

    log_min = np.log10(meta["pos_min"])
    log_max = np.log10(meta["pos_max"])
    snr_train = 50
    out_dir_gen = Path(f"data/spectrograms/snr_{snr_train}/gerados")

    with torch.no_grad():
        for dist_m in range(10, 65, 5):
            cond_val = (np.log10(dist_m) - log_min) / (log_max - log_min)
            c = torch.full((spc_cfg["n_gen_per_class"], 1), cond_val, dtype=torch.float32, device=device)
            z = torch.randn(spc_cfg["n_gen_per_class"], meta["z_dim"], device=device)
            fakes = [denorm_signal(s, meta) for s in G(z, c).squeeze(1).cpu().numpy()]
            save_spectrograms_as_png(
                fakes,
                out_dir_gen / f"pos_{dist_m}m",
                vmin, vmax,
                spc_cfg["fs"], spc_cfg["nperseg"], spc_cfg["noverlap"],
                mask, spc_cfg["img_size"],
                snr_db=snr_train,
            )
            print(f"  pos_{dist_m}m done")

    print("All spectrograms saved.")


if __name__ == "__main__":
    main()

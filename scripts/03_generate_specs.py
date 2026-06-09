"""Generate synthetic and real spectrograms for all positions.

Loads the trained GAN to synthesise signals at 11 distances (10–60 m),
computes a global colour scale from the generated pool, and saves PNG
spectrograms for both the synthetic and real (train + test) data splits.

Usage:
    python scripts/03_generate_specs.py --config configs/exp_train_50.yaml
"""

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.core.config_parser import load_config
from src.data_loaders.gan_dataset import denorm_signal
from src.data_prep.spectro_maker import (
    calc_global_scale,
    get_freq_mask,
    save_spectrograms_as_png,
)
from src.models.gan_components import Gen1D


def _load_folder(folder: str) -> dict:
    data = {}
    for pkl_file in Path(folder).glob("*.pkl"):
        with open(pkl_file, "rb") as f:
            data[pkl_file.stem] = pickle.load(f)
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp_train_50.yaml")
    parser.add_argument(
        "--skip-real-test", action="store_true",
        help="Do not overwrite data/spectrograms/real_test (use for ablation runs)"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    spc_cfg = cfg["spectro"]
    data_cfg = cfg["data"]
    gan_cfg = cfg["gan"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pct = int(round(data_cfg["split_ratio"] * 100))
    spec_root = data_cfg.get("spec_root", "data/spectrograms")

    # Frequency mask 25–80 kHz applied to all spectrograms so ViT always sees
    # the same band in generated and real signals.
    mask = get_freq_mask(
        spc_cfg["fs"], spc_cfg["nperseg"], spc_cfg["f_min"], spc_cfg["f_max"]
    )

    # Combine train + test splits to build the full real-signal pool.
    # Real spectrograms are split-agnostic; only generated ones differ by split.
    train_data = _load_folder(data_cfg["train_dir"])
    test_data = _load_folder(data_cfg["test_dir"])
    all_keys = set(train_data) | set(test_data)
    real_data = {k: train_data.get(k, []) + test_data.get(k, []) for k in all_keys}

    # Test-split leak signals (held-out for ViT evaluation).
    test_leak = {k: v for k, v in test_data.items() if "_v1" in k}
    # All leak signals (combined) used only for the real/ visualisation folder.
    leak_data = {k: v for k, v in real_data.items() if "_v1" in k}
    # Normal signals: train split only — test-split v0 never enters ViT training.
    train_normal = {k: v for k, v in train_data.items() if "_v0" in k}

    normal_pool: list = []
    for v in train_normal.values():
        normal_pool.extend(v)
    random.seed(data_cfg.get("seed", 42))
    random.shuffle(normal_pool)
    normal_pool = normal_pool[: spc_cfg["n_norm_samples"]]

    # ── Step 1: synthesise signals via GAN ────────────────────────────────────
    gan_model_dir = Path(data_cfg["gan_model_dir"])
    meta = joblib.load(gan_model_dir / f"meta_norm_{pct}_pct.save")
    G = Gen1D(
        z_dim=meta["z_dim"], cond_dim=gan_cfg["cond_dim"], seq_len=meta["SEQ_LEN"]
    ).to(device)
    G.load_state_dict(
        torch.load(gan_model_dir / f"generator_wgan_{pct}_pct.pth", map_location=device)
    )
    G.eval()

    log_min = np.log10(meta["pos_min"])
    log_max = np.log10(meta["pos_max"])

    print(f"[1/3] Generating synthetic signals with GAN ({pct}% split)...")
    generated: dict = {}
    with torch.no_grad():
        for dist_m in range(10, 65, 5):
            cond_val = (np.log10(dist_m) - log_min) / (log_max - log_min)
            c = torch.full(
                (spc_cfg["n_gen_per_class"], 1),
                cond_val,
                dtype=torch.float32,
                device=device,
            )
            z = torch.randn(spc_cfg["n_gen_per_class"], meta["z_dim"], device=device)
            generated[dist_m] = [
                denorm_signal(s, meta) for s in G(z, c).squeeze(1).cpu().numpy()
            ]
            print(f"  pos_{dist_m}m done")

    # ── Step 2: global colour scale ───────────────────────────────────────────
    # Include real train-split leak signals so the scale covers real pos_60m
    # amplitude (attenuated after 60 m propagation). Percentile 2/98 removes
    # the log(1e-12) noise floor. Test-split signals excluded to prevent leakage.
    scale_signals = {f"pos_{d}m": generated[d] for d in generated}
    scale_signals["normal"] = normal_pool
    vmin, vmax = calc_global_scale(
        scale_signals, spc_cfg["fs"], spc_cfg["nperseg"], spc_cfg["noverlap"], mask=mask
    )
    print(f"Colour scale 2nd/98th pct (25–80 kHz): {vmin:.2f} → {vmax:.2f} dB")

    scale_path = Path(f"{spec_root}/scale_{pct}pct.json")
    scale_path.parent.mkdir(parents=True, exist_ok=True)
    scale_path.write_text(
        json.dumps(
            {
                "vmin": vmin,
                "vmax": vmax,
                "pct": pct,
                "nperseg": spc_cfg["nperseg"],
                "noverlap": spc_cfg["noverlap"],
            }
        )
    )

    # ── Step 3a: save real spectrograms at multiple SNRs (robustness study) ────
    # These go under noise_eval/ and are never used for ViT training or main eval.
    snrs = [50, 40, 30, 20, 10]
    pos_map = data_cfg["pos_map"]  # {sensor_index: distance_m}

    print("[2/3] Saving real spectrograms at multiple SNRs (noise_eval)...")
    for snr in snrs:
        out_dir_noise = Path(f"data/spectrograms/noise_eval/snr{snr}/real")
        for key, series_list in leak_data.items():
            idx = int(key.split("_")[1])
            if idx not in pos_map:
                continue
            save_spectrograms_as_png(
                series_list,
                out_dir_noise / f"pos_{pos_map[idx]}m",
                vmin,
                vmax,
                spc_cfg["fs"],
                spc_cfg["nperseg"],
                spc_cfg["noverlap"],
                mask,
                spc_cfg["img_size"],
                snr_db=snr,
            )
        save_spectrograms_as_png(
            normal_pool,
            out_dir_noise / "normal",
            vmin,
            vmax,
            spc_cfg["fs"],
            spc_cfg["nperseg"],
            spc_cfg["noverlap"],
            mask,
            spc_cfg["img_size"],
            snr_db=None,
        )

    # ── Step 3b: save generated spectrograms ──────────────────────────────────
    out_dir_gen = Path(f"{spec_root}/generated_{pct}pct")
    snr_gen_60m = spc_cfg.get("snr_gen_60m", None)

    print("[3/3] Saving synthetic spectrograms...")
    for dist_m, fakes in generated.items():
        snr_here = snr_gen_60m if (dist_m >= 60 and snr_gen_60m is not None) else None
        save_spectrograms_as_png(
            fakes,
            out_dir_gen / f"pos_{dist_m}m",
            vmin,
            vmax,
            spc_cfg["fs"],
            spc_cfg["nperseg"],
            spc_cfg["noverlap"],
            mask,
            spc_cfg["img_size"],
            snr_db=snr_here,
        )
        print(f"  pos_{dist_m}m done (snr={snr_here})")

    # ── Step 3c: save real train-split leak spectrograms (for injection exp) ──
    # Training-split only → zero leakage with test evaluation.
    train_leak = {k: v for k, v in train_data.items() if "_v1" in k}
    out_dir_real_train = Path(f"{spec_root}/real_train_{pct}pct")
    for key, series_list in train_leak.items():
        idx = int(key.split("_")[1])
        if idx not in pos_map:
            continue
        save_spectrograms_as_png(
            series_list,
            out_dir_real_train / f"pos_{pos_map[idx]}m",
            vmin,
            vmax,
            spc_cfg["fs"],
            spc_cfg["nperseg"],
            spc_cfg["noverlap"],
            mask,
            spc_cfg["img_size"],
            snr_db=None,
        )
    print(f"Real train-split spectrograms saved to {out_dir_real_train}")

    # ── Step 3d: save held-out test-split leak spectrograms ───────────────────
    # Save normal spectrograms used by ViT for training/evaluation.
    out_dir_normal = Path(f"{spec_root}/real/normal")
    save_spectrograms_as_png(
        normal_pool,
        out_dir_normal,
        vmin, vmax,
        spc_cfg["fs"], spc_cfg["nperseg"], spc_cfg["noverlap"],
        mask, spc_cfg["img_size"],
    )
    print(f"Normal spectrograms saved to {out_dir_normal}")

    if args.skip_real_test:
        print("[+] Skipping real_test (--skip-real-test flag set)")
    else:
        print("[+] Recreating real_test from held-out test split...")
        out_dir_real_test = Path(f"{spec_root}/real_test")
        for key, series_list in test_leak.items():
            idx = int(key.split("_")[1])
            if idx not in pos_map:
                continue
            save_spectrograms_as_png(
                series_list,
                out_dir_real_test / f"pos_{pos_map[idx]}m",
                vmin,
                vmax,
                spc_cfg["fs"],
                spc_cfg["nperseg"],
                spc_cfg["noverlap"],
                mask,
                spc_cfg["img_size"],
                snr_db=None,
            )

    print("All spectrograms saved.")


if __name__ == "__main__":
    main()

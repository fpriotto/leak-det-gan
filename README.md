# Leak Detection with WGAN-GP and ViT

A WGAN-GP dual-critic model generates synthetic acoustic emission signals. The signals (real and synthetic) are converted to log-spectrograms and evaluated under AWGN noise at multiple SNRs using a cascade ViT classifier and CNN baselines.

## Pipeline

All scripts must be run from the repository root.

| Script | Purpose |
|---|---|
| `scripts/00_convert_npy_to_pkl.py` | Convert raw zlib-compressed sensor dumps (1 MHz) to `.pkl` |
| `scripts/01_prepare_data.py` | Butterworth downsampling + Chebyshev bandpass, 10 ms windowing |
| `scripts/02_train_wgan.py` | Train WGAN-GP generator |
| `scripts/03_generate_specs.py` | Render log-spectrograms (real + synthetic) at SNR 10–50 dB |
| `scripts/04_train_evaluate_vit.py` | Train and evaluate the cascade ViT (BinaryViT → LeakPosViT) |
| `scripts/05_model_complexity.py` | Report FLOPs and parameter counts for all models |

## Setup

```bash
pip install -r requirements.txt
pip install thop
```

## Data layout

```
data/
  raw/
    Vazamento_5_bar/      # raw leak recordings (1 MHz, zlib)
    Normalidade/          # raw normal recordings
  processed/
    treino_50/            # 50% split used for GAN training
    teste_50/             # held-out 50% — never seen by GAN
  modelo_gan/             # saved generator weights and normalization metadata
  spectrograms/
    snr_50/
      reais/              # real spectrograms at 50 dB SNR
      reais_teste/        # test-set-only real spectrograms (pos_10m, pos_25m, pos_60m)
      gerados/            # GAN-generated spectrograms (6 positions × 1000 samples)
```

## Configuration

Active experiment: `configs/exp_treino_50.yaml` (50% data split, WGAN-GP).

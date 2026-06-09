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
| `scripts/06_threshold_sweep.py` | Sweep binary threshold on held-out test pkl; find optimal operating point |
| `scripts/07_train_regression.py` | Train and evaluate the cascade ViT regressor (BinaryViT → LeakRegressorViT) |
| `scripts/08_run_ablation.py` | Run leave-one-sensor-out ablations (no_10m / no_25m / no_60m) |

## Setup

```bash
pip install -r requirements.txt
```

## Data layout

```
data/
  raw/
    leak_5bar/        # raw leak recordings (1 MHz, zlib)
    normal/           # raw normal recordings
  processed/
    train_50/         # 50% split used for GAN training
    test_50/          # held-out 50% — never seen by GAN
  gan_model/          # saved generator weights and normalization metadata
  spectrograms/
    snr_50/
      real/           # real spectrograms at 50 dB SNR
      real_test/      # test-set-only real spectrograms (pos_10m, pos_25m, pos_60m)
      generated_50pct/  # GAN-generated spectrograms (6 positions × 1000 samples)
```

## Configuration

Active experiment: `configs/exp_train_50.yaml` (50% data split, WGAN-GP).

Key parameters:
- `spectro.nperseg: 128` / `noverlap: 96` — 75% overlap STFT; 76 temporal frames per window
- `eval_model.bin_threshold: 0.05` — operating threshold for binary stage (safety-driven: false alarms preferred over missed leaks)
- `eval_model.epochs: 30` — ViT trained with CosineAnnealingLR

## Running

Run the full pipeline for a single split (e.g. 50%) from the repository root:

```bash
python scripts/01_prepare_data.py       --config configs/exp_train_50.yaml
python scripts/02_train_wgan.py         --config configs/exp_train_50.yaml
python scripts/03_generate_specs.py     --config configs/exp_train_50.yaml
python scripts/04_train_evaluate_vit.py --config configs/exp_train_50.yaml
python scripts/07_train_regression.py   --config configs/exp_train_50.yaml --splits 50
```

Other splits use the matching config (`configs/exp_train_{20,35,50,65,80}.yaml`).

Threshold sweep (after ViT training) and leave-one-sensor-out ablations:

```bash
python scripts/06_threshold_sweep.py
python scripts/08_run_ablation.py
```

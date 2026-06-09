"""Run the full ablation pipeline for no_10m, no_25m and no_60m experiments.

Orchestrates scripts 02 → 03 → 04 → 07 for each ablation config.
Each ablation retrains the GAN without one real sensor position,
regenerates spectrograms, trains and evaluates the cascade ViT,
and trains the regression model.

NOTE: 03_generate_specs.py is always called with --skip-real-test so that
data/spectrograms/real_test (populated by the standard 80% run) is never
overwritten. Ablation evaluation therefore tests against all three real sensors,
which is the correct experimental design.

Results:
    outputs/ablation/no_10m/  — GAN trained without 10 m sensor
    outputs/ablation/no_25m/  — GAN trained without 25 m sensor
    outputs/ablation/no_60m/  — GAN trained without 60 m sensor

Usage:
    python scripts/08_run_ablation.py
    python scripts/08_run_ablation.py --ablation no_25m
    python scripts/08_run_ablation.py --ablation no_60m --start-from vit
"""

from __future__ import annotations

import argparse
import subprocess
import sys

_ABLATIONS = {
    "no_10m": "configs/ablation_no_10m_80.yaml",
    "no_25m": "configs/ablation_no_25m_80.yaml",
    "no_60m": "configs/ablation_no_60m_80.yaml",
}
_STAGES = ["gan", "specs", "vit", "regression"]


def _run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[ERROR] Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def run_ablation(name: str, config: str, start_from: str = "gan") -> None:
    """Run one ablation experiment from a given stage onward."""
    stages = _STAGES[_STAGES.index(start_from):]
    print(f"\n{'=' * 60}")
    print(f"  Ablation: {name} | config: {config}")
    print(f"  Stages: {' → '.join(stages)}")
    print(f"{'=' * 60}")

    python = sys.executable

    if "gan" in stages:
        print("\n[1/4] Training WGAN-GP (ablation GAN)...")
        _run([python, "scripts/02_train_wgan.py", "--config", config])

    if "specs" in stages:
        print("\n[2/4] Generating spectrograms (ablation)...")
        _run([python, "scripts/03_generate_specs.py", "--config", config, "--skip-real-test"])

    if "vit" in stages:
        print("\n[3/4] Training and evaluating cascade ViT (ablation)...")
        _run([python, "scripts/04_train_evaluate_vit.py", "--config", config])

    if "regression" in stages:
        print("\n[4/4] Training regression model (ablation)...")
        _run([python, "scripts/07_train_regression.py", "--config", config, "--splits", "80"])

    print(f"\n[DONE] Ablation '{name}' complete.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation",
        choices=list(_ABLATIONS.keys()) + ["all"],
        default="all",
        help="Which ablation to run (default: all)",
    )
    parser.add_argument(
        "--start-from",
        choices=_STAGES,
        default="gan",
        help="Stage to start from (default: gan)",
    )
    args = parser.parse_args()

    names = list(_ABLATIONS.keys()) if args.ablation == "all" else [args.ablation]
    for name in names:
        run_ablation(name, _ABLATIONS[name], start_from=args.start_from)


if __name__ == "__main__":
    main()

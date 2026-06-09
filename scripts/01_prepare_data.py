"""Filter and split raw sensor signals into train/test pickle files.

Usage:
    python scripts/01_prepare_data.py --config configs/exp_train_50.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.core.config_parser import load_config
from src.data_prep.signal_filters import process_sensor_dict
from src.data_prep.splitter import extract_channel_2, load_pkl_folder, split_and_save


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp_train_50.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    filter_cfg = cfg["filters"]

    print("Loading raw .pkl files...")
    leak_data = load_pkl_folder(data_cfg["raw_leak_dir"])
    normal_data = load_pkl_folder(data_cfg["raw_normal_dir"])

    print("Extracting channel 2...")
    sensor_data = extract_channel_2(leak_data=leak_data, normal_data=normal_data)

    print("Windowing and filtering...")
    filtered = process_sensor_dict(sensor_data, filter_cfg)

    print("Splitting and saving...")
    split_and_save(
        sensor_data=filtered,
        train_dir=data_cfg["train_dir"],
        test_dir=data_cfg["test_dir"],
        split_ratio=data_cfg["split_ratio"],
        seed=data_cfg["seed"],
    )
    print("Done.")


if __name__ == "__main__":
    main()

import os
import pickle
import random
from pathlib import Path


def load_pkl_folder(folder_path: str) -> dict:
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Warning: folder not found -> {folder_path}")
        return {}
    result = {}
    for f in os.listdir(folder):
        if f.endswith(".pkl"):
            with open(folder / f, "rb") as fh:
                result[f.replace(".pkl", "")] = pickle.load(fh)
    return result


def extract_channel_2(leak_data: dict, normal_data: dict = None) -> dict:
    sensor_data = {}
    if leak_data:
        for i in range(1, 5):
            key = f"c1_qingcheng_{i}_c2_vallen_{i}"
            if key in leak_data:
                sensor_data[f"vallen_{i}_v1"] = [df["Canal 2"] for df in leak_data[key]]
    if normal_data:
        for i in range(1, 4):
            key = f"c1_qingcheng_{i}_c2_vallen_{i}"
            if key in normal_data:
                sensor_data[f"vallen_{i}_v0"] = [df["Canal 2"] for df in normal_data[key]]
    return sensor_data


def split_and_save(
    sensor_data: dict,
    train_dir: str,
    test_dir: str,
    split_ratio: float = 0.8,
    seed: int = 42,
):
    Path(train_dir).mkdir(parents=True, exist_ok=True)
    Path(test_dir).mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    for key, series_list in sensor_data.items():
        indices = list(range(len(series_list)))
        random.shuffle(indices)
        split = int(len(series_list) * split_ratio)
        for subset, idxs, out_dir in [
            ("train", indices[:split], train_dir),
            ("test", indices[split:], test_dir),
        ]:
            with open(Path(out_dir) / f"{key}.pkl", "wb") as f:
                pickle.dump([series_list[i] for i in idxs], f)
    print(f"Split {int(split_ratio * 100)}/{100 - int(split_ratio * 100)} done.")

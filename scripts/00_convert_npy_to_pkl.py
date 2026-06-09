"""Convert raw sensor .npy recordings to pickle format.

Usage:
    python scripts/00_convert_npy_to_pkl.py
"""

import io
import pickle
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))


def _normalize_folder_name(name: str) -> str:
    return name.lower().replace("canal", "c").replace("__", "_")


def _read_raw_file(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        raw = f.read()

    # Sensor dumps are zlib-compressed raw float64 DAQ frames (2 channels interleaved).
    if raw.startswith(b"x\x9c"):
        data = np.frombuffer(zlib.decompress(raw), dtype=np.float64)
        return data.reshape(-1, 2) if len(data) % 2 == 0 else data

    try:
        return np.load(io.BytesIO(raw), allow_pickle=True)
    except Exception as exc:
        raise ValueError(f"Unrecognized format in {path}: {exc}") from exc


def convert_folder_to_pkl(input_dir: str, output_dir: str) -> None:
    """Convert all .npy subfolders in input_dir to a single .pkl per subfolder."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Warning: directory not found: {input_path}")
        return

    for subfolder in input_path.iterdir():
        if not subfolder.is_dir():
            continue
        frames = []
        for npy_file in subfolder.glob("*.npy"):
            try:
                arr = _read_raw_file(npy_file)
                if getattr(arr, "ndim", 1) == 0:
                    item = arr.item()
                    df = item if isinstance(item, pd.DataFrame) else pd.DataFrame(item)
                elif isinstance(arr, np.ndarray):
                    if arr.ndim > 1 and arr.shape[1] >= 2:
                        df = pd.DataFrame(
                            {"channel_1": arr[:, 0], "channel_2": arr[:, 1]}
                        )
                    else:
                        df = pd.DataFrame({"channel_2": arr.flatten()})
                else:
                    df = arr if isinstance(arr, pd.DataFrame) else pd.DataFrame(arr)
                frames.append(df)
            except Exception as exc:
                print(f"Error reading {npy_file.name}: {exc}")

        if frames:
            out_path = output_path / f"{_normalize_folder_name(subfolder.name)}.pkl"
            with open(out_path, "wb") as f:
                pickle.dump(frames, f)
            print(f"Saved: {out_path.name} ({len(frames)} frames)")


def main() -> None:
    base_raw = Path("data/raw")
    convert_folder_to_pkl(base_raw / "leak_5bar", base_raw / "leak_5bar/data_pkl")
    convert_folder_to_pkl(base_raw / "normal", base_raw / "normal/data_pkl")


if __name__ == "__main__":
    main()

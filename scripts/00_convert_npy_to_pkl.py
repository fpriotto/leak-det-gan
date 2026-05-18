import sys
import numpy as np
import pandas as pd
import pickle
import zlib
import io
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))


def normalize_folder_name(name: str) -> str:
    return name.lower().replace("canal", "c").replace("__", "_")


def read_raw_file(path):
    with open(path, "rb") as f:
        raw = f.read()

    # Sensor dumps are zlib-compressed raw float64 DAQ frames (2 channels interleaved).
    if raw.startswith(b"x\x9c"):
        data = np.frombuffer(zlib.decompress(raw), dtype=np.float64)
        return data.reshape(-1, 2) if len(data) % 2 == 0 else data

    try:
        return np.load(io.BytesIO(raw), allow_pickle=True)
    except Exception as e:
        raise ValueError(f"Unrecognized format in {path}: {e}")


def convert_npy_folder_to_pkl(input_dir: str, output_dir: str):
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
                arr = read_raw_file(npy_file)
                if getattr(arr, "ndim", 1) == 0:
                    item = arr.item()
                    df = item if isinstance(item, pd.DataFrame) else pd.DataFrame(item)
                elif isinstance(arr, np.ndarray):
                    if arr.ndim > 1 and arr.shape[1] >= 2:
                        df = pd.DataFrame({"Canal 1": arr[:, 0], "Canal 2": arr[:, 1]})
                    else:
                        df = pd.DataFrame({"Canal 2": arr.flatten()})
                else:
                    df = arr if isinstance(arr, pd.DataFrame) else pd.DataFrame(arr)
                frames.append(df)
            except Exception as e:
                print(f"Error reading {npy_file.name}: {e}")

        if frames:
            out_path = output_path / f"{normalize_folder_name(subfolder.name)}.pkl"
            with open(out_path, "wb") as f:
                pickle.dump(frames, f)
            print(f"Saved: {out_path.name} ({len(frames)} frames)")


def main():
    base_raw = Path("data/raw")
    convert_npy_folder_to_pkl(base_raw / "Vazamento_5_bar", base_raw / "Vazamento_5_bar/data_pkl")
    convert_npy_folder_to_pkl(base_raw / "Normalidade", base_raw / "Normalidade/data_pkl")


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt, cheby2, filtfilt


def apply_butter_downsample(signal, orig_fs, target_fs, cutoff, order):
    factor = orig_fs // target_fs
    Wn = cutoff / (orig_fs / 2)
    sos = butter(N=order, Wn=Wn, btype="low", output="sos")
    sig = signal.values if hasattr(signal, "values") else np.asarray(signal)
    return pd.Series(sosfilt(sos, sig)[::factor])


def apply_chebyshev_bandpass(signal, fs, lowcut, highcut, order, ripple_db):
    nyquist = 0.5 * fs
    b, a = cheby2(order, ripple_db, [lowcut / nyquist, highcut / nyquist], btype="band")
    sig = signal.values if hasattr(signal, "values") else np.asarray(signal)
    return pd.Series(filtfilt(b, a, sig), name=getattr(signal, "name", None))


def chunk_signal(signal_series: pd.Series, chunk_size: int) -> list:
    array = signal_series.values
    n = len(array) // chunk_size
    return [pd.Series(c) for c in np.split(array[: n * chunk_size], n)]


def process_sensor_dict(sensor_data: dict, filter_cfg: dict) -> dict:
    orig_chunk_size = int(filter_cfg["orig_fs"] * 0.01)
    processed = {}
    for key, series_list in sensor_data.items():
        chunks = []
        for serie in series_list:
            for window in chunk_signal(serie, orig_chunk_size):
                s = apply_butter_downsample(
                    window,
                    orig_fs=filter_cfg["orig_fs"],
                    target_fs=filter_cfg["target_fs"],
                    cutoff=filter_cfg["butter_cutoff"],
                    order=filter_cfg["butter_order"],
                )
                s = apply_chebyshev_bandpass(
                    s,
                    fs=filter_cfg["target_fs"],
                    lowcut=filter_cfg["cheby_lowcut"],
                    highcut=filter_cfg["cheby_highcut"],
                    order=filter_cfg["cheby_order"],
                    ripple_db=filter_cfg["cheby_ripple"],
                )
                chunks.append(s)
        processed[key] = chunks
    return processed

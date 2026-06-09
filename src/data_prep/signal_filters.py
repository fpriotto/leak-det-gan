"""Signal filtering and windowing utilities for raw acoustic emission data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any

from scipy.signal import butter, cheby2, filtfilt, sosfilt


def apply_butter_downsample(
    signal: pd.Series | np.ndarray,
    orig_fs: int,
    target_fs: int,
    cutoff: float,
    order: int,
) -> pd.Series:
    """Low-pass filter then downsample a signal by integer factor.

    Args:
        signal: Input time series.
        orig_fs: Original sampling rate (Hz).
        target_fs: Target sampling rate (Hz); must divide orig_fs evenly.
        cutoff: Low-pass cutoff frequency (Hz).
        order: Butterworth filter order.

    Returns:
        Downsampled signal as a pd.Series.
    """
    factor = orig_fs // target_fs
    Wn = cutoff / (orig_fs / 2)
    sos = butter(N=order, Wn=Wn, btype="low", output="sos")
    sig = signal.values if hasattr(signal, "values") else np.asarray(signal)
    return pd.Series(sosfilt(sos, sig)[::factor])


def apply_chebyshev_bandpass(
    signal: pd.Series | np.ndarray,
    fs: int,
    lowcut: float,
    highcut: float,
    order: int,
    ripple_db: float,
) -> pd.Series:
    """Apply a Chebyshev Type II bandpass filter (zero-phase via filtfilt).

    Args:
        signal: Input time series.
        fs: Sampling rate (Hz).
        lowcut: Lower passband edge (Hz).
        highcut: Upper passband edge (Hz).
        order: Filter order.
        ripple_db: Minimum stopband attenuation (dB).

    Returns:
        Filtered signal as a pd.Series.
    """
    nyquist = 0.5 * fs
    b, a = cheby2(order, ripple_db, [lowcut / nyquist, highcut / nyquist], btype="band")
    sig = signal.values if hasattr(signal, "values") else np.asarray(signal)
    return pd.Series(filtfilt(b, a, sig), name=getattr(signal, "name", None))


def chunk_signal(signal_series: pd.Series, chunk_size: int) -> list[pd.Series]:
    """Split a series into non-overlapping fixed-length chunks.

    Trailing samples that do not fill a complete chunk are discarded.

    Args:
        signal_series: Input time series.
        chunk_size: Number of samples per chunk.

    Returns:
        List of equal-length pd.Series chunks.
    """
    array = signal_series.values
    n = len(array) // chunk_size
    return [pd.Series(c) for c in np.split(array[: n * chunk_size], n)]


def process_sensor_dict(
    sensor_data: dict[str, list[pd.Series]],
    filter_cfg: dict[str, Any],
) -> dict[str, list[pd.Series]]:
    """Window, downsample, and bandpass-filter all signals in a sensor dict.

    Each raw recording is split into 10 ms windows at the original sampling
    rate, then each window is downsampled and bandpass-filtered.

    Args:
        sensor_data: Mapping of sensor key → list of raw signal series.
        filter_cfg: Filter parameters from the YAML config (``filters`` block).

    Returns:
        Same structure as ``sensor_data`` but with processed, shorter signals.
    """
    downsample_factor = filter_cfg["orig_fs"] // filter_cfg["target_fs"]
    orig_chunk_size = filter_cfg["chunk_size"] * downsample_factor
    processed: dict[str, list[pd.Series]] = {}
    for key, series_list in sensor_data.items():
        chunks: list[pd.Series] = []
        for serie in series_list:
            for window in chunk_signal(serie, orig_chunk_size):
                s = apply_butter_downsample(
                    window,
                    orig_fs=filter_cfg["orig_fs"],
                    target_fs=filter_cfg["target_fs"],
                    cutoff=filter_cfg["butter_cutoff"],
                    order=filter_cfg["butter_order"],
                )
                if filter_cfg.get("use_cheby", True):
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

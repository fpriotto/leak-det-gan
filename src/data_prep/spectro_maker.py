"""Spectrogram computation, colour scaling, and PNG export utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import cm
from PIL import Image
from scipy.signal import spectrogram
from tqdm import tqdm


def get_freq_mask(
    fs: int,
    nperseg: int,
    f_min: float,
    f_max: float,
) -> np.ndarray:
    """Return a boolean mask selecting frequency bins in [f_min, f_max].

    Args:
        fs: Sampling rate (Hz).
        nperseg: STFT window length (samples).
        f_min: Lower frequency bound (Hz).
        f_max: Upper frequency bound (Hz).

    Returns:
        Boolean array of shape ``(nperseg // 2 + 1,)``.
    """
    f = np.fft.rfftfreq(nperseg, d=1 / fs)
    return (f >= f_min) & (f <= f_max)


def calc_global_scale(
    series_dict: dict[str, list[Any]],
    fs: int,
    nperseg: int,
    noverlap: int,
    mask: np.ndarray | None = None,
    pct_low: float = 2,
    pct_high: float = 98,
) -> tuple[float, float]:
    """Compute a robust global log-power colour scale across all signals.

    Args:
        series_dict: Mapping of class name → list of signal series.
        fs: Sampling rate (Hz).
        nperseg: STFT window length (samples).
        noverlap: Number of overlapping samples between windows.
        mask: Optional frequency mask to restrict the scale computation.
        pct_low: Lower percentile for vmin.
        pct_high: Upper percentile for vmax.

    Returns:
        Tuple of (vmin, vmax) in dB.
    """
    vals: list[np.ndarray] = []
    for series_list in series_dict.values():
        for serie in series_list:
            sig = serie.values if hasattr(serie, "values") else np.asarray(serie)
            _, _, Sxx = spectrogram(sig, fs=fs, nperseg=nperseg, noverlap=noverlap)
            S = Sxx[mask] if mask is not None else Sxx
            vals.append((10 * np.log10(S + 1e-12)).ravel())
    all_vals = np.concatenate(vals)
    return float(np.percentile(all_vals, pct_low)), float(np.percentile(all_vals, pct_high))


def add_awgn(signal: np.ndarray, snr_db: float | None) -> np.ndarray:
    """Add additive white Gaussian noise at a given SNR.

    Args:
        signal: Input signal array.
        snr_db: Target SNR in dB. Pass None to return the signal unchanged.

    Returns:
        Noisy signal (or original if snr_db is None or signal power is zero).
    """
    if snr_db is None:
        return signal
    sig_power = np.mean(signal**2)
    if sig_power == 0:
        return signal
    noise_power = sig_power / (10 ** (snr_db / 10))
    return signal + np.random.normal(0, np.sqrt(noise_power), len(signal))


def save_spectrograms_as_png(
    series_list: list[Any],
    out_folder: str | Path,
    vmin: float,
    vmax: float,
    fs: int,
    nperseg: int,
    noverlap: int,
    mask: np.ndarray | None,
    img_size: tuple[int, int] | list[int],
    snr_db: float | None = None,
) -> None:
    """Compute log-spectrograms and save each as a magma-coloured PNG.

    Args:
        series_list: List of signal series to process.
        out_folder: Destination directory (created if absent).
        vmin: Global colour scale minimum (dB).
        vmax: Global colour scale maximum (dB).
        fs: Sampling rate (Hz).
        nperseg: STFT window length (samples).
        noverlap: Overlap between consecutive windows (samples).
        mask: Frequency mask for cropping the spectrogram rows. Pass None
            to use the full spectrum.
        img_size: Output image dimensions as (width, height) or [w, h].
        snr_db: If set, adds AWGN at this SNR before computing the STFT.
    """
    Path(out_folder).mkdir(parents=True, exist_ok=True)
    for idx, serie in enumerate(tqdm(series_list, desc=str(Path(out_folder).name))):
        sig = serie.values if hasattr(serie, "values") else np.asarray(serie)
        sig = add_awgn(sig, snr_db)
        _, _, Sxx = spectrogram(sig, fs=fs, nperseg=nperseg, noverlap=noverlap)
        S_crop = Sxx[mask, :] if mask is not None else Sxx
        if np.isnan(S_crop).any():
            continue
        S_log = np.flipud(10 * np.log10(S_crop + 1e-12))
        norm = np.clip((S_log - vmin) / (vmax - vmin), 0, 1)
        rgb = (cm.magma(norm)[..., :3] * 255).astype(np.uint8)
        img = Image.fromarray(rgb).resize(tuple(img_size), Image.LANCZOS)
        img.save(Path(out_folder) / f"{idx:04d}.png", format="PNG", optimize=True)

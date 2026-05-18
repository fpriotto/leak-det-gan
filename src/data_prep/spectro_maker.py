import numpy as np
from scipy.signal import spectrogram
from matplotlib import cm
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def get_freq_mask(fs, nperseg, f_min, f_max):
    f = np.fft.rfftfreq(nperseg, d=1 / fs)
    return (f >= f_min) & (f <= f_max)


def calc_global_scale(series_dict, fs, nperseg, noverlap, mask):
    vals = []
    for series_list in series_dict.values():
        for serie in series_list:
            sig = serie.values if hasattr(serie, "values") else np.asarray(serie)
            _, _, Sxx = spectrogram(sig, fs=fs, nperseg=nperseg, noverlap=noverlap)
            vals.append((10 * np.log10(Sxx[mask] + 1e-12)).ravel())
    vals = np.concatenate(vals)
    return vals.min(), vals.max()


def add_awgn(signal, snr_db):
    if snr_db is None:
        return signal
    sig_power = np.mean(signal**2)
    if sig_power == 0:
        return signal
    noise_power = sig_power / (10 ** (snr_db / 10))
    return signal + np.random.normal(0, np.sqrt(noise_power), len(signal))


def save_spectrograms_as_png(
    series_list, out_folder, vmin, vmax, fs, nperseg, noverlap, mask, img_size, snr_db=None
):
    Path(out_folder).mkdir(parents=True, exist_ok=True)
    for idx, serie in enumerate(tqdm(series_list, desc=str(Path(out_folder).name))):
        sig = serie.values if hasattr(serie, "values") else np.asarray(serie)
        sig = add_awgn(sig, snr_db)
        _, _, Sxx = spectrogram(sig, fs=fs, nperseg=nperseg, noverlap=noverlap)
        S_crop = Sxx[mask, :]
        if np.isnan(S_crop).any():
            continue
        S_log = np.flipud(10 * np.log10(S_crop + 1e-12))
        norm = np.clip((S_log - vmin) / (vmax - vmin), 0, 1)
        rgb = (cm.magma(norm)[..., :3] * 255).astype(np.uint8)
        img = Image.fromarray(rgb).resize(tuple(img_size), Image.LANCZOS)
        img.save(Path(out_folder) / f"{idx:04d}.png", format="PNG", optimize=True)

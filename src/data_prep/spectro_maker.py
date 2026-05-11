"""
Arquivo: src/data_prep/spectro_maker.py
Descrição: Converte séries temporais filtradas 1D em matrizes de espectrogramas e salva como imagens .png (escala magma).
"""

import numpy as np
from scipy.signal import spectrogram
from matplotlib import cm
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def get_freq_mask(fs, nperseg, f_min, f_max):
    """Cria a máscara booleana para recortar apenas a banda de frequência desejada."""
    f = np.fft.rfftfreq(nperseg, d=1 / fs)
    return (f >= f_min) & (f <= f_max)


def calc_global_scale(series_dict, fs, nperseg, noverlap, mask):
    """Calcula o VMIN e VMAX global para que todas as imagens tenham a mesma escala de cor."""
    vals = []
    for key, series_list in series_dict.items():
        for serie in series_list:
            sig = serie.values if hasattr(serie, "values") else np.asarray(serie)
            _, _, Sxx = spectrogram(sig, fs=fs, nperseg=nperseg, noverlap=noverlap)
            S_log = 10 * np.log10(Sxx[mask] + 1e-12)
            vals.append(S_log.ravel())

    vals = np.concatenate(vals)
    return vals.min(), vals.max()


def save_spectrograms_as_png(
    series_list,
    out_folder,
    vmin,
    vmax,
    fs,
    nperseg,
    noverlap,
    mask,
    img_size,
    snr_db=None,
):
    """Converte sinais em espectrogramas logarítmicos (com ou sem AWGN) e salva como PNG."""
    Path(out_folder).mkdir(parents=True, exist_ok=True)

    for idx, serie in enumerate(
        tqdm(series_list, desc=f"Salvando em {Path(out_folder).name}")
    ):
        sig = serie.values if hasattr(serie, "values") else np.asarray(serie)

        # INJEÇÃO DE RUÍDO AQUI
        sig = add_awgn(sig, snr_db)

        _, _, Sxx = spectrogram(sig, fs=fs, nperseg=nperseg, noverlap=noverlap)

        S_crop = Sxx[mask, :]
        if np.isnan(S_crop).any():
            continue

        S_log = 10 * np.log10(S_crop + 1e-12)
        S_log = np.flipud(S_log)

        norm = np.clip((S_log - vmin) / (vmax - vmin), 0, 1)
        rgb = (cm.magma(norm)[..., :3] * 255).astype(np.uint8)

        img = Image.fromarray(rgb).resize(tuple(img_size), Image.LANCZOS)
        img.save(Path(out_folder) / f"{idx:04d}.png", format="PNG", optimize=True)


def add_awgn(signal, snr_db):
    """Adiciona Ruído Branco Gaussiano Aditivo (AWGN) ao sinal 1D baseado no SNR em dB."""
    if snr_db is None:
        return signal

    sig_power = np.mean(signal**2)

    # Previne divisão por zero caso o sinal seja puro silêncio
    if sig_power == 0:
        return signal

    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise

"""
Arquivo: src/data_prep/signal_filters.py
Descrição: Funções matemáticas de processamento de sinal (Filtros Butterworth, Chebyshev Tipo II e downsampling).
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt, cheby2, filtfilt


def apply_butter_downsample(signal, orig_fs, target_fs, cutoff, order):
    factor = orig_fs // target_fs
    nyquist = orig_fs / 2
    Wn = cutoff / nyquist

    sos = butter(N=order, Wn=Wn, btype="low", output="sos")
    sig_array = signal.values if hasattr(signal, "values") else np.asarray(signal)

    filtered = sosfilt(sos, sig_array)
    downsampled = filtered[::factor]

    return pd.Series(downsampled)


def apply_chebyshev_bandpass(signal, fs, lowcut, highcut, order, ripple_db):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = cheby2(order, ripple_db, [low, high], btype="band")
    sig_array = signal.values if hasattr(signal, "values") else np.asarray(signal)
    filtered = filtfilt(b, a, sig_array)

    return pd.Series(filtered, name=getattr(signal, "name", None))


def chunk_signal(signal_series: pd.Series, chunk_size: int) -> list:
    """Divide um sinal longo em dezenas/centenas de pequenas fatias (janelas)."""
    array = signal_series.values
    total_chunks = len(array) // chunk_size

    # Descarta o 'farelo' no final para manter pacotes de tamanho exato
    array = array[: total_chunks * chunk_size]

    # Divide num formato de lista de arrays e devolve como Séries
    chunks = np.split(array, total_chunks)
    return [pd.Series(c) for c in chunks]


def process_sensor_dict(sensor_data: dict, filter_cfg: dict) -> dict:
    processed_data = {}

    # Chunk original em 1MHz para 10ms = 10.000 amostras
    orig_chunk_size = int(filter_cfg["orig_fs"] * 0.01)

    for key, series_list in sensor_data.items():
        processed_list = []
        for serie in series_list:
            # 1. FATIA PRIMEIRO (Garante a consistência metodológica)
            fatias = chunk_signal(serie, chunk_size=orig_chunk_size)

            # 2. APLICA OS FILTROS EM CADA FATIA
            for fatia in fatias:
                # Downsampling de 1MHz para 250kHz (fatia cai de 10.000 para 2.500 pontos)
                s1 = apply_butter_downsample(
                    fatia,
                    orig_fs=filter_cfg["orig_fs"],
                    target_fs=filter_cfg["target_fs"],
                    cutoff=filter_cfg["butter_cutoff"],
                    order=filter_cfg["butter_order"],
                )

                # Aplica o passa-faixa para focar na banda de 25-80 kHz
                s2 = apply_chebyshev_bandpass(
                    s1,
                    fs=filter_cfg["target_fs"],
                    lowcut=filter_cfg["cheby_lowcut"],
                    highcut=filter_cfg["cheby_highcut"],
                    order=filter_cfg["cheby_order"],
                    ripple_db=filter_cfg["cheby_ripple"],
                )

                processed_list.append(s2)

        processed_data[key] = processed_list

    return processed_data

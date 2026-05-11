"""
Arquivo: src/data_prep/splitter.py
Descrição: Lógica de separação dos sinais originais .pkl em bases de treino e teste estratificadas.
"""

import os
import pickle
import random
from pathlib import Path


def carregar_pasta_pkl(pasta_path: str) -> dict:
    """Carrega todos os arquivos .pkl de uma pasta para um dicionário."""
    pasta = Path(pasta_path)
    if not pasta.exists():
        print(f"Aviso: Pasta não encontrada -> {pasta_path}")
        return {}

    dict_loaded = {}
    pkl_files = [f for f in os.listdir(pasta) if f.endswith(".pkl")]

    for file in pkl_files:
        key = file.replace(".pkl", "")
        file_path = pasta / file
        with open(file_path, "rb") as f:
            dict_loaded[key] = pickle.load(f)

    return dict_loaded


def extrair_canal_2(dict_vazamento: dict, dict_normalidade: dict = None) -> dict:
    """Extrai apenas o 'Canal 2' dos sensores Vallen para vazamento e normalidade."""
    sensor_data = {}

    # Extrai Vazamento (v1)
    if dict_vazamento:
        for i in range(1, 5):  # 1 a 4
            chave_orig = f"c1_qingcheng_{i}_c2_vallen_{i}"
            if chave_orig in dict_vazamento:
                sensor_data[f"vallen_{i}_v1"] = [
                    df["Canal 2"] for df in dict_vazamento[chave_orig]
                ]

    # Extrai Normalidade (v0) - baseado no seu segundo notebook
    if dict_normalidade:
        for i in range(1, 4):  # 1 a 3 (no seu código de normalidade ia até 3)
            chave_orig = f"c1_qingcheng_{i}_c2_vallen_{i}"
            if chave_orig in dict_normalidade:
                sensor_data[f"vallen_{i}_v0"] = [
                    df["Canal 2"] for df in dict_normalidade[chave_orig]
                ]

    return sensor_data


def split_e_salvar(
    sensor_data: dict,
    pasta_treino: str,
    pasta_teste: str,
    split_ratio: float = 0.8,
    seed: int = 42,
):
    """Divide os dados e salva em pastas separadas de treino e teste."""
    Path(pasta_treino).mkdir(parents=True, exist_ok=True)
    Path(pasta_teste).mkdir(parents=True, exist_ok=True)

    random.seed(seed)

    for key, series_list in sensor_data.items():
        total = len(series_list)
        indices = list(range(total))
        random.shuffle(indices)

        split_point = int(total * split_ratio)
        treino_idxs = indices[:split_point]
        teste_idxs = indices[split_point:]

        treino_data = [series_list[i] for i in treino_idxs]
        teste_data = [series_list[i] for i in teste_idxs]

        with open(Path(pasta_treino) / f"{key}.pkl", "wb") as f_out:
            pickle.dump(treino_data, f_out)

        with open(Path(pasta_teste) / f"{key}.pkl", "wb") as f_out:
            pickle.dump(teste_data, f_out)

    print(
        f"Divisão {int(split_ratio * 100)}/{(100 - int(split_ratio * 100))} concluída com sucesso!"
    )

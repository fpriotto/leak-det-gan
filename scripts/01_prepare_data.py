"""
Script: scripts/01_prepare_data.py
Descrição: Pipeline de ETL inicial. Lê os dados brutos (.pkl), extrai os sinais, fatia em janelas de 10ms, aplica os filtros em cada janela, divide em treino/teste e salva na pasta processed/.
Uso: python scripts/01_prepare_data.py --config configs/exp_treino_50.yaml
"""

import sys
from pathlib import Path

# Garante que o python ache a pasta 'src'
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.core.config_parser import load_config
from src.data_prep.splitter import carregar_pasta_pkl, extrair_canal_2, split_e_salvar
from src.data_prep.signal_filters import process_sensor_dict


def main():
    config_path = "configs/exp_treino_50.yaml"
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    filter_cfg = cfg["filters"]

    print("Iniciando preparação de dados...")

    # 1. Carrega dados brutos convertidos
    print("Carregando arquivos .pkl originais...")
    dict_vaz = carregar_pasta_pkl(data_cfg["raw_vazamento_dir"])
    dict_norm = carregar_pasta_pkl(data_cfg["raw_normalidade_dir"])

    # 2. Extrai as features de interesse (Canal 2)
    print("Extraindo Canal 2 dos sensores...")
    sensor_data = extrair_canal_2(dict_vazamento=dict_vaz, dict_normalidade=dict_norm)

    # 3. Fatiamento e Filtros (Nova Ordem da Dissertação!)
    print(
        "Fatiando os sinais em janelas de 10ms e aplicando filtros Butterworth e Chebyshev (isso pode demorar uns segundos)..."
    )
    sensor_data_filtrado = process_sensor_dict(sensor_data, filter_cfg)

    # 4. Divide e salva
    print("Dividindo as fatias e salvando em treino/teste...")
    split_e_salvar(
        sensor_data=sensor_data_filtrado,
        pasta_treino=data_cfg["treino_dir"],
        pasta_teste=data_cfg["teste_dir"],
        split_ratio=data_cfg["split_ratio"],
        seed=data_cfg["seed"],
    )
    print("Pipeline de preparação finalizado!")


if __name__ == "__main__":
    main()

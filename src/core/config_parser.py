"""
Arquivo: src/core/config_parser.py
Descrição: Responsável por ler os arquivos .yaml da pasta configs/ e transformar em dicionários Python.
"""

import yaml
from pathlib import Path


def load_config(config_path: str) -> dict:
    """Lê um arquivo .yaml e retorna um dicionário Python."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo de configuração não encontrado: {config_path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

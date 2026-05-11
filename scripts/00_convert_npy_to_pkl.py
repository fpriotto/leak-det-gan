"""
Script: scripts/00_convert_npy_to_pkl.py
Descrição: Descompacta dumps de memória raw (zlib) de sensores de 1MHz e converte para .pkl.
Uso: python scripts/00_convert_npy_to_pkl.py
"""

import os
import numpy as np
import pandas as pd
import pickle
import zlib
import io
from pathlib import Path


def padronizar_nome_pasta(nome_pasta: str) -> str:
    """Padroniza 'Canal1_QingCheng_1__Canal2_Vallen_1' para 'c1_qingcheng_1_c2_vallen_1'."""
    nome = nome_pasta.lower()
    nome = nome.replace("canal", "c")
    nome = nome.replace("__", "_")
    return nome


def ler_arquivo_teimoso(caminho):
    """Lê o arquivo lidando com a compressão zlib e o raw memory dump (sem header)."""
    with open(caminho, "rb") as f:
        raw_bytes = f.read()

    # 1. É ZLIB! (Confirmado pelo Hex '78 9c')
    if raw_bytes.startswith(b"x\x9c"):
        decompressed = zlib.decompress(raw_bytes)

        # Como vimos no debug, não há header Pickle nem Numpy. É um RAW memory dump.
        try:
            # Transforma os bytes puros em float32 (formato padrão de aquisição DAQ)
            arr = np.frombuffer(decompressed, dtype=np.float64)

            # Sabemos que são 2 canais, então transformamos o vetor 1D em matriz (N, 2)
            if len(arr) % 2 == 0:
                return arr.reshape(-1, 2)
            else:
                return arr  # Se por acaso for 1 canal só

        except Exception as e:
            # Fallback caso a precisão seja double (float64) ao invés de float32
            arr = np.frombuffer(decompressed, dtype=np.float64)
            if len(arr) % 2 == 0:
                return arr.reshape(-1, 2)
            return arr

    # 2. Se não for zlib, tenta Numpy padrão
    try:
        f_io = io.BytesIO(raw_bytes)
        return np.load(f_io, allow_pickle=True)
    except Exception as e:
        raise ValueError(f"Não é ZLIB e o Numpy recusou: {str(e)}")


def converter_pasta_npy_para_pkl(input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"⚠️ Diretório não encontrado: {input_path}")
        return

    for subpasta in input_path.iterdir():
        if not subpasta.is_dir():
            continue

        lista_dfs = []
        for npy_file in subpasta.glob("*.npy"):
            try:
                array_data = ler_arquivo_teimoso(npy_file)

                # Independente do formato que saiu da caixa, transforma em DataFrame
                if getattr(array_data, "ndim", 1) == 0:
                    data = array_data.item()
                    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
                elif isinstance(array_data, np.ndarray):
                    if array_data.ndim > 1 and array_data.shape[1] >= 2:
                        df = pd.DataFrame(
                            {"Canal 1": array_data[:, 0], "Canal 2": array_data[:, 1]}
                        )
                    else:
                        df = pd.DataFrame({"Canal 2": array_data.flatten()})
                else:
                    df = (
                        array_data
                        if isinstance(array_data, pd.DataFrame)
                        else pd.DataFrame(array_data)
                    )

                lista_dfs.append(df)
            except Exception as e:
                print(f"❌ Erro ao ler {npy_file.name}: {e}")

        if lista_dfs:
            nome_pkl = f"{padronizar_nome_pasta(subpasta.name)}.pkl"
            caminho_pkl = output_path / nome_pkl
            with open(caminho_pkl, "wb") as f:
                pickle.dump(lista_dfs, f)
            print(f"✅ Salvo: {caminho_pkl.name} ({len(lista_dfs)} arquivos agrupados)")


def main():
    print("Iniciando conversão de Dumps de Memória (Raw Float32)...")

    base_raw = Path("data/raw")
    vazamento_input = base_raw / "Vazamento_5_bar"
    vazamento_output = base_raw / "Vazamento_5_bar/data_pkl"
    normalidade_input = base_raw / "Normalidade"
    normalidade_output = base_raw / "Normalidade/data_pkl"

    print("\n--- Processando Vazamento ---")
    converter_pasta_npy_para_pkl(vazamento_input, vazamento_output)

    print("\n--- Processando Normalidade ---")
    converter_pasta_npy_para_pkl(normalidade_input, normalidade_output)

    print("\nConversão finalizada! Dados resgatados com sucesso do Limbo.")


if __name__ == "__main__":
    main()

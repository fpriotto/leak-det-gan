"""
Script: scripts/03_generate_specs.py
Descrição: Usa os dados processados e a GAN treinada para gerar as imagens .png dos espectrogramas,
           aplicando diferentes níveis de AWGN (SNR) para validação de robustez.
Uso: python scripts/03_generate_specs.py
"""

import sys
import torch
import joblib
import pickle
import random
import numpy as np
from pathlib import Path

# Garante que o python ache a pasta 'src'
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.core.config_parser import load_config
from src.models.gan_components import Gen1D
from src.data_prep.spectro_maker import (
    get_freq_mask,
    calc_global_scale,
    save_spectrograms_as_png,
)


def carregar_dados_treino(pasta_treino):
    data = {}
    for pkl_file in Path(pasta_treino).glob("*.pkl"):
        key = pkl_file.stem
        with open(pkl_file, "rb") as f:
            data[key] = pickle.load(f)
    return data


def main():
    print("Iniciando a Fábrica de Espectrogramas (Modo Interpolação + AWGN)...")

    cfg = load_config("configs/exp_treino_50.yaml")
    spc_cfg = cfg["spectro"]
    data_cfg = cfg["data"]
    gan_cfg = cfg["gan"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Carregando dados reais fatiados...")
    dados_reais = carregar_dados_treino(data_cfg["treino_dir"])
    mask = get_freq_mask(
        spc_cfg["fs"], spc_cfg["nperseg"], spc_cfg["f_min"], spc_cfg["f_max"]
    )

    print("Calculando VMIN e VMAX global...")
    # Calcula a escala com os dados originais (limpos) para manter a consistência de cor
    vmin, vmax = calc_global_scale(
        dados_reais, spc_cfg["fs"], spc_cfg["nperseg"], spc_cfg["noverlap"], mask
    )
    print(f"🎨 Escala global de cor: {vmin:.2f} → {vmax:.2f} dB")

    # Níveis de SNR para avaliação
    snrs_teste = [50, 40, 30, 20, 10]

    print("\n[1/2] Gerando PNGs dos Sinais Reais (Múltiplos SNRs para Teste)...")
    for snr in snrs_teste:
        print(f"\n--- Processando Sinais Reais com SNR {snr}dB ---")
        out_dir_real = Path(f"data/spectrograms/snr_{snr}/reais")

        # Pote para agrupar todas as normalidades antes de salvar
        pool_normalidade = []

        for key, series_list in dados_reais.items():
            if "_v0" in key:
                # Se for normalidade, joga no pote coletivo
                pool_normalidade.extend(series_list)
            else:
                # Se for vazamento, processa por posição normalmente
                map_idx = {1: 10, 2: 25, 3: 60}
                idx = int(key.split("_")[1])
                if idx not in map_idx:
                    continue

                pasta_out = out_dir_real / f"pos_{map_idx[idx]}m"
                save_spectrograms_as_png(
                    series_list,
                    pasta_out,
                    vmin,
                    vmax,
                    spc_cfg["fs"],
                    spc_cfg["nperseg"],
                    spc_cfg["noverlap"],
                    mask,
                    spc_cfg["img_size"],
                    snr_db=snr,  # Injeção do ruído
                )

        # Após o loop, embaralha a normalidade e salva apenas a quantidade limite 1 ÚNICA VEZ
        random.seed(
            data_cfg.get("seed", 42)
        )  # Garante que o embaralhamento seja sempre igual
        random.shuffle(pool_normalidade)

        lista_norm_limitada = pool_normalidade[: spc_cfg["n_norm_samples"]]
        pasta_out_norm = out_dir_real / "normalidade"

        save_spectrograms_as_png(
            lista_norm_limitada,
            pasta_out_norm,
            vmin,
            vmax,
            spc_cfg["fs"],
            spc_cfg["nperseg"],
            spc_cfg["noverlap"],
            mask,
            spc_cfg["img_size"],
            snr_db=snr,  # Injeção do ruído
        )

    print(
        "\n[2/2] Gerando sinais sintéticos INTERPOLADOS com a GAN (Apenas 50dB para Treino)..."
    )
    meta_path = Path("data/modelo_gan/meta_normalizacao_80_pct.save")
    meta = joblib.load(meta_path)

    G = Gen1D(
        z_dim=meta["z_dim"], cond_dim=gan_cfg["cond_dim"], seq_len=meta["SEQ_LEN"]
    ).to(device)
    G.load_state_dict(
        torch.load("data/modelo_gan/generator_wgan_80_pct.pth", map_location=device)
    )
    G.eval()

    distancias_alvo = list(range(10, 65, 5))
    log_min = np.log10(meta["pos_min"])
    log_max = np.log10(meta["pos_max"])

    def denorm_time(sig_scaled):
        return (sig_scaled + 1) / 2 * (meta["GLOBAL_MAX"] - meta["GLOBAL_MIN"]) + meta[
            "GLOBAL_MIN"
        ]

    # Geramos dados sintéticos APENAS para o SNR de 50dB (Base Line de Treinamento)
    snr_treino = 50
    out_dir_gen = Path(f"data/spectrograms/snr_{snr_treino}/gerados")

    with torch.no_grad():
        for dist_m in distancias_alvo:
            print(f"Forjando vazamentos sintéticos para pos_{dist_m}m...")

            log_dist = np.log10(dist_m)
            cond_val = (log_dist - log_min) / (log_max - log_min)

            c_tensor = torch.full(
                (spc_cfg["n_gen_per_class"], 1),
                cond_val,
                dtype=torch.float32,
                device=device,
            )
            z_tensor = torch.randn(
                spc_cfg["n_gen_per_class"], meta["z_dim"], device=device
            )

            fakes_tensor = G(z_tensor, c_tensor).squeeze(1).cpu().numpy()
            fakes_list = [denorm_time(fake_sig) for fake_sig in fakes_tensor]

            pasta_out = out_dir_gen / f"pos_{dist_m}m"
            save_spectrograms_as_png(
                fakes_list,
                pasta_out,
                vmin,
                vmax,
                spc_cfg["fs"],
                spc_cfg["nperseg"],
                spc_cfg["noverlap"],
                mask,
                spc_cfg["img_size"],
                snr_db=snr_treino,  # Injeção do ruído base de treino
            )

    print(
        "\n✅ Todos os Espectrogramas Reais e Sintéticos criados e organizados por SNR!"
    )


if __name__ == "__main__":
    main()

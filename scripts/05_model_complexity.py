import sys
import torch
from pathlib import Path
from thop import profile, clever_format

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.core.config_parser import load_config
from src.models.cascade_vit import BinaryViT, LeakRegressorViT
from src.models.cnn_baselines import ResNetBaseline


def calcular_complexidade(modelo, nome_modelo, img_size):
    device = torch.device("cpu")
    modelo = modelo.to(device)

    # Cria uma imagem falsa com 1 canal (escala de cinza)
    entrada_falsa = torch.randn(1, 1, img_size, img_size).to(device)

    # Calcula os MACs e Parâmetros usando a biblioteca thop
    macs, params = profile(modelo, inputs=(entrada_falsa,), verbose=False)

    macs_fmt, params_fmt = clever_format([macs, params], "%.3f")
    flops = (macs * 2) / 1e9  # 1 MAC ~= 2 FLOPs

    print(f"\n=== Complexidade: {nome_modelo} (Input: {img_size}x{img_size}) ===")
    print(f"Parâmetros: {params_fmt} ({int(params):,} parâmetros)")
    print(f"MACs:       {macs_fmt}")
    print(f"FLOPs:      {flops:.4f} GFLOPs")
    print("=" * 60)


def main():
    cfg = load_config("configs/exp_treino_50.yaml")
    eval_cfg = cfg["eval_model"]
    img_size = eval_cfg["img_size"]
    vp = eval_cfg["vit_params"]

    print("Calculando complexidade dos modelos para a dissertação...\n")

    # 1. Avalia ViT Binário
    vit_bin = BinaryViT(
        img_size=img_size,
        patch_size=vp["patch_size"],
        emb_dim=vp["emb_dim"],
        n_layers=vp["n_layers_bin"],
        n_heads=vp["n_heads"],
    )
    calcular_complexidade(vit_bin, "Vision Transformer (Binário)", img_size)

    # 2. Avalia ViT Regressão
    vit_reg = LeakRegressorViT(
        img_size=img_size,
        patch_size=vp["patch_size"],
        emb_dim=vp["emb_dim"],
        n_layers=vp["n_layers_reg"],
        n_heads=vp["n_heads"],
    )
    calcular_complexidade(vit_reg, "Vision Transformer (Regressão)", img_size)

    # 3. Avalia ResNet-18 (Benchmark)
    resnet = ResNetBaseline(is_classifier=True)
    calcular_complexidade(resnet, "ResNet-18 (Classificador)", img_size)


if __name__ == "__main__":
    main()

"""Report FLOPs and parameter counts for ViT and CNN baseline models.

Usage:
    python scripts/05_model_complexity.py
"""

import sys
import torch
from pathlib import Path
from thop import profile, clever_format

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.core.config_parser import load_config
from src.models.cascade_vit import BinaryViT, LeakPosViT
from src.models.cnn_baselines import SimpleCNN, ResNetBaseline


def report(model, name: str, img_size: int):
    model = model.cpu()
    dummy = torch.randn(1, 1, img_size, img_size)
    macs, params = profile(model, inputs=(dummy,), verbose=False)
    macs_fmt, params_fmt = clever_format([macs, params], "%.3f")
    print(f"\n{name}  ({img_size}×{img_size})")
    print(f"  Params : {params_fmt}  ({int(params):,})")
    print(f"  MACs   : {macs_fmt}")
    print(f"  FLOPs  : {(macs * 2) / 1e9:.4f} G")


def main():
    cfg = load_config("configs/exp_train_50.yaml")
    vp = cfg["eval_model"]["vit_params"]
    img_size = cfg["eval_model"]["img_size"]
    n_pos = 6

    report(
        BinaryViT(
            img_size, vp["patch_size"], vp["emb_dim"], vp["n_layers_bin"], vp["n_heads"]
        ),
        "ViT Binary",
        img_size,
    )
    report(
        LeakPosViT(
            img_size,
            vp["patch_size"],
            vp["emb_dim"],
            vp["n_layers_bin"],
            vp["n_heads"],
            n_pos,
        ),
        "ViT Position",
        img_size,
    )
    report(SimpleCNN(n_classes=2), "SimpleCNN Binary", img_size)
    report(SimpleCNN(n_classes=n_pos), "SimpleCNN Position", img_size)
    report(ResNetBaseline(n_classes=2), "ResNet-18 Binary", img_size)
    report(ResNetBaseline(n_classes=n_pos), "ResNet-18 Position", img_size)


if __name__ == "__main__":
    main()

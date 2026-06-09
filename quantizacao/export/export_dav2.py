#!/usr/bin/env python
"""
Export ONNX do Depth Anything V2 (ViT-S metric) fine-tunado no ICL Ground Robot.

Carrega o checkpoint do projeto (model_state_dict) e exporta:
    rgb[0,1] -> DepthAnythingV2.forward -> profundidade [B,1,H,W] em metros.

ATENÇÃO à normalização: o DAV2 NÃO normaliza dentro do forward (a norm.
ImageNet fica no image2tensor/infer_image, que o treino não usou). Logo o
modelo foi treinado com [0,1] cru e o ONNX recebe [0,1] cru — coerente.

Resolução: múltiplo de 14 (patch do DINOv2). O DINOv2 interpola o
pos-encoding para a grade pedida; com HxW fixo isso vira constante no grafo.

Uso:
    python export/export_dav2.py --encoder vits --max_depth 20 \
        --height 364 --width 518 --output onnx/dav2_vits_364x518.onnx --simplify
"""

from __future__ import annotations

import argparse
import torch
import torch.nn as nn

from _common import (add_external_paths, resolve_checkpoint, export_onnx,
                     sanity_forward)

add_external_paths()

DAV2_ENCODER_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


class DAV2Exportable(nn.Module):
    """Garante saída [B,1,H,W] (o forward nativo devolve [B,H,W])."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        depth = self.model(rgb)          # [B,H,W] em metros (já * max_depth)
        return depth.unsqueeze(1)        # [B,1,H,W]


def build_dav2(checkpoint, encoder: str, max_depth: float) -> nn.Module:
    from depth_anything_v2.dpt import DepthAnythingV2

    cfg = DAV2_ENCODER_CONFIGS[encoder]
    model = DepthAnythingV2(**cfg, max_depth=max_depth)

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = state["model_state_dict"] if "model_state_dict" in state else state
    model.load_state_dict(sd)
    vm = state.get("val_metrics", {}) if isinstance(state, dict) else {}
    print(f"[dav2] checkpoint carregado (encoder={encoder}, max_depth={max_depth}, "
          f"val abs_rel={vm.get('abs_rel', float('nan')):.4f})")

    return DAV2Exportable(model).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="default: modelos_treinados/dav2_vits_*_best.pth (escolha frozen/trainable)")
    ap.add_argument("--encoder", default="vits", choices=["vits", "vitb", "vitl"])
    ap.add_argument("--max_depth", type=float, default=20.0,
                    help="20m do checkpoint Hypersim usado no treino.")
    ap.add_argument("--height", type=int, default=364, help="múltiplo de 14")
    ap.add_argument("--width", type=int, default=518, help="múltiplo de 14")
    ap.add_argument("--output", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--simplify", action="store_true")
    args = ap.parse_args()

    if args.height % 14 or args.width % 14:
        raise SystemExit("[erro] DAV2 (DINOv2) exige HxW múltiplos de 14.")

    # default = trainable (melhor); troque com --checkpoint para o frozen.
    ckpt = resolve_checkpoint(
        args.checkpoint, "dav2_vits_trainable_20260605_152446_best.pth")
    model = build_dav2(ckpt, args.encoder, args.max_depth)

    dummy = torch.zeros(1, 3, args.height, args.width, dtype=torch.float32)
    sanity_forward(model, dummy)
    export_onnx(model, dummy, args.output, opset=args.opset,
                simplify=args.simplify)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Export ONNX do Monodepth2 fine-tunado no ICL Ground Robot.

Carrega o checkpoint do projeto (encoder_state_dict + decoder_state_dict) e
exporta um grafo único: rgb[0,1] -> ResnetEncoder -> DepthDecoder -> disp(scale 0)
-> disp_to_depth -> profundidade [B,1,H,W] em metros.

NOTA sobre resolução
--------------------
No treino o forward redimensionava o input para 192x640 (resolução nativa do
mono_640x192). Aqui o ONNX roda a rede DIRETAMENTE na resolução HxW pedida
(o ResnetEncoder é totalmente convolucional, aceita qualquer múltiplo de 32).
Rodar em resolução diferente de 192x640 é justamente o objetivo do estudo de
Pareto — mas lembre que a escala muda, então a avaliação DEVE usar median-align
(o Monodepth2 prediz profundidade só até um fator de escala).

Uso:
    python export/export_monodepth2.py --height 192 --width 640 \
        --output onnx/monodepth2_192x640.onnx --simplify
"""

from __future__ import annotations

import argparse
import torch
import torch.nn as nn

from _common import (add_external_paths, resolve_checkpoint, export_onnx,
                     sanity_forward, PROJECT_ROOT)

add_external_paths()

# Constantes do Monodepth2 (convenção disp->depth do paper Godard et al. 2019)
MD2_MIN_DEPTH = 0.1
MD2_MAX_DEPTH = 100.0


def disp_to_depth(disp: torch.Tensor,
                  min_depth: float = MD2_MIN_DEPTH,
                  max_depth: float = MD2_MAX_DEPTH) -> torch.Tensor:
    min_disp = 1.0 / max_depth
    max_disp = 1.0 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    return 1.0 / scaled_disp


class Monodepth2Exportable(nn.Module):
    """Encoder + Decoder + conversão p/ profundidade num único módulo exportável."""

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        # rgb em [0,1] NCHW; a normalização (x-0.45)/0.225 é feita dentro do
        # ResnetEncoder (networks/resnet_encoder.py), então não a aplicamos aqui.
        features = self.encoder(rgb)
        outputs = self.decoder(features)
        disp = outputs[("disp", 0)]          # [B,1,H,W], disparidade normalizada
        depth = disp_to_depth(disp)          # metros (até escala)
        return depth


def build_monodepth2(checkpoint) -> nn.Module:
    from networks.resnet_encoder import ResnetEncoder
    from networks.depth_decoder import DepthDecoder

    encoder = ResnetEncoder(num_layers=18, pretrained=False)
    decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    encoder.load_state_dict(state["encoder_state_dict"])
    decoder.load_state_dict(state["decoder_state_dict"])
    print(f"[monodepth2] checkpoint carregado (epoch "
          f"{state.get('epoch', '?')}, val abs_rel="
          f"{state.get('val_metrics', {}).get('abs_rel', float('nan')):.4f})")

    model = Monodepth2Exportable(encoder, decoder).eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="default: modelos_treinados/monodepth2_finetune_*_best.pth")
    ap.add_argument("--height", type=int, default=192, help="múltiplo de 32")
    ap.add_argument("--width", type=int, default=640, help="múltiplo de 32")
    ap.add_argument("--output", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--simplify", action="store_true")
    args = ap.parse_args()

    if args.height % 32 or args.width % 32:
        raise SystemExit("[erro] Monodepth2 exige HxW múltiplos de 32.")

    ckpt = resolve_checkpoint(
        args.checkpoint, "monodepth2_finetune_20260518_005713_best.pth")
    model = build_monodepth2(ckpt)

    dummy = torch.zeros(1, 3, args.height, args.width, dtype=torch.float32)
    sanity_forward(model, dummy)
    export_onnx(model, dummy, args.output, opset=args.opset,
                simplify=args.simplify)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Export ONNX do ZoeDepth (Zoe_N, n_bins=32) fine-tunado no ICL Ground Robot.

Carrega o checkpoint do projeto (model_state_dict) e exporta:
    rgb[0,1] -> ZoeDepth.forward -> metric_depth [B,1,H,W] em metros.

A normalização (mean/std=0.5) e o resize interno acontecem dentro do
MidasCore.prep, então ficam embutidos no grafo — o ONNX recebe [0,1] cru.

⚠️  CASO DIFÍCIL: o backbone é DPT_BEiT_L_384 (~352M params). O export ONNX
do BEiT costuma esbarrar em ops de interpolação do relative position bias e
em meshgrid/gather. Se falhar, tente:
    - opset 16 ou 18 (--opset)
    - rodar SEM --simplify primeiro (isola se o erro é do onnxsim)
    - manter a resolução nativa de treino (384x384)
E lembre: mesmo exportando, ZoeDepth no Orin Nano (8GB) tende a rodar a
poucos FPS e não recomendamos INT8 (BEiT degrada sem QAT).

Uso:
    python export/export_zoedepth.py --height 384 --width 384 \
        --output onnx/zoedepth_trainable_384x384.onnx --simplify
"""

from __future__ import annotations

import argparse
import torch
import torch.nn as nn

from _common import (add_external_paths, resolve_checkpoint, export_onnx,
                     sanity_forward)

add_external_paths()


class ZoeDepthExportable(nn.Module):
    """Extrai metric_depth do dict de saída e garante [B,1,H,W]."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        out = self.model(rgb)
        pred = out["metric_depth"] if isinstance(out, dict) else out
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        return pred


def build_zoedepth(checkpoint, n_bins: int, img_size: int,
                   min_depth: float = 0.001, max_depth: float = 10.0) -> nn.Module:
    """
    Reconstrói a arquitetura (igual ao treino) e carrega NOSSO checkpoint por
    cima. Não baixa o ZoeD_M12_N — os pesos vêm do fine-tune.
    """
    from zoedepth.models.builder import build_model
    from zoedepth.utils.config import get_config

    conf = get_config("zoedepth", "train", dataset=None)
    conf.n_bins = n_bins
    conf.min_depth = min_depth
    conf.max_depth = max_depth
    conf.img_size = [img_size, img_size]
    conf.train_midas = False
    # Precisamos da arquitetura do MiDaS/BEiT (vem do cache do torch.hub, já
    # baixado no treino). Os pesos serão sobrescritos pelo nosso checkpoint.
    conf.use_pretrained_midas = True
    conf.pretrained_resource = ""

    model = build_model(conf)

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sd = state["model_state_dict"] if "model_state_dict" in state else state
    missing, unexpected = model.load_state_dict(sd, strict=False)
    vm = state.get("val_metrics", {}) if isinstance(state, dict) else {}
    print(f"[zoedepth] checkpoint carregado (n_bins={n_bins}, "
          f"val abs_rel={vm.get('abs_rel', float('nan')):.4f}; "
          f"missing={len(missing)}, unexpected={len(unexpected)})")
    if missing:
        print(f"[zoedepth][warn] chaves faltando: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    return ZoeDepthExportable(model).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="default: modelos_treinados/zoedepth_trainable5_final_best.pth")
    ap.add_argument("--n_bins", type=int, default=32)
    ap.add_argument("--height", type=int, default=384, help="múltiplo de 32")
    ap.add_argument("--width", type=int, default=384, help="múltiplo de 32")
    ap.add_argument("--output", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--simplify", action="store_true")
    args = ap.parse_args()

    if args.height % 32 or args.width % 32:
        raise SystemExit("[erro] ZoeDepth exige HxW múltiplos de 32.")

    ckpt = resolve_checkpoint(
        args.checkpoint, "zoedepth_trainable5_final_best.pth")
    model = build_zoedepth(ckpt, n_bins=args.n_bins, img_size=args.height)

    dummy = torch.zeros(1, 3, args.height, args.width, dtype=torch.float32)
    sanity_forward(model, dummy)
    try:
        export_onnx(model, dummy, args.output, opset=args.opset,
                    simplify=args.simplify)
    except Exception as e:
        raise SystemExit(
            f"\n[zoedepth][FALHA no export ONNX] {type(e).__name__}: {e}\n"
            f"Esse é o caso difícil (BEiT). Tente: --opset 16/18, sem "
            f"--simplify, ou resolução nativa 384x384. Veja o cabeçalho do "
            f"script para detalhes."
        )


if __name__ == "__main__":
    main()

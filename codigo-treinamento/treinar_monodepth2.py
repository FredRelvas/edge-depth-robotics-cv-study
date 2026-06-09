"""
Treino do Monodepth2 no ICL Ground Robot (fine-tune supervisionado).

Replica o experimento Monodepth2 de Vizzotto et al. (RITA 2025):
  - Carrega encoder ResNet-18 + decoder do mono_640x192 (pré-treinado no KITTI).
  - Fine-tune supervisionado usando o GT do ICL.
  - 20 épocas, Adam + StepLR.

Sobre a divergência com o paper original do Monodepth2 (Godard et al. 2019):
  O Monodepth2 original é self-supervised (loss fotométrico entre frames).
  O paper RITA 2025 faz "fine-tune supervisionado" usando o GT do ICL, então
  trocamos a função de perda por SILog (mesma do ZoeDepth) sobre o GT.
  O modelo continua predizendo disparidade [0,1] na saída do decoder; convertemos
  pra depth via disparidade -> depth (1/disp) e re-escalamos por alinhamento de
  mediana durante a avaliação (median_align=True nas métricas).

Pré-requisito: ter o repo nianticlabs/monodepth2 clonado em external/monodepth2/.
O script run_monodepth2.sh faz isso automaticamente.

Uso direto:
    uv run python codigo-treinamento/treinar_monodepth2.py \\
        --epochs 20 --batch_size 8 --lr 1e-4

Uso recomendado:
    bash scripts/run_monodepth2.sh
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "codigo-treinamento"))
sys.path.insert(0, str(PROJECT_ROOT / "metricas"))
sys.path.insert(0, str(PROJECT_ROOT / "external" / "monodepth2"))

from dataloader import build_icl_dataloaders                                  # noqa: E402
from metricas import compute_depth_metrics, aggregate_batch_metrics, format_metrics  # noqa: E402
from treino_utils import (                                                    # noqa: E402
    ExperimentConfig, TrainLogger, BestMetricTracker, save_checkpoint
)


# Constantes do Monodepth2 (mono_640x192) — herdadas do paper original.
MD2_HEIGHT = 192
MD2_WIDTH  = 640
MD2_MIN_DEPTH = 0.1
MD2_MAX_DEPTH = 100.0


# ---------------------------------------------------------------------------
# Carregamento dos pesos pré-treinados
# ---------------------------------------------------------------------------

def load_pretrained_monodepth2(weights_dir: Path, device: str):
    """
    Carrega ResnetEncoder + DepthDecoder do mono_640x192.

    O Monodepth2 vem com dois checkpoints separados:
        weights_dir/encoder.pth     (encoder + intrinsics + dims)
        weights_dir/depth.pth       (decoder)
    """
    from networks.resnet_encoder import ResnetEncoder
    from networks.depth_decoder import DepthDecoder

    # ResnetEncoder com pesos pré-treinados em ImageNet, depois sobrescritos pelo KITTI
    encoder = ResnetEncoder(num_layers=18, pretrained=False)
    enc_ckpt_path = weights_dir / "encoder.pth"
    if not enc_ckpt_path.exists():
        raise FileNotFoundError(
            f"Não encontrei {enc_ckpt_path}. Rode 'bash scripts/run_monodepth2.sh' "
            f"ou baixe manualmente o mono_640x192 do repo nianticlabs/monodepth2."
        )
    enc_state = torch.load(enc_ckpt_path, map_location=device, weights_only=False)
    # O checkpoint inclui 'height'/'width'/'use_stereo' além dos pesos do modelo.
    enc_model_keys = {k: v for k, v in enc_state.items() if k in encoder.state_dict()}
    encoder.load_state_dict(enc_model_keys)

    decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
    dec_state = torch.load(weights_dir / "depth.pth", map_location=device,
                           weights_only=False)
    decoder.load_state_dict(dec_state)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    n_enc = sum(p.numel() for p in encoder.parameters())
    n_dec = sum(p.numel() for p in decoder.parameters())
    print(f"[monodepth2] encoder={n_enc/1e6:.1f}M  decoder={n_dec/1e6:.1f}M  "
          f"input={MD2_HEIGHT}x{MD2_WIDTH}")

    return encoder, decoder


# ---------------------------------------------------------------------------
# Disparidade -> profundidade (convenção Monodepth2)
# ---------------------------------------------------------------------------

def disp_to_depth(disp: torch.Tensor,
                  min_depth: float = MD2_MIN_DEPTH,
                  max_depth: float = MD2_MAX_DEPTH) -> torch.Tensor:
    """
    Converte disparidade normalizada [0, 1] em profundidade em metros.
    Convenção do Monodepth2 (Godard et al. 2019, eq. 8).
    """
    min_disp = 1.0 / max_depth
    max_disp = 1.0 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1.0 / scaled_disp
    return depth


# ---------------------------------------------------------------------------
# Loss SILog (mesma do ZoeDepth — paper usa "smooth factor 1e-5" como aux)
# ---------------------------------------------------------------------------

class SILogLoss(nn.Module):
    def __init__(self, lambd: float = 0.85, eps: float = 1e-7):
        super().__init__()
        self.lambd = lambd
        self.eps = eps

    def forward(self, pred, gt, valid):
        valid = valid & (gt > self.eps) & (pred > self.eps)
        if valid.sum() < 10:
            return pred.sum() * 0.0
        d = torch.log(pred[valid]) - torch.log(gt[valid])
        return torch.sqrt((d ** 2).mean() - self.lambd * (d.mean() ** 2)) * 10.0


# ---------------------------------------------------------------------------
# Forward consistente: encoder -> decoder -> disp(scale=0) -> depth
# ---------------------------------------------------------------------------

def forward_pred(encoder, decoder, rgb: torch.Tensor,
                 target_size: tuple) -> torch.Tensor:
    """
    Pipeline completo: rgb -> features -> disps -> depth na resolução do GT.

    O decoder do Monodepth2 produz disparidades em 4 escalas; usamos a escala 0
    (resolução de input). Convertemos pra metros e fazemos upscale pra resolução
    do GT (do dataloader, 256x256 por padrão).
    """
    # Monodepth2 espera input 192x640; redimensionamos antes.
    rgb_resized = F.interpolate(rgb, size=(MD2_HEIGHT, MD2_WIDTH),
                                mode="bilinear", align_corners=False)
    features = encoder(rgb_resized)
    outputs = decoder(features)
    disp = outputs[("disp", 0)]   # [B, 1, MD2_HEIGHT, MD2_WIDTH]
    depth = disp_to_depth(disp)   # em metros
    # Upscale pra resolução do GT
    depth = F.interpolate(depth, size=target_size, mode="bilinear",
                          align_corners=False)
    return depth


# ---------------------------------------------------------------------------
# Validação
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_validation(encoder, decoder, val_loader, device: str,
                   median_align: bool) -> dict:
    encoder.eval()
    decoder.eval()
    batch_metrics = []
    for batch in val_loader:
        rgb   = batch["rgb"].to(device, non_blocking=True)
        gt    = batch["depth"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)

        pred = forward_pred(encoder, decoder, rgb, target_size=gt.shape[-2:])
        batch_metrics.append(compute_depth_metrics(
            pred, gt, valid=valid, median_align=median_align,
        ))
    return aggregate_batch_metrics(batch_metrics)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_step_size", type=int, default=15,
                    help="StepLR step_size (paper: ~15 épocas para 20 totais)")
    ap.add_argument("--lr_gamma", type=float, default=0.1)
    ap.add_argument("--image_size", type=int, default=256,
                    help="Resolução do GT/avaliação. Modelo redimensiona internamente.")
    ap.add_argument("--median_align", type=lambda s: s.lower() != "false", default=True,
                    help="Alinhar pred * median(gt)/median(pred) na avaliação. "
                         "True por padrão (Monodepth2 prediz até uma escala).")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--data_root", type=str,
                    default=str(PROJECT_ROOT / "dados" / "icl_ground_robot"))
    ap.add_argument("--pretrained_dir", type=str,
                    default=str(PROJECT_ROOT / "external" / "monodepth2" / "models" / "mono_640x192"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={device}")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    # ----- Dataloaders -----
    loaders = build_icl_dataloaders(
        root=args.data_root,
        scenes=("deer", "diamond"),
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # ----- Modelo -----
    encoder, decoder = load_pretrained_monodepth2(Path(args.pretrained_dir), device)
    loss_fn = SILogLoss()

    # ----- Otimizador / scheduler -----
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = Adam(params, lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    # ----- Logger -----
    cfg = ExperimentConfig(
        name=Path(args.run_dir).name,
        model="monodepth2",
        mode="supervised",
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        image_size=args.image_size,
        extra={
            "input_resolution": f"{MD2_HEIGHT}x{MD2_WIDTH}",
            "loss": "SILog",
            "scheduler": f"StepLR(step={args.lr_step_size}, gamma={args.lr_gamma})",
            "median_align_eval": args.median_align,
            "pretrained": "mono_640x192 (KITTI)",
        },
    )
    logger = TrainLogger(cfg, run_dir=args.run_dir)
    tracker = BestMetricTracker(key="abs_rel", mode="min")

    # ----- Loop de treino -----
    steps_per_epoch = len(loaders["train"])
    global_step = 0
    for epoch in range(args.epochs):
        encoder.train()
        decoder.train()
        epoch_loss_sum = 0.0
        epoch_t0 = time.time()

        for step, batch in enumerate(loaders["train"]):
            rgb   = batch["rgb"].to(device, non_blocking=True)
            gt    = batch["depth"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = forward_pred(encoder, decoder, rgb, target_size=gt.shape[-2:])
            loss = loss_fn(pred, gt, valid)
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            loss_val = loss.item()
            epoch_loss_sum += loss_val
            logger.log_scalar("train/loss", loss_val, step=global_step)
            logger.log_scalar("train/lr", optimizer.param_groups[0]["lr"],
                              step=global_step)
            global_step += 1

            if step % 20 == 0:
                print(f"  [epoch {epoch+1}/{args.epochs}] "
                      f"step {step+1}/{steps_per_epoch}  loss={loss_val:.4f}")

        scheduler.step()
        avg_train_loss = epoch_loss_sum / max(1, steps_per_epoch)
        dt = time.time() - epoch_t0

        # ----- Validação -----
        val_metrics = run_validation(encoder, decoder, loaders["val"], device,
                                     median_align=args.median_align)
        is_best = tracker.update(val_metrics, epoch)

        print(f"\n[epoch {epoch+1}/{args.epochs}] "
              f"({dt:.0f}s) train_loss={avg_train_loss:.4f}  "
              f"val: {format_metrics(val_metrics)}"
              f"  {'(BEST)' if is_best else ''}\n")

        logger.log_epoch(epoch, val_metrics, train_loss=avg_train_loss,
                         learning_rate=optimizer.param_groups[0]["lr"])

        save_checkpoint(
            state={
                "epoch": epoch,
                "encoder_state_dict": encoder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_metrics": val_metrics,
                "config": cfg.to_dict(),
            },
            run_dir=logger.run_dir,
            is_best=is_best,
        )

    # ----- Avaliação final no test -----
    print("\n=== Avaliação no conjunto de teste ===")
    best_ckpt = Path(logger.run_dir) / "checkpoints" / "best.pth"
    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location=device, weights_only=False)
        encoder.load_state_dict(state["encoder_state_dict"])
        decoder.load_state_dict(state["decoder_state_dict"])
        print(f"[test] Carregado best.pth (epoch {state['epoch']+1}, "
              f"val abs_rel={tracker.best:.4f})")

    test_metrics = run_validation(encoder, decoder, loaders["test"], device,
                                  median_align=args.median_align)
    print(f"[test] {format_metrics(test_metrics)}")
    (Path(logger.run_dir) / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2)
    )

    logger.close()
    print(f"\n[done] resultados em {logger.run_dir}")


if __name__ == "__main__":
    main()
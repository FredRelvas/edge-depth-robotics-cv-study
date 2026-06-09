"""
Treino do Depth Anything V2 (ViT-S, Indoor) no ICL Ground Robot.

Experimento adicional ao paper RITA 2025: avaliação do estado da arte atual
(Yang et al., NeurIPS 2024) sobre o mesmo dataset, para comparação justa
com ZoeDepth e Monodepth2.

Modos:
  --mode frozen     : encoder ViT-S congelado, treina só a DPT head metric.
                      ~20 épocas, AdamW + OneCycleLR, lr_max=1e-4.
  --mode trainable  : fine-tune completo (encoder + head).
                      ~10 épocas (early stopping), AdamW + OneCycleLR, lr_max=5e-6.

Pré-requisito: repo DepthAnything/Depth-Anything-V2 clonado em external/DAV2/
e checkpoint baixado em external/DAV2/checkpoints/.
O script run_dav2_*.sh faz isso automaticamente.

Uso direto:
    uv run python codigo-treinamento/treinar_depth_anything_v2.py \\
        --mode frozen --epochs 20 --batch_size 8 --lr 1e-4

Uso recomendado:
    bash scripts/run_dav2_frozen.sh
    bash scripts/run_dav2_trainable.sh
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "codigo-treinamento"))
sys.path.insert(0, str(PROJECT_ROOT / "metricas"))
# O repo da DAV2 tem o módulo em metric_depth/depth_anything_v2/
sys.path.insert(0, str(PROJECT_ROOT / "external" / "DAV2" / "metric_depth"))

from dataloader import build_icl_dataloaders                                  # noqa: E402
from metricas import compute_depth_metrics, aggregate_batch_metrics, format_metrics  # noqa: E402
from treino_utils import (                                                    # noqa: E402
    ExperimentConfig, TrainLogger, BestMetricTracker, save_checkpoint
)


# ---------------------------------------------------------------------------
# Configurações dos encoders (do README do DAV2)
# ---------------------------------------------------------------------------

DAV2_ENCODER_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64,
             "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128,
             "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256,
             "out_channels": [256, 512, 1024, 1024]},
}

# Tamanho de entrada do DAV2: divisível por 14 (patch size do DINOv2).
# 518 é o default do repo, mas 392 cabe melhor em VRAM e é múltiplo de 14*4.
DAV2_DEFAULT_INPUT_SIZE = 518


# ---------------------------------------------------------------------------
# Construção do modelo
# ---------------------------------------------------------------------------

def build_dav2(mode: str, encoder: str = "vits",
               max_depth: float = 20.0,
               pretrained_path: Path | None = None) -> nn.Module:
    """
    Constrói o DepthAnythingV2 metric e carrega pesos pré-treinados em Hypersim.

    Args:
        mode: 'frozen' (congela encoder) | 'trainable' (treina tudo).
        encoder: 'vits' | 'vitb' | 'vitl'.
        max_depth: profundidade máxima predita em metros. Hypersim usa 20m,
                   mas o ICL Ground Robot só vai até 10m — usamos 20 pra ter
                   margem e bater com o checkpoint pré-treinado.
        pretrained_path: caminho pro arquivo .pth (default = external/DAV2/checkpoints/).
    """
    from depth_anything_v2.dpt import DepthAnythingV2

    if encoder not in DAV2_ENCODER_CONFIGS:
        raise ValueError(f"encoder inválido: {encoder!r}")

    cfg = DAV2_ENCODER_CONFIGS[encoder]
    model = DepthAnythingV2(**cfg, max_depth=max_depth)

    # Carrega pesos pré-treinados (Hypersim — indoor metric).
    if pretrained_path is None:
        pretrained_path = (PROJECT_ROOT / "external" / "DAV2" / "checkpoints"
                           / f"depth_anything_v2_metric_hypersim_{encoder}.pth")
    if not pretrained_path.exists():
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {pretrained_path}. "
            f"Rode 'bash scripts/run_dav2_*.sh' que faz o download automaticamente, "
            f"ou baixe manualmente do Hugging Face "
            f"(depth-anything/Depth-Anything-V2-Metric-{encoder.upper()}-hypersim)."
        )
    print(f"[dav2] Carregando pesos pré-treinados: {pretrained_path.name}")
    state = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)

    # Congelamento do encoder no modo frozen (mantém só a DPT head treinável)
    if mode == "frozen":
        for name, p in model.named_parameters():
            # No DAV2, o encoder é model.pretrained (DINOv2). A DPT head é model.depth_head.
            if name.startswith("pretrained."):
                p.requires_grad = False

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[dav2] encoder={encoder} mode={mode}  "
          f"params total={n_total/1e6:.1f}M  "
          f"trainable={n_train/1e6:.1f}M ({100*n_train/n_total:.1f}%)")

    return model


# ---------------------------------------------------------------------------
# Loss SILog (mesma usada nos outros experimentos pra comparação justa)
# ---------------------------------------------------------------------------

class SILogLoss(nn.Module):
    def __init__(self, lambd: float = 0.85, eps: float = 1e-7):
        super().__init__()
        self.lambd = lambd
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        valid = valid & (gt > self.eps) & (pred > self.eps)
        if valid.sum() < 10:
            return pred.sum() * 0.0
        d = torch.log(pred[valid]) - torch.log(gt[valid])
        return torch.sqrt((d ** 2).mean() - self.lambd * (d.mean() ** 2)) * 10.0


# ---------------------------------------------------------------------------
# Forward: o modelo já devolve profundidade em metros [B, H, W]
# ---------------------------------------------------------------------------

def forward_pred(model: nn.Module, rgb: torch.Tensor,
                 target_size: tuple) -> torch.Tensor:
    """
    DepthAnythingV2.forward devolve [B, H, W] em metros (escala absoluta).
    Garantimos shape [B, 1, target_H, target_W].
    """
    pred = model(rgb)
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if pred.shape[-2:] != target_size:
        pred = F.interpolate(pred, size=target_size,
                             mode="bilinear", align_corners=False)
    return pred


# ---------------------------------------------------------------------------
# Validação
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_validation(model, val_loader, device: str) -> dict:
    model.eval()
    batch_metrics = []
    for batch in val_loader:
        rgb   = batch["rgb"].to(device, non_blocking=True)
        gt    = batch["depth"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)
        pred = forward_pred(model, rgb, target_size=gt.shape[-2:])
        batch_metrics.append(compute_depth_metrics(
            pred, gt, valid=valid, median_align=False,
        ))
    return aggregate_batch_metrics(batch_metrics)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["frozen", "trainable"])
    ap.add_argument("--encoder", default="vits", choices=["vits", "vitb", "vitl"])
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--amp", action="store_true",
                    help="Mixed precision bfloat16 (recomendado para vitl/vitb).")
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--max_depth", type=float, default=20.0,
                    help="20m do checkpoint Hypersim (ICL vai só até 10m).")
    ap.add_argument("--image_size", type=int, default=518,
                    help="DAV2 default: 518x518 (518 = 14×37, divisível pelo patch).")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--data_root", type=str,
                    default=str(PROJECT_ROOT / "dados" / "icl_ground_robot"))
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
    model = build_dav2(mode=args.mode, encoder=args.encoder,
                       max_depth=args.max_depth).to(device)
    loss_fn = SILogLoss()

    # ----- Otimizador / scheduler -----
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = len(loaders["train"])
    optimizer_steps_per_epoch = ((steps_per_epoch + args.grad_accum_steps - 1)
                                 // args.grad_accum_steps)
    scheduler = OneCycleLR(
        optimizer, max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=optimizer_steps_per_epoch,
        pct_start=0.3, anneal_strategy="cos",
    )

    # ----- Logger -----
    cfg = ExperimentConfig(
        name=Path(args.run_dir).name,
        model=f"depth_anything_v2_{args.encoder}",
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        image_size=args.image_size,
        extra={
            "encoder": args.encoder,
            "max_depth": args.max_depth,
            "pretrained": f"depth_anything_v2_metric_hypersim_{args.encoder}",
            "loss": "SILog",
            "scheduler": "OneCycleLR",
            "weight_decay": args.weight_decay,
            "grad_accum_steps": args.grad_accum_steps,
            "amp": args.amp,
        },
    )
    logger = TrainLogger(cfg, run_dir=args.run_dir)
    tracker = BestMetricTracker(key="abs_rel", mode="min")

    # ----- Avaliação zero-shot (sem treinar, baseline interessante pro relatório) -----
    print("\n=== Avaliação zero-shot (sem fine-tune) ===")
    zs_metrics = run_validation(model, loaders["val"], device)
    print(f"[zero-shot val] {format_metrics(zs_metrics)}")
    (Path(logger.run_dir) / "zero_shot_metrics.json").write_text(
        json.dumps(zs_metrics, indent=2)
    )

    amp_dtype = torch.bfloat16 if args.amp else None
    print(f"\n[setup] batch_size={args.batch_size} grad_accum={args.grad_accum_steps} "
          f"-> batch efetivo={args.batch_size * args.grad_accum_steps}  "
          f"amp={'bf16' if args.amp else 'off'}")

    # ----- Loop de treino -----
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loaders["train"]):
            rgb   = batch["rgb"].to(device, non_blocking=True)
            gt    = batch["depth"].to(device, non_blocking=True)
            valid = batch["valid"].to(device, non_blocking=True)

            if args.amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    pred = forward_pred(model, rgb, target_size=gt.shape[-2:])
                    loss = loss_fn(pred, gt, valid)
            else:
                pred = forward_pred(model, rgb, target_size=gt.shape[-2:])
                loss = loss_fn(pred, gt, valid)

            (loss / args.grad_accum_steps).backward()

            is_accum_boundary = ((step + 1) % args.grad_accum_steps == 0) or \
                                (step + 1 == steps_per_epoch)
            if is_accum_boundary:
                nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            loss_val = loss.item()
            epoch_loss_sum += loss_val
            logger.log_scalar("train/loss", loss_val, step=global_step)
            logger.log_scalar("train/lr", scheduler.get_last_lr()[0], step=global_step)
            global_step += 1

            if step % 20 == 0:
                print(f"  [epoch {epoch+1}/{args.epochs}] "
                      f"step {step+1}/{steps_per_epoch}  loss={loss_val:.4f}  "
                      f"lr={scheduler.get_last_lr()[0]:.2e}")

        avg_train_loss = epoch_loss_sum / max(1, steps_per_epoch)
        dt = time.time() - epoch_t0

        # ----- Validação -----
        val_metrics = run_validation(model, loaders["val"], device)
        is_best = tracker.update(val_metrics, epoch)

        print(f"\n[epoch {epoch+1}/{args.epochs}] "
              f"({dt:.0f}s) train_loss={avg_train_loss:.4f}  "
              f"val: {format_metrics(val_metrics)}"
              f"  {'(BEST)' if is_best else ''}\n")

        logger.log_epoch(epoch, val_metrics, train_loss=avg_train_loss,
                         learning_rate=scheduler.get_last_lr()[0])

        save_checkpoint(
            state={
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_metrics": val_metrics,
                "config": cfg.to_dict(),
            },
            run_dir=logger.run_dir, is_best=is_best,
        )

    # ----- Avaliação final no test -----
    print("\n=== Avaliação no conjunto de teste ===")
    best_ckpt = Path(logger.run_dir) / "checkpoints" / "best.pth"
    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        print(f"[test] Carregado best.pth (epoch {state['epoch']+1}, "
              f"val abs_rel={tracker.best:.4f})")

    test_metrics = run_validation(model, loaders["test"], device)
    print(f"[test] {format_metrics(test_metrics)}")
    (Path(logger.run_dir) / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2)
    )

    logger.close()
    print(f"\n[done] resultados em {logger.run_dir}")


if __name__ == "__main__":
    main()
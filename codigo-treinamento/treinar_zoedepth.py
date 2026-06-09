"""
Treino do ZoeDepth no ICL Ground Robot.

Replica os experimentos Frozen-32 e Trainable-5 de Vizzotto et al. (RITA 2025):

  Frozen-32 (--mode frozen):
    - Backbone congelado, treina só a metric head (32 bins).
    - 20 épocas, AdamW + OneCycleLR, lr_max=1e-4, flip augmentation.

  Trainable-5 (--mode trainable):
    - Fine-tune completo (backbone + metric head).
    - 40 épocas, AdamW + OneCycleLR, lr_max=1e-5, flip augmentation.
    - É o melhor modelo do paper (RMSE log 0.144).

Pré-requisito: ter o repo isl-org/ZoeDepth clonado em external/ZoeDepth/.
O script run_zoedepth_*.sh faz isso automaticamente.

Uso direto:
    uv run python codigo-treinamento/treinar_zoedepth.py \\
        --mode frozen --epochs 20 --batch_size 8 --lr 1e-4

Uso recomendado (via bash):
    bash scripts/run_zoedepth_frozen.sh
    bash scripts/run_zoedepth_trainable.sh
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

# Resolve paths relativos à raiz do projeto (este arquivo está em codigo-treinamento/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "codigo-treinamento"))
sys.path.insert(0, str(PROJECT_ROOT / "metricas"))
sys.path.insert(0, str(PROJECT_ROOT / "external" / "ZoeDepth"))

from dataloader import build_icl_dataloaders                 # noqa: E402
from metricas import compute_depth_metrics, aggregate_batch_metrics, format_metrics  # noqa: E402
from treino_utils import (                                    # noqa: E402
    ExperimentConfig, TrainLogger, BestMetricTracker, save_checkpoint
)


# ---------------------------------------------------------------------------
# Construção do modelo ZoeDepth
# ---------------------------------------------------------------------------

def build_zoedepth(mode: str, n_bins: int = 32, min_depth: float = 0.001,
                   max_depth: float = 10.0) -> nn.Module:
    """
    Carrega ZoeDepth (Zoe_N pré-treinado em NYU Depth v2) e configura conforme o modo.

    Args:
        mode:      'frozen' (congela backbone) ou 'trainable' (treina tudo).
        n_bins:    número de bins da metric head (paper usa 32).
        min_depth: profundidade mínima do range (m).
        max_depth: profundidade máxima do range (m).

    Nota sobre o número de bins:
        O checkpoint oficial `ZoeD_M12_N.pt` foi treinado com 64 bins. Para usar
        n_bins != 64 (ex.: 32, como o paper RITA 2025 pede), carregamos os pesos
        com strict=False, deixando a metric head ser inicializada do zero. O
        backbone (DPT_BEiT_L_384) e as camadas comuns continuam vindo pré-treinadas.
    """
    import torch
    from zoedepth.models.builder import build_model
    from zoedepth.utils.config import get_config

    # Carrega config do Zoe_N. O paper RITA usa 32 bins, mas o checkpoint upstream
    # foi treinado com 64, então construímos o modelo com 32 e carregamos os pesos
    # depois com strict=False (ver bloco abaixo).
    conf = get_config("zoedepth", "train", dataset=None)
    conf.n_bins = n_bins
    conf.min_depth = min_depth
    conf.max_depth = max_depth
    conf.train_midas = (mode == "trainable")
    conf.use_pretrained_midas = True
    # Deixa pretrained_resource vazio aqui pra evitar o load automático que falha
    # com size mismatch — vamos fazer o load manual logo abaixo com strict=False.
    conf.pretrained_resource = ""

    model = build_model(conf)

    # Carrega manualmente os pesos pré-treinados (ZoeD_M12_N.pt) ignorando chaves
    # da metric head cujas dimensões dependem de n_bins.
    print(f"[zoedepth] Carregando pesos pré-treinados (n_bins={n_bins}, "
          f"strict=False — metric head será inicializada do zero)...")
    ckpt = torch.hub.load_state_dict_from_url(
        "https://github.com/isl-org/ZoeDepth/releases/download/v1.0/ZoeD_M12_N.pt",
        map_location="cpu",
        progress=True,
    )
    # Alguns releases salvam como dict {'model': state_dict, ...}, outros como state_dict puro.
    state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    # Remove o prefixo 'module.' se existir (salvo de DataParallel)
    state_dict = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
                  for k, v in state_dict.items()}

    # Filtra chaves cujo shape não bate com o modelo atual (acontece quando
    # n_bins != 64, que é o usado para treinar o checkpoint upstream). strict=False
    # sozinho NÃO cobre size mismatch — só cobre chaves missing/unexpected, então
    # precisamos remover essas chaves manualmente antes de chamar load_state_dict.
    model_state = model.state_dict()
    filtered = {}
    skipped_shape = []
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape != v.shape:
            skipped_shape.append((k, tuple(v.shape), tuple(model_state[k].shape)))
            continue
        filtered[k] = v

    if skipped_shape:
        print(f"[zoedepth] Pulando {len(skipped_shape)} chaves com shape diferente "
              f"(dependentes de n_bins, serão inicializadas do zero):")
        for k, ckpt_shape, model_shape in skipped_shape:
            print(f"           {k}: checkpoint {ckpt_shape} != modelo {model_shape}")

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"[zoedepth] backbone carregado OK "
          f"(missing={len(missing)}, unexpected={len(unexpected)})")

    # Congelamento explícito do backbone MiDaS no modo frozen.
    if mode == "frozen":
        for name, p in model.named_parameters():
            # Em ZoeDepth, o backbone fica em model.core (MiDaS DPT_BEiT_L_384).
            # A metric head fica em model.seed_bin_regressor, model.bin_centers_module,
            # model.attractors etc. Congelamos só o core.
            if "core" in name and "midas" in name.lower():
                p.requires_grad = False
            elif name.startswith("core."):
                p.requires_grad = False

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[zoedepth] mode={mode}  params total={n_total/1e6:.1f}M  "
          f"trainable={n_train/1e6:.1f}M ({100*n_train/n_total:.1f}%)")

    return model


# ---------------------------------------------------------------------------
# Loss conforme paper (SILog — o loss padrão do ZoeDepth, herdado do AdaBins)
# ---------------------------------------------------------------------------

class SILogLoss(nn.Module):
    """
    Scale-Invariant Logarithmic loss (Eigen et al. 2014), parametrização do AdaBins:

        L = sqrt( mean(d_i^2) - λ * mean(d_i)^2 ) * 10,    d_i = log(pred_i) - log(gt_i)

    Com λ = 0.85 (default do AdaBins/ZoeDepth).
    Atua só nos pixels válidos.
    """

    def __init__(self, lambd: float = 0.85, eps: float = 1e-7):
        super().__init__()
        self.lambd = lambd
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt: torch.Tensor,
                valid: torch.Tensor) -> torch.Tensor:
        valid = valid & (gt > self.eps) & (pred > self.eps)
        if valid.sum() < 10:
            return pred.sum() * 0.0   # evita NaN em batches degenerados

        d = torch.log(pred[valid]) - torch.log(gt[valid])
        loss = torch.sqrt((d ** 2).mean() - self.lambd * (d.mean() ** 2)) * 10.0
        return loss


# ---------------------------------------------------------------------------
# Inferência consistente: garante saída em [B, 1, H, W]
# ---------------------------------------------------------------------------

def forward_pred(model: nn.Module, rgb: torch.Tensor) -> torch.Tensor:
    """Forward que devolve sempre [B, 1, H, W] em metros."""
    out = model(rgb)
    if isinstance(out, dict):
        pred = out.get("metric_depth", out.get("out", out.get("rel_depth")))
    else:
        pred = out
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    return pred


# ---------------------------------------------------------------------------
# Validação
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_validation(model: nn.Module, val_loader, device: str) -> dict:
    model.eval()
    batch_metrics = []
    for batch in val_loader:
        rgb   = batch["rgb"].to(device, non_blocking=True)
        gt    = batch["depth"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)

        pred = forward_pred(model, rgb)
        if pred.shape[-2:] != gt.shape[-2:]:
            pred = nn.functional.interpolate(pred, size=gt.shape[-2:],
                                             mode="bilinear", align_corners=False)
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
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum_steps", type=int, default=1,
                    help="Acumula gradientes por N passos antes de optimizer.step(). "
                         "Permite simular batch maior sem custo de VRAM. "
                         "Batch efetivo = batch_size * grad_accum_steps.")
    ap.add_argument("--amp", action="store_true",
                    help="Mixed precision (autocast bfloat16). Reduz VRAM ~40%%.")
    ap.add_argument("--lr", type=float, required=True, help="lr_max do OneCycleLR")
    ap.add_argument("--weight_decay", type=float, default=1e-2,
                    help="L2 weight decay do AdamW. Aumentar (ex.: 1e-1) ajuda a "
                         "controlar overfit em fine-tunes longos.")
    ap.add_argument("--n_bins", type=int, default=32)
    ap.add_argument("--image_size", type=int, default=384,
                    help="ZoeDepth Zoe_N foi treinado em 384 (DPT). Paper usa 384.")
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
    model = build_zoedepth(mode=args.mode, n_bins=args.n_bins).to(device)
    loss_fn = SILogLoss()

    # ----- Otimizador / scheduler -----
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = len(loaders["train"])
    # Com gradient accumulation, o optimizer.step() acontece a cada N batches,
    # então o scheduler deve dar 1 step apenas a cada N batches também.
    optimizer_steps_per_epoch = (steps_per_epoch + args.grad_accum_steps - 1) // args.grad_accum_steps
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=optimizer_steps_per_epoch,
        pct_start=0.3,
        anneal_strategy="cos",
    )

    # ----- Logger -----
    cfg = ExperimentConfig(
        name=Path(args.run_dir).name,
        model="zoedepth",
        mode=args.mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        image_size=args.image_size,
        extra={"n_bins": args.n_bins, "loss": "SILog",
               "scheduler": "OneCycleLR", "weight_decay": args.weight_decay,
               "grad_accum_steps": args.grad_accum_steps, "amp": args.amp},
    )
    logger = TrainLogger(cfg, run_dir=args.run_dir)
    tracker = BestMetricTracker(key="abs_rel", mode="min")

    # AMP context: bfloat16 funciona em Ampere+ (4090 é Ada, suporta). Sem GradScaler
    # porque bf16 tem range parecido com fp32 — não overflow.
    amp_dtype = torch.bfloat16 if args.amp else None

    print(f"[setup] batch_size={args.batch_size} grad_accum={args.grad_accum_steps} "
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

            # Forward + loss (em AMP se ativado)
            if args.amp:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    pred = forward_pred(model, rgb)
                    if pred.shape[-2:] != gt.shape[-2:]:
                        pred = nn.functional.interpolate(
                            pred, size=gt.shape[-2:],
                            mode="bilinear", align_corners=False
                        )
                    loss = loss_fn(pred, gt, valid)
            else:
                pred = forward_pred(model, rgb)
                if pred.shape[-2:] != gt.shape[-2:]:
                    pred = nn.functional.interpolate(
                        pred, size=gt.shape[-2:],
                        mode="bilinear", align_corners=False
                    )
                loss = loss_fn(pred, gt, valid)

            # Escala a loss para que o gradiente acumulado tenha a mesma magnitude
            # que se fosse um único batch grande.
            (loss / args.grad_accum_steps).backward()

            # Só dá step a cada grad_accum_steps batches (ou no último batch da época)
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
            run_dir=logger.run_dir,
            is_best=is_best,
        )

    # ----- Avaliação final no test -----
    print("\n=== Avaliação no conjunto de teste ===")
    # Carrega o melhor checkpoint
    best_ckpt = Path(logger.run_dir) / "checkpoints" / "best.pth"
    if best_ckpt.exists():
        state = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        print(f"[test] Carregado best.pth (epoch {state['epoch']+1}, "
              f"val abs_rel={tracker.best:.4f})")

    test_metrics = run_validation(model, loaders["test"], device)
    print(f"[test] {format_metrics(test_metrics)}")
    # Salva métricas finais em JSON pra fácil leitura posterior
    import json
    (Path(logger.run_dir) / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2)
    )

    logger.close()
    print(f"\n[done] resultados em {logger.run_dir}")


if __name__ == "__main__":
    main()
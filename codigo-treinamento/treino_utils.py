"""
Utilitários compartilhados pelos scripts de treino.

Inclui um logger que escreve simultaneamente em TensorBoard (events.out.tfevents)
e em um CSV human-readable. Cada experimento tem sua própria pasta de log:

    runs/<experimento>/
        events.out.tfevents.*    (TensorBoard)
        metricas.csv             (uma linha por época de validação)
        config.json              (hiperparâmetros + git commit)
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.tensorboard import SummaryWriter


# ---------------------------------------------------------------------------
# Configuração do experimento
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Hiperparâmetros e metadados do experimento, salvos em config.json."""
    name: str                            # ex.: "zoedepth_frozen32"
    model: str                            # ex.: "zoedepth"
    mode: str                             # ex.: "frozen" | "trainable" | "supervised"
    epochs: int
    batch_size: int
    learning_rate: float
    image_size: int
    scenes: tuple = ("deer", "diamond")
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["scenes"] = list(d["scenes"])
        return d


def _git_commit() -> str:
    """Hash do commit atual (útil pra rastrear qual versão do código gerou o resultado)."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class TrainLogger:
    """
    Logger que escreve em TensorBoard e CSV. Uso:

        cfg = ExperimentConfig(name='zoedepth_frozen32', ...)
        logger = TrainLogger(cfg, run_dir='runs/zoedepth_frozen32')

        for epoch in range(N):
            for step, batch in enumerate(train_loader):
                ...
                logger.log_scalar('train/loss', loss.item(), step=global_step)
                global_step += 1

            val_metrics = run_validation(...)
            logger.log_epoch(epoch, val_metrics, train_loss=avg_train_loss)

        logger.close()
    """

    def __init__(self, cfg: ExperimentConfig, run_dir: str | Path):
        self.cfg = cfg
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        self.tb = SummaryWriter(log_dir=str(self.run_dir))

        # CSV
        self.csv_path = self.run_dir / "metricas.csv"
        self._csv_header_written = self.csv_path.exists()

        # Config: salvar uma vez no início
        cfg_path = self.run_dir / "config.json"
        cfg_dict = cfg.to_dict()
        cfg_dict["git_commit"] = _git_commit()
        cfg_dict["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        cfg_dict["cuda_device"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        )
        cfg_path.write_text(json.dumps(cfg_dict, indent=2))

        print(f"[logger] run_dir = {self.run_dir}")
        print(f"[logger] config  = {cfg_path}")

    # -----------------------------------------------------------------
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log de um valor durante o treino (loss, lr, etc.). Vai só pro TensorBoard."""
        self.tb.add_scalar(tag, value, step)

    def log_epoch(
        self,
        epoch: int,
        val_metrics: Dict[str, float],
        train_loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
    ) -> None:
        """
        Registra uma linha de métricas de validação no CSV e no TensorBoard.

        val_metrics deve conter as 7 chaves de compute_depth_metrics():
            abs_rel, sq_rel, rmse, rmse_log, delta1, delta2, delta3
        """
        # TensorBoard
        for k, v in val_metrics.items():
            self.tb.add_scalar(f"val/{k}", v, epoch)
        if train_loss is not None:
            self.tb.add_scalar("train/loss_epoch", train_loss, epoch)
        if learning_rate is not None:
            self.tb.add_scalar("train/lr", learning_rate, epoch)
        self.tb.flush()

        # CSV
        row = {"epoch": epoch}
        if train_loss is not None:
            row["train_loss"] = round(train_loss, 6)
        if learning_rate is not None:
            row["lr"] = learning_rate
        for k in ("abs_rel", "sq_rel", "rmse", "rmse_log", "delta1", "delta2", "delta3"):
            row[k] = round(val_metrics[k], 6)

        is_new_file = not self._csv_header_written
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if is_new_file:
                writer.writeheader()
                self._csv_header_written = True
            writer.writerow(row)

    def close(self) -> None:
        self.tb.close()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    state: Dict[str, Any],
    run_dir: str | Path,
    is_best: bool = False,
    epoch: Optional[int] = None,
) -> Path:
    """
    Salva um checkpoint em <run_dir>/checkpoints/.

    Por convenção, mantém:
        latest.pth   - sempre sobrescrito (última época)
        best.pth     - sobrescrito quando is_best=True (melhor val abs_rel)
        epoch_N.pth  - opcional (se epoch for fornecido)
    """
    ckpt_dir = Path(run_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    latest = ckpt_dir / "latest.pth"
    torch.save(state, latest)

    if is_best:
        torch.save(state, ckpt_dir / "best.pth")

    if epoch is not None:
        torch.save(state, ckpt_dir / f"epoch_{epoch:03d}.pth")

    return latest


# ---------------------------------------------------------------------------
# Tracker de melhor métrica (early stopping / save best)
# ---------------------------------------------------------------------------

class BestMetricTracker:
    """
    Mantém a melhor val abs_rel vista até agora e indica quando atualizar best.pth.

    Por convenção do paper, usamos abs_rel como métrica primária — quanto menor melhor.
    """

    def __init__(self, key: str = "abs_rel", mode: str = "min"):
        if mode not in ("min", "max"):
            raise ValueError("mode deve ser 'min' ou 'max'")
        self.key = key
        self.mode = mode
        self.best = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = -1

    def update(self, metrics: Dict[str, float], epoch: int) -> bool:
        """Retorna True se essa época bateu o recorde."""
        value = metrics.get(self.key)
        if value is None:
            return False
        is_better = (
            (self.mode == "min" and value < self.best) or
            (self.mode == "max" and value > self.best)
        )
        if is_better:
            self.best = value
            self.best_epoch = epoch
        return is_better


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        cfg = ExperimentConfig(
            name="dummy_test",
            model="dummy",
            mode="frozen",
            epochs=3,
            batch_size=4,
            learning_rate=1e-4,
            image_size=256,
        )
        logger = TrainLogger(cfg, run_dir=os.path.join(tmp, "test_run"))
        tracker = BestMetricTracker()

        import random
        for epoch in range(3):
            for step in range(5):
                logger.log_scalar("train/loss", random.random(), step=epoch * 5 + step)

            fake_metrics = {
                "abs_rel": 0.3 - epoch * 0.05,
                "sq_rel":  0.2,
                "rmse":    0.7,
                "rmse_log": 0.3,
                "delta1":  0.5 + epoch * 0.1,
                "delta2":  0.8,
                "delta3":  0.9,
            }
            is_best = tracker.update(fake_metrics, epoch)
            logger.log_epoch(epoch, fake_metrics, train_loss=0.5, learning_rate=1e-4)
            save_checkpoint({"epoch": epoch}, logger.run_dir, is_best=is_best)
            print(f"  epoch {epoch}: abs_rel={fake_metrics['abs_rel']:.3f} (best={is_best})")

        logger.close()

        print("\nConteúdo do run_dir:")
        for p in sorted(Path(logger.run_dir).rglob("*")):
            print(" ", p.relative_to(logger.run_dir))
        print("\nCSV:")
        print(open(Path(logger.run_dir) / "metricas.csv").read())
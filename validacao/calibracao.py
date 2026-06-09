"""
calibracao.py — carregamento da calibração (intrínsecas + extrínseca LiDAR↔câmera).

Fonte única de verdade para o K, a resolução de referência e a extrínseca 4×4,
compartilhada pelo gerador de bag sintético e pelo avaliador. Numa validação
real, este arquivo é produzido pela Frente 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass
class Calibracao:
    K: np.ndarray            # (3, 3) intrínseca
    largura: int             # largura de referência (px)
    altura: int              # altura de referência (px)
    T_lidar_cam: np.ndarray  # (4, 4) leva pontos do LiDAR ao frame da câmera


def carregar_calibracao(caminho: str | Path) -> Calibracao:
    """Lê o YAML de calibração e devolve uma Calibracao."""
    with open(caminho, "r") as f:
        cfg = yaml.safe_load(f)

    cam = cfg["camera"]
    K = np.asarray(cam["k"], dtype=np.float64).reshape(3, 3)
    T = np.asarray(cfg["extrinseca_lidar_cam"], dtype=np.float64).reshape(4, 4)
    return Calibracao(
        K=K,
        largura=int(cam["largura"]),
        altura=int(cam["altura"]),
        T_lidar_cam=T,
    )

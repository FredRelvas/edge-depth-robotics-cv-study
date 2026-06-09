"""
avaliacao.py — núcleo do cálculo offline das métricas.

Para cada frame: projeta o LiDAR nos pixels, amostra y_oak e y_ia nesses pixels,
unifica unidades em metros, alinha y_ia conforme o modelo (por frame) e acumula
os pares (predição, ground truth). Ao final calcula as métricas do paper para
dois confrontos: **IA vs LiDAR** e **OAK-D (baseline) vs LiDAR**, reusando o
metricas/metricas.py do repositório.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

import geometria
import alinhamento

# Reusa as métricas do paper (metricas/metricas.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "metricas"))
from metricas import compute_depth_metrics, format_metrics, MIN_DEPTH_M, MAX_DEPTH_M  # noqa: E402


@dataclass
class ParesAcumulados:
    """Pares (predição, ground truth) em metros, acumulados sobre todos os frames."""
    pred: List[np.ndarray] = field(default_factory=list)
    gt: List[np.ndarray] = field(default_factory=list)

    def adicionar(self, pred: np.ndarray, gt: np.ndarray):
        if pred.size:
            self.pred.append(np.asarray(pred, dtype=np.float64))
            self.gt.append(np.asarray(gt, dtype=np.float64))

    def metricas(self, min_depth: float, max_depth: float) -> Dict[str, float]:
        if not self.pred:
            return {k: float("nan") for k in
                    ("abs_rel", "sq_rel", "rmse", "rmse_log", "delta1", "delta2", "delta3")}
        pred = torch.from_numpy(np.concatenate(self.pred))
        gt = torch.from_numpy(np.concatenate(self.gt))
        return compute_depth_metrics(pred, gt, valid=None, median_align=False,
                                     min_depth=min_depth, max_depth=max_depth)

    @property
    def n_pontos(self) -> int:
        return int(sum(p.size for p in self.pred))


def _oak_para_metros(oak: np.ndarray) -> np.ndarray:
    """OAK-D 16UC1 vem em milímetros; converte para metros. 32FC1 já é metros."""
    if np.issubdtype(oak.dtype, np.integer):
        return oak.astype(np.float64) / 1000.0
    return oak.astype(np.float64)


def avaliar_bag(
    amostras,
    T_lidar_cam: np.ndarray,
    modelo: str,
    fonte_escala: str = "lidar",
    altura_camera: float = None,
    min_depth: float = MIN_DEPTH_M,
    max_depth: float = MAX_DEPTH_M,
) -> Dict:
    """
    Processa as amostras sincronizadas e devolve as métricas IA-vs-LiDAR e OAK-vs-LiDAR.

    Args:
        amostras:      lista de AmostraSincronizada (de leitor_rosbag.ler_bag).
        T_lidar_cam:   extrínseca 4×4 LiDAR→câmera (da calibração).
        modelo:        zoedepth | monodepth2 | depthanything.
        fonte_escala:  como a escala da IA é recuperada:
                       "lidar"  → alinhamento contra o LiDAR (padrão; usa o GT como régua).
                       "altura" → escala pelo plano do chão + altura conhecida da câmera,
                                  independente do LiDAR (mais honesto). Não se aplica ao
                                  Depth-Anything (2 graus de liberdade) → cai no LiDAR.
        altura_camera: altura da câmera ao chão em metros (obrigatório se fonte_escala="altura").
    Returns:
        dict com 'ia', 'oak' (métricas), 'modelo', 'n_frames', 'n_pontos_*' e 'fonte_escala'.
    """
    modo = alinhamento.modo_para_modelo(modelo)
    # Depth-Anything (afim) não fecha só com a altura → mantém-se no LiDAR.
    usa_altura = (fonte_escala == "altura" and modo != "afim_disparidade")
    if fonte_escala == "altura" and altura_camera is None:
        raise ValueError("fonte_escala='altura' exige altura_camera (metros).")

    pares_ia = ParesAcumulados()
    pares_oak = ParesAcumulados()
    frames_usados = 0
    frames_altura_falhou = 0

    for a in amostras:
        # LiDAR → pontos no frame da câmera → pixels (GT esparso).
        pts_lidar = geometria.laserscan_para_pontos(
            a.scan.ranges, a.scan.angle_min, a.scan.angle_increment,
            a.scan.range_min, a.scan.range_max,
        )
        if pts_lidar.shape[0] == 0:
            continue
        pts_cam = geometria.transformar(pts_lidar, T_lidar_cam)
        u, v, z_gt = geometria.projetar(pts_cam, a.K, a.largura_ref, a.altura_ref)
        if u.size == 0:
            continue

        # Amostra as duas fontes nos mesmos pixels.
        ia_raw = geometria.amostrar(a.ia_depth, u, v, a.largura_ref, a.altura_ref).astype(np.float64)
        oak_m = geometria.amostrar(_oak_para_metros(a.oak_depth), u, v,
                                   a.largura_ref, a.altura_ref)

        # Limpeza: GT na faixa válida; descarta OAK=0 e valores não-finitos.
        valido = (z_gt >= min_depth) & (z_gt <= max_depth) & np.isfinite(ia_raw)
        valido_oak = valido & (oak_m > 0) & np.isfinite(oak_m)
        if not np.any(valido):
            continue
        frames_usados += 1

        # IA: recupera a escala e acumula (alinhamento por frame).
        if usa_altura:
            # Escala vem do plano do chão (mapa inteiro) + altura conhecida — sem o LiDAR.
            try:
                s, _ = geometria.escala_por_altura(a.ia_depth, a.K, altura_camera)
                ia_m = ia_raw[valido] / s
            except ValueError:
                frames_altura_falhou += 1
                continue  # chão não detectável neste frame → descarta
        else:
            # Escala/forma ajustada contra o LiDAR (modelo relativo) ou direto (métrico).
            ia_m = alinhamento.alinhar(ia_raw[valido], z_gt[valido], modo)
        pares_ia.adicionar(ia_m, z_gt[valido])

        # OAK-D: baseline métrico, direto.
        pares_oak.adicionar(oak_m[valido_oak], z_gt[valido_oak])

    fonte_ia = "altura" if usa_altura else "lidar"
    return {
        "modelo": modelo,
        "modo_alinhamento": modo,
        "fonte_escala": fonte_ia,
        "n_frames": frames_usados,
        "frames_altura_falhou": frames_altura_falhou,
        "n_pontos_ia": pares_ia.n_pontos,
        "n_pontos_oak": pares_oak.n_pontos,
        "ia": pares_ia.metricas(min_depth, max_depth),
        "oak": pares_oak.metricas(min_depth, max_depth),
    }


def formatar_relatorio(resultado: Dict) -> str:
    """Tabela legível comparando IA (modelo) vs OAK-D (baseline) vs LiDAR (GT)."""
    extra = ""
    if resultado.get("frames_altura_falhou"):
        extra = f"  (chão não detectado em {resultado['frames_altura_falhou']} frame(s))"
    linhas = [
        "=" * 70,
        f"  Validação: {resultado['modelo']}  "
        f"(alinhamento: {resultado['modo_alinhamento']} · escala via: {resultado.get('fonte_escala','lidar')})",
        f"  Frames usados: {resultado['n_frames']}{extra}  |  "
        f"pontos IA: {resultado['n_pontos_ia']}  pontos OAK: {resultado['n_pontos_oak']}",
        "=" * 70,
        f"  IA   vs LiDAR : {format_metrics(resultado['ia'])}",
        f"  OAK  vs LiDAR : {format_metrics(resultado['oak'])}",
        "=" * 70,
    ]
    return "\n".join(linhas)

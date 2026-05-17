# TO-DO: Implementar código de cálculo de métricas

"""
Métricas de avaliação para estimação de profundidade.

Implementa as 7 métricas usadas em Vizzotto et al. (RITA, vol. 32 n.1, 2025)
para avaliação de modelos no ICL Ground Robot dataset:

    Erros de regressão (menor é melhor):
        - Abs Rel    : erro absoluto relativo médio
        - Sq Rel     : erro quadrático relativo médio
        - RMSE       : raiz do erro quadrático médio (em metros)
        - RMSE log   : RMSE em espaço logarítmico

    Acurácias com threshold (maior é melhor):
        - δ < 1.25
        - δ < 1.25²
        - δ < 1.25³

Faixa válida de profundidade conforme paper: 1 mm a 10 m.

Uso típico no loop de validação/teste:

    from metricas import compute_depth_metrics

    all_metrics = []
    for batch in loader:
        rgb, gt, valid = batch["rgb"].cuda(), batch["depth"].cuda(), batch["valid"].cuda()
        with torch.no_grad():
            pred = model(rgb)
        m = compute_depth_metrics(pred, gt, valid=valid, median_align=False)
        all_metrics.append(m)

    # média final do split
    final = {k: float(np.mean([m[k] for m in all_metrics])) for k in all_metrics[0]}
    print(final)

Sobre `median_align`:
    - ZoeDepth (supervisionado, métrico):    median_align=False  (avaliação direta).
    - Monodepth2 (autossupervisionado):      median_align=True   (convenção Godard
                                              et al. 2019: alinha a escala global
                                              pela razão das medianas antes de medir,
                                              porque o modelo prediz depth até uma
                                              escala arbitrária).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Constantes do paper (Vizzotto et al., RITA 2025)
# ---------------------------------------------------------------------------

MIN_DEPTH_M = 0.001     # 1 mm
MAX_DEPTH_M = 10.0      # 10 m
DELTA_THRESHOLDS = (1.25, 1.25 ** 2, 1.25 ** 3)


# ---------------------------------------------------------------------------
# Núcleo: cálculo das métricas a partir de dois vetores de pixels válidos
# ---------------------------------------------------------------------------

def _metrics_from_pixels(pred: Tensor, gt: Tensor) -> Dict[str, float]:
    """
    Calcula as 7 métricas a partir de dois tensores 1D já alinhados,
    contendo apenas pixels válidos (sem máscara) e em metros.

    Internal helper; o resto do módulo se preocupa com batching e mascaramento.
    """
    n = pred.numel()
    if n == 0:
        return {k: float("nan") for k in (
            "abs_rel", "sq_rel", "rmse", "rmse_log",
            "delta1", "delta2", "delta3",
        )}

    diff = pred - gt
    abs_rel = (diff.abs() / gt).mean()
    sq_rel  = ((diff ** 2) / gt).mean()
    rmse    = torch.sqrt((diff ** 2).mean())
    rmse_log = torch.sqrt(((torch.log(pred) - torch.log(gt)) ** 2).mean())

    # δ < thr = fração de pixels com max(d/d*, d*/d) < thr
    ratio = torch.maximum(pred / gt, gt / pred)
    delta1 = (ratio < DELTA_THRESHOLDS[0]).float().mean()
    delta2 = (ratio < DELTA_THRESHOLDS[1]).float().mean()
    delta3 = (ratio < DELTA_THRESHOLDS[2]).float().mean()

    return {
        "abs_rel":  abs_rel.item(),
        "sq_rel":   sq_rel.item(),
        "rmse":     rmse.item(),
        "rmse_log": rmse_log.item(),
        "delta1":   delta1.item(),
        "delta2":   delta2.item(),
        "delta3":   delta3.item(),
    }


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------

def compute_depth_metrics(
    pred: Tensor,
    gt: Tensor,
    valid: Optional[Tensor] = None,
    median_align: bool = False,
    min_depth: float = MIN_DEPTH_M,
    max_depth: float = MAX_DEPTH_M,
) -> Dict[str, float]:
    """
    Calcula as 7 métricas de profundidade do paper para um batch.

    Args:
        pred:         predição do modelo, em metros. Shape [B, 1, H, W] ou [B, H, W].
        gt:           ground truth, em metros. Mesmo shape de pred.
        valid:        máscara booleana opcional indicando pixels válidos do GT.
                      Se None, é criada a partir de min_depth/max_depth.
                      Mesmo shape de pred (ou broadcastable).
        median_align: se True, escala globalmente o pred multiplicando pela razão
                      median(gt) / median(pred). Usar para modelos self-supervised
                      como o Monodepth2. Aplica-se ANTES do clamp final.
        min_depth:    profundidade mínima válida em metros (default: 0.001).
        max_depth:    profundidade máxima válida em metros (default: 10.0).

    Returns:
        dict com chaves: abs_rel, sq_rel, rmse, rmse_log, delta1, delta2, delta3.
        Valores são floats (Python). Médias sobre todos os pixels válidos do batch.

    Notas:
        - As métricas são computadas sobre os pixels válidos AGREGADOS de todas as
          imagens do batch (não é "média de médias"), conforme convenção do paper.
        - pred precisa ser positivo nos pixels válidos (log requer isso); um clamp
          em [min_depth, max_depth] é aplicado antes da medição.
    """
    if pred.shape != gt.shape:
        raise ValueError(f"pred {tuple(pred.shape)} e gt {tuple(gt.shape)} devem ter mesmo shape.")

    # Constrói máscara: pixels com GT na faixa válida do paper, e (se fornecida)
    # pixels válidos da máscara externa.
    base_valid = (gt >= min_depth) & (gt <= max_depth)
    if valid is not None:
        if valid.dtype != torch.bool:
            valid = valid.bool()
        base_valid = base_valid & valid

    # Alinhamento por mediana (opcional). Feito por imagem do batch.
    if median_align:
        pred = _apply_median_align(pred, gt, base_valid)

    # Clamp do pred na faixa válida — evita log(0), divisões por zero e valores
    # negativos que podem aparecer em redes não-saturadas.
    pred = pred.clamp(min=min_depth, max=max_depth)

    # Achata e filtra; cálculo único sobre todos os pixels válidos do batch.
    pred_flat = pred[base_valid]
    gt_flat = gt[base_valid]

    return _metrics_from_pixels(pred_flat, gt_flat)


def _apply_median_align(pred: Tensor, gt: Tensor, valid: Tensor) -> Tensor:
    """
    Escala cada imagem do batch pela razão de medianas: pred *= med(gt)/med(pred).

    Convenção de Godard et al. (2019) usada em todos os papers de self-supervised
    depth estimation. Operação por amostra (não global) para que cada imagem ganhe
    o melhor fator de escala individualmente.
    """
    if pred.dim() == 3:  # [B, H, W]
        b = pred.shape[0]
        scaled = pred.clone()
        for i in range(b):
            v = valid[i]
            if v.any():
                scale = gt[i][v].median() / pred[i][v].median().clamp(min=1e-8)
                scaled[i] = pred[i] * scale
        return scaled

    if pred.dim() == 4:  # [B, 1, H, W]
        b = pred.shape[0]
        scaled = pred.clone()
        for i in range(b):
            v = valid[i]
            if v.any():
                scale = gt[i][v].median() / pred[i][v].median().clamp(min=1e-8)
                scaled[i] = pred[i] * scale
        return scaled

    raise ValueError(f"pred deve ter 3 ou 4 dimensões, recebido {pred.dim()}")


# ---------------------------------------------------------------------------
# Agregação entre batches (média ponderada pelo número de pixels)
# ---------------------------------------------------------------------------

def aggregate_batch_metrics(batch_metrics: list) -> Dict[str, float]:
    """
    Agrega uma lista de dicts de métricas (um por batch) em uma média final.

    Implementação simples (média não ponderada). Para a maioria dos casos com
    batches do mesmo tamanho e mesma resolução, isto é equivalente à média
    ponderada e bate com o que o paper reporta.

    Args:
        batch_metrics: lista de dicts retornados por compute_depth_metrics().

    Returns:
        dict com as mesmas chaves, contendo a média sobre os batches.
    """
    if not batch_metrics:
        return {}
    keys = batch_metrics[0].keys()
    out = {}
    for k in keys:
        vals = [m[k] for m in batch_metrics if not np.isnan(m[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


def format_metrics(metrics: Dict[str, float]) -> str:
    """Imprime as métricas no mesmo formato das Tabelas 1-3 do paper."""
    return (
        f"Abs Rel: {metrics['abs_rel']:.3f}  "
        f"Sq Rel: {metrics['sq_rel']:.3f}  "
        f"RMSE: {metrics['rmse']:.3f}m  "
        f"RMSE log: {metrics['rmse_log']:.3f}  "
        f"δ<1.25: {metrics['delta1']:.3f}  "
        f"δ<1.25²: {metrics['delta2']:.3f}  "
        f"δ<1.25³: {metrics['delta3']:.3f}"
    )


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Teste 1: predição perfeita deve dar 0 nos erros e 1 nas acurácias.
    torch.manual_seed(0)
    gt = torch.rand(4, 1, 64, 64) * 5.0 + 0.5   # depth em [0.5, 5.5]m (1.30x cabe sem clamp)
    pred_perfect = gt.clone()
    m = compute_depth_metrics(pred_perfect, gt)
    print("[teste 1] predição perfeita:")
    print("  ", format_metrics(m))
    assert m["abs_rel"] < 1e-6
    assert m["delta1"] > 0.999

    # Teste 2: predição com erro uniforme de 20%.
    pred_off = gt * 1.20
    m = compute_depth_metrics(pred_off, gt)
    print("[teste 2] pred = 1.20 * gt:")
    print("  ", format_metrics(m))
    # ratio = max(1.20, 0.833) = 1.20 -> dentro de 1.25, todos pixels.
    assert abs(m["abs_rel"] - 0.20) < 1e-4
    assert abs(m["delta1"] - 1.0) < 1e-4

    # Teste 3: predição com erro de 30% — deve cair a δ<1.25.
    pred_off = gt * 1.30
    m = compute_depth_metrics(pred_off, gt)
    print("[teste 3] pred = 1.30 * gt:")
    print("  ", format_metrics(m))
    # ratio = 1.30 > 1.25 -> δ<1.25 deve ser ~0.
    assert m["delta1"] < 0.01
    # mas 1.30 < 1.25² = 1.5625 -> δ<1.25² deve ser ~1.
    assert m["delta2"] > 0.999

    # Teste 4: median_align corrige uma escala constante.
    pred_scaled = gt * 3.0   # predição 3x maior que o real
    m_no_align = compute_depth_metrics(pred_scaled, gt, median_align=False)
    m_align    = compute_depth_metrics(pred_scaled, gt, median_align=True)
    print("[teste 4] pred = 3.0 * gt:")
    print("   sem alinhamento:", format_metrics(m_no_align))
    print("   com alinhamento:", format_metrics(m_align))
    assert m_align["abs_rel"] < 1e-4   # depois do alinhamento, fica ~perfeito

    # Teste 5: máscara de validade exclui pixels.
    valid = torch.ones_like(gt, dtype=torch.bool)
    valid[:, :, :32, :] = False         # invalida metade
    m = compute_depth_metrics(gt.clone(), gt, valid=valid)
    assert m["abs_rel"] < 1e-6
    print("[teste 5] máscara funciona: ok")

    print("\nTodos os testes passaram. ✓")
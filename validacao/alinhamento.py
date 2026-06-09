"""
alinhamento.py — alinhamento de escala da predição contra o ground truth.

Cada família de modelo prediz profundidade num espaço diferente, então antes de
comparar com o LiDAR é preciso levar a predição para metros:

    - "direto"            : a predição já é métrica. ZoeDepth (supervisionado) e o
                            baseline estéreo da OAK-D. Nada a fazer.
    - "mediana"           : a predição tem escala arbitrária (mas linear). Monodepth2
                            (autossupervisionado). Escala por median(gt)/median(pred),
                            convenção de Godard et al. (2019).
    - "afim_disparidade"  : a predição é profundidade inversa (disparidade), invariante
                            a transformação afim. Depth-Anything / MiDaS. Ajusta
                            a·pred + b ≈ 1/gt por mínimos quadrados e devolve
                            1/(a·pred + b).

O alinhamento é aplicado **por frame**, sobre os pares esparsos (pred, gt) casados
naquele frame, e a métrica é calculada depois com median_align=False.
"""

from __future__ import annotations

import numpy as np

# Mapa modelo → modo de alinhamento.
MODO_POR_MODELO = {
    "zoedepth":      "direto",
    "monodepth2":    "mediana",
    "depthanything": "afim_disparidade",
}


def modo_para_modelo(modelo: str) -> str:
    """Retorna o modo de alinhamento de um modelo (zoedepth/monodepth2/depthanything)."""
    try:
        return MODO_POR_MODELO[modelo]
    except KeyError:
        raise ValueError(
            f"Modelo desconhecido: {modelo!r}. "
            f"Opções: {sorted(MODO_POR_MODELO)}"
        )


def alinhar(pred: np.ndarray, gt: np.ndarray, modo: str) -> np.ndarray:
    """
    Alinha a predição ao ground truth segundo o modo dado.

    Args:
        pred: predição bruta do modelo nos pixels casados (1D).
              Em metros para "direto"/"mediana"; disparidade para "afim_disparidade".
        gt:   ground truth do LiDAR nos mesmos pixels, em metros (1D).
        modo: "direto" | "mediana" | "afim_disparidade".
    Returns:
        Predição alinhada, em metros (1D, mesmo tamanho de pred).
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)

    if modo == "direto":
        return pred.copy()

    if modo == "mediana":
        if pred.size == 0:
            return pred.copy()
        med_pred = np.median(pred)
        if abs(med_pred) < 1e-8:
            return pred.copy()
        escala = np.median(gt) / med_pred
        return pred * escala

    if modo == "afim_disparidade":
        if pred.size < 2:
            return pred.copy()
        disp_gt = 1.0 / gt
        # Mínimos quadrados: [pred, 1] · [a, b]ᵀ ≈ disp_gt
        A = np.stack([pred, np.ones_like(pred)], axis=1)
        (a, b), *_ = np.linalg.lstsq(A, disp_gt, rcond=None)
        disp_pred = a * pred + b
        # Evita divisão por zero/negativo em disparidades degeneradas.
        disp_pred = np.where(disp_pred > 1e-8, disp_pred, 1e-8)
        return 1.0 / disp_pred

    raise ValueError(f"Modo de alinhamento desconhecido: {modo!r}")


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    gt = rng.uniform(1.0, 6.0, size=500)  # profundidades em [1, 6] m

    # Teste 1: "direto" não altera a predição.
    pred = gt.copy()
    out = alinhar(pred, gt, "direto")
    assert np.allclose(out, gt)
    print("[teste 1] direto: ok")

    # Teste 2: "mediana" recupera uma escala oculta constante.
    pred = gt * 4.2  # escala arbitrária desconhecida
    out = alinhar(pred, gt, "mediana")
    erro = np.abs(out - gt).mean()
    print(f"[teste 2] mediana — erro médio após alinhar: {erro:.2e}")
    assert erro < 1e-9

    # Teste 3: "afim_disparidade" recupera profundidade de uma disparidade afim.
    a0, b0 = 2.5, 0.3
    pred_disp = a0 * (1.0 / gt) + b0          # o que a rede emitiria (disparidade afim)
    out = alinhar(pred_disp, gt, "afim_disparidade")
    erro = np.abs(out - gt).mean()
    print(f"[teste 3] afim_disparidade — erro médio após alinhar: {erro:.2e}")
    assert erro < 1e-6

    # Teste 4: afim com ruído pequeno continua próximo.
    pred_disp_ruido = pred_disp + rng.normal(0, 1e-3, size=gt.shape)
    out = alinhar(pred_disp_ruido, gt, "afim_disparidade")
    erro = np.abs(out - gt).mean()
    print(f"[teste 4] afim_disparidade c/ ruído — erro médio: {erro:.2e}")
    assert erro < 0.1

    # Teste 5: mapa modelo → modo.
    assert modo_para_modelo("zoedepth") == "direto"
    assert modo_para_modelo("monodepth2") == "mediana"
    assert modo_para_modelo("depthanything") == "afim_disparidade"
    print("[teste 5] mapa modelo→modo: ok")

    print("\nTodos os testes de alinhamento passaram. ✓")

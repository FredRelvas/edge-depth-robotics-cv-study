"""
demo_conceito.py — visualização didática dos espaços de saída e do alinhamento.

NÃO faz parte do pipeline de avaliação; é um material de apoio para entender o que
cada modelo prediz e por que cada um usa um modo de alinhamento diferente. Usa a
mesma cena sintética do gerar_bag_sintetico.py (um plano de profundidade conhecida)
e gera duas figuras em resultados/:

    conceito_mapas.png    — os mapas de profundidade: GT vs saída bruta de cada modelo
    conceito_relacao.png  — saída bruta × profundidade real (mostra os graus de
                            liberdade) e o resultado depois do alinhamento

Uso:
    uv run python validacao/demo_conceito.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import alinhamento
from calibracao import carregar_calibracao
from gerar_bag_sintetico import _campo_profundidade, ESCALA_MONO, AFIM_A, AFIM_B

CALIB = Path(__file__).parent / "config" / "calibracao_exemplo.yaml"
SAIDA = Path(__file__).resolve().parent.parent / "resultados"


def cenario():
    """Profundidade GT (m) e a saída BRUTA de cada modelo, sem ruído, para a cena."""
    calib = carregar_calibracao(CALIB)
    gt = _campo_profundidade(calib, d0=3.0)            # metros, shape (H, W)
    saidas = {
        "ZoeDepth\n(métrico)":            gt.copy(),               # já em metros
        "Monodepth2\n(× escala oculta)":  gt * ESCALA_MONO,        # métrico × constante
        "Depth-Anything\n(disparidade)":  AFIM_A * (1.0 / gt) + AFIM_B,  # 1/profundidade afim
    }
    return gt, saidas


def figura_mapas(gt, saidas):
    """Os 4 mapas lado a lado, cada um com sua própria barra de cores/unidade."""
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.2))
    dados = [("Ground truth\n(LiDAR, metros)", gt, "viridis", "m")] + [
        (nome, arr, "viridis", "") for nome, arr in saidas.items()
    ]
    for ax, (titulo, arr, cmap, unidade) in zip(axs, dados):
        im = ax.imshow(arr, cmap=cmap)
        ax.set_title(titulo, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        rotulo = f"[{arr.min():.2f} .. {arr.max():.2f}] {unidade}".strip()
        cb.ax.set_xlabel(rotulo, fontsize=8)
    fig.suptitle(
        "A MESMA cena, o que cada rede de fato emite por pixel\n"
        "A estrutura é a mesma; os números (a régua) é que mudam — repare nas faixas das barras de cores",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    cam = SAIDA / "conceito_mapas.png"
    fig.savefig(cam, dpi=110); plt.close(fig)
    return cam


def figura_relacao(gt, saidas):
    """
    Linha 1: saída bruta × profundidade real → a FORMA mostra os graus de liberdade.
    Linha 2: depois de alinhar → tudo colapsa na diagonal y=x (vira metros).
    """
    modos = ["direto", "mediana", "afim_disparidade"]
    nomes = list(saidas.keys())
    # Faixa AMPLA de profundidade (1–10 m) para a relação inversa do Depth-Anything
    # aparecer como curva, e não como uma reta quase plana de um intervalo estreito.
    g = np.linspace(1.0, 10.0, 2500)
    brutos = {
        "direto":           g.copy(),                 # ZoeDepth: já em metros
        "mediana":          g * ESCALA_MONO,          # Monodepth2: métrico × escala
        "afim_disparidade": AFIM_A * (1.0 / g) + AFIM_B,  # Depth-Anything: disparidade
    }

    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    diag = np.linspace(g.min(), g.max(), 50)

    for col, (nome, modo) in enumerate(zip(nomes, modos)):
        bruto = brutos[modo]

        # ── Linha 1: relação bruta ──────────────────────────────────────────
        ax = axs[0, col]
        ax.scatter(g, bruto, s=6, alpha=0.4, color="tab:blue")
        ax.set_xlabel("profundidade real (m)")
        ax.set_ylabel("saída bruta da rede")
        rotulos = {
            "direto":           "0 graus de liberdade\nreta y = x (já é metros)",
            "mediana":          "1 grau de liberdade\nreta pela origem y = escala·x",
            "afim_disparidade": "2 graus de liberdade\ncurva: saída ∝ 1/profundidade",
        }
        ax.set_title(f"{nome.splitlines()[0]}\n{rotulos[modo]}", fontsize=10)
        ax.grid(alpha=0.3)

        # ── Linha 2: depois do alinhamento ──────────────────────────────────
        ax2 = axs[1, col]
        alinhado = alinhamento.alinhar(bruto, g, modo)
        ax2.scatter(g, alinhado, s=6, alpha=0.4, color="tab:green")
        ax2.plot(diag, diag, "r--", lw=1.5, label="y = x (perfeito)")
        ax2.set_xlabel("profundidade real (m)")
        ax2.set_ylabel("profundidade estimada (m)")
        erro = np.abs(alinhado - g).mean()
        ax2.set_title(f"depois de alinhar ({modo})\nvirou metros · erro médio {erro:.3f} m",
                      fontsize=10)
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    fig.suptitle(
        "Por que cada modelo usa um alinhamento diferente\n"
        "Linha de cima: o que a rede emite (a FORMA da nuvem = nº de incógnitas para chegar a metros).  "
        "Linha de baixo: após resolver essas incógnitas com o LiDAR, tudo vira metros.",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    cam = SAIDA / "conceito_relacao.png"
    fig.savefig(cam, dpi=110); plt.close(fig)
    return cam


def main():
    SAIDA.mkdir(parents=True, exist_ok=True)
    gt, saidas = cenario()
    c1 = figura_mapas(gt, saidas)
    c2 = figura_relacao(gt, saidas)
    print(f"Figuras geradas:\n  {c1}\n  {c2}")


if __name__ == "__main__":
    main()

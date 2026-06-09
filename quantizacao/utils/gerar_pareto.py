#!/usr/bin/env python
"""
Gera a curva de Pareto acurácia × latência a partir dos JSONs produzidos por:
    - benchmark.py    -> <base>.bench.json   (lat_ms_p50, fps, ...)
    - avaliar_engine.py -> <base>.eval.json  (metrics.abs_rel, delta1, ...)

Cruza os dois pelo nome do engine/fonte, monta uma tabela (CSV) e plota
abs_rel (↓) × latência p50 (↓), destacando a fronteira de Pareto (canto
inferior-esquerdo = melhor). Marcadores por precisão, cores por modelo.

Uso (na pasta quantizacao/, após rodar benchmark + avaliar nos engines):
    python utils/gerar_pareto.py --dir engines/ \
        --csv_out engines/pareto.csv --fig_out engines/pareto.png
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

_RES = re.compile(r"(\d+x\d+)")
_PREC = re.compile(r"_(fp16|int8|fp32)\b")


def _key(name: str) -> str:
    """Normaliza o identificador: remove extensões .engine/.onnx/.bench/.eval/.json."""
    n = name
    for suf in (".json", ".bench", ".eval", ".engine", ".onnx"):
        if n.endswith(suf):
            n = n[: -len(suf)]
    return n


def _parse_name(key: str) -> dict:
    res = _RES.search(key)
    prec = _PREC.search(key)
    resolution = res.group(1) if res else "?"
    precision = prec.group(1) if prec else "fp32"
    # modelo = tudo antes da resolução, sem a precisão
    model = key
    if res:
        model = key[: res.start()].rstrip("_")
    return {"model": model, "resolution": resolution, "precision": precision}


def load_records(d: Path) -> pd.DataFrame:
    bench, evalm = {}, {}
    for f in d.glob("*.bench.json"):
        data = json.loads(f.read_text())
        bench[_key(data.get("engine", f.name))] = data
    for f in d.glob("*.eval.json"):
        data = json.loads(f.read_text())
        evalm[_key(data.get("source", f.name))] = data

    keys = sorted(set(bench) | set(evalm))
    rows = []
    for k in keys:
        row = _parse_name(k)
        row["id"] = k
        b, e = bench.get(k), evalm.get(k)
        if b:
            row["lat_ms_p50"] = b.get("lat_ms_p50")
            row["lat_ms_p95"] = b.get("lat_ms_p95")
            row["fps"] = b.get("fps")
        if e:
            m = e.get("metrics", {})
            row["abs_rel"] = m.get("abs_rel")
            row["rmse"] = m.get("rmse")
            row["delta1"] = m.get("delta1")
        rows.append(row)
    return pd.DataFrame(rows)


def pareto_front(df: pd.DataFrame) -> pd.Series:
    """Marca pontos não-dominados: nenhum outro tem lat E abs_rel menores."""
    mask = []
    for _, r in df.iterrows():
        if pd.isna(r.get("lat_ms_p50")) or pd.isna(r.get("abs_rel")):
            mask.append(False)
            continue
        dominated = (
            (df["lat_ms_p50"] <= r["lat_ms_p50"]) &
            (df["abs_rel"] <= r["abs_rel"]) &
            ((df["lat_ms_p50"] < r["lat_ms_p50"]) | (df["abs_rel"] < r["abs_rel"]))
        ).any()
        mask.append(not dominated)
    return pd.Series(mask, index=df.index)


def plot(df: pd.DataFrame, fig_out: Path) -> None:
    plottable = df.dropna(subset=["lat_ms_p50", "abs_rel"])
    if plottable.empty:
        print("[pareto][warn] nada com lat+abs_rel para plotar (rode benchmark E avaliar).")
        return

    df = df.copy()
    df["on_front"] = pareto_front(df)

    fig, ax = plt.subplots(figsize=(8, 6))
    models = sorted(plottable["model"].unique())
    cmap = plt.get_cmap("tab10")
    colors = {m: cmap(i % 10) for i, m in enumerate(models)}
    markers = {"fp16": "o", "int8": "s", "fp32": "^"}

    for _, r in plottable.iterrows():
        ax.scatter(r["lat_ms_p50"], r["abs_rel"],
                   color=colors[r["model"]],
                   marker=markers.get(r["precision"], "o"),
                   s=90, edgecolors="black",
                   linewidths=1.5 if df.loc[r.name, "on_front"] else 0.4,
                   zorder=3)
        ax.annotate(f"{r['resolution']}", (r["lat_ms_p50"], r["abs_rel"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=7)

    # linha da fronteira de Pareto
    front = df[df["on_front"]].dropna(subset=["lat_ms_p50", "abs_rel"]) \
                              .sort_values("lat_ms_p50")
    if len(front) >= 2:
        ax.plot(front["lat_ms_p50"], front["abs_rel"], "--",
                color="gray", zorder=2, label="fronteira de Pareto")

    # legendas
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[m],
                      markeredgecolor="black", markersize=9, label=m) for m in models]
    handles += [Line2D([0], [0], marker=mk, color="w", markerfacecolor="gray",
                       markeredgecolor="black", markersize=9, label=p)
                for p, mk in markers.items()]
    if len(front) >= 2:
        handles.append(Line2D([0], [0], ls="--", color="gray", label="Pareto"))
    ax.legend(handles=handles, fontsize=8, loc="upper right")

    ax.set_xlabel("Latência p50 (ms) — Jetson Orin Nano")
    ax.set_ylabel("AbsRel (↓ melhor)")
    ax.set_title("Pareto acurácia × latência — depth no ICL Ground Robot")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_out, dpi=150)
    print(f"[pareto] figura salva em {fig_out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("engines"),
                    help="pasta com *.bench.json e *.eval.json")
    ap.add_argument("--csv_out", type=Path, default=None)
    ap.add_argument("--fig_out", type=Path, default=Path("engines/pareto.png"))
    args = ap.parse_args()

    df = load_records(args.dir)
    if df.empty:
        raise SystemExit(f"[erro] nenhum *.bench.json/*.eval.json em {args.dir}")

    cols = ["model", "resolution", "precision", "abs_rel", "rmse", "delta1",
            "lat_ms_p50", "lat_ms_p95", "fps"]
    cols = [c for c in cols if c in df.columns]
    table = df[cols].sort_values(["model", "resolution", "precision"])
    print(table.to_string(index=False))

    if args.csv_out:
        table.to_csv(args.csv_out, index=False)
        print(f"[pareto] tabela salva em {args.csv_out}")

    plot(df, args.fig_out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Separa imagens do split de TREINO do ICL Ground Robot para calibração INT8.

(O README-template falava em "LMData", mas o dataset real deste projeto é o
ICL Ground Robot — deer + diamond. Usamos as mesmas imagens vistas no treino
para a calibração ser representativa da distribuição de operação.)

As imagens são copiadas CRUAS (PNG). O pré-processamento (/255, resize, NCHW)
é feito pelo calibrador dentro do build_engine.py, garantindo que a
calibração use exatamente o mesmo contrato de entrada [0,1] dos modelos.

Uso:
    python export/preparar_calibracao.py --n 500 --out calibracao/
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "codigo-treinamento"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500,
                    help="número de imagens de calibração (~500 é suficiente).")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "calibracao"))
    ap.add_argument("--data_root", default=str(PROJECT_ROOT / "dados" / "icl_ground_robot"))
    ap.add_argument("--image_size", type=int, default=256,
                    help="só usado para instanciar o dataset; copiamos o PNG original.")
    ap.add_argument("--scene", nargs="+", default=["deer", "diamond"],
                    help="cenas a usar (default: deer diamond).")
    args = ap.parse_args()

    from dataloader import ICLGroundRobotDataset

    ds = ICLGroundRobotDataset(
        root=args.data_root, scene=tuple(args.scene),
        split="train", image_size=args.image_size, augment=False,
    )
    n = min(args.n, len(ds))
    # Amostra uniforme ao longo do split (cobre as duas cenas e a trajetória).
    idxs = [round(i * (len(ds) - 1) / max(1, n - 1)) for i in range(n)]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # limpa PNGs antigos (mantém .gitkeep)
    for old in out.glob("*.png"):
        old.unlink()

    copied = 0
    for j, idx in enumerate(idxs):
        src = Path(ds.samples[idx]["rgb_path"])
        if not src.exists():
            continue
        dst = out / f"calib_{j:04d}_{ds.samples[idx]['scene']}.png"
        shutil.copy(src, dst)
        copied += 1

    print(f"[calib] {copied} imagens copiadas para {out}")
    print(f"[calib] use no build_engine.py: --calib_dir {out}")


if __name__ == "__main__":
    main()

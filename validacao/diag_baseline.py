"""
diag_baseline.py — diagnóstico do baseline no bag "morto".

Responde a duas perguntas, ambas SEM depender da IA (que está quebrada):
  (A) Quanto o alinhamento por mediana "ajuda" o baseline? (mede a assimetria)
  (B) O bag foi gravado com align_depth? (frames depth vs color)

Uso: uv run python validacao/diag_baseline.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "validacao"))
sys.path.insert(0, str(RAIZ / "metricas"))

import geometria as geo
from leitor_rosbag import ler_bag, TOPICOS_DEFAULT, _decodificar_imagem
from calibracao import carregar_calibracao
from metricas import compute_depth_metrics  # type: ignore
import torch

MIN_D, MAX_D = 0.15, 10.0


def metr(pred, gt):
    p = torch.tensor(pred, dtype=torch.float32)
    g = torch.tensor(gt, dtype=torch.float32)
    m = compute_depth_metrics(p, g, median_align=False, min_depth=MIN_D, max_depth=MAX_D)
    return m


def main():
    calib = carregar_calibracao(RAIZ / "validacao/config/calibracao_realsense_d415.yaml")
    amostras = ler_bag(RAIZ / "dados", intrinseca_fallback=calib.K,
                       dims_fallback=(calib.largura, calib.altura))
    T = calib.T_lidar_cam

    gt_all, oak_raw_all, oak_align_all = [], [], []
    descartados = 0
    for a in amostras:
        pts = geo.laserscan_para_pontos(a.scan.ranges, a.scan.angle_min,
                                        a.scan.angle_increment,
                                        a.scan.range_min, a.scan.range_max)
        pts_cam = geo.transformar(pts, T)
        u, v, z = geo.projetar(pts_cam, calib.K, calib.largura, calib.altura)
        if u.size == 0:
            continue
        oak = geo.amostrar(a.oak_depth, u, v, calib.largura, calib.altura).astype(np.float64)
        oak_m = oak / 1000.0  # 16UC1 mm → m

        bom = (np.isfinite(z) & (z >= MIN_D) & (z <= MAX_D)
               & np.isfinite(oak_m) & (oak_m > MIN_D) & (oak_m <= MAX_D))
        descartados += int((~bom).sum())
        z, oak_m = z[bom], oak_m[bom]
        if z.size < 5:
            continue

        # (A) alinhamento por mediana aplicado ao baseline (mesmo da IA).
        s = np.median(z) / np.median(oak_m)
        gt_all.append(z)
        oak_raw_all.append(oak_m)
        oak_align_all.append(oak_m * s)

    gt = np.concatenate(gt_all)
    oak_raw = np.concatenate(oak_raw_all)
    oak_align = np.concatenate(oak_align_all)

    print("=" * 64)
    print(f"  Pontos baseline usados: {gt.size}   (descartados: {descartados})")
    print("=" * 64)
    for nome, pred in [("baseline CRU      ", oak_raw), ("baseline ALINHADO ", oak_align)]:
        m = metr(pred, gt)
        print(f"  {nome} | Abs Rel: {m['abs_rel']:.3f}  RMSE: {m['rmse']:.3f}m  "
              f"δ<1.25: {m['delta1']:.3f}")
    print("=" * 64)

    # (B) align_depth? frames das imagens.
    print("\n[ Verificação align_depth — frames das imagens do bag ]")
    from rosbags.highlevel import AnyReader
    from rosbags.typesys import Stores, get_typestore
    ts = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([RAIZ / "dados"], default_typestore=ts) as r:
        nomes = [c.topic for c in r.connections]
        for topico in ["/camera/camera/color/image_raw",
                       "/camera/camera/depth/image_rect_raw"]:
            conns = [c for c in r.connections if c.topic == topico]
            for con, _t, d in r.messages(connections=conns):
                m = r.deserialize(d, con.msgtype)
                print(f"  {topico}: frame_id={m.header.frame_id!r}")
                break
        tem_aligned = any("aligned_depth" in n for n in nomes)
        print(f"  tópico aligned_depth_to_color presente? {tem_aligned}")


if __name__ == "__main__":
    main()

"""
inspecionar_bag.py — raio-x de um rosbag antes de rodar a validação.

É a primeira coisa a rodar quando o bag real do robô chegar. Faz um levantamento
do que o bag CONTÉM (tópicos, tipos, encodings, resolução de cada stream, frame_ids,
taxas e faixas de valores) e confere contra o contrato que o pipeline da Frente 4
espera, sinalizando problemas — em especial:

    • ia/depth_map normalizado em [0,1] (deveria ser depth bruto em metros);
    • stereo/depth que parece disparidade, ou em frame diferente do RGB (não alinhado);
    • ausência de camera_info / scan / tf_static;
    • nomes de tópico ou namespace diferentes do esperado.

A partir desse relatório dá para decidir, com base em evidência, quais adaptações
fazer — em vez de construí-las no escuro.

Uso:
    uv run python validacao/inspecionar_bag.py <caminho_do_bag>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

sys.path.insert(0, str(Path(__file__).resolve().parent))
from leitor_rosbag import _decodificar_imagem  # reusa o decodificador de imagem

VERDE, AMARELO, VERMELHO, CINZA, FIM = "\033[0;32m", "\033[1;33m", "\033[0;31m", "\033[0;90m", "\033[0m"
def ok(m):   print(f"  {VERDE}✓{FIM} {m}")
def warn(m): print(f"  {AMARELO}⚠{FIM} {m}")
def fail(m): print(f"  {VERMELHO}✗{FIM} {m}")
def info(m): print(f"  {CINZA}·{FIM} {m}")


def _stats_depth(arr: np.ndarray) -> dict:
    """Estatísticas de um mapa de profundidade (ignora nan/inf no min/max/mean)."""
    a = arr.astype(np.float64).ravel()
    finito = np.isfinite(a)
    fin = a[finito]
    return {
        "min": float(fin.min()) if fin.size else float("nan"),
        "max": float(fin.max()) if fin.size else float("nan"),
        "mean": float(fin.mean()) if fin.size else float("nan"),
        "pct_zero": 100.0 * float(np.mean(a == 0)),
        "pct_nan": 100.0 * float(np.mean(np.isnan(a))),
        "pct_inf": 100.0 * float(np.mean(np.isinf(a))),
    }


def _primeira_msg(reader, topico):
    """Deserializa a primeira mensagem de um tópico."""
    conns = [c for c in reader.connections if c.topic == topico]
    for con, _t, dados in reader.messages(connections=conns):
        return reader.deserialize(dados, con.msgtype)
    return None


def inspecionar(caminho: Path):
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    with AnyReader([caminho], default_typestore=typestore) as reader:
        dur_s = reader.duration / 1e9 if reader.duration else 0.0

        print("=" * 74)
        print(f"  INSPEÇÃO DE BAG: {caminho}")
        print("=" * 74)
        print(f"  Mensagens: {reader.message_count}   Duração: {dur_s:.1f} s   "
              f"Tópicos: {len(reader.topics)}")
        print()

        # Classifica os tópicos por tipo de mensagem.
        por_tipo = {}
        for nome, meta in reader.topics.items():
            por_tipo.setdefault(meta.msgtype, []).append((nome, meta))

        # ── Lista geral ──────────────────────────────────────────────────────
        print("[ Tópicos ]")
        for nome, meta in sorted(reader.topics.items()):
            taxa = meta.msgcount / dur_s if dur_s > 0 else 0.0
            print(f"  {nome:<48} {meta.msgtype:<26} {meta.msgcount:>5} msg  {taxa:5.1f} Hz")
        print()

        imagens = por_tipo.get("sensor_msgs/msg/Image", [])
        scans = por_tipo.get("sensor_msgs/msg/LaserScan", [])
        caminfos = por_tipo.get("sensor_msgs/msg/CameraInfo", [])
        tfs = (por_tipo.get("tf2_msgs/msg/TFMessage", []))

        # ── Imagens (ia / oak / rgb) ─────────────────────────────────────────
        print("[ Streams de imagem ]")
        info_imgs = {}
        for nome, _meta in imagens:
            msg = _primeira_msg(reader, nome)
            if msg is None:
                continue
            enc = msg.encoding
            res = f"{msg.width}x{msg.height}"
            frame = msg.header.frame_id
            linha = f"{nome}  enc={enc}  res={res}  frame_id={frame!r}"
            if enc in ("32FC1", "16UC1", "mono16"):
                arr = _decodificar_imagem(msg)
                s = _stats_depth(arr)
                linha += (f"\n      valores: min={s['min']:.3f} max={s['max']:.3f} "
                          f"mean={s['mean']:.3f}  zeros={s['pct_zero']:.0f}% "
                          f"nan={s['pct_nan']:.0f}% inf={s['pct_inf']:.0f}%")
                info_imgs[nome] = {"enc": enc, "res": res, "frame": frame, "stats": s}
            else:
                info_imgs[nome] = {"enc": enc, "res": res, "frame": frame, "stats": None}
            info(linha)
        print()

        # ── CameraInfo ───────────────────────────────────────────────────────
        print("[ CameraInfo ]")
        for nome, _meta in caminfos:
            msg = _primeira_msg(reader, nome)
            K = np.asarray(msg.k, dtype=float).reshape(3, 3)
            info(f"{nome}  res={msg.width}x{msg.height}  "
                 f"fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}  "
                 f"frame_id={msg.header.frame_id!r}")
        if not caminfos:
            warn("Nenhum CameraInfo no bag.")
        print()

        # ── LaserScan ────────────────────────────────────────────────────────
        print("[ LaserScan ]")
        for nome, _meta in scans:
            msg = _primeira_msg(reader, nome)
            r = np.asarray(msg.ranges, dtype=float)
            n = r.size
            pct_inf = 100.0 * np.mean(~np.isfinite(r))
            fin = r[np.isfinite(r)]
            info(f"{nome}  feixes={n}  fov=[{np.degrees(msg.angle_min):.0f}°,"
                 f"{np.degrees(msg.angle_max):.0f}°]  range=[{msg.range_min:.2f},"
                 f"{msg.range_max:.2f}]m  inválidos={pct_inf:.0f}%"
                 + (f"  válidos[min={fin.min():.2f} max={fin.max():.2f}]" if fin.size else ""))
        if not scans:
            warn("Nenhum LaserScan no bag.")
        print()

        # ── TF ───────────────────────────────────────────────────────────────
        print("[ TF (árvore de frames) ]")
        frames_tf = set()
        for nome, _meta in tfs:
            msg = _primeira_msg(reader, nome)
            for tr in msg.transforms:
                frames_tf.add((tr.header.frame_id, tr.child_frame_id))
                info(f"{nome}: {tr.header.frame_id} → {tr.child_frame_id}")
        if not tfs:
            warn("Nenhum tópico TF — extrínseca terá de vir do arquivo de calibração.")
        print()

        # ── Diagnóstico contra o contrato esperado ───────────────────────────
        print("=" * 74)
        print("  DIAGNÓSTICO (contra o que o pipeline espera)")
        print("=" * 74)
        _diagnostico(info_imgs, caminfos, scans, frames_tf)


def _palpite(info_imgs, *chaves):
    """Adivinha o tópico cujo nome contém alguma das chaves."""
    for nome in info_imgs:
        baixo = nome.lower()
        if any(k in baixo for k in chaves):
            return nome
    return None


def _diagnostico(info_imgs, caminfos, scans, frames_tf):
    # IA depth map
    ia = _palpite(info_imgs, "ia", "depth_map")
    if not ia:
        fail("ia/depth_map: NÃO encontrado (procurei 'ia'/'depth_map' no nome).")
    else:
        d = info_imgs[ia]
        if d["enc"] != "32FC1":
            warn(f"{ia}: encoding {d['enc']} (esperado 32FC1).")
        s = d["stats"]
        if s and np.isfinite(s["max"]):
            if s["max"] <= 1.01 and s["min"] >= 0.0:
                fail(f"{ia}: valores em [{s['min']:.2f},{s['max']:.2f}] → PARECE NORMALIZADO [0,1]! "
                     "Precisa ser depth BRUTO (correção da Frente 1 no depth_node).")
            elif 1.2 < s["max"] < 100:
                ok(f"{ia}: faixa {s['min']:.2f}–{s['max']:.2f} parece profundidade métrica/relativa válida.")
            else:
                warn(f"{ia}: faixa {s['min']:.2f}–{s['max']:.2f} — confira se faz sentido.")

    # OAK stereo depth
    oak = _palpite(info_imgs, "stereo", "oakd/depth", "depth/depth")
    if not oak:
        warn("stereo/depth (baseline OAK-D): não encontrado pelo nome. Confirme o tópico.")
    else:
        d = info_imgs[oak]
        s = d["stats"]
        if d["enc"] == "16UC1" and s and s["max"] > 100:
            ok(f"{oak}: 16UC1, max={s['max']:.0f} → profundidade em mm (será /1000).")
        elif d["enc"] == "32FC1" and s and s["max"] < 100:
            ok(f"{oak}: 32FC1, max={s['max']:.1f} → profundidade em metros.")
        else:
            warn(f"{oak}: enc={d['enc']} max={s['max'] if s else '?'} — confirme se é PROFUNDIDADE "
                 "(não disparidade).")

    # Alinhamento estéreo↔RGB pelo frame_id
    if ia and oak:
        if info_imgs[ia]["frame"] != info_imgs[oak]["frame"]:
            warn(f"frame_id difere (ia={info_imgs[ia]['frame']!r} vs oak={info_imgs[oak]['frame']!r}): "
                 "o stereo pode NÃO estar alinhado ao RGB → talvez precise projeção no frame estéreo.")
        else:
            ok("ia e oak no mesmo frame_id (provável alinhamento ao RGB).")

    # Resoluções
    if ia and oak:
        if info_imgs[ia]["res"] != info_imgs[oak]["res"]:
            info(f"Resoluções diferentes (ia={info_imgs[ia]['res']}, oak={info_imgs[oak]['res']}) — "
                 "tratado pela amostragem com escala, mas confira o camera_info correspondente.")

    # CameraInfo / scan / tf
    ok("camera_info presente.") if caminfos else fail("camera_info AUSENTE — sem intrínsecas não há projeção.")
    ok("scan (LiDAR) presente.") if scans else fail("scan AUSENTE — sem ground truth do LiDAR.")
    if frames_tf:
        nomes = {f for par in frames_tf for f in par}
        tem_lidar = any("lidar" in f.lower() or "laser" in f.lower() or "rplidar" in f.lower() for f in nomes)
        tem_cam = any("cam" in f.lower() or "oak" in f.lower() or "optical" in f.lower() for f in nomes)
        if tem_lidar and tem_cam:
            ok("TF tem frames de LiDAR e câmera → extrínseca pode ser derivada do TF.")
        else:
            warn("TF presente, mas não identifiquei frames claros de LiDAR e câmera — verifique a árvore.")
    print("=" * 74)


def main():
    if len(sys.argv) < 2:
        print("Uso: uv run python validacao/inspecionar_bag.py <caminho_do_bag>")
        sys.exit(1)
    inspecionar(Path(sys.argv[1]))


if __name__ == "__main__":
    main()

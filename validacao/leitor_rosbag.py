"""
leitor_rosbag.py — leitura e sincronização de um rosbag de validação.

Lê um bag rosbag2 (.mcap ou .db3) com a lib `rosbags` (puro Python, sem ROS
instalado) e sincroniza, para cada frame da IA, o scan do LiDAR e a profundidade
da OAK-D mais próximos no tempo (dentro de uma tolerância) — equivalente a um
ApproximateTime. O CameraInfo é lido uma vez (latched).

Tópicos esperados (default, espelham o record_bag.sh do repo TB4):
    /robot4/ia/depth_map               sensor_msgs/Image   (32FC1)  → y_ia
    /robot4/stereo/depth               sensor_msgs/Image   (16UC1)  → y_oak (mm)
    /robot4/scan                       sensor_msgs/LaserScan        → LiDAR
    /robot4/oakd/rgb/preview/camera_info  sensor_msgs/CameraInfo    → intrínsecas K
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from rosbags.highlevel import AnyReader


# Tópicos default (sobrescrevíveis em ler_bag).
TOPICOS_DEFAULT = {
    "ia":          "/robot4/ia/depth_map",
    "oak":         "/robot4/stereo/depth",
    "scan":        "/robot4/scan",
    "camera_info": "/robot4/oakd/rgb/preview/camera_info",
}


@dataclass
class Scan2D:
    """Dados mínimos de um LaserScan para a projeção."""
    ranges: np.ndarray
    angle_min: float
    angle_increment: float
    range_min: float
    range_max: float


@dataclass
class AmostraSincronizada:
    """Um frame da IA com o scan e a profundidade da OAK-D casados no tempo."""
    stamp_ns: int
    ia_depth: np.ndarray      # (H, W) float32 — predição bruta do modelo
    oak_depth: np.ndarray     # (H, W) — profundidade estéreo bruta (16UC1, mm)
    scan: Scan2D
    K: np.ndarray             # (3, 3) intrínseca da câmera de referência
    largura_ref: int          # largura da imagem de referência (camera_info)
    altura_ref: int           # altura da imagem de referência (camera_info)


# ---------------------------------------------------------------------------
# Decodificação de mensagens
# ---------------------------------------------------------------------------

def _stamp_ns(msg) -> int:
    """Timestamp do header da mensagem em nanossegundos."""
    s = msg.header.stamp
    return int(s.sec) * 1_000_000_000 + int(s.nanosec)


# encoding → (dtype, nº de canais)
_ENCODINGS = {
    "rgb8":  (np.uint8, 3),
    "bgr8":  (np.uint8, 3),
    "mono8": (np.uint8, 1),
    "32FC1": (np.float32, 1),
    "16UC1": (np.uint16, 1),
    "mono16": (np.uint16, 1),
}


def _decodificar_imagem(msg) -> np.ndarray:
    """Converte um sensor_msgs/Image em array numpy (H, W) ou (H, W, C)."""
    enc = msg.encoding
    if enc not in _ENCODINGS:
        raise ValueError(f"Encoding de imagem não suportado: {enc!r}")
    dtype, canais = _ENCODINGS[enc]

    raw = np.asarray(msg.data, dtype=np.uint8)
    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    bytes_pix = np.dtype(dtype).itemsize * canais

    # Respeita o stride (step) e descarta padding ao fim de cada linha.
    linhas = raw.reshape(h, step)[:, : w * bytes_pix].copy()
    img = linhas.view(dtype).reshape(h, w, canais)
    return img[:, :, 0] if canais == 1 else img


# ---------------------------------------------------------------------------
# Leitura + sincronização
# ---------------------------------------------------------------------------

def _mais_proximo(alvo_ns: int, lista, tol_ns: int):
    """Retorna o item (stamp, valor) de `lista` mais próximo de alvo_ns dentro da tolerância, ou None."""
    melhor = None
    melhor_d = None
    for stamp, valor in lista:
        d = abs(stamp - alvo_ns)
        if d <= tol_ns and (melhor_d is None or d < melhor_d):
            melhor, melhor_d = valor, d
    return melhor


def ler_bag(
    caminho: str | Path,
    topicos: Optional[Dict[str, str]] = None,
    tolerancia_ms: float = 50.0,
) -> List[AmostraSincronizada]:
    """
    Lê o bag e devolve a lista de amostras sincronizadas.

    Args:
        caminho:       diretório do rosbag2 (.mcap/.db3) ou arquivo.
        topicos:       dict com chaves ia/oak/scan/camera_info (default TOPICOS_DEFAULT).
        tolerancia_ms: janela máxima de casamento temporal por mensagem.
    Returns:
        Lista de AmostraSincronizada, uma por frame da IA com scan e OAK casados.
    """
    topicos = {**TOPICOS_DEFAULT, **(topicos or {})}
    tol_ns = int(tolerancia_ms * 1_000_000)

    ia_msgs: List = []
    oak_msgs: List = []
    scan_msgs: List = []
    K = None
    largura_ref = altura_ref = None

    caminho = Path(caminho)
    with AnyReader([caminho]) as reader:
        alvos = {topicos["ia"], topicos["oak"], topicos["scan"], topicos["camera_info"]}
        conexoes = [c for c in reader.connections if c.topic in alvos]
        for conexao, _t, dados in reader.messages(connections=conexoes):
            msg = reader.deserialize(dados, conexao.msgtype)
            topico = conexao.topic

            if topico == topicos["camera_info"]:
                if K is None:
                    K = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
                    largura_ref, altura_ref = int(msg.width), int(msg.height)
            elif topico == topicos["ia"]:
                ia_msgs.append((_stamp_ns(msg), _decodificar_imagem(msg)))
            elif topico == topicos["oak"]:
                oak_msgs.append((_stamp_ns(msg), _decodificar_imagem(msg)))
            elif topico == topicos["scan"]:
                scan = Scan2D(
                    ranges=np.asarray(msg.ranges, dtype=np.float64),
                    angle_min=float(msg.angle_min),
                    angle_increment=float(msg.angle_increment),
                    range_min=float(msg.range_min),
                    range_max=float(msg.range_max),
                )
                scan_msgs.append((_stamp_ns(msg), scan))

    if K is None:
        raise ValueError(
            f"CameraInfo não encontrado no tópico {topicos['camera_info']!r}. "
            "Sem intrínsecas não há projeção."
        )

    amostras: List[AmostraSincronizada] = []
    for stamp, ia_depth in sorted(ia_msgs):
        oak = _mais_proximo(stamp, oak_msgs, tol_ns)
        scan = _mais_proximo(stamp, scan_msgs, tol_ns)
        if oak is None or scan is None:
            continue  # frame sem par válido — ignorado
        amostras.append(AmostraSincronizada(
            stamp_ns=stamp,
            ia_depth=ia_depth,
            oak_depth=oak,
            scan=scan,
            K=K,
            largura_ref=largura_ref,
            altura_ref=altura_ref,
        ))
    return amostras


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python validacao/leitor_rosbag.py <caminho_do_bag>")
        print("(o teste end-to-end é via gerar_bag_sintetico.py + avaliar.py)")
        sys.exit(0)
    amostras = ler_bag(sys.argv[1])
    print(f"{len(amostras)} amostras sincronizadas lidas.")
    if amostras:
        a = amostras[0]
        print(f"  ref={a.largura_ref}x{a.altura_ref}  ia={a.ia_depth.shape}  "
              f"oak={a.oak_depth.shape}  feixes={a.scan.ranges.shape[0]}")

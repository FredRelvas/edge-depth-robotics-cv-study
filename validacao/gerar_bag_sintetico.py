"""
gerar_bag_sintetico.py — escreve um rosbag2 .mcap sintético com geometria conhecida.

Serve para validar o pipeline da Frente 4 de ponta a ponta sem o robô: a cena é
um plano inclinado de profundidade conhecida, o scan do LiDAR é derivado por
interseção raio-plano (fisicamente consistente com a câmera), e as imagens de
profundidade (y_oak e y_ia) são geradas a partir do mesmo ground truth.

A variante de y_ia depende do --modelo, para exercitar cada modo de alinhamento:
    zoedepth      → y_ia = depth_gt + ruído            (métrico → alinhamento direto)
    monodepth2    → y_ia = depth_gt · escala + ruído   (escala oculta → mediana)
    depthanything → y_ia = a·(1/depth_gt) + b + ruído  (disparidade → afim)

Com ruído pequeno, o avaliar.py deve reportar métricas ≈ 0 após o alinhamento.

Uso:
    uv run python validacao/gerar_bag_sintetico.py --modelo zoedepth --out /tmp/sim_zoe
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
from rosbags.rosbag2 import Writer
from rosbags.rosbag2.writer import StoragePlugin
from rosbags.typesys import Stores, get_typestore

from calibracao import carregar_calibracao

# Plano da cena no frame da câmera: n·X = d0. Normal fixa; d0 varia por frame.
NORMAL_PLANO = np.array([0.15, -0.4, 1.0])
D0_BASE = 3.0
D0_PASSO = 0.15  # variação de d0 entre frames (simula o robô andando)

# Parâmetros do "LiDAR" sintético.
ANGLE_MIN = -np.pi
ANGLE_MAX = np.pi
N_FEIXES = 720
RANGE_MIN = 0.1
RANGE_MAX = 12.0

# Escala/afim ocultas injetadas em cada variante de modelo.
ESCALA_MONO = 2.7
AFIM_A, AFIM_B = 3.0, 0.2

PROF_MIN, PROF_MAX = 0.4, 9.5  # clip do campo de profundidade

# Cena "parede_chao": uma parede inclinada (o LiDAR bate nela) + um chão horizontal
# a ALTURA_CHAO abaixo da câmera (a câmera vê nos pixels de baixo; o LiDAR horizontal
# nunca o atinge). Serve para testar a ancoragem de escala pela altura da câmera.
ALTURA_CHAO = 0.395
NORMAL_PAREDE = np.array([0.3, 0.0, 1.0])   # plano "vertical" (sem componente y), inclinado em x
NORMAL_CHAO = np.array([0.0, 1.0, 0.0])     # plano horizontal, normal no eixo y (para baixo)


def _depth_plano(calib, normal: np.ndarray, c: float) -> np.ndarray:
    """Profundidade por pixel (Z) para o plano normal·X = c. Sem interseção válida → inf."""
    fx, fy = calib.K[0, 0], calib.K[1, 1]
    cx, cy = calib.K[0, 2], calib.K[1, 2]
    uu, vv = np.meshgrid(np.arange(calib.largura), np.arange(calib.altura))
    dx = (uu - cx) / fx
    dy = (vv - cy) / fy
    denom = normal[0] * dx + normal[1] * dy + normal[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = c / denom
    depth[~np.isfinite(depth) | (depth <= 0)] = np.inf
    return depth


def _campo_profundidade(calib, d0: float) -> np.ndarray:
    """Profundidade GT (metros) por pixel para o plano inclinado único. Shape (H, W)."""
    depth = _depth_plano(calib, NORMAL_PLANO, d0)
    return np.clip(depth, PROF_MIN, PROF_MAX).astype(np.float64)


def _campo_parede_chao(calib, d0: float) -> np.ndarray:
    """Profundidade GT da cena parede+chão: o pixel vê a superfície mais próxima."""
    parede = _depth_plano(calib, NORMAL_PAREDE, d0)          # c_parede = d0
    chao = _depth_plano(calib, NORMAL_CHAO, ALTURA_CHAO)     # c_chao = altura
    depth = np.minimum(parede, chao)
    return np.clip(depth, PROF_MIN, PROF_MAX).astype(np.float64)


def _scan_plano(calib, normal: np.ndarray, c: float) -> np.ndarray:
    """Ranges do LaserScan por interseção raio-plano com normal·X = c. Sem hit → inf."""
    T = calib.T_lidar_cam
    R = T[:3, :3]
    origem_cam = T[:3, 3]
    n_o = normal @ origem_cam

    angulos = ANGLE_MIN + np.arange(N_FEIXES) * ((ANGLE_MAX - ANGLE_MIN) / N_FEIXES)
    ranges = np.full(N_FEIXES, np.inf)
    for i, th in enumerate(angulos):
        dir_lidar = np.array([np.cos(th), np.sin(th), 0.0])
        dir_cam = R @ dir_lidar
        denom = normal @ dir_cam
        if abs(denom) < 1e-9:        # feixe paralelo ao plano (ex.: LiDAR vs chão) → sem hit
            continue
        s = (c - n_o) / denom
        if RANGE_MIN <= s <= RANGE_MAX:
            ranges[i] = s
    return ranges


def _scan_do_plano(calib, d0: float) -> np.ndarray:
    """Scan da cena de plano único (LiDAR bate no plano inclinado)."""
    return _scan_plano(calib, NORMAL_PLANO, d0)


def _scan_parede_chao(calib, d0: float) -> np.ndarray:
    """Scan da cena parede+chão: o LiDAR horizontal só atinge a parede, nunca o chão."""
    return _scan_plano(calib, NORMAL_PAREDE, d0)


def _variante_ia(depth_gt: np.ndarray, modelo: str, rng, ruido: float) -> np.ndarray:
    """Gera y_ia (32FC1) a partir do GT, conforme o espaço de saída do modelo."""
    n = rng.normal(0.0, ruido, size=depth_gt.shape)
    if modelo == "zoedepth":
        return (depth_gt + n).astype(np.float32)
    if modelo == "monodepth2":
        return (depth_gt * ESCALA_MONO + n).astype(np.float32)
    if modelo == "depthanything":
        disp = 1.0 / depth_gt
        return (AFIM_A * disp + AFIM_B + n).astype(np.float32)
    raise ValueError(f"Modelo desconhecido: {modelo}")


def gerar(out: Path, modelo: str, n_frames: int, ruido: float, calib_path: Path,
          cena: str = "plano", vies_lidar: float = 1.0, normalizar_ia: bool = False):
    calib = carregar_calibracao(calib_path)
    campo = _campo_parede_chao if cena == "parede_chao" else _campo_profundidade
    scan_fn = _scan_parede_chao if cena == "parede_chao" else _scan_do_plano
    ts = get_typestore(Stores.ROS2_HUMBLE)
    Header = ts.types["std_msgs/msg/Header"]
    Time = ts.types["builtin_interfaces/msg/Time"]
    Image = ts.types["sensor_msgs/msg/Image"]
    CameraInfo = ts.types["sensor_msgs/msg/CameraInfo"]
    LaserScan = ts.types["sensor_msgs/msg/LaserScan"]
    RegionOfInterest = ts.types["sensor_msgs/msg/RegionOfInterest"]

    rng = np.random.default_rng(42)
    W, H = calib.largura, calib.altura

    def header(stamp_ns, frame):
        return Header(stamp=Time(sec=stamp_ns // 1_000_000_000,
                                 nanosec=stamp_ns % 1_000_000_000),
                      frame_id=frame)

    if out.exists():
        shutil.rmtree(out)

    with Writer(out, version=9, storage_plugin=StoragePlugin.MCAP) as writer:
        c_ci   = writer.add_connection("/robot4/oakd/rgb/preview/camera_info",
                                       "sensor_msgs/msg/CameraInfo", typestore=ts)
        c_ia   = writer.add_connection("/robot4/ia/depth_map",
                                       "sensor_msgs/msg/Image", typestore=ts)
        c_oak  = writer.add_connection("/robot4/stereo/depth",
                                       "sensor_msgs/msg/Image", typestore=ts)
        c_scan = writer.add_connection("/robot4/scan",
                                       "sensor_msgs/msg/LaserScan", typestore=ts)

        t0 = 1_000_000_000
        dt = 100_000_000  # 100 ms entre frames

        # CameraInfo uma vez (latched-like).
        ci = CameraInfo(
            header=header(t0, "camera_optical"),
            height=H, width=W, distortion_model="plumb_bob",
            d=np.zeros(5, dtype=np.float64),
            k=calib.K.flatten().astype(np.float64),
            r=np.eye(3, dtype=np.float64).flatten(),
            p=np.concatenate([calib.K, np.zeros((3, 1))], axis=1).flatten().astype(np.float64),
            binning_x=0, binning_y=0,
            roi=RegionOfInterest(x_offset=0, y_offset=0, height=0, width=0, do_rectify=False),
        )
        writer.write(c_ci, t0, ts.serialize_cdr(ci, ci.__msgtype__))

        for f in range(n_frames):
            stamp = t0 + f * dt
            d0 = D0_BASE + D0_PASSO * f
            depth_gt = campo(calib, d0)

            # y_ia (32FC1)
            ia = _variante_ia(depth_gt, modelo, rng, ruido)
            if normalizar_ia:  # simula o bug do depth_node (min-max por frame) — y_ia inútil
                ia = ((ia - ia.min()) / (ia.max() - ia.min())).astype(np.float32)
            ia_msg = Image(
                header=header(stamp, "camera_optical"),
                height=H, width=W, encoding="32FC1", is_bigendian=0,
                step=W * 4, data=np.frombuffer(ia.tobytes(), dtype=np.uint8),
            )
            writer.write(c_ia, stamp, ts.serialize_cdr(ia_msg, ia_msg.__msgtype__))

            # y_oak (16UC1, mm)
            oak_m = depth_gt + rng.normal(0.0, ruido, size=depth_gt.shape)
            oak_mm = np.clip(oak_m * 1000.0, 0, 65535).astype(np.uint16)
            oak_msg = Image(
                header=header(stamp, "camera_optical"),
                height=H, width=W, encoding="16UC1", is_bigendian=0,
                step=W * 2, data=np.frombuffer(oak_mm.tobytes(), dtype=np.uint8),
            )
            writer.write(c_oak, stamp, ts.serialize_cdr(oak_msg, oak_msg.__msgtype__))

            # LaserScan — vies_lidar (α) simula um erro sistemático de alcance do LiDAR.
            ranges = scan_fn(calib, d0) * vies_lidar
            scan_msg = LaserScan(
                header=header(stamp, "lidar"),
                angle_min=float(ANGLE_MIN), angle_max=float(ANGLE_MAX),
                angle_increment=float((ANGLE_MAX - ANGLE_MIN) / N_FEIXES),
                time_increment=0.0, scan_time=0.1,
                range_min=float(RANGE_MIN), range_max=float(RANGE_MAX),
                ranges=ranges.astype(np.float32),
                intensities=np.zeros(N_FEIXES, dtype=np.float32),
            )
            writer.write(c_scan, stamp, ts.serialize_cdr(scan_msg, scan_msg.__msgtype__))

    print(f"Bag sintético escrito em: {out}")
    print(f"  modelo={modelo}  cena={cena}  frames={n_frames}  ruído={ruido}  "
          f"viés_lidar={vies_lidar}  resolução={W}x{H}")


def main():
    ap = argparse.ArgumentParser(description="Gera um rosbag .mcap sintético para testar a Frente 4.")
    ap.add_argument("--modelo", required=True,
                    choices=["zoedepth", "monodepth2", "depthanything"])
    ap.add_argument("--out", type=Path, required=True, help="diretório do bag de saída")
    ap.add_argument("--cena", choices=["plano", "parede_chao"], default="plano",
                    help="plano: um plano inclinado · parede_chao: parede + chão (testa ancoragem por altura)")
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--ruido", type=float, default=0.005, help="desvio do ruído gaussiano (m)")
    ap.add_argument("--vies-lidar", type=float, default=1.0,
                    help="erro sistemático de alcance do LiDAR (α): ranges são multiplicados por isto")
    ap.add_argument("--normalizar-ia", action="store_true",
                    help="DEBUG: normaliza y_ia em [0,1] por frame (simula o bug do depth_node)")
    ap.add_argument("--calibracao", type=Path,
                    default=Path(__file__).parent / "config" / "calibracao_exemplo.yaml")
    args = ap.parse_args()
    gerar(args.out, args.modelo, args.n_frames, args.ruido, args.calibracao,
          cena=args.cena, vies_lidar=args.vies_lidar, normalizar_ia=args.normalizar_ia)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    main()

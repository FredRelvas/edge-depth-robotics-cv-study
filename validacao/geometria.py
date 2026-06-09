"""
geometria.py — projeção do LiDAR 2D sobre o plano de imagem da câmera.

Convenções de frame (padrão ROS):
    LiDAR  : x para frente, y para a esquerda, z para cima. O scan vive no plano
             z=0; o feixe i aponta para o ângulo θ = angle_min + i·angle_increment,
             gerando o ponto (r·cosθ, r·sinθ, 0).
    Câmera : frame óptico — x para a direita, y para baixo, z para frente (o eixo
             óptico). A profundidade de um ponto é a sua coordenada Z.

Fluxo: LaserScan → pontos no frame do LiDAR → (extrínseca 4×4) → pontos no frame
da câmera → (projeção pinhole com K) → pixels (u, v) com profundidade Z.

O ground truth de profundidade num pixel é o **Z** do ponto no frame da câmera,
NÃO o alcance radial r medido pelo LiDAR.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def laserscan_para_pontos(
    ranges: np.ndarray,
    angle_min: float,
    angle_increment: float,
    range_min: float,
    range_max: float,
) -> np.ndarray:
    """
    Converte os alcances de um LaserScan em pontos 3D no frame do LiDAR.

    Descarta feixes inválidos (inf, nan, fora de [range_min, range_max]).

    Returns:
        Array (N, 3) de pontos (x, y, z=0) no frame do LiDAR, só com feixes válidos.
    """
    ranges = np.asarray(ranges, dtype=np.float64)
    n = ranges.shape[0]
    angles = angle_min + np.arange(n) * angle_increment

    valido = np.isfinite(ranges) & (ranges >= range_min) & (ranges <= range_max)
    r = ranges[valido]
    a = angles[valido]

    x = r * np.cos(a)
    y = r * np.sin(a)
    z = np.zeros_like(x)
    return np.stack([x, y, z], axis=1)


def transformar(pontos: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Aplica uma transformação homogênea 4×4 a um conjunto de pontos (N, 3).

    Args:
        pontos: (N, 3) no frame de origem.
        T:      (4, 4) que leva do frame de origem ao frame de destino
                (ex.: T_lidar_cam leva pontos do LiDAR para o frame da câmera).
    Returns:
        (N, 3) no frame de destino.
    """
    pontos = np.asarray(pontos, dtype=np.float64)
    if pontos.shape[0] == 0:
        return pontos.reshape(0, 3)
    hom = np.concatenate([pontos, np.ones((pontos.shape[0], 1))], axis=1)  # (N,4)
    out = hom @ np.asarray(T, dtype=np.float64).T
    return out[:, :3]


def projetar(
    pontos_cam: np.ndarray,
    K: np.ndarray,
    largura: int,
    altura: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Projeta pontos do frame da câmera nos pixels via modelo pinhole.

    u = fx·X/Z + cx ,  v = fy·Y/Z + cy ,  com profundidade = Z.
    Mantém apenas pontos à frente da câmera (Z > 0) e dentro da imagem.

    Args:
        pontos_cam: (N, 3) no frame óptico da câmera.
        K:          (3, 3) matriz intrínseca.
        largura:    largura da imagem de referência (px).
        altura:     altura da imagem de referência (px).
    Returns:
        (u, v, z): três arrays 1D de mesmo tamanho M ≤ N, com as coordenadas de
        pixel (float) e a profundidade Z (metros) dos pontos visíveis.
    """
    pontos_cam = np.asarray(pontos_cam, dtype=np.float64)
    if pontos_cam.shape[0] == 0:
        vazio = np.empty(0, dtype=np.float64)
        return vazio, vazio, vazio

    X, Y, Z = pontos_cam[:, 0], pontos_cam[:, 1], pontos_cam[:, 2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    frente = Z > 1e-6
    X, Y, Z = X[frente], Y[frente], Z[frente]

    u = fx * X / Z + cx
    v = fy * Y / Z + cy

    dentro = (u >= 0) & (u <= largura - 1) & (v >= 0) & (v <= altura - 1)
    return u[dentro], v[dentro], Z[dentro]


def amostrar(
    img: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    largura_ref: int,
    altura_ref: int,
) -> np.ndarray:
    """
    Amostra uma imagem de profundidade nos pixels (u, v) dados na resolução de
    referência (a da câmera RGB / camera_info).

    A imagem amostrada (ia_depth ou stereo/depth) pode ter resolução diferente da
    de referência; as coordenadas são escaladas proporcionalmente antes da
    amostragem por vizinho mais próximo.

    Args:
        img:         imagem 2D (H, W) a amostrar.
        u, v:        coordenadas de pixel na resolução de referência (float).
        largura_ref: largura da imagem de referência.
        altura_ref:  altura da imagem de referência.
    Returns:
        Array 1D com o valor de img em cada (u, v).
    """
    h, w = img.shape[:2]
    sx = w / largura_ref
    sy = h / altura_ref

    iu = np.clip(np.round(u * sx).astype(int), 0, w - 1)
    iv = np.clip(np.round(v * sy).astype(int), 0, h - 1)
    return img[iv, iu]


def matriz_intrinseca(k_flat) -> np.ndarray:
    """Constrói a matriz K (3×3) a partir do campo `k` do CameraInfo (9 valores, row-major)."""
    return np.asarray(k_flat, dtype=np.float64).reshape(3, 3)


# ---------------------------------------------------------------------------
# Ancoragem de escala pela altura da câmera ao chão (alternativa ao LiDAR)
# ---------------------------------------------------------------------------

def retroprojetar(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Retroprojeta um mapa de profundidade em uma nuvem de pontos 3D no frame da câmera.

    Para cada pixel (u, v) com profundidade Z válida (finita, > 0):
        X = (u - cx)/fx · Z ,  Y = (v - cy)/fy · Z ,  Z = Z.

    Args:
        depth: (H, W) profundidade por pixel.
        K:     (3, 3) intrínseca.
    Returns:
        (N, 3) pontos no frame da câmera; também devolve os índices (v, u) usados.
    """
    h, w = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth.astype(np.float64)
    valido = np.isfinite(Z) & (Z > 0)
    u = uu[valido]; v = vv[valido]; z = Z[valido]
    X = (u - cx) / fx * z
    Y = (v - cy) / fy * z
    return np.stack([X, Y, z], axis=1), (v, u)


def ransac_plano(pontos: np.ndarray, n_iter: int = 200, limiar: float = 0.02,
                 rng=None):
    """
    Ajusta o plano dominante de uma nuvem de pontos por RANSAC.

    Returns:
        (normal, c, inliers): plano definido por  normal·X = c  (|normal| = 1),
        e máscara booleana dos inliers. A distância da câmera (origem) ao plano é |c|.
    """
    rng = rng or np.random.default_rng(0)
    n = pontos.shape[0]
    if n < 3:
        raise ValueError("Pontos insuficientes para ajustar um plano.")

    melhor_inliers = None
    melhor_n = None
    melhor_c = None
    for _ in range(n_iter):
        idx = rng.choice(n, size=3, replace=False)
        p1, p2, p3 = pontos[idx]
        normal = np.cross(p2 - p1, p3 - p1)
        norma = np.linalg.norm(normal)
        if norma < 1e-9:
            continue
        normal = normal / norma
        c = normal @ p1
        dist = np.abs(pontos @ normal - c)
        inliers = dist < limiar
        if melhor_inliers is None or inliers.sum() > melhor_inliers.sum():
            melhor_inliers, melhor_n, melhor_c = inliers, normal, c

    # Refina com todos os inliers (plano por SVD: minimiza distância ortogonal).
    pts_in = pontos[melhor_inliers]
    centro = pts_in.mean(axis=0)
    _, _, vt = np.linalg.svd(pts_in - centro, full_matrices=False)
    normal = vt[-1]
    normal = normal / np.linalg.norm(normal)
    c = normal @ centro
    return normal, float(c), melhor_inliers


def escala_por_altura(
    depth: np.ndarray,
    K: np.ndarray,
    altura_real: float,
    apenas_inferior: bool = True,
    limiar_normal: float = 0.7,
    rng=None,
):
    """
    Estima o fator de escala de um mapa de profundidade a partir da altura conhecida
    da câmera ao chão — alternativa ao LiDAR, e independente dele.

    Ideia: para um modelo de escala (ex.: Monodepth2), a nuvem retroprojetada é a
    nuvem verdadeira multiplicada por uma constante `s`. O plano do chão fica a uma
    distância `|c| = s · altura_real` da câmera. Logo `s = |c| / altura_real`, e a
    profundidade métrica é `depth / s`. Para um modelo já métrico (ZoeDepth), `s ≈ 1`.

    Args:
        depth:           (H, W) profundidade predita (bruta para Monodepth2).
        K:               (3, 3) intrínseca.
        altura_real:     altura da câmera ao chão em metros (ex.: 0.395).
        apenas_inferior: usa só a metade inferior da imagem (onde o chão aparece).
        limiar_normal:   exige |normal·eixo_y| ≥ isto (chão ≈ horizontal no frame óptico).
    Returns:
        (s, info): fator de escala e dict com normal, distância e nº de inliers.
    """
    pontos, (v, _u) = retroprojetar(depth, K)
    if apenas_inferior:
        cy = K[1, 2]
        mask = v > cy
        pontos = pontos[mask]
    if pontos.shape[0] < 3:
        raise ValueError("Sem pontos suficientes na região do chão para estimar a escala.")

    normal, c, inliers = ransac_plano(pontos, rng=rng)
    # No frame óptico, o chão é ~horizontal → normal aponta no eixo y (para baixo/cima).
    if abs(normal[1]) < limiar_normal:
        raise ValueError(
            f"Plano dominante não parece o chão (|n_y|={abs(normal[1]):.2f} < {limiar_normal}). "
            "O chão pode não estar visível o suficiente."
        )
    distancia = abs(c)
    s = distancia / altura_real
    return s, {"normal": normal, "distancia": distancia,
               "n_inliers": int(inliers.sum()), "n_pontos": int(pontos.shape[0])}


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # K simples 640x480, fx=fy=500, centro no meio.
    K = np.array([[500.0, 0.0, 320.0],
                  [0.0, 500.0, 240.0],
                  [0.0, 0.0, 1.0]])

    # Teste 1: projeção de pontos conhecidos no frame da câmera.
    # Ponto no eixo óptico a 2 m → centro da imagem, Z=2.
    pts = np.array([
        [0.0, 0.0, 2.0],     # centro
        [1.0, 0.0, 5.0],     # à direita: u = 500*1/5 + 320 = 420
        [0.0, -1.0, 4.0],    # acima: v = 500*(-1)/4 + 240 = 115
        [0.0, 0.0, -1.0],    # atrás da câmera → descartado
        [100.0, 0.0, 1.0],   # muito à direita → fora da imagem, descartado
    ])
    u, v, z = projetar(pts, K, 640, 480)
    print(f"[teste 1] {len(u)} pontos visíveis (esperado 3)")
    assert len(u) == 3
    assert abs(u[0] - 320) < 1e-6 and abs(v[0] - 240) < 1e-6 and abs(z[0] - 2.0) < 1e-6
    assert abs(u[1] - 420) < 1e-6 and abs(z[1] - 5.0) < 1e-6
    assert abs(v[2] - 115) < 1e-6

    # Teste 2: transformar com identidade não altera pontos.
    assert np.allclose(transformar(pts[:3], np.eye(4)), pts[:3])

    # Teste 3: round-trip LaserScan → pontos → (identidade) → projeção.
    # Um feixe reto para frente do LiDAR vira, com a rotação lidar→câmera, um Z>0.
    # Rotação: lidar x(frente)→cam z, y(esq)→cam -x, z(cima)→cam -y.
    R = np.array([[0, -1, 0],
                  [0, 0, -1],
                  [1, 0, 0]], dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = R
    ranges = np.array([np.inf, 3.0, 0.0, 2.0])  # 2 válidos (3.0 e 2.0)
    pontos_l = laserscan_para_pontos(ranges, angle_min=0.0, angle_increment=0.1,
                                     range_min=0.1, range_max=10.0)
    print(f"[teste 3] feixes válidos: {len(pontos_l)} (esperado 2)")
    assert len(pontos_l) == 2
    pontos_c = transformar(pontos_l, T)
    # feixe θ=0.1 (índice 1): ponto lidar (3cos0.1, 3sin0.1, 0) → cam: Z = x_lidar = 3cos0.1
    assert abs(pontos_c[0, 2] - 3.0 * np.cos(0.1)) < 1e-6
    assert np.all(pontos_c[:, 2] > 0)  # à frente da câmera

    # Teste 4: amostragem com escala de resolução.
    img = np.arange(100, dtype=np.float32).reshape(10, 10)  # 10x10
    # referência 100x100; pixel (50,30) → escala 0.1 → img[3,5] = 35
    val = amostrar(img, np.array([50.0]), np.array([30.0]), 100, 100)
    assert val[0] == img[3, 5]
    print("[teste 4] amostragem com escala: ok")

    # Teste 5: escala pela altura do chão.
    # Chão a h=0.395 m abaixo da câmera (câmera olhando na horizontal); só aparece
    # na metade inferior (v > cy). depth = h·fy/(v - cy).
    h_real = 0.395
    H, W = 480, 640          # consistente com o K do teste 1 (cx=320, cy=240)
    fy, cy = K[1, 1], K[1, 2]  # 500, 240
    depth = np.zeros((H, W), dtype=np.float64)
    vs = np.arange(H)
    inferior = vs > cy
    depth[inferior, :] = (h_real * fy / (vs[inferior] - cy))[:, None]
    # modelo já métrico → s ≈ 1
    s, info = escala_por_altura(depth, K, h_real)
    print(f"[teste 5] escala (métrico): s={s:.4f}  (esperado ~1.0)  inliers={info['n_inliers']}")
    assert abs(s - 1.0) < 1e-3
    # modelo com escala oculta 2.7 → s ≈ 2.7, e depth/s recupera o métrico
    s2, _ = escala_por_altura(depth * 2.7, K, h_real)
    print(f"[teste 5] escala (× 2.7 oculto): s={s2:.4f}  (esperado ~2.7)")
    assert abs(s2 - 2.7) < 1e-2

    print("\nTodos os testes de geometria passaram. ✓")

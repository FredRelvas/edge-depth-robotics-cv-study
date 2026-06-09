"""
avaliar.py — CLI da validação offline (Frente 4).

Lê um rosbag de um run, projeta o LiDAR, monta a tabela y_lidar/y_oak/y_ia e
imprime as métricas comparando o modelo com o baseline (OAK-D) e o ground truth
(LiDAR). Opcionalmente salva o resultado em JSON.

Exemplo:
    uv run python validacao/avaliar.py \\
        --bag /tmp/sim_zoe --modelo zoedepth \\
        --calibracao validacao/config/calibracao_exemplo.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Permite importar os módulos irmãos quando rodado como script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from calibracao import carregar_calibracao
from leitor_rosbag import ler_bag
from avaliacao import avaliar_bag, formatar_relatorio


def main():
    ap = argparse.ArgumentParser(description="Validação offline de profundidade (Frente 4).")
    ap.add_argument("--bag", type=Path, required=True, help="diretório/arquivo do rosbag2")
    ap.add_argument("--modelo", required=True,
                    choices=["zoedepth", "monodepth2", "depthanything"])
    ap.add_argument("--calibracao", type=Path,
                    default=Path(__file__).parent / "config" / "calibracao_exemplo.yaml")
    ap.add_argument("--tolerancia-ms", type=float, default=50.0,
                    help="janela de sincronização entre tópicos (ms)")
    ap.add_argument("--fonte-escala", choices=["lidar", "altura"], default="lidar",
                    help="lidar: escala da IA ajustada pelo LiDAR · "
                         "altura: escala pela altura da câmera ao chão (independente do LiDAR)")
    ap.add_argument("--altura-camera", type=float, default=None,
                    help="altura da câmera ao chão em metros (obrigatório com --fonte-escala altura)")
    ap.add_argument("--saida", type=Path, default=None, help="salva o resultado em JSON")
    args = ap.parse_args()

    calib = carregar_calibracao(args.calibracao)
    amostras = ler_bag(args.bag, tolerancia_ms=args.tolerancia_ms)
    if not amostras:
        print("Nenhuma amostra sincronizada encontrada no bag — verifique tópicos e tolerância.")
        sys.exit(1)

    resultado = avaliar_bag(amostras, calib.T_lidar_cam, args.modelo,
                            fonte_escala=args.fonte_escala,
                            altura_camera=args.altura_camera)
    print(formatar_relatorio(resultado))

    if args.saida:
        args.saida.parent.mkdir(parents=True, exist_ok=True)
        with open(args.saida, "w") as f:
            json.dump(resultado, f, indent=2, ensure_ascii=False)
        print(f"\nResultado salvo em: {args.saida}")


if __name__ == "__main__":
    main()

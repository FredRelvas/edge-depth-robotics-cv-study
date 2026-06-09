#!/usr/bin/env bash
# =============================================================================
# FASE 2 (Jetson Orin Nano): constrói todos os engines a partir dos ONNX.
#
#   - FP16 para TODOS os ONNX (precisão padrão do benchmark).
#   - INT8 só para Monodepth2 e DAV2 na maior resolução (ZoeDepth/BEiT degrada
#     sem QAT, então não geramos INT8 dele).
#
# Rode da pasta quantizacao/ na Jetson:
#     cd quantizacao && bash build_trt/construir_todos.sh
#
# Pré-requisitos: TensorRT (vem com o JetPack) + pip install pycuda numpy opencv-python
# e ter rodado preparar_calibracao.py (no desktop) e copiado calibracao/.
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."          # -> quantizacao/
mkdir -p engines
WS=2048

echo "==================== FP16 (todos) ===================="
shopt -s nullglob
for onnx_file in onnx/*.onnx; do
    base=$(basename "$onnx_file" .onnx)
    python build_trt/build_engine.py \
        --onnx "$onnx_file" \
        --output "engines/${base}_fp16.engine" \
        --precision fp16 --workspace_mb $WS \
        || echo "[construir_todos][warn] falha em ${base} (FP16); seguindo."
done

echo "==================== INT8 (Monodepth2 + DAV2, maior resolução) ===="
if [ ! -d calibracao ] || [ -z "$(ls -A calibracao/*.png 2>/dev/null)" ]; then
    echo "[construir_todos][warn] calibracao/ vazia — pulando INT8."
    echo "  Gere no desktop: python export/preparar_calibracao.py --n 500"
    exit 0
fi

# Escolhe automaticamente o ONNX de maior resolução de cada um, se existir.
for prefix in monodepth2 dav2_vits; do
    onnx_file=$(ls -1 onnx/${prefix}_*.onnx 2>/dev/null | sort | tail -n1 || true)
    [ -z "$onnx_file" ] && continue
    base=$(basename "$onnx_file" .onnx)
    python build_trt/build_engine.py \
        --onnx "$onnx_file" \
        --output "engines/${base}_int8.engine" \
        --precision int8 \
        --calib_dir calibracao/ \
        --calib_cache "engines/calib_${prefix}.cache" \
        || echo "[construir_todos][warn] falha em ${base} (INT8); seguindo."
done

echo
echo "[ok] engines gerados:"
ls -lh engines/*.engine 2>/dev/null || true

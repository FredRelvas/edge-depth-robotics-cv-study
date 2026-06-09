#!/usr/bin/env bash
# ============================================================================
# Depth Anything V2 (ViT-S) — Frozen: DPT head treinada, encoder DINOv2 congelado.
#
# Experimento adicional ao paper RITA 2025: avalia o estado da arte atual
# (Yang et al., NeurIPS 2024) sobre o ICL Ground Robot.
#
# Setup: ~280 MB de download para o checkpoint pré-treinado (Hypersim, indoor).
# Tempo de treino: ~20-30 min em RTX 4090 (encoder ViT-S é leve).
#
# Uso:
#     bash scripts/run_dav2_frozen.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXPERIMENT_NAME="dav2_vits_frozen"
RUN_DIR="runs/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}"

# ----- Clonar Depth-Anything-V2 -----
if [ ! -d "external/DAV2" ]; then
    echo "[setup] Clonando DepthAnything/Depth-Anything-V2 em external/DAV2..."
    mkdir -p external
    git clone --depth 1 https://github.com/DepthAnything/Depth-Anything-V2.git external/DAV2
fi

# ----- Baixar checkpoint pré-treinado (Hypersim, indoor, ViT-S) -----
CKPT_DIR="external/DAV2/checkpoints"
CKPT_FILE="${CKPT_DIR}/depth_anything_v2_metric_hypersim_vits.pth"
if [ ! -f "${CKPT_FILE}" ]; then
    echo "[setup] Baixando checkpoint pré-treinado (Hypersim ViT-S, ~98MB)..."
    mkdir -p "${CKPT_DIR}"
    curl -L -o "${CKPT_FILE}" \
        "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth"
fi

if [ ! -d "dados/icl_ground_robot/deer/frames" ]; then
    echo "[erro] Dataset não encontrado. Rode primeiro:"
    echo "       uv run python codigo-treinamento/baixar_dados.py"
    exit 1
fi

# ----- Hiperparâmetros (alinhados com o ZoeDepth Frozen-32 para comparação justa) -----
EPOCHS=20
BATCH_SIZE=8
LR=1e-4
ENCODER="vits"
IMAGE_SIZE=518

echo "============================================================"
echo " Experimento: ${EXPERIMENT_NAME}"
echo " Run dir:     ${RUN_DIR}"
echo " Encoder:     ${ENCODER} (~25M params)"
echo " Modo:        frozen (encoder DINOv2 congelado, treina só a DPT head)"
echo " Épocas:      ${EPOCHS}"
echo " Batch size:  ${BATCH_SIZE}"
echo " LR (max):    ${LR}"
echo " Image size:  ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "============================================================"

uv run python codigo-treinamento/treinar_depth_anything_v2.py \
    --mode frozen \
    --encoder "${ENCODER}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --image_size "${IMAGE_SIZE}" \
    --run_dir "${RUN_DIR}" \
    2>&1 | tee "${RUN_DIR}/treino.log"

echo "[done] ${EXPERIMENT_NAME} finalizado. Resultados em ${RUN_DIR}/"
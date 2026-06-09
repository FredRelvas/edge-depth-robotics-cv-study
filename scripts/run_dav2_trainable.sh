#!/usr/bin/env bash
# ============================================================================
# Depth Anything V2 (ViT-S) — Trainable: fine-tune completo (encoder + head).
#
# Aplicamos o early stopping de 10 épocas (vs ZoeDepth de 40 épocas reportadas
# pelo paper), pelo mesmo motivo discutido no relatório do ZoeDepth Trainable-5:
# o protocolo de split sequencial 70/10/20 do paper RITA 2025 favorece overfit
# por proximidade temporal entre validation e training.
#
# Como o ViT-S é menor que o DPT-BEiT-L do ZoeDepth (24.8M vs 345M), o risco
# de overfit é menor — ainda assim, mantemos a configuração conservadora.
#
# Tempo de treino: ~25-35 min em RTX 4090.
#
# Uso:
#     bash scripts/run_dav2_trainable.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXPERIMENT_NAME="dav2_vits_trainable"
RUN_DIR="runs/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}"

if [ ! -d "external/DAV2" ]; then
    echo "[setup] Clonando DepthAnything/Depth-Anything-V2 em external/DAV2..."
    mkdir -p external
    git clone --depth 1 https://github.com/DepthAnything/Depth-Anything-V2.git external/DAV2
fi

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

# ----- Hiperparâmetros -----
# - LR menor que Frozen (5e-6 vs 1e-4) porque fine-tune completo precisa lr baixo
#   para não destruir as features pré-treinadas no DINOv2.
# - Batch efetivo 8 = 4 físico × 2 accum (ViT-S em 518x518 não cabe batch=8 cheio).
# - bf16 ativado pra economizar VRAM e acelerar.
EPOCHS=10
BATCH_SIZE=4
GRAD_ACCUM_STEPS=2
LR=5e-6
ENCODER="vits"
IMAGE_SIZE=518

echo "============================================================"
echo " Experimento: ${EXPERIMENT_NAME}"
echo " Run dir:     ${RUN_DIR}"
echo " Encoder:     ${ENCODER} (~25M params, fine-tune completo)"
echo " Modo:        trainable (treina encoder + head)"
echo " Épocas:      ${EPOCHS}  (early stopping vs paper, evita overfit temporal)"
echo " Batch size:  ${BATCH_SIZE} físico x ${GRAD_ACCUM_STEPS} accum = $((BATCH_SIZE * GRAD_ACCUM_STEPS)) efetivo"
echo " LR (max):    ${LR}  (10x menor que Frozen, padrão fine-tune)"
echo " AMP:         bf16"
echo " Image size:  ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "============================================================"

uv run python codigo-treinamento/treinar_depth_anything_v2.py \
    --mode trainable \
    --encoder "${ENCODER}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --grad_accum_steps "${GRAD_ACCUM_STEPS}" \
    --amp \
    --lr "${LR}" \
    --image_size "${IMAGE_SIZE}" \
    --run_dir "${RUN_DIR}" \
    2>&1 | tee "${RUN_DIR}/treino.log"

echo "[done] ${EXPERIMENT_NAME} finalizado. Resultados em ${RUN_DIR}/"
#!/usr/bin/env bash
# ============================================================================
# Monodepth2 — fine-tune do modelo mono_640x192 (pré-treinado no KITTI).
#
# - ResNet-18 encoder + DepthDecoder Monodepth2.
# - Fine-tune supervisionado com SILog loss sobre o GT do ICL.
# - 20 épocas, Adam + StepLR (step=15, gamma=0.1), lr_inicial=1e-4.
# - Avaliação com median_align=True (modelo prediz disp/depth até uma escala).
#
# Uso:
#     bash scripts/run_monodepth2.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

EXPERIMENT_NAME="monodepth2_finetune"
RUN_DIR="runs/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S)"

# ----- Clonar Monodepth2 e baixar pesos pré-treinados -----
if [ ! -d "external/monodepth2" ]; then
    echo "[setup] Clonando nianticlabs/monodepth2 em external/monodepth2..."
    mkdir -p external
    git clone --depth 1 https://github.com/nianticlabs/monodepth2.git external/monodepth2
fi

MD2_WEIGHTS_DIR="external/monodepth2/models/mono_640x192"
if [ ! -f "${MD2_WEIGHTS_DIR}/encoder.pth" ]; then
    echo "[setup] Baixando pesos pré-treinados mono_640x192..."
    mkdir -p "${MD2_WEIGHTS_DIR}"
    curl -L -o "/tmp/mono_640x192.zip" \
        "https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip"
    unzip -o /tmp/mono_640x192.zip -d "${MD2_WEIGHTS_DIR}/"
    rm /tmp/mono_640x192.zip
fi

if [ ! -d "dados/icl_ground_robot/deer/frames" ]; then
    echo "[erro] Dataset não encontrado. Rode primeiro:"
    echo "       uv run python codigo-treinamento/baixar_dados.py"
    exit 1
fi

# ----- Hiperparâmetros (conforme paper) -----
EPOCHS=20
BATCH_SIZE=8
LR=1e-4
LR_STEP_SIZE=15
LR_GAMMA=0.1

echo "============================================================"
echo " Experimento: ${EXPERIMENT_NAME}"
echo " Run dir:     ${RUN_DIR}"
echo " Épocas:      ${EPOCHS}"
echo " Batch size:  ${BATCH_SIZE}"
echo " LR inicial:  ${LR}, step a cada ${LR_STEP_SIZE} épocas (γ=${LR_GAMMA})"
echo "============================================================"

mkdir -p "${RUN_DIR}"
uv run python codigo-treinamento/treinar_monodepth2.py \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --lr_step_size "${LR_STEP_SIZE}" \
    --lr_gamma "${LR_GAMMA}" \
    --run_dir "${RUN_DIR}" \
    --pretrained_dir "${MD2_WEIGHTS_DIR}" \
    2>&1 | tee -a "${RUN_DIR}/treino.log"

echo "[done] ${EXPERIMENT_NAME} finalizado. Resultados em ${RUN_DIR}/"
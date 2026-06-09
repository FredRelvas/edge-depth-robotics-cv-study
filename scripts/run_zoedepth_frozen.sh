#!/usr/bin/env bash
# ============================================================================
# ZoeDepth Frozen-32 — replicação do experimento de Vizzotto et al. (RITA 2025).
#
# - Backbone MiDaS (DPT_BEiT_L_384) CONGELADO.
# - Treina apenas a metric head com 32 bins.
# - 20 épocas, AdamW + OneCycleLR, lr_max=1e-4.
# - Horizontal flip como augmentation (prob=0.5, já no dataloader).
#
# Uso (da raiz do projeto):
#     bash scripts/run_zoedepth_frozen.sh
# ============================================================================
set -euo pipefail

# Resolve a raiz do projeto independentemente de onde o script foi chamado.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

EXPERIMENT_NAME="zoedepth_frozen32"
RUN_DIR="runs/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}"

# ----- Clonar ZoeDepth se ainda não existir -----
if [ ! -d "external/ZoeDepth" ]; then
    echo "[setup] Clonando isl-org/ZoeDepth em external/ZoeDepth..."
    mkdir -p external
    git clone --depth 1 https://github.com/isl-org/ZoeDepth.git external/ZoeDepth
fi

# Garante que timm está na versão compatível com o MiDaS usado pelo ZoeDepth.
# Versões novas do timm removeram o buffer 'relative_position_index' do BeitAttention,
# o que quebra o load do dpt_beit_large_384.pt. A 0.6.12 é a fixada no README do ZoeDepth.
uv pip install --quiet "timm==0.6.12"

# ----- Verificar dataset -----
if [ ! -d "dados/icl_ground_robot/deer/frames" ]; then
    echo "[erro] Dataset não encontrado. Rode primeiro:"
    echo "       uv run python codigo-treinamento/baixar_dados.py"
    exit 1
fi

# ----- Hiperparâmetros (conforme paper) -----
EPOCHS=20
BATCH_SIZE=8
LR=1e-4
N_BINS=32
IMAGE_SIZE=384

echo "============================================================"
echo " Experimento: ${EXPERIMENT_NAME}"
echo " Run dir:     ${RUN_DIR}"
echo " Épocas:      ${EPOCHS}"
echo " Batch size:  ${BATCH_SIZE}"
echo " LR (max):    ${LR}"
echo " N bins:      ${N_BINS}"
echo " Image size:  ${IMAGE_SIZE}"
echo "============================================================"

uv run python codigo-treinamento/treinar_zoedepth.py \
    --mode frozen \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --n_bins "${N_BINS}" \
    --image_size "${IMAGE_SIZE}" \
    --run_dir "${RUN_DIR}" \
    2>&1 | tee "${RUN_DIR}/treino.log"

echo "[done] ${EXPERIMENT_NAME} finalizado. Resultados em ${RUN_DIR}/"
# ----- Pós-treino: versão slim do best.pth (sem optimizer/scheduler) -----
BEST="${RUN_DIR}/checkpoints/best.pth"
[ -f "${BEST}" ] && uv run python scripts/slim_checkpoint.py "${BEST}"

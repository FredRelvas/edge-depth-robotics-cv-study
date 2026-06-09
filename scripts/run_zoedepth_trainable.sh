#!/usr/bin/env bash
# ============================================================================
# ZoeDepth Trainable-5 — melhor modelo do paper Vizzotto et al. (RITA 2025).
#
# - Fine-tune COMPLETO: backbone MiDaS + metric head.
# - 32 bins, 40 épocas, AdamW + OneCycleLR, lr_max=1e-5.
# - Resultado esperado: RMSE log ~0.144 (melhor do paper).
#
# Atenção: este experimento é mais pesado (treina ~340M de parâmetros).
# Estimativa: ~2-3h em uma RTX 4090 com batch_size=8.
#
# Uso:
#     bash scripts/run_zoedepth_trainable.sh
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

EXPERIMENT_NAME="zoedepth_trainable5"
RUN_DIR="runs/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RUN_DIR}"

if [ ! -d "external/ZoeDepth" ]; then
    echo "[setup] Clonando isl-org/ZoeDepth em external/ZoeDepth..."
    mkdir -p external
    git clone --depth 1 https://github.com/isl-org/ZoeDepth.git external/ZoeDepth
fi

# Versão compatível do timm (a atual quebra ao carregar dpt_beit_large_384.pt).
uv pip install --quiet "timm==0.6.12"

if [ ! -d "dados/icl_ground_robot/deer/frames" ]; then
    echo "[erro] Dataset não encontrado. Rode primeiro:"
    echo "       uv run python codigo-treinamento/baixar_dados.py"
    exit 1
fi

# ----- Hiperparâmetros (Trainable-5 + regularização forte contra overfit) -----
# Variação sobre o paper: usamos weight_decay 10x maior (1e-1 vs 1e-2 padrão) para
# combater o overfit observado em fine-tunes longos sobre 2240 frames. Justifica-se
# porque o protocolo de split sequencial do paper (70/10/20 por cena) deixa o val
# temporalmente próximo do train, facilitando overfit que só aparece no test.
# Batch efetivo = BATCH_SIZE * GRAD_ACCUM_STEPS = 2 * 4 = 8 (mesmo do paper).
EPOCHS=10
BATCH_SIZE=2
GRAD_ACCUM_STEPS=4
LR=1e-5
WEIGHT_DECAY=1e-2
N_BINS=32
IMAGE_SIZE=384

echo "============================================================"
echo " Experimento: ${EXPERIMENT_NAME}"
echo " Run dir:     ${RUN_DIR}"
echo " Épocas:      ${EPOCHS}"
echo " Batch size:  ${BATCH_SIZE} físico x ${GRAD_ACCUM_STEPS} accum = $((BATCH_SIZE * GRAD_ACCUM_STEPS)) efetivo"
echo " LR (max):    ${LR}"
echo " Weight decay:${WEIGHT_DECAY}  (10x mais forte — combate overfit)"
echo " N bins:      ${N_BINS}"
echo " AMP:         bf16 (mixed precision)"
echo "============================================================"

mkdir -p "${RUN_DIR}"
uv run python codigo-treinamento/treinar_zoedepth.py \
    --mode trainable \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --grad_accum_steps "${GRAD_ACCUM_STEPS}" \
    --amp \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --n_bins "${N_BINS}" \
    --image_size "${IMAGE_SIZE}" \
    --run_dir "${RUN_DIR}" \
    2>&1 | tee "${RUN_DIR}/treino.log"

echo "[done] ${EXPERIMENT_NAME} finalizado. Resultados em ${RUN_DIR}/"
# ----- Pós-treino: versão slim do best.pth (sem optimizer/scheduler) -----
BEST="${RUN_DIR}/checkpoints/best.pth"
[ -f "${BEST}" ] && uv run python scripts/slim_checkpoint.py "${BEST}"

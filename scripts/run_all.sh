#!/usr/bin/env bash
# ============================================================================
# Master — roda os 3 experimentos do paper Vizzotto et al. (RITA 2025) em
# sequência: ZoeDepth Frozen, ZoeDepth Trainable, Monodepth2.
#
# Estimativa total de tempo em uma RTX 4090: ~4-5h
#   - ZoeDepth Frozen-32:     ~30-45min  (20 épocas, só metric head)
#   - ZoeDepth Trainable-5:   ~2-3h      (40 épocas, fine-tune completo)
#   - Monodepth2 fine-tune:   ~30-45min  (20 épocas, ResNet-18 leve)
#
# Uso:
#     bash scripts/run_all.sh                  # roda todos
#     bash scripts/run_all.sh --skip-trainable # pula o mais demorado
#
# Comportamento em caso de falha: registra no resumo e continua para o próximo
# experimento. Assim uma noite de treino não é perdida por um único erro.
# ============================================================================
set -uo pipefail   # nota: sem '-e' para permitir continuar após falhas

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Argumentos opcionais
SKIP_FROZEN=false
SKIP_TRAINABLE=false
SKIP_MONODEPTH2=false
for arg in "$@"; do
    case "${arg}" in
        --skip-frozen)     SKIP_FROZEN=true ;;
        --skip-trainable)  SKIP_TRAINABLE=true ;;
        --skip-monodepth2) SKIP_MONODEPTH2=true ;;
        -h|--help)
            grep '^#' "$0" | head -n 25
            exit 0
            ;;
        *) echo "[warn] argumento desconhecido: ${arg}" ;;
    esac
done

SUMMARY_FILE="runs/_resumo_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p runs
echo "Resumo da execução dos experimentos — $(date)" > "${SUMMARY_FILE}"
echo "========================================================" >> "${SUMMARY_FILE}"

run_one() {
    local name="$1"
    local script="$2"
    local t0=$(date +%s)

    echo ""
    echo "########################################################"
    echo "# Iniciando: ${name}"
    echo "########################################################"

    if bash "${script}"; then
        local dt=$(( $(date +%s) - t0 ))
        local mm=$(( dt / 60 ))
        local ss=$(( dt % 60 ))
        echo "[${name}] OK em ${mm}m${ss}s" | tee -a "${SUMMARY_FILE}"
    else
        echo "[${name}] FALHOU — veja o log em runs/" | tee -a "${SUMMARY_FILE}"
    fi
}

# ----- Executa os experimentos -----
if [ "${SKIP_FROZEN}" = false ]; then
    run_one "ZoeDepth Frozen-32" "scripts/run_zoedepth_frozen.sh"
fi

if [ "${SKIP_TRAINABLE}" = false ]; then
    run_one "ZoeDepth Trainable-5" "scripts/run_zoedepth_trainable.sh"
fi

if [ "${SKIP_MONODEPTH2}" = false ]; then
    run_one "Monodepth2" "scripts/run_monodepth2.sh"
fi

echo ""
echo "========================================================"
echo " Resumo final"
echo "========================================================"
cat "${SUMMARY_FILE}"
echo ""
echo "Use 'tensorboard --logdir runs/' para visualizar as métricas."
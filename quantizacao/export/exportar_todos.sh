#!/usr/bin/env bash
# =============================================================================
# FASE 1 (desktop / RTX 4090): exporta os 3 modelos para ONNX em várias
# resoluções, gerando o estudo de Pareto acurácia × latência.
#
# Rode da pasta quantizacao/:
#     cd quantizacao && bash export/exportar_todos.sh
#
# Pré-requisitos (env de export):
#     pip install onnx onnxsim onnxruntime
# e ter os checkpoints em ../modelos_treinados/ e os repos em ../external/.
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."          # -> quantizacao/
mkdir -p onnx
SIMP="--simplify"

echo "==================== Monodepth2 (mult. de 32) ===================="
for res in "192 640" "192 256" "256 320"; do
    H=${res% *}; W=${res#* }
    python export/export_monodepth2.py \
        --height "$H" --width "$W" \
        --output "onnx/monodepth2_${H}x${W}.onnx" $SIMP
done

echo "==================== DAV2 ViT-S (mult. de 14) ===================="
for res in "252 364" "308 420" "364 518"; do
    H=${res% *}; W=${res#* }
    python export/export_dav2.py \
        --encoder vits --max_depth 20 \
        --height "$H" --width "$W" \
        --output "onnx/dav2_vits_${H}x${W}.onnx" $SIMP
done

echo "==================== ZoeDepth (mult. de 32, caso difícil) ========"
# Resolução nativa de treino primeiro (a mais provável de exportar OK).
for res in "384 384" "384 512"; do
    H=${res% *}; W=${res#* }
    python export/export_zoedepth.py \
        --height "$H" --width "$W" \
        --output "onnx/zoedepth_trainable_${H}x${W}.onnx" $SIMP \
        || echo "[exportar_todos][warn] ZoeDepth ${H}x${W} falhou (esperado p/ BEiT); seguindo."
done

echo
echo "[ok] ONNX gerados em onnx/:"
ls -lh onnx/*.onnx 2>/dev/null || true
echo
echo "Próximo passo: transferir para a Jetson ->"
echo "    scp -r onnx/ ceia-jetson@<jetson-ip>:~/quantizacao/"

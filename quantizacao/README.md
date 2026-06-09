# Pipeline de Quantização → TensorRT (Jetson Orin Nano)

Exporta os modelos treinados neste projeto (**Monodepth2**, **ZoeDepth-N** e
**Depth Anything V2 ViT-S**) para ONNX no desktop e os compila em engines
TensorRT FP16/INT8 na Jetson Orin Nano, para o estudo de latência × acurácia.

> **Este pipeline carrega os checkpoints REAIS do projeto** (em
> `../modelos_treinados/`), treinados no **ICL Ground Robot**. Ele difere de
> um template genérico de ZoeDepth/DAV2: aqui a variante do ZoeDepth é a **N
> com `n_bins=32`**, o Monodepth2 é **fine-tunado** (prediz até escala → exige
> *median-align* na avaliação) e o DAV2 usa **`max_depth=20`**.

## Contrato de entrada (importante)

Os três modelos foram treinados com **RGB cru em `[0,1]`, NCHW, sem
normalização ImageNet externa** (o dataloader só faz `img/255`). Cada modelo
aplica internamente sua própria normalização:

| Modelo     | Normalização | Onde |
|------------|--------------|------|
| Monodepth2 | `(x-0.45)/0.225` | dentro do `ResnetEncoder` |
| ZoeDepth   | `mean/std = 0.5` + resize | dentro do `MidasCore.prep` |
| DAV2       | **nenhuma** (treinado com `[0,1]` cru) | — |

Por isso **todos os ONNX recebem `[0,1]` NCHW** e qualquer pré-processamento
interno fica embutido no grafo. O `validate.py` e o calibrador INT8 alimentam
exatamente `[0,1]` (img/255 + resize) — nada de mean/std ImageNet.

## Estrutura

```
quantizacao/
├── export/                       # FASE 1 — desktop (RTX 4090)
│   ├── _common.py                # helpers (paths, export+simplify, sanity)
│   ├── export_monodepth2.py
│   ├── export_zoedepth.py
│   ├── export_dav2.py
│   ├── preparar_calibracao.py    # separa imagens do ICL p/ INT8
│   └── exportar_todos.sh         # orquestra os 3 em várias resoluções
├── build_trt/                    # FASE 2 — Jetson Orin Nano
│   ├── build_engine.py           # FP16/INT8 (TensorRT 10.x)
│   └── construir_todos.sh
├── utils/
│   ├── validate.py               # sanity ONNX vs engine
│   ├── benchmark.py              # latência p50/p95 + FPS
│   ├── avaliar_engine.py         # 7 métricas no split de teste do ICL
│   └── gerar_pareto.py           # curva acurácia × latência + CSV
├── onnx/        # (gitignored) gerado pelo export
├── engines/     # (gitignored) gerado pelo build
└── calibracao/  # (gitignored) imagens p/ INT8
```

## Resoluções (restrição de grade por modelo)

| Modelo       | Múltiplo | Resoluções sugeridas (HxW) |
|--------------|----------|----------------------------|
| Monodepth2   | 32       | 192x640 (nativa), 192x256, 256x320 |
| ZoeDepth     | 32       | 384x384 (nativa), 384x512 |
| DAV2 (ViT-S) | 14       | 252x364, 308x420, 364x518 |

> Monodepth2 prediz até escala: ao avaliar fora de 192x640, a escala muda —
> use *median-align*. ZoeDepth tem resize interno (~384) no `prep`, então
> resoluções maiores não fazem o backbone rodar maior; comece pela nativa.

## FASE 1 — Export ONNX (desktop)

```bash
# Env de export (separado do .venv do treino)
pip install onnx onnxsim onnxruntime opencv-python numpy
# (torch/torchvision já presentes no ambiente do projeto)

cd quantizacao
bash export/exportar_todos.sh          # gera onnx/*.onnx
python export/preparar_calibracao.py --n 500   # gera calibracao/ (p/ INT8)

# Transfere para a Jetson
scp -r onnx/ calibracao/ ceia-jetson@<jetson-ip>:~/quantizacao/
```

Export individual (ex.: ZoeDepth na resolução nativa):

```bash
python export/export_zoedepth.py --height 384 --width 384 \
    --output onnx/zoedepth_trainable_384x384.onnx --simplify
```

## FASE 2 — Build de engines (Jetson Orin Nano, JetPack 6.2 / TRT 10.x)

```bash
pip install pycuda numpy opencv-python    # TensorRT já vem no JetPack

cd quantizacao
bash build_trt/construir_todos.sh         # FP16 p/ todos + INT8 p/ MD2 e DAV2
```

Build individual:

```bash
python build_trt/build_engine.py \
    --onnx onnx/dav2_vits_364x518.onnx \
    --output engines/dav2_vits_364x518_fp16.engine \
    --precision fp16 --workspace_mb 2048
```

Alternativa rápida com `trtexec` (CLI do TRT):

```bash
trtexec --onnx=onnx/dav2_vits_364x518.onnx --fp16 \
        --saveEngine=engines/dav2_fp16.engine \
        --memPoolSize=workspace:2048 --useCudaGraph
```

## Sanity check e benchmark

```bash
# Confirma que o engine bate com o ONNX (FP16 ~1% de diferença)
python utils/validate.py --engine engines/dav2_vits_364x518_fp16.engine \
    --image amostra.png --height 364 --width 518

# Latência / FPS (mede energia com `tegrastats` em paralelo)
python utils/benchmark.py --engine engines/dav2_vits_364x518_fp16.engine \
    --runs 200 --json_out engines/dav2_364x518_fp16.bench.json
```

## Avaliação das métricas + curva de Pareto

Depois de gerar os engines, meça acurácia e latência e cruze tudo:

```bash
# 1) Métricas (7 do paper) no split de teste do ICL — por engine
python utils/avaliar_engine.py \
    --engine engines/dav2_vits_364x518_fp16.engine \
    --json_out engines/dav2_vits_364x518_fp16.eval.json
# (median-align liga sozinho p/ monodepth2; force com --median_align / --no_median_align)

# 2) Latência/FPS — por engine
python utils/benchmark.py \
    --engine engines/dav2_vits_364x518_fp16.engine \
    --json_out engines/dav2_vits_364x518_fp16.bench.json

# 3) Curva de Pareto: cruza *.eval.json + *.bench.json -> CSV + PNG
python utils/gerar_pareto.py --dir engines/ \
    --csv_out engines/pareto.csv --fig_out engines/pareto.png
```

`avaliar_engine.py` reutiliza o `metricas.py` e o `dataloader.py` do projeto,
então os números são comparáveis com os `test_metrics.json` do treino (FP32).
Roda tanto com `--engine` (Jetson) quanto com `--onnx` (desktop, p/ conferir
paridade antes de compilar).

## Notas de viabilidade

- **Monodepth2** (ResNet-18) e **DAV2 ViT-S** são os candidatos a tempo real
  no Orin Nano; ambos exportam bem e aceitam INT8.
- **ZoeDepth (BEiT-Large, ~352M params)** é o caso difícil: o export ONNX do
  BEiT é frágil (interpolação de pos-bias, meshgrid/gather) e, mesmo
  funcionando, roda a poucos FPS nos 8 GB do Orin Nano. **Não use INT8** nele
  (BEiT degrada sem QAT) — fique em FP16. Se o export falhar, veja as dicas no
  cabeçalho de `export/export_zoedepth.py`.
- **INT8**: calibre com as imagens do **ICL Ground Robot** (`preparar_calibracao.py`),
  não com COCO/ImageNet — a distribuição de operação importa.

## Fluxo completo (resumo)

```
desktop:  exportar_todos.sh  ->  onnx/        ->  scp p/ Jetson
jetson:   construir_todos.sh ->  engines/
jetson:   benchmark.py + avaliar_engine.py por engine  ->  *.bench.json + *.eval.json
desktop:  gerar_pareto.py    ->  pareto.csv + pareto.png
```

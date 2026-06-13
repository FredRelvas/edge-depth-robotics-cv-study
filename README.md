# Estimativa de profundidade monocular embarcada em robótica móvel

Avaliação de modelos de **estimativa monocular de profundidade** executados
**a bordo de um TurtleBot4 real**, comparando suas predições com o *ground truth*
de um **LiDAR 2D** e com a profundidade do **sensor RGB-D nativo** (baseline).

O trabalho é uma **extensão direta** de Vizzotto et al. (RITA 2025) —
*Case Study of Deep Learning Methods for Depth Estimation in Indoor Ground
Robotics* —, que avaliou os modelos **em simulação/offline**. Aqui levamos a
avaliação para o **embarcado**: os modelos são quantizados e rodam numa
**Jetson Orin Nano** dentro do robô, e a comparação é feita contra sensores
físicos durante trajetórias reais.

## Modelos avaliados

| Modelo | Tipo | Saída | Alinhamento de escala |
|---|---|---|---|
| **Monodepth2** (fine-tunado) | autossupervisionado | até-escala | mediana (1 grau de liberdade) |
| **Depth-Anything v2** *frozen* | relativo | disparidade | afim (escala + *shift*) |
| **Depth-Anything v2** *trainable* | relativo | disparidade | afim (escala + *shift*) |

> **ZoeDepth** (o segundo modelo do RITA 2025) **não** foi avaliado a bordo:
> como modelo métrico mais pesado, esbarrou em restrições de hardware e de
> quantização no embarcado. O código de treino/exportação dele permanece no
> repositório (`codigo-treinamento/`, `quantizacao/`) documentando a tentativa,
> e a sua inclusão fica como trabalho futuro (ver o artigo).

## Hardware

- **Robô:** TurtleBot4
- **Computação embarcada:** NVIDIA Jetson Orin Nano
- **Câmera:** Intel RealSense D415 (RGB + depth nativo → *baseline*)
- **Ground truth:** RPLiDAR A1 (LiDAR 2D)

## Pipeline (visão geral)

```
treino/fine-tuning        quantização            execução no robô        validação offline
(codigo-treinamento/)  →  (quantizacao/)     →   (Jetson + ROS2)     →   (validacao/)
   dataset ICL             ONNX → TensorRT         grava rosbags          rosbag → métricas
                           FP16/INT8                                       (IA × LiDAR × baseline)
```

A inferência da IA é gravada **ao vivo** no rosbag durante o run. A validação
(`validacao/`) **não** recarrega modelos: ela lê os arrays do bag, projeta o
LiDAR sobre os pixels da imagem (gerando o *ground truth* esparso), aplica o
**mesmo alinhamento de escala** à predição da IA e ao baseline, e calcula as
métricas. Esse alinhamento simétrico é o que torna a comparação IA × baseline
justa por construção (detalhes na metodologia do artigo).

## Estrutura do repositório

```
.
├── codigo-treinamento/   # fine-tuning dos modelos no dataset ICL Ground Robot
│   └── README.md         #   (documentação detalhada do treino e do dataloader)
├── quantizacao/          # exportação ONNX + compilação TensorRT (FP16/INT8) p/ Jetson
├── validacao/            # pipeline offline: rosbag → projeção LiDAR → métricas
├── metricas/             # as 7 métricas de profundidade (Abs Rel, RMSE, δ<1.25, ...)
├── scripts/              # orquestradores run_*.sh dos experimentos
├── dados/                # rosbags e dataset (ignorados pelo git)
├── modelos_treinados/    # checkpoints .pth/.onnx/.engine (ignorados pelo git)
├── resultados/           # saídas de validação em JSON (ignoradas pelo git)
├── pyproject.toml / uv.lock
└── README.md
```

## Instalação

Usa [`uv`](https://docs.astral.sh/uv/) para um ambiente reproduzível:

```bash
# instalar o uv (macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/FredRelvas/edge-depth-robotics-cv-study
cd edge-depth-robotics-cv-study
uv sync
```

O `uv sync` cria o `.venv/` e instala as dependências fixadas no `uv.lock`.
Para detalhes de CUDA/GPU e do fluxo de **treino**, ver
[`codigo-treinamento/README.md`](codigo-treinamento/README.md).

## Como reproduzir a validação

```bash
# 1. (uma vez) sanity-check das métricas
uv run python metricas/metricas.py

# 2. validar um rosbag contra LiDAR + baseline
uv run python validacao/avaliar.py \
    --bag dados/<rosbag_da_run> \
    --modelo monodepth2 \
    --calibracao validacao/config/calibracao_realsense_d415.yaml \
    --saida resultados/monodepth2.json
```

Trocar `--modelo` por `dav2_frozen` ou `dav2_trainable` para os demais.
Antes de rodar com um bag novo, inspecione o conteúdo com
`uv run python validacao/inspecionar_bag.py`. Ver
[`validacao/README.md`](validacao/README.md) para os módulos e o contrato dos
tópicos.

## Disponibilidade de código e dados

- **Pipeline de validação (este repo):** <https://github.com/FredRelvas/edge-depth-robotics-cv-study>
- **Setup do robô (ROS2):** <https://github.com/JoaoGChv/TB4>
- **Checkpoints treinados:** Hugging Face — `SrRyan/depth-icl-ground-robot`
- **Dataset de treino/fine-tuning:** <https://www.kaggle.com/datasets/fredericorelvas/datasets-depht-models>

## Referências

- Vizzotto, F. L. et al. *Case Study of Deep Learning Methods for Depth
  Estimation in Indoor Ground Robotics*. RITA, 32(1), 166–172, 2025.
  [DOI: 10.22456/2175-2745.143443](https://doi.org/10.22456/2175-2745.143443)
- [Monodepth2](https://github.com/nianticlabs/monodepth2) (Godard et al., 2019)
- [Depth-Anything v2](https://github.com/DepthAnything/Depth-Anything-V2) (Yang et al., 2024)
- [ZoeDepth](https://github.com/isl-org/ZoeDepth) (Bhat et al., 2023)

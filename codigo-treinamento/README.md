# Trabalho Final Visão

Replicação do estudo de Vizzotto et al. (RITA 2025) — *Case Study of Deep
Learning Methods for Depth Estimation in Indoor Ground Robotics* — comparando
**ZoeDepth** (supervisionado, métrico) e **Monodepth2** (autossupervisionado)
no dataset **ICL Ground Robot** do Pering Laboratory (Imperial College London).

## Estrutura do repositório

```
edge-depth-robotics-cv-study/
├── codigo-treinamento/
│   ├── baixar_dados.py     # baixa e organiza o dataset ICL
│   └── dataloader.py       # ICLGroundRobotDataset (PyTorch)
├── dados/
│   └── icl_ground_robot/   # criado por baixar_dados.py (~3GB, ignorado pelo git)
├── metricas/
│   └── metricas.py         # 7 métricas do paper (Abs Rel, RMSE, δ<1.25 etc.)
├── modelos-treinados/      # checkpoints (.pth/.onnx/.engine) ignorados pelo git
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Instalação

### 1. Instalar o uv (caso não tenha)

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Após instalar, reinicie o terminal (ou rode `export PATH="$HOME/.local/bin:$PATH"`)
para que o comando `uv` fique disponível.

### 2. Clonar o repositório

```bash
git clone <url-do-repositorio>
cd edge-depth-robotics-cv-study
```

### 3. Instalar as dependências

```bash
uv sync
```

O `uv` cria automaticamente um ambiente virtual em `.venv/` e instala todas as
dependências fixadas no `uv.lock`, garantindo o mesmo ambiente em qualquer máquina.

### 4. Verificar a GPU

```bash
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Esperado:

```
CUDA: True | GPU: NVIDIA GeForce RTX 4090
```

Se vier `CUDA: False`, consulte a seção [GPU / CUDA](#gpu--cuda) abaixo.

---

## Fluxo de uso

### 1. Baixar o dataset ICL Ground Robot

```bash
uv run python codigo-treinamento/baixar_dados.py
```

Baixa as cenas **Deer** e **Diamond** (1600 frames RGB-D cada, ~3GB no total) do
Pering Laboratory e organiza em `dados/icl_ground_robot/`. O dataset **não é
versionado** pelo git — cada máquina baixa localmente.

Layout resultante:

```
dados/icl_ground_robot/
├── deer/
│   ├── frames/deer_robot/
│   │   ├── cam0/data/       # 1600 PNGs RGB (640x480)
│   │   ├── depth0/data/     # 1600 PNGs depth (uint16, escala 1mm)
│   │   ├── cameraInfo.txt
│   │   └── poses.gt
│   └── trajectory.gt
└── diamond/
    └── ... (mesma estrutura)
```

### 2. Testar o dataloader

```bash
uv run python codigo-treinamento/dataloader.py --scene deer
```

Saída esperada (split 70/10/20 conforme paper, faixa de profundidade plausível):

```
[deer   /train]  N=1120  rgb=(3, 256, 256)  depth=(1, 256, 256)  depth_min=0.264m  depth_max=6.688m  valid_frac=1.000
[deer   /val  ]  N= 160  rgb=(3, 256, 256)  depth=(1, 256, 256)  depth_min=0.265m  depth_max=8.547m  valid_frac=1.000
[deer   /test ]  N= 320  rgb=(3, 256, 256)  depth=(1, 256, 256)  depth_min=0.273m  depth_max=4.379m  valid_frac=1.000
```

### 3. Testar as métricas

```bash
uv run python metricas/metricas.py
```

Roda 5 sanity tests (predição perfeita, erro proporcional, threshold delta,
alinhamento por mediana, máscara externa). Todos devem passar.

### 4. Usar no seu script de treino

```python
import sys
sys.path.insert(0, "codigo-treinamento")
sys.path.insert(0, "metricas")

from dataloader import build_icl_dataloaders
from metricas import compute_depth_metrics, aggregate_batch_metrics, format_metrics

# DataLoaders prontos com split do paper
loaders = build_icl_dataloaders(
    scenes=("deer", "diamond"),
    image_size=256,        # ZoeDepth DPT_SwinV2_T_256
    batch_size=8,
    num_workers=4,
)

# Loop de treino
for epoch in range(40):
    model.train()
    for batch in loaders["train"]:
        rgb   = batch["rgb"].cuda(non_blocking=True)    # [B, 3, 256, 256]
        gt    = batch["depth"].cuda(non_blocking=True)  # [B, 1, 256, 256] em metros
        valid = batch["valid"].cuda(non_blocking=True)  # [B, 1, 256, 256] máscara
        ...

    # Validação
    model.eval()
    batch_metrics = []
    with torch.no_grad():
        for batch in loaders["val"]:
            pred = model(batch["rgb"].cuda())
            batch_metrics.append(compute_depth_metrics(
                pred, batch["depth"].cuda(), valid=batch["valid"].cuda(),
                median_align=False,   # True para Monodepth2
            ))
    print(f"Época {epoch}:", format_metrics(aggregate_batch_metrics(batch_metrics)))
```

---

## GPU / CUDA

Por padrão, o `pyproject.toml` está configurado para **CUDA 12.6** (compatível
com a RTX 4090 e drivers modernos).

Se sua workstation usa outra versão de CUDA, edite o `pyproject.toml` e
substitua `cu126` pela sua versão (4 ocorrências):

| CUDA | Substituir por |
|------|----------------|
| 12.1 | `cu121` |
| 12.4 | `cu124` |
| 12.6 | `cu126` (default) |

Depois rode:

```bash
uv sync --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio
```

Em **macOS** (Apple Silicon), o `uv` cai automaticamente no PyPI padrão (que já
vem com suporte MPS) — não precisa fazer nada.

---

## Dependências principais

| Categoria | Bibliotecas |
|-----------|-------------|
| Deep learning | `torch`, `torchvision`, `torchaudio` |
| Visão computacional | `opencv-python`, `Pillow`, `albumentations` |
| Dados | `numpy`, `scikit-learn`, `pandas` |
| Visualização | `matplotlib`, `seaborn` |
| Experimentos | `tensorboard` |
| Utilitários | `tqdm`, `PyYAML` |

Para adicionar uma nova biblioteca:

```bash
uv add <nome-do-pacote>
```

---

## Referências

- **Paper original:** Vizzotto, F. L. et al. *Case Study of Deep Learning Methods
  for Depth Estimation in Indoor Ground Robotics*. Revista de Informática Teórica
  e Aplicada, 32(1), 166–172, 2025.
  [DOI: 10.22456/2175-2745.143443](https://doi.org/10.22456/2175-2745.143443)
- **Paper do dataset:** Saeedi, S. et al. *Characterizing Visual Localization
  and Mapping Datasets*. ICRA 2019.
- **Dataset:** [Pering Laboratory — LMData](https://peringlab.org/lmdata/)
- **Modelos avaliados:**
  - [ZoeDepth](https://github.com/isl-org/ZoeDepth) (Bhat et al., 2023)
  - [Monodepth2](https://github.com/nianticlabs/monodepth2) (Godard et al., 2019)
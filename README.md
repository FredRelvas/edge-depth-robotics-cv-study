# Trabalho Final Visão

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

Após instalar, reinicie o terminal ou execute `source $HOME/.local/bin/env` (macOS/Linux) para o comando `uv` ficar disponível.

### 2. Clonar o repositório

```bash
git clone <url-do-repositorio>
cd trabalho-final
```

### 3. Instalar as dependências

```bash
uv sync
```

O `uv` cria automaticamente um ambiente virtual em `.venv/` e instala todas as dependências fixadas no `uv.lock`, garantindo o mesmo ambiente em qualquer máquina.

### 4. Executar scripts

```bash
uv run python codigo-treinamento/seu_script.py
```

Ou ative o ambiente virtual manualmente:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

---

## GPU (opcional)

Por padrão, o PyTorch é instalado na versão CPU (compatível com Apple MPS automaticamente). Para usar CUDA no Linux, edite o `pyproject.toml` e adicione o índice correspondente à sua versão de CUDA em `[tool.uv.sources]`:

| CUDA | URL do índice |
|------|---------------|
| 12.1 | `https://download.pytorch.org/whl/cu121` |
| 12.4 | `https://download.pytorch.org/whl/cu124` |
| 12.6 | `https://download.pytorch.org/whl/cu126` |

Depois rode `uv sync` novamente.

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

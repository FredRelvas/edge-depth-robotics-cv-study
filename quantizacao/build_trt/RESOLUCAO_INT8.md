# Resolução dos Erros no Build INT8 (TRT 10.3 / Jetson Orin Nano SM 8.7)

## Contexto

Plataforma: Jetson Orin Nano — JetPack 6.2, CUDA 12.5, TensorRT 10.3.0, SM 8.7 (Ampere).

O build de engines INT8 via `IInt8EntropyCalibrator2` falhava consistentemente com dois erros no log:

```
[TRT] [E] Skipping tactic 0x... due to exception Cask Pooling Runner Execute Failure
[TRT] [E] IBuilder::buildSerializedNetwork: Error Code 10: Internal Error
         (Could not find any implementation for node node_max_pool2d.)
```

---

## Causa Raiz 1 — Conflito de Contexto CUDA

### O problema

O TensorRT e o PyCUDA precisam usar **o mesmo contexto CUDA** durante todo o ciclo de vida do builder. O código original importava `pycuda.autoinit` dentro do construtor da classe `ImageCalibrator`:

```python
# ❌ CÓDIGO ORIGINAL — autoinit dentro do __init__
class ImageCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, ...):
        super().__init__()
        import pycuda.driver as cuda
        import pycuda.autoinit   # ← cria um novo contexto CUDA aqui
```

O fluxo de execução era:

1. `trt.Builder(TRT_LOGGER)` — TRT inicializa e adquire o **contexto A**
2. `config.int8_calibrator = ImageCalibrator(...)` — `pycuda.autoinit` executa e cria o **contexto B**
3. `builder.build_serialized_network(...)` — TRT já opera no contexto B, diferente do contexto A em que foi criado

O próprio TRT avisa sobre isso no log:

```
[TRT] [W] The CUDA context changed between createInferBuilder and
buildSerializedNetwork. A Builder holds CUDA resources which cannot be
shared across CUDA contexts...
```

Com dois contextos ativos, os kernels CASK (usados para Pooling INT8) não conseguiam encontrar nem reservar memória corretamente, resultando em `Cask Pooling Runner Execute Failure` para **todas** as tácticas testadas.

### A correção

Mover o `import pycuda.autoinit` para o **topo do módulo**, antes de qualquer código TRT:

```python
# ✅ CORREÇÃO — autoinit antes do tensorrt
import pycuda.autoinit   # cria o contexto antes do TRT Builder
import pycuda.driver as _cuda_drv
import tensorrt as trt   # TRT reutiliza o contexto já existente
```

Isso garante que quando `trt.Builder()` for chamado, o contexto PyCUDA já existe e o TRT o reutiliza — um único contexto compartilhado por todo o build.

---

## Causa Raiz 2 — Ausência de Kernel INT8 para `max_pool2d` no SM 8.7

### O problema

Mesmo com o contexto corrigido, o TRT 10.3 não possui um kernel INT8 estável para `MaxPool2D` no SM 8.7 (Orin Nano). A biblioteca CASK tenta e falha; nenhuma outra táctica (cuDNN, cuBLAS) tem implementação INT8 de Pooling disponível para essa arquitetura nessa versão do TRT.

Isso não é um bug de contexto — é uma limitação real do suporte INT8 do TRT 10.3 para SM 8.7.

### A correção

Forçar as camadas de Pooling para **FP16** dentro de um build INT8, usando o mecanismo de precisão por camada do TRT. O restante da rede continua em INT8:

```python
# ✅ CORREÇÃO — forçar Pooling para FP16 dentro do build INT8
config.set_flag(trt.BuilderFlag.INT8)
config.set_flag(trt.BuilderFlag.FP16)
config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

n_pool = 0
for i in range(network.num_layers):
    layer = network.get_layer(i)
    if layer.type == trt.LayerType.POOLING:
        layer.precision = trt.DataType.HALF        # input da camada em FP16
        layer.set_output_type(0, trt.DataType.HALF) # output da camada em FP16
        n_pool += 1
```

Significado das flags:
| Flag | Efeito |
|------|--------|
| `INT8` | Habilita quantização INT8 para a rede |
| `FP16` | Habilita FP16; TRT escolhe INT8 ou FP16 por camada conforme disponibilidade |
| `GPU_FALLBACK` | Camadas sem kernel INT8/FP16 podem cair para FP32 em vez de abortar |
| `PREFER_PRECISION_CONSTRAINTS` | Respeita as precisões definidas por camada quando possível |

A iteração sobre `network.num_layers` é feita **após** o parser ONNX ter populado a rede, então todas as camadas já estão disponíveis.

---

## Tentativas Anteriores que Não Funcionaram

| Tentativa | Motivo da Falha |
|-----------|-----------------|
| Adicionar apenas `GPU_FALLBACK` | Não resolve: CASK **lança exceção** (crash), não retorna "não suportado". O fallback só age quando o kernel sinaliza indisponibilidade normalmente. |
| `config.set_tactic_sources(CUDNN \| CUBLAS \| CUBLAS_LT)` | Desabilita CASK por via de `TacticSource`, mas `TacticSource` em TRT 10.3 não inclui entrada nomeada "CASK" — só `CUBLAS`, `CUBLAS_LT`, `CUDNN`, `EDGE_MASK_CONVOLUTIONS`, `JIT_CONVOLUTIONS`. O CASK é um conjunto de tácticas internas, não um `TacticSource` configurável. |
| `layer.precision = FP16` sem `PREFER_PRECISION_CONSTRAINTS` | TRT ignora restrições por camada sem essa flag. |

---

## Resultado Final

Com as duas correções aplicadas:

1. **Monodepth2 192×640 INT8**: build bem-sucedido — 15.9 MB (vs 29 MB FP16, redução ~45%).
2. **DAV2 ViT-S 364×518 INT8**: calibração e build em andamento sem erros.

A rede Monodepth2 (ResNet-18) tem 1 camada `MaxPool2D` que passou a rodar em FP16. O DAV2 ViT-S não usa Pooling convencional (é um Vision Transformer), então o workaround não foi necessário para ele — mas é inócuo aplicar.

---

## Nota sobre Depreciação

O `IInt8EntropyCalibrator2` foi marcado como depreciado no TRT 10.1. A API recomendada a partir do TRT 10.x é **Explicit Quantization** (inserir nós Q/DQ no grafo ONNX antes de exportar). Para os fins deste projeto, a API de calibração implícita ainda funciona, apenas exibe um aviso em tempo de execução.

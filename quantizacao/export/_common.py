"""
Utilitários compartilhados pelos scripts de export ONNX (FASE 1, desktop).

CONTRATO DE ENTRADA (importante!)
---------------------------------
Os três modelos deste projeto foram TREINADOS recebendo RGB cru em [0, 1],
NCHW, SEM normalização ImageNet externa (ver codigo-treinamento/dataloader.py,
que só faz img/255). Cada modelo aplica (ou não) sua própria normalização
DENTRO do forward:

    - Monodepth2 : ResnetEncoder normaliza (x-0.45)/0.225 internamente.
    - ZoeDepth   : MidasCore.prep normaliza (mean/std=0.5) + resize internamente.
    - DAV2       : NÃO normaliza no forward (foi treinado com [0,1] cru mesmo).

Portanto TODOS os ONNX exportados aqui recebem [0,1] NCHW float32 e qualquer
pré-processamento interno fica embutido no grafo. O utils/validate.py e o
INT8 calibrator devem alimentar exatamente [0,1] NCHW (img/255), nada de
mean/std ImageNet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# quantizacao/export/_common.py -> raiz = parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def add_external_paths() -> None:
    """Coloca os repos externos clonados no sys.path (mesma convenção do treino)."""
    paths = [
        PROJECT_ROOT / "codigo-treinamento",
        PROJECT_ROOT / "external" / "ZoeDepth",
        PROJECT_ROOT / "external" / "DAV2" / "metric_depth",
        PROJECT_ROOT / "external" / "monodepth2",
    ]
    for p in paths:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def resolve_checkpoint(cli_value: str | None, default_name: str) -> Path:
    """
    Resolve o caminho do checkpoint. Se o usuário não passar --checkpoint,
    procura em modelos_treinados/<default_name>.
    """
    if cli_value:
        p = Path(cli_value).expanduser().resolve()
    else:
        p = PROJECT_ROOT / "modelos_treinados" / default_name
    if not p.exists():
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {p}\n"
            f"Passe --checkpoint <caminho> ou coloque o arquivo em "
            f"modelos_treinados/."
        )
    return p


def export_onnx(
    model: torch.nn.Module,
    dummy: torch.Tensor,
    output: Path,
    opset: int = 17,
    simplify: bool = True,
    input_name: str = "rgb",
    output_name: str = "depth",
    dynamic_batch: bool = False,
) -> Path:
    """
    Exporta `model` para ONNX em `output`. Entrada estática HxW (TensorRT na
    Jetson trabalha melhor com shapes fixos). `dynamic_batch=True` deixa só a
    dimensão de batch dinâmica.

    Retorna o caminho final (já simplificado se simplify=True e onnxsim presente).
    """
    import onnx

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {input_name: {0: "batch"}, output_name: {0: "batch"}}

    print(f"[export] -> {output}  (input {tuple(dummy.shape)}, opset {opset})")
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(output),
            input_names=[input_name],
            output_names=[output_name],
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
        )

    # Validação estrutural
    onnx_model = onnx.load(str(output))
    onnx.checker.check_model(onnx_model)

    if simplify:
        try:
            from onnxsim import simplify as onnxsim_simplify
            print("[export] simplificando com onnxsim...")
            simplified, ok = onnxsim_simplify(onnx_model)
            if ok:
                onnx.save(simplified, str(output))
                print("[export] onnxsim OK")
            else:
                print("[export][warn] onnxsim não convergiu; mantendo ONNX original.")
        except ImportError:
            print("[export][warn] onnxsim não instalado; pulei a simplificação "
                  "(pip install onnxsim).")

    mb = output.stat().st_size / 1e6
    print(f"[export] concluído: {output} ({mb:.1f} MB)")
    return output


@torch.no_grad()
def sanity_forward(model: torch.nn.Module, dummy: torch.Tensor) -> None:
    """Roda 1 forward em PyTorch antes de exportar e imprime o range de profundidade."""
    model.eval()
    out = model(dummy)
    print(f"[sanity] saída PyTorch: shape={tuple(out.shape)}  "
          f"min={out.min().item():.3f}  max={out.max().item():.3f}  "
          f"mean={out.mean().item():.3f}")

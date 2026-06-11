#!/usr/bin/env python
"""
FASE 2 (Jetson Orin Nano, JetPack 6.2 / TensorRT 10.x):
constrói um engine TensorRT a partir de um ONNX.

Suporta FP16 (padrão do benchmark) e INT8 (com calibração por entropia).
O contrato de entrada é [0,1] NCHW — o calibrador pré-processa as imagens de
calibração exatamente assim (img/255 + resize), igual ao treino.

Exemplos:
    # FP16
    python build_trt/build_engine.py \
        --onnx onnx/dav2_vits_364x518.onnx \
        --output engines/dav2_vits_364x518_fp16.engine \
        --precision fp16 --workspace_mb 2048

    # INT8 (só recomendado p/ Monodepth2 e DAV2; NÃO p/ ZoeDepth/BEiT)
    python build_trt/build_engine.py \
        --onnx onnx/monodepth2_192x640.onnx \
        --output engines/monodepth2_192x640_int8.engine \
        --precision int8 --calib_dir calibracao/ \
        --calib_cache calib_monodepth2.cache
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
# pycuda.autoinit DEVE ser importado antes do tensorrt para garantir
# que ambos usem o mesmo contexto CUDA (TRT avisa se o contexto mudar).
import pycuda.autoinit  # noqa: F401
import pycuda.driver as _cuda_drv
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


# ---------------------------------------------------------------------------
# Pré-processamento (idêntico ao validate.py e ao contrato de treino)
# ---------------------------------------------------------------------------

def preprocess(path: str, h: int, w: int) -> np.ndarray:
    """Carrega imagem -> RGB [0,1] NCHW float32 (1,3,h,w)."""
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)        # BGR HxWx3
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None]             # 1,3,h,w
    return np.ascontiguousarray(img)


# ---------------------------------------------------------------------------
# Calibrador INT8 por entropia (EntropyCalibrator2)
# ---------------------------------------------------------------------------

class ImageCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calib_dir: Path, shape, cache_file: Path):
        super().__init__()
        self.cuda = _cuda_drv
        self.cache_file = Path(cache_file)
        _, c, h, w = shape
        self.h, self.w = h, w
        self.batch_size = 1
        self.files = sorted(Path(calib_dir).glob("*.png")) + \
                     sorted(Path(calib_dir).glob("*.jpg"))
        if not self.files:
            raise RuntimeError(f"Sem imagens de calibração em {calib_dir}")
        print(f"[calib] {len(self.files)} imagens, input {c}x{h}x{w}")
        self.idx = 0
        self.device_input = self.cuda.mem_alloc(
            int(np.prod((self.batch_size, c, h, w)) * 4))   # float32

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.idx >= len(self.files):
            return None
        batch = preprocess(self.files[self.idx], self.h, self.w)
        self.idx += 1
        if self.idx % 50 == 0:
            print(f"[calib]   {self.idx}/{len(self.files)}")
        self.cuda.memcpy_htod(self.device_input, batch)
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if self.cache_file.exists():
            print(f"[calib] usando cache {self.cache_file}")
            return self.cache_file.read_bytes()
        return None

    def write_calibration_cache(self, cache):
        self.cache_file.write_bytes(cache)
        print(f"[calib] cache salvo em {self.cache_file}")


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(onnx_path: Path, output: Path, precision: str,
          workspace_mb: int, calib_dir: Path | None, calib_cache: Path | None):
    builder = trt.Builder(TRT_LOGGER)
    # TRT 10: rede com batch explícito (o flag legado foi removido; create_network()
    # já é explícito). Mantemos compat com TRT 8/9 via try.
    try:
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)
    except AttributeError:
        network = builder.create_network(0)

    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"[parser] {parser.get_error(i)}")
            raise RuntimeError(f"Falha ao parsear {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                 workspace_mb * (1 << 20))

    in_shape = network.get_input(0).shape
    print(f"[build] {onnx_path.name}  input={tuple(in_shape)}  precision={precision}")

    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            print("[build][warn] plataforma sem FP16 rápido.")
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        if calib_dir is None:
            raise SystemExit("[erro] --precision int8 exige --calib_dir.")
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
        # TRT 10.3 + SM 8.7 (Orin Nano): não existe kernel INT8 para max_pool2d.
        # Força cada camada de Pooling para FP16; o restante da rede fica INT8.
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        n_pool = 0
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if layer.type == trt.LayerType.POOLING:
                layer.precision = trt.DataType.HALF
                layer.set_output_type(0, trt.DataType.HALF)
                n_pool += 1
        if n_pool:
            print(f"[build] {n_pool} camada(s) Pooling forçadas para FP16 (workaround SM 8.7)")
        cache = calib_cache or output.with_suffix(".cache")
        config.int8_calibrator = ImageCalibrator(calib_dir, in_shape, cache)
    # fp32 = sem flags

    print("[build] construindo engine (pode levar minutos)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build_serialized_network retornou None.")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(serialized)
    print(f"[build] OK -> {output} ({output.stat().st_size/1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--precision", default="fp16", choices=["fp16", "int8", "fp32"])
    ap.add_argument("--workspace_mb", type=int, default=2048)
    ap.add_argument("--calib_dir", type=Path, default=None)
    ap.add_argument("--calib_cache", type=Path, default=None)
    args = ap.parse_args()

    build(args.onnx, args.output, args.precision, args.workspace_mb,
          args.calib_dir, args.calib_cache)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Sanity check: roda 1 imagem por um ONNX (desktop) OU por um engine TensorRT
(Jetson) e imprime o range de profundidade. Use para confirmar que o engine
FP16/INT8 produz profundidades coerentes com o ONNX (FP16 ~1% de diferença
esperada; INT8 pode ter mais).

Entrada sempre [0,1] NCHW (img/255 + resize) — mesmo contrato do treino.

Exemplos:
    # ONNX (desktop, precisa onnxruntime)
    python utils/validate.py --onnx onnx/dav2_vits_364x518.onnx \
        --image amostra.png --height 364 --width 518

    # Engine (Jetson, precisa tensorrt + pycuda)
    python utils/validate.py --engine engines/dav2_vits_364x518_fp16.engine \
        --image amostra.png --height 364 --width 518
"""

from __future__ import annotations

import argparse
import numpy as np


def preprocess(path: str, h: int, w: int) -> np.ndarray:
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Não consegui ler a imagem: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    return np.ascontiguousarray(np.transpose(img, (2, 0, 1))[None])


def report(name: str, depth: np.ndarray) -> None:
    d = depth.reshape(-1)
    d = d[np.isfinite(d)]
    print(f"[{name}] shape={depth.shape}  min={d.min():.3f}  "
          f"max={d.max():.3f}  mean={d.mean():.3f}  median={np.median(d):.3f}")


def run_onnx(path, x):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CUDAExecutionProvider",
                                                 "CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return sess.run(None, {name: x})[0]


def run_engine(path, x):
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401  (inicializa contexto CUDA)

    logger = trt.Logger(trt.Logger.WARNING)
    with open(path, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    # TRT 10: API por nome de tensor.
    in_name = out_name = None
    for i in range(engine.num_io_tensors):
        nm = engine.get_tensor_name(i)
        if engine.get_tensor_mode(nm) == trt.TensorIOMode.INPUT:
            in_name = nm
        else:
            out_name = nm

    context.set_input_shape(in_name, x.shape)
    out_shape = tuple(context.get_tensor_shape(out_name))
    out = np.empty(out_shape, dtype=np.float32)

    d_in = cuda.mem_alloc(x.nbytes)
    d_out = cuda.mem_alloc(out.nbytes)
    stream = cuda.Stream()

    cuda.memcpy_htod_async(d_in, x, stream)
    context.set_tensor_address(in_name, int(d_in))
    context.set_tensor_address(out_name, int(d_out))
    context.execute_async_v3(stream.handle)
    cuda.memcpy_dtoh_async(out, d_out, stream)
    stream.synchronize()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx")
    ap.add_argument("--engine")
    ap.add_argument("--image", required=True)
    ap.add_argument("--height", type=int, required=True)
    ap.add_argument("--width", type=int, required=True)
    args = ap.parse_args()

    if not (args.onnx or args.engine):
        raise SystemExit("[erro] passe --onnx OU --engine.")

    x = preprocess(args.image, args.height, args.width)
    print(f"[input] {x.shape}  range [{x.min():.2f}, {x.max():.2f}]")

    if args.onnx:
        report("onnx", run_onnx(args.onnx, x))
    if args.engine:
        report("engine", run_engine(args.engine, x))


if __name__ == "__main__":
    main()

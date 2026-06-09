#!/usr/bin/env python
"""
Benchmark de latência/FPS de um engine TensorRT na Jetson Orin Nano.

Mede latência de inferência pura (GPU) com CUDA events: warmup + N execuções,
reporta p50/p95/média e FPS. Para medir energia/temperatura em paralelo, rode
`tegrastats` em outro terminal (ver --tegrastats_hint).

Exemplo:
    python utils/benchmark.py \
        --engine engines/dav2_vits_364x518_fp16.engine \
        --runs 200 --warmup 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, type=Path)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--json_out", type=Path, default=None,
                    help="se passado, salva o resumo em JSON (p/ a curva de Pareto).")
    args = ap.parse_args()

    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401

    logger = trt.Logger(trt.Logger.WARNING)
    with open(args.engine, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    in_name = out_name = None
    for i in range(engine.num_io_tensors):
        nm = engine.get_tensor_name(i)
        if engine.get_tensor_mode(nm) == trt.TensorIOMode.INPUT:
            in_name = nm
        else:
            out_name = nm

    in_shape = tuple(context.get_tensor_shape(in_name))
    if -1 in in_shape:  # batch dinâmico -> fixa em 1
        in_shape = (1,) + in_shape[1:]
        context.set_input_shape(in_name, in_shape)
    out_shape = tuple(context.get_tensor_shape(out_name))

    x = np.ascontiguousarray(np.random.rand(*in_shape).astype(np.float32))
    out = np.empty(out_shape, dtype=np.float32)
    d_in = cuda.mem_alloc(x.nbytes)
    d_out = cuda.mem_alloc(out.nbytes)
    stream = cuda.Stream()
    cuda.memcpy_htod(d_in, x)
    context.set_tensor_address(in_name, int(d_in))
    context.set_tensor_address(out_name, int(d_out))

    start, end = cuda.Event(), cuda.Event()

    for _ in range(args.warmup):
        context.execute_async_v3(stream.handle)
    stream.synchronize()

    lat_ms = []
    for _ in range(args.runs):
        start.record(stream)
        context.execute_async_v3(stream.handle)
        end.record(stream)
        end.synchronize()
        lat_ms.append(end.time_since(start))

    lat = np.array(lat_ms)
    summary = {
        "engine": args.engine.name,
        "input_shape": list(in_shape),
        "runs": args.runs,
        "lat_ms_p50": float(np.percentile(lat, 50)),
        "lat_ms_p95": float(np.percentile(lat, 95)),
        "lat_ms_mean": float(lat.mean()),
        "fps": float(1000.0 / lat.mean()),
    }
    print(f"\n[benchmark] {summary['engine']}  input={summary['input_shape']}")
    print(f"  latência: p50={summary['lat_ms_p50']:.2f} ms  "
          f"p95={summary['lat_ms_p95']:.2f} ms  média={summary['lat_ms_mean']:.2f} ms")
    print(f"  throughput: {summary['fps']:.1f} FPS")
    print("\n[dica] energia/temperatura: rode `tegrastats` em outro terminal "
          "durante o benchmark e registre POM_5V_IN (mW).")

    if args.json_out:
        args.json_out.write_text(json.dumps(summary, indent=2))
        print(f"[benchmark] resumo salvo em {args.json_out}")


if __name__ == "__main__":
    main()

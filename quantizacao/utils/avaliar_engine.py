#!/usr/bin/env python
"""
Avaliação das 7 métricas (AbsRel, SqRel, RMSE, RMSE log, δ1/2/3) de um engine
TensorRT (Jetson) OU de um ONNX (desktop) no split de TESTE do ICL Ground Robot.

Reutiliza exatamente o metricas.py e o dataloader.py do projeto, então os
números são comparáveis com os test_metrics.json do treino em FP32.

Fluxo por frame:
    rgb[0,1] @gt -> resize p/ entrada do engine -> inferência -> depth
    -> resize de volta p/ resolução do GT -> compute_depth_metrics(valid)

median-align: ligado automaticamente para Monodepth2 (prediz até escala);
desligado para ZoeDepth/DAV2 (métricos). Force com --median_align/--no_median_align.

Exemplos:
    # Engine (Jetson)
    python utils/avaliar_engine.py \
        --engine engines/dav2_vits_364x518_fp16.engine \
        --json_out engines/dav2_vits_364x518_fp16.eval.json

    # ONNX (desktop, confere a paridade antes de ir p/ Jetson)
    python utils/avaliar_engine.py \
        --onnx onnx/monodepth2_192x640.onnx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "codigo-treinamento"))
sys.path.insert(0, str(PROJECT_ROOT / "metricas"))

from dataloader import ICLGroundRobotDataset                       # noqa: E402
from metricas import (compute_depth_metrics, aggregate_batch_metrics,  # noqa: E402
                      format_metrics)


# ---------------------------------------------------------------------------
# Backends de inferência (ONNX ou TensorRT), com a mesma interface .infer()
# ---------------------------------------------------------------------------

class OnnxBackend:
    def __init__(self, path: str):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(
            path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.name = inp.name
        # shape ONNX: [N,3,H,W] (H/W podem ser estáticos)
        _, _, h, w = inp.shape
        self.in_hw = (int(h), int(w))

    def infer(self, x: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {self.name: x})[0]


class TrtBackend:
    def __init__(self, path: str):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401
        self.trt, self.cuda = trt, cuda

        logger = trt.Logger(trt.Logger.WARNING)
        with open(path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.in_name = self.out_name = None
        for i in range(self.engine.num_io_tensors):
            nm = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(nm) == trt.TensorIOMode.INPUT:
                self.in_name = nm
            else:
                self.out_name = nm
        in_shape = tuple(self.context.get_tensor_shape(self.in_name))
        if -1 in in_shape:
            in_shape = (1,) + in_shape[1:]
            self.context.set_input_shape(self.in_name, in_shape)
        self.in_hw = (int(in_shape[2]), int(in_shape[3]))
        self.stream = cuda.Stream()
        self._d_in = self._d_out = None

    def infer(self, x: np.ndarray) -> np.ndarray:
        cuda = self.cuda
        self.context.set_input_shape(self.in_name, x.shape)
        out_shape = tuple(self.context.get_tensor_shape(self.out_name))
        out = np.empty(out_shape, dtype=np.float32)
        if self._d_in is None:
            self._d_in = cuda.mem_alloc(x.nbytes)
            self._d_out = cuda.mem_alloc(out.nbytes)
        cuda.memcpy_htod_async(self._d_in, np.ascontiguousarray(x), self.stream)
        self.context.set_tensor_address(self.in_name, int(self._d_in))
        self.context.set_tensor_address(self.out_name, int(self._d_out))
        self.context.execute_async_v3(self.stream.handle)
        cuda.memcpy_dtoh_async(out, self._d_out, self.stream)
        self.stream.synchronize()
        return out


# ---------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------

def evaluate(backend, gt_hw, median_align: bool, data_root: str,
             max_frames: int | None) -> dict:
    in_h, in_w = backend.in_hw
    gt_h, gt_w = gt_hw

    ds = ICLGroundRobotDataset(
        root=data_root, scene=("deer", "diamond"), split="test",
        image_size=(gt_h, gt_w), augment=False,
    )
    n = len(ds) if max_frames is None else min(max_frames, len(ds))
    print(f"[eval] {n} frames de teste  entrada={in_h}x{in_w}  GT={gt_h}x{gt_w}  "
          f"median_align={median_align}")

    batch_metrics = []
    for i in range(n):
        s = ds[i]
        rgb = s["rgb"].unsqueeze(0)        # [1,3,gt_h,gt_w] em [0,1]
        gt = s["depth"].unsqueeze(0)       # [1,1,gt_h,gt_w]
        valid = s["valid"].unsqueeze(0)

        # resize p/ a entrada do modelo
        if (gt_h, gt_w) != (in_h, in_w):
            rgb_in = F.interpolate(rgb, size=(in_h, in_w),
                                   mode="bilinear", align_corners=False)
        else:
            rgb_in = rgb
        x = np.ascontiguousarray(rgb_in.numpy().astype(np.float32))

        pred = torch.from_numpy(np.ascontiguousarray(backend.infer(x)))
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if pred.shape[-2:] != (gt_h, gt_w):
            pred = F.interpolate(pred, size=(gt_h, gt_w),
                                 mode="bilinear", align_corners=False)

        batch_metrics.append(compute_depth_metrics(
            pred, gt, valid=valid, median_align=median_align))

        if (i + 1) % 50 == 0:
            print(f"[eval]   {i+1}/{n}")

    return aggregate_batch_metrics(batch_metrics)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine")
    ap.add_argument("--onnx")
    ap.add_argument("--gt_height", type=int, default=None,
                    help="resolução do GT/métrica (default = entrada do modelo).")
    ap.add_argument("--gt_width", type=int, default=None)
    ap.add_argument("--median_align", dest="median_align", action="store_true",
                    default=None)
    ap.add_argument("--no_median_align", dest="median_align", action="store_false")
    ap.add_argument("--data_root", default=str(PROJECT_ROOT / "dados" / "icl_ground_robot"))
    ap.add_argument("--max_frames", type=int, default=None,
                    help="limita nº de frames (debug). Default: split inteiro.")
    ap.add_argument("--json_out", type=Path, default=None)
    args = ap.parse_args()

    src = args.engine or args.onnx
    if not src:
        raise SystemExit("[erro] passe --engine OU --onnx.")

    backend = TrtBackend(args.engine) if args.engine else OnnxBackend(args.onnx)

    # median-align: auto p/ monodepth2 se não especificado
    median_align = args.median_align
    if median_align is None:
        median_align = "monodepth2" in Path(src).name.lower()

    gt_h = args.gt_height or backend.in_hw[0]
    gt_w = args.gt_width or backend.in_hw[1]

    metrics = evaluate(backend, (gt_h, gt_w), median_align,
                       args.data_root, args.max_frames)
    print(f"\n[eval] {Path(src).name}")
    print(f"[eval] {format_metrics(metrics)}")

    if args.json_out:
        payload = {
            "source": Path(src).name,
            "input_hw": list(backend.in_hw),
            "gt_hw": [gt_h, gt_w],
            "median_align": median_align,
            "metrics": metrics,
        }
        args.json_out.write_text(json.dumps(payload, indent=2))
        print(f"[eval] salvo em {args.json_out}")


if __name__ == "__main__":
    main()

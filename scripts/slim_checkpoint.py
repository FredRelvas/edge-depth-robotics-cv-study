#!/usr/bin/env python
"""Gera best_slim.pth (sem optimizer/scheduler) e, p/ ZoeDepth, best_zoedepth_oficial.pt."""
import sys, torch
from pathlib import Path

src = Path(sys.argv[1])
ckpt = torch.load(src, map_location="cpu", weights_only=False)
slim = ({k: v for k, v in ckpt.items()
         if k not in {"optimizer_state_dict", "scheduler_state_dict"}}
        if isinstance(ckpt, dict) else ckpt)
out = src.with_name("best_slim.pth")
torch.save(slim, out)
# cópia no formato oficial do ZoeDepth ({"model": ...}) p/ usar com load_wts/evaluate.py
if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    torch.save({"model": ckpt["model_state_dict"]}, src.with_name("best_zoedepth_oficial.pt"))
print(f"[slim] {src.stat().st_size/1e6:.0f} MB -> {out.stat().st_size/1e6:.0f} MB ({out})")

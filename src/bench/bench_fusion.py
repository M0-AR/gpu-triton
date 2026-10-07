"""EXP-02 fusion: unfused (add then relu, 20 B/elem) vs fused (12 B/elem).

Prediction before running: fused moves 40% fewer bytes -> up to 1.67x faster
in the memory-bound regime. Records bytes, timings, GB/s and speedup.
"""
from __future__ import annotations
import torch
from .common import device, time_fn, gbps
from src.kernels.fused_add_relu import triton_fused_add_relu, torch_unfused_add_relu


def run(sizes, repeats=30):
    dev = device()
    out = {"device": str(dev), "rows": []}
    for n in sorted(sizes):
        row = {"n": n,
               "unfused_ms": None,
               "unfused_gbps": None,
               "fused_ms": None, "fused_gbps": None, "speedup": None, "max_err": None,
               "status": "ok"}
        try:
            torch.manual_seed(1)
            a = torch.rand(n, device=dev, dtype=torch.float32) * 2 - 1
            b = torch.rand(n, device=dev, dtype=torch.float32) * 2 - 1
            ref = torch_unfused_add_relu(a, b)
            t_un = time_fn(lambda: torch_unfused_add_relu(a, b), repeats=repeats)
            row.update(unfused_ms=t_un["median_ms"],
                       unfused_gbps=gbps(n, 20, t_un["median_ms"]))
            if dev.type == "cuda":
                c = triton_fused_add_relu(a, b)
                err = (c - ref).abs().max().item()
                assert err == 0.0, f"fused mismatch: {err}"
                t_fu = time_fn(lambda: triton_fused_add_relu(a, b), repeats=repeats)
                row.update(fused_ms=t_fu["median_ms"], fused_gbps=gbps(n, 12, t_fu["median_ms"]),
                           speedup=t_un["median_ms"] / t_fu["median_ms"], max_err=err)
            del a, b, ref
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            row["status"] = "skipped_oom_shared_gpu"
        out["rows"].append(row)
    return out

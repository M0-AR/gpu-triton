"""EXP-01 vector-add: correctness + Triton vs torch bandwidth (GB/s).

Method (Triton 01-vector-add perf_report convention, verified 2026-10-07):
  GB/s = 3 * N * 4B / seconds  (read x, read y, write out).
"""
from __future__ import annotations
import torch
from .common import device, time_fn, gbps
from src.kernels.vector_add import triton_add, torch_add


def run(sizes, block_sizes=(1024,), repeats=30):
    dev = device()
    out = {"device": str(dev), "rows": []}
    for n in sorted(sizes):
        row = {"n": n, "torch_ms": None, "torch_gbps": None, "variants": [],
               "status": "ok"}
        try:
            torch.manual_seed(0)
            a = torch.rand(n, device=dev, dtype=torch.float32)
            b = torch.rand(n, device=dev, dtype=torch.float32)
            ref = torch_add(a, b)
            t = time_fn(lambda: torch_add(a, b), repeats=repeats)
            row["torch_ms"] = t["median_ms"]
            row["torch_gbps"] = gbps(n, 12, t["median_ms"])
            assert torch.allclose(ref, torch_add(a, b)), "torch self-check failed"
            if dev.type == "cuda":
                for bs in block_sizes:
                    c = triton_add(a, b, block_size=bs)  # correctness first
                    max_err = (c - ref).abs().max().item()
                    assert max_err == 0.0, f"triton mismatch bs={bs}: {max_err}"
                    t2 = time_fn(lambda: triton_add(a, b, block_size=bs), repeats=repeats)
                    row["variants"].append({"block": bs, "ms": t2["median_ms"],
                                            "gbps": gbps(n, 12, t2["median_ms"]),
                                            "max_err": max_err})
            del a, b, ref
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            row["status"] = "skipped_oom_shared_gpu"
        out["rows"].append(row)
    return out

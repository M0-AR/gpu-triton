"""Shared timing + device helpers. All GPU timings synchronize (else ~0ms lies)."""
from __future__ import annotations
import time
import torch


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def time_fn(fn, repeats: int = 30, warmup: int = 10) -> dict:
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = []
    use_cuda = torch.cuda.is_available()
    for _ in range(repeats):
        if use_cuda:
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            torch.cuda.synchronize()
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            dt.append(s.elapsed_time(e))
        else:
            t0 = time.perf_counter()
            fn()
            dt.append((time.perf_counter() - t0) * 1e3)
    dt.sort()
    return {"median_ms": dt[len(dt) // 2], "min_ms": dt[0], "max_ms": dt[-1]}


def gbps(n_elem: int, bytes_per_elem: int, ms: float) -> float:
    return n_elem * bytes_per_elem * 1e-9 / (ms * 1e-3) if ms > 0 else 0.0

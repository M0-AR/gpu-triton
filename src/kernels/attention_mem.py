"""Attention memory scaling (Lecture 1, Part D): standard N^2 vs Flash O(N).

Standard attention must materialize the N x N score grid (lecture capacity
figures: 64 MiB @1k, 1 GiB @4k, 16 GiB @16k, 64 GiB @32k for 32 fp16 heads —
binary GiB, ONE resident N x N matrix). End-to-end traffic is ~4 full-matrix
passes (write S, reread S + write P, reread P for O=PV): 256 GiB/layer @32k.
FlashAttention (Dao et al. 2205.14135; FlashAttention-2 work partitioning):
  never writes N x N; one output-sized write + running (m, l) statistics.

  std_extra_bytes(N, H)   = N^2 * H * bytes_per_score   (resident grid)
  flash_extra_bytes(N, H) = ~one SRAM tile (does not grow with N)
"""
from __future__ import annotations


def std_attention_extra_bytes(n: int, heads: int = 32, bytes_per_score: int = 2) -> int:
    return n * n * heads * bytes_per_score


def flash_attention_extra_bytes(n: int, heads: int = 32, tile: int = 1024) -> int:
    # One resident score tile + O(heads*d) statistics; independent of N.
    _ = (n, heads)
    return 2 * tile * tile


def std_traffic_bytes(n: int, d: int = 128, heads: int = 32,
                      bytes_per_score: int = 2, passes: int = 4) -> int:
    # ~4 full-matrix passes over the N x N grid per layer (lecture's figure).
    return passes * n * n * heads * bytes_per_score

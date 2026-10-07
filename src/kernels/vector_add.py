"""Canonical Triton vector-add (Lecture 1: C = A + B).

Source of truth: https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html
(verified 2026-10-07 via websearch + DuckDuckGo-lite fallback; mirrors the
triton-lang/triton 01-vector-add.py tutorial and the torch.compile recipe).

Kernel semantics are intentionally UNCHANGED from the tutorial so this repo
tests the lecture, not a variant of it:
  pid = program_id(0); offsets = pid*BLOCK + arange(BLOCK); mask = offsets < n
  x,y = load(masked); store(x+y, masked); grid = cdiv(n, BLOCK).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(out_ptr + offsets, output, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    assert x.device == y.device, "inputs must share a device"
    out = torch.empty_like(x)
    n = out.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, out, n, BLOCK_SIZE=block_size)
    return out


def torch_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x + y

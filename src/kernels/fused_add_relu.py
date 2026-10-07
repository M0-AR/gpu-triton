"""Fused add+ReLU: y = relu(a + b) in ONE kernel (Lecture 1, Part C: fusion).

Unfused baseline (plain PyTorch, 2 kernels):
  t = a + b        # kernel 1: read a,b (8B) + write t (4B)
  y = relu(t)      # kernel 2: read t (4B) + write y (4B)  -> 20 B/elem total
Fused (this file, 1 kernel):
  y = relu(a+b) with t kept in a register -> 12 B/elem total (40% less traffic).

Expected ceiling speedup from traffic alone: 20/12 = 1.667x (memory-bound regime).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def fused_add_relu_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    t = a + b  # stays in registers, never touches HBM
    y = tl.maximum(t, 0.0)
    tl.store(out_ptr + offsets, y, mask=mask)


def triton_fused_add_relu(a: torch.Tensor, b: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    out = torch.empty_like(a)
    n = out.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    fused_add_relu_kernel[grid](a, b, out, n, BLOCK_SIZE=block_size)
    return out


def torch_unfused_add_relu(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.relu(a + b)

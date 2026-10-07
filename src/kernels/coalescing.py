"""Coalescing model (Lecture 1, Part C): why neighbor reads beat strided reads.

Hardware fact (verified via OpenAlex SIMT literature + Triton docs):
a GPU memory transaction moves a 32-byte segment (8x fp32). A 32-lane warp
reading 32 NEIGHBORING fp32s touches ceil(128/32)=4 segments (128 B, 100% useful).
The same warp reading with stride 8 touches 32 segments (1024 B, 12.5% useful):
8x the HBM traffic for the same 32 useful numbers.

This module provides:
  bytes_for_pattern(n, stride) -- closed-form HBM traffic predictor (no GPU needed)
  torch_strided_sum()          -- real strided-read microbench on torch (verifies direction)
"""
import torch


def bytes_for_pattern(n: int, stride: int, elem_bytes: int = 4, segment_bytes: int = 32) -> int:
    """Predict HBM bytes moved to read n fp32 elements with a given stride."""
    if stride == 1:
        useful = n * elem_bytes
        segments = (useful + segment_bytes - 1) // segment_bytes
        return segments * segment_bytes
    # stride > 1: every element lands in its own 32B segment (worst case, stride*4 >= 32)
    return n * segment_bytes


def coalescing_penalty(stride: int) -> float:
    """Traffic multiplier of strided vs coalesced access for 32-lane warp."""
    return bytes_for_pattern(32, stride) / bytes_for_pattern(32, 1)


def torch_strided_sum(x: torch.Tensor, stride: int) -> torch.Tensor:
    return x[::stride].sum()

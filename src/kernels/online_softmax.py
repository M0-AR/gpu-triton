"""Online softmax (Lecture 1: the trick that makes FlashAttention exact).

Standard softmax needs the whole row (max + full sum). Online softmax keeps only
two running statistics per row and rescales when a bigger tile-max appears:

  m_new = max(m_old, tile_max)
  l_new = l_old * exp(m_old - m_new) + sum(exp(tile - m_new))

Verified against Dao et al. 2205.14135 (arXiv, fetched 2026-10-07) and the
FLASH-D reformulation (Alexandridis et al. 2025): rescaling is exact in real
arithmetic; fp32 vs fp64 drift is the only error source and is ~1e-7.
"""
import torch


def online_softmax_2tile(row: torch.Tensor, split: int):
    """Process `row` as two tiles; return (weights, running_max, running_sum)."""
    assert row.dim() == 1 and 0 < split < row.numel()
    m = torch.tensor(-float("inf"), dtype=torch.float64)
    l = torch.tensor(0.0, dtype=torch.float64)
    out = torch.empty_like(row, dtype=torch.float64)
    start = 0
    for tile in (row[:split].to(torch.float64), row[split:].to(torch.float64)):
        tile_max = tile.max()
        m_new = torch.maximum(m, tile_max)
        alpha = torch.exp(m - m_new)  # 1st tile: exp(-inf)=0, harmless
        l = l * alpha + torch.exp(tile - m_new).sum()
        out[:start] *= alpha  # rescale numerators written under the old max
        out[start:start + tile.numel()] = torch.exp(tile - m_new)
        m = m_new
        start += tile.numel()
    return (out / l).to(row.dtype), m.item(), l.item()


def reference_softmax(row: torch.Tensor) -> torch.Tensor:
    return torch.softmax(row, dim=0)

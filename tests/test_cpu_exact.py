import torch
from src.kernels.coalescing import bytes_for_pattern, coalescing_penalty
from src.kernels.attention_mem import std_attention_extra_bytes, flash_attention_extra_bytes
from src.kernels.online_softmax import online_softmax_2tile, reference_softmax
from src.bench.bench_waves import waves


def test_coalescing_math():
    assert bytes_for_pattern(32, 1) == 128
    assert bytes_for_pattern(32, 8) == 1024
    assert coalescing_penalty(8) == 8.0


def test_attention_scales_quadratically():
    a = std_attention_extra_bytes(1024)
    b = std_attention_extra_bytes(4096)
    assert b / a == 16.0
    assert flash_attention_extra_bytes(32768) == flash_attention_extra_bytes(1024)
    assert flash_attention_extra_bytes(32768) < std_attention_extra_bytes(32768) / 1e3


def test_online_softmax_exact():
    # Seeds 0..7: covers both orders (2nd-tile max bigger AND smaller).
    # Regression: an early version forgot to rescale earlier numerators and
    # failed seed 7 with err 0.71 while passing seed 0 -- one seed is not enough.
    for seed in range(8):
        torch.manual_seed(seed)
        row = torch.randn(512) * 3
        w, _, _ = online_softmax_2tile(row, 200)
        err = (w - reference_softmax(row)).abs().max().item()
        assert err < 1e-6, (seed, err)


def test_wave_math():
    w = waves(20, 4, 2)
    assert w["waves"] == 3 and w["last_wave_util"] == 0.5

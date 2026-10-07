"""EXP-05 attention memory scaling: verify the N^2 blowup numerically.

Lecture's figures (32 heads, fp16 scores, 80 GB HBM):
  N=1024 -> 64 MB | N=4096 -> 1 GB | N=16384 -> 16 GB | N=32768 -> 64 GB.
Assert ratio scaling = 16x per 4x N (quadratic) and flash stays flat (~0).
Also verifies online-softmax exactness on a random row (max err < 1e-6).
"""
from __future__ import annotations
import torch
from src.kernels.attention_mem import (std_attention_extra_bytes, flash_attention_extra_bytes,
                                       std_traffic_bytes)
from src.kernels.online_softmax import online_softmax_2tile, reference_softmax


def run(lengths=(1024, 4096, 16384, 32768), heads=32):
    rows = []
    for n in lengths:
        std_B = std_attention_extra_bytes(n, heads)
        fl_B = flash_attention_extra_bytes(n, heads)
        rows.append({"n": n, "std_GiB": std_B / 2**30, "flash_KiB": fl_B / 2**10,
                     "traffic_GiB": std_traffic_bytes(n, heads=heads) / 2**30})
    # quadratic check: 1024->4096 must be exactly 16x
    assert rows[1]["std_GiB"] / rows[0]["std_GiB"] == 16.0
    # lecture spot values (GiB, binary): 1/16, 1, 16, 64
    assert abs(rows[0]["std_GiB"] - 0.0625) < 1e-9
    assert abs(rows[1]["std_GiB"] - 1.0) < 1e-9
    assert abs(rows[2]["std_GiB"] - 16.0) < 1e-9
    assert abs(rows[3]["std_GiB"] - 64.0) < 1e-9
    # lecture traffic figure: ~256 GiB/layer @32k over 4 passes
    assert abs(rows[3]["traffic_GiB"] - 256.0) < 1e-9
    # online softmax exactness
    torch.manual_seed(7)
    row = torch.randn(512) * 3
    w_online, _, _ = online_softmax_2tile(row, 200)
    err = (w_online - reference_softmax(row)).abs().max().item()
    assert err < 1e-6, err
    return {"rows": rows, "online_softmax_max_err": err,
            "lecture_claim": "10x tokens -> 100x scores; 32k tokens ~ most of 80 GB"}

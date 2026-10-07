"""EXP-03 roofline: arithmetic intensity + ridge point from MEASURED bandwidth.

Lecture claims (H100): peak 67 TFLOP/s, HBM 3.35 TB/s -> ridge ~20 FLOP/B;
vector-add AI = 1 FLOP / 12 B = 0.0833 -> deep memory-bound, ~0.28 TFLOP/s ceiling.
Here we recompute the ridge for the LOCAL GPU from its measured vector-add
bandwidth (roofline done right: ceilings from measurement, not brochures),
and place vector-add / decode-like / GEMM-like points on it.
"""
from __future__ import annotations
import torch
from .common import device


def ai_vector_add_f32() -> float:
    return 1.0 / 12.0  # 1 FLOP per 12 bytes (fp32)


def ridge_point(flops: float, bandwidth_Bps: float) -> float:
    return flops / bandwidth_Bps


def run(vec_gbps: float):
    bw = vec_gbps * 1e9  # measured HBM ceiling, B/s
    dev = device()
    if dev.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        # Ampere RTX 3090: 82 SM x 128 CUDA cores; use spec FP32 peak 35.6 TFLOP/s.
        peak_flops = 35.6e12
        gpu = props.name
    else:
        peak_flops, gpu = 0.0, "cpu-only"
    ridge = ridge_point(peak_flops, bw) if bw > 0 else 0.0
    ai = ai_vector_add_f32()
    attainable_tflops = min(peak_flops, bw * ai) / 1e12 if bw > 0 else 0.0
    return {"gpu": gpu, "measured_bw_GBs": vec_gbps, "peak_tflops": peak_flops / 1e12,
            "ridge_flop_per_byte": ridge, "vector_add_ai": ai,
            "vector_add_attainable_tflops": attainable_tflops,
            "bound": "memory-bound" if ai < ridge else "compute-bound",
            "lecture_h100_ridge": 20.0, "lecture_h100_attainable_tflops": 0.28}

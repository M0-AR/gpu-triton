# METHOD — how each lecture minute maps to one runnable check

| Lecture segment | File | What it proves |
|---|---|---|
| `C = A+B`, 1 kernel, 1M parallel adds | `src/kernels/vector_add.py` + `src/bench/bench_vector_add.py` | grid=`cdiv`, mask tail, max_err 0.0 |
| Latency vs throughput (car/bus) | `src/bench/bench_vector_add.py` small-N rows | 4k: 16 GB/s (launch-bound) → 16M: 836 GB/s (bandwidth-bound) |
| SIMT warp, 32 lanes, one instruction | `src/kernels/coalescing.py` | 32 B segments → 128 B vs 1024 B |
| Blocks/grid/waves/tail | `src/bench/bench_waves.py` | 20 blocks on 4 SM×2 slots = 3 waves, 50% tail |
| `i = blockIdx.x*blockDim.x + threadIdx.x` + mask | `test_gpu_kernels.py` N=98,432 case | ragged tail exact |
| HBM vs PCIe (copy once) | README §7 vendor vote | 3.35 TB/s vs 64 GB/s ≈ 52× |
| Roofline, AI=1/12, ridge 20 | `src/bench/bench_roofline.py` | recomputed from measured BW |
| Fusion 20 B → 12 B | `src/kernels/fused_add_relu.py` + `bench_fusion.py` | 1.677× @16.7M |
| Online softmax + tiling | `src/kernels/online_softmax.py` + `bench_attention.py` | err 1.19e-07; N² table exact |
| Live-data leg | `src/bench/bench_market.py` + `scripts/fetch_live_data.py` | 0.0 on BTC/AAPL |

Conventions: CUDA-event timing, warmup 10, median 30, synchronized;
`triton.cdiv` grids; power-of-two BLOCKs; fp32; seed-pinned synthetic legs,
timestamped live legs. Every assert in `tests/` and `src/bench/` is a
falsifiable lecture sentence.

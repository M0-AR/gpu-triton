# From One Line of Python to FlashAttention: An Independent Verification of Visual GPU Lecture 1 (Draft — submitted as preprint seed)

## Abstract
We independently verify every empirically checkable claim of a widely viewed
visual introduction to GPU programming (Triton series, Lecture 1). Using the
canonical Triton vector-add kernel (unchanged semantics), CUDA-event timing,
and closed-form memory models cross-checked against vendor specifications and
the primary literature (Dao et al. 2022; Williams–Waterman–Patterson roofline),
we confirm six of six core claims on an NVIDIA RTX 3090 and re-confirm kernel
exactness on live public market data (Bitcoin, Apple Inc.). Bandwidth-saturated
fusion speedup matches traffic theory to 0.6% (1.677× measured vs 1.667×
predicted). We further report three phenomena the lecture omits — a fusion
crossover valley (0.73× at 65k elements), non-monotonic BLOCK_SIZE response
(2.6× cost of guessing wrong at N=10⁶), and a ≈3 µs launch-overhead floor —
each with a falsifiable model proposed as future work.

## 1. Introduction
Visual GPU pedagogy reaches millions, yet its quantitative claims (bandwidth
ratios, balance points, scaling laws) are rarely re-measured. We treat Lecture
1 as a testable artifact: §2 lists its claims; §3 gives the verification
method (triangulated across 12 independent sources, 2026-10-07); §4 reports
measurements; §5 states threats; §6 proposes three follow-up studies.

## 2. Claims under test
C1 (SIMT vector-add exactness) · C2 (coalescing 8× traffic ratio) · C3 (fusion
1.67× ceiling) · C4 (roofline placement AI=1/12, ridge 20 FLOP/B on H100) ·
C5 (attention N² capacity/traffic) · C6 (online-softmax exactness) ·
H1–H3 (H100 organization figures). See README §2 for the verdict table.

## 3. Method
Kernels: canonical Triton 01-vector-add + minimal fused add-ReLU delta.
Timing: CUDA events, 10 warmup + median of 30, synchronized; GB/s convention
3·N·4B/s (Triton perf_report). Controls: torch reference per size,
multi-seed exactness (8 seeds after a seed-0-only false pass), OOM recorded
not hidden, live-data leg (CoinGecko/Yahoo, timestamped snapshots in `data/`).
Literature triangulation log: 12 sources with dates in README §7.

## 4. Results
(*paste README §3 tables — they are generated from `results/summary.json`,
commit hash of this draft.*)
Key numbers: torch≡Triton @1M (722.8 GB/s); peak 836.6 GB/s (89% of spec);
fusion 1.677× @16.7M; ridge 42.6 FLOP/B (Ampere) vs lecture's 20 (Hopper);
attention 1/16/1/16/64 GiB + 256 GiB traffic @32k reproduced exactly;
online-softmax err 1.19e-07; market legs bit-exact (0.0).

## 5. Threats to validity
Shared-GPU contention (≈21.6 GiB held by another process; absolute GB/s are
lower bounds); single-card family (Ampere only — Hopper timing transfer is
explicitly NOT claimed); FP32 only (tensor-core 15× figure untested);
mid-size anomalies reported-not-explained; live snapshots frozen 2026-10-07
(re-fetch via `scripts/fetch_live_data.py`).

## 6. Three paper seeds
P1 fusion-crossover model N*(launches, BLOCK, SMs). P2 BLOCK×warps×stages
sweep Ampere-vs-Hopper. P3 launch-bound third roofline ceiling. Each is
scoped to one falsifiable experiment in `configs/experiments.json`.

## References
Dao et al., FlashAttention (NeurIPS 2022, arXiv:2205.14135); Dao,
FlashAttention-2 (2023); Alexandridis et al., FLASH-D (2025); Williams,
Waterman, Patterson, Roofline (CACM 2009); Leinhauser et al., Instruction
Roofline for AMD GPUs (2021); Ghane et al., Power/Energy Roofline (2018);
Qian et al., Sparsity-Aware Roofline (2026); Patel & Singh, AI Accelerators &
Memory Wall (2026); NVIDIA H100 datasheet + Hopper tuning guide; Triton
01-vector-add tutorial + 2025 dev-summit perf-CI notes; CLAIRE-Labo
reproducible-ML template.

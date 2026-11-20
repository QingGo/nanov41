# Training-kernel status: time breakdown, ceilings, overlap, and what is left

Status note (2026-11-06), based on the R69 W2 measurements
([`benchmarks.md`](benchmarks.md) §3 R69, `runs/r69_mfu_5090.json`) and the R55 tail census.
Shape: d12, b=16 × seq=1024, graph replay + optimizer outside the graph, torch 2.14.1+cu130,
tilelang 0.1.15. The headline: **raw bf16 MFU is already 59.4% (raw-60% wall ≈ reached); the
project's primary metric is discounted MFU, and that is a different, much harder target.**

## 1. Where the time goes (device side)

R69 step = **57.5 ms** (graph replay + optimizer). CUDA-graph replay alone = **54.3 ms / 2847
kernels**; the optimizer is **≈4.4 ms (7.6%)** outside the graph (`--skip-opt` 53.75 vs 58.14 ms),
and the remaining step−device gap is ~3 ms of optimizer/launch tail.

| segment | ms (device ≈54.3) | share | type | headroom / ceiling |
|---|---:|---:|---|---|
| head FP8 (K1 fwd GEMM 5.3, K3 dW 4.7, K2 dX/CE 4.2) | **13.8** | 25% | compute (fp8) | ~83% of fp8 peak already; ≤0.5 ms via epilogue/quant fusion (FP4 head tested: no win at d12) |
| MoE (dw13 3.3, swiglu 1.6+1.1, aggemm 1.6+0.8, dw_reduce 1.5, dw 1.5, tail/dws ~0.6) | **~10** | 18% | compute (bf16) | 78–84% of bf16 peak; ~0.4–0.6 ms via deterministic counting-sort router (argsort 0.58 ms), dw-reduce fusion |
| attention (TileLang kernels x47 4.95 + rope_tail 0.87 + reduce) | **~5.3** | 10% | memory-bound (~19% peak) | ~0.3–0.5 ms via dkv_win query-stationary; one fewer sort (CUB 0.83 ms) |
| mHC (post_bwd 2.58, mixes_pre_bwd 1.65, post_fwd 1.42, mixes_pre_fwd 0.8, sinkhorn 0.36) | **~7.8** | 14% | bandwidth-floor | ~0.2–0.4 ms only with cross-layer fusion (tried, slower) / grad-accum fold |
| small cutlass GEMMs (60× 64x64x32, 61× tn, 37× wmma, 36× 64x128x64, …) | **~6.6** | 12% | launch/latency-bound | projection fusion was **negative** (R56/R69); ~0.3–0.8 ms only via shape/batch packing |
| elementwise + copy + sort (100× vectorized add 2.13, copies 0.88, CUB sorts 0.83, misc 0.6) | **~4.4** | 8% | bandwidth/launch | ~0.5–1.0 ms via fused cats/copies + a custom deterministic scatter; the autograd adds are hard to remove |
| rms + rope fusion (rms_fwd/bwd, rope) | **~1.3** | 2% | bandwidth | already fused; ~0.1 ms |
| misc / indexer / launch tail | **~4.8** | 9% | mixed | ~0.3–0.5 ms by kernel-count reduction |
| **optimizer (outside graph)** | **4.4** | 7.6% | bandwidth/ALU | 1–2 ms: single-pass fused/capturable AdamW (bf16 states or capturable step), or in-graph with a fixed parity protocol |

## 2. Ceilings

- **Compute floor** (100% peak, all GEMMs): total training FLOPs/step 6.945 TFLOP.
  - all-bf16 peak 209.5 TF ⇒ **33.2 ms**;
  - with the FP8 head (≈70% of FLOPs at 2× peak) ⇒ **≈21.6 ms** mix floor.
- **Memory floor**: the non-GEMM tail (mHC, attention, elementwise, sorts, copies) is bandwidth-bound;
  R55/R58 estimated the practical 5090 floor for this mix at **42–46 ms** (~48–53% discounted MFU).
  R69 device is 54.3 ms ⇒ ~20–25% above that floor; ~8–12 ms of the gap is structural (mHC
  residual passes, attention bytes, autograd adds, launch tail).
- **Raw MFU**: 59.4% — essentially the raw-60% target (55.25 ms). **Discounted MFU** (project
  primary metric, α=1/0.5/0.25) is only ≈38.5% for the current mix (FP8 head share ~70%), nowhere
  near 60%; reaching 60% discounted on sm_120 was already judged unreachable by tiling/fusion alone
  (R58) and would need FP8/FP4 linears (negative at nano GEMM shapes in R51/R54) or a platform/
  algorithmic change.

## 3. Overlap and launch analysis (what is not overlapping)

- The training graph runs on **one stream**: compute-bound GEMMs and memory-bound mHC/attention/
  elementwise phases execute **sequentially**, no compute↔memory overlap.
- CUDA graph removes the per-launch **CPU** cost (eager 113 ms → 54.3 ms device), but **device-side
  dispatch** for 2847 kernels still costs ~1–2 µs each, i.e. potentially **3–6 ms/step** of launch
  tail; the 54.3 ms "device" number already contains it (sum of kernel durations), so the only way to
  recover it is fewer kernels or better batching.
- Optimizer is outside the graph and serialized after the last backward; with 4× gradient
  accumulation it runs once per 4 micro-batches, amortizing its 4.4 ms to ≈1.1 ms/16K tokens.
- **Next diagnostic** (cheap): one Nsight Systems timeline of a graph replay (and of eager) to get
  real launch gaps, stream utilization, DRAM read/write throughput per phase, and whether any kernel
  is idle-waiting. ~1 h GPU (≈¥3). This decides whether fusion work is worth it.

## 4. Realistic remaining training wins

| item | expected | effort | verdict |
|---|---:|---|---|
| optimizer: single-pass fused/capturable AdamW (or in-graph with parity protocol) | **1–2 ms** | 2–3 days | best ROI |
| head epilogue: fwd/dW accumulate in-kernel + fused weight amax/abs/quant (saves add_ 0.54 + quant ~0.5) | 0.3–0.5 ms | 2–3 days | do if staying |
| deterministic counting-sort router + scatter (removes CUB argsort 0.58 / sort 0.83) | 0.4–0.6 ms | 3–5 days | medium |
| attention dkv_win query-stationary + fewer launches | 0.3–0.5 ms | 3–5 days | medium |
| kernel-count reduction: merge copies/cats/elementwise tails | 0.5–1.0 ms | 1–2 weeks | risky |
| mHC cross-layer fold | 0.2–0.4 ms | 1 week; earlier attempt slower | skip |
| total | **≈3–5 ms** (step ~53–55 ms, raw MFU ~62–64%) | 2–4 weeks | — |

**Conclusion for the training path**: after R69 the remaining kernel work is high-effort/low-ROI
(~5–8% for weeks of GPU-kernel engineering), the raw-60% goal is effectively reached, and the
project's discounted-MFU target cannot be met by more tiling. Under the OPD plan
([`opd-strategy.md`](opd-strategy.md)) the student trains on only a few B tokens, so training-kernel
savings are worth ≈¥5–10 in total — **redirect effort to the inference/rollout engine and
long-context (262K 117k tok/s, 1M 98k tok/s) paths**, where wall-clock and cost actually scale with
the product.

## 5. If we do one last training pass

1. Nsight timeline (~1 h GPU) to quantify launch gaps and DRAM utilization.
2. Optimizer pass (biggest single win, also cuts the step-level tail).
3. Head epilogue fusion (safe, local).
4. Deterministic sort/scatter replacement (also helps graph parity culture).
Stop there; do not reopen projection fusion, MoE FP8, or mHC cross-layer fusion (all measured
negative).

# runs/ — curated benchmark evidence

Only a small, hand-picked subset of benchmark output is version-controlled (see `.gitignore`).
Files fall into three groups: W1 step breakdown (`bench_mfu.py`, p0/p1), W2 best training
steps (internal graph-training driver, r42/r48/r49/r50), and the 2026-10-28 GPU re-validation
(NVFP4/FP8 bench + CUDA parity logs).

| File | Config | Shape | Key numbers |
|---|---|---|---|
| `p0_b8.json` | d12, bf16, fused head + shift CE, MoE loop | b=8, seq=1024 | 236.93 ms/step, 34,576 tok/s, 7.0% bf16 MFU, device-busy 85.1% |
| `p1_b16_s512.json` | d12, bf16, fused head + shift CE, MoE loop | b=16, seq=512 | 199.65 ms/step, 41,031 tok/s, 8.3% bf16 MFU, device-busy 84.3% |
| `p0_b8_5090_rerun_2026-10-28.json` | same command as `p0_b8.json`, fresh 5090 host | b=8, seq=1024 | 275.64 ms/step, 29,720 tok/s, 6.0% bf16 MFU, device-busy 54.6% — host-CPU-bound step, hence slower than the original run |
| `tl_kernels_5090_2026-10-28.json` | `experiments/tl_kernels/bench.py` (NVFP4/FP8 kernels + correctness) | head 8192×129280×384 | kernel-only 1.66–1.72 ms / 632–655 TFLOPS, quantizer 0.53 ms (17.3–17.7×), FP8 cast 2.10 ms (2.3×); FP8 bytes 100% equal |
| `parity_cuda_fp32_5090.log` | `parity.py --dtype float32` on CUDA | tiny_all_topk | ALL PASS (T1 2.384e-07, T2 0.0, T3 PASS) |
| `parity_cuda_bf16_5090.log` | `parity.py --dtype bf16` on CUDA | tiny_all_topk | ALL PASS (T1 1.788e-07, T2 0.0, T3 PASS) |
| `r42_best_final.json` | W2 graph-training, first best | b=16, seq=1024 | 92.08 ms/step, 75.43 TFLOP/s, 36.0% bf16-MFU |
| `r48_best_b16.json` | W2 graph-training best | b=16, seq=1024 | 60.92 ms/step, 114.01 TFLOP/s, 54.42% bf16-MFU |
| `r50_base.json` | R49 best re-measured (R50 baseline) | b=16, seq=1024 | 60.94 ms/step, 113.96 TFLOP/s, 54.40% bf16-MFU |
| `r50_best_b16.json` | **W2 best** (+ MoE NOPAD), 3-step parity 0.0 | b=16, seq=1024 | **60.43 ms/step, 114.93 TFLOP/s, 54.86% bf16-MFU** |
| `r50_best_b8.json` | W2 best at b=8 | b=8, seq=1024 | 34.87 ms/step, 99.58 TFLOP/s, 47.53% bf16-MFU |
| `r50_fp8.json` | MoE fp8 v1 (negative result) | b=16, seq=1024 | 60.94 ms/step, 54.40% — no gain vs bf16 |
| `r51_fp4_expert_microbench_5090.json` | R51: MXFP4-QAT expert weights + FP8 compute / FP8 wgrad microbench | d12 expert shapes + full V4.1 expert shapes | nano shapes: FP8 not faster (0.0103 vs 0.0083 ms graph-replay wgrad); full V4.1 shapes: FP8 1.76–2.43× faster; 30-step QAT loss 0.2499→0.1486 vs bf16 →0.1412 |
| `r52_grouped_probe_5090.json` | R52 Step-0: grouped/batched execution probe (grouped vs low precision) | 1 layer × 16 experts, nano + full V4.1 shapes | nano: eager 36–45 TF → graph/bmm 83–105 TF (2.0–2.3×); full-scale: no gain (already 206 TF); `torch._scaled_grouped_mm` FP8 refuses sm_120 |
| `r54_precision_mix_5090.json` | R54 decision gate: FP8 vs bf16 head × batch size | d12, b=16–48, seq=1024 | FP8 head b=16 61.12 ms/54.24% raw/35.18% discounted; all-bf16 86.32 ms/38.41%; b≥24 FP8 falls back (int32 TP·V limit); batch asymptote ~41% discounted |
| `r55_tail_census_5090.json` | R55 residual-tail census + replacement candidates | d12 b=16, R50-best graph | device 57.63 ms = head 13.93 + MoE 9.88 + mHC 9.48 + attention v2 5.70 + small GEMMs 6.47 + elementwise 5.80 + rms/rope 1.52 + misc 4.85; official mHC slower; official sparse_attn fwd-only (no bwd); head tiles default best |
| `r55_tail_census_5090.json` | R55 residual-tail census + replacement candidates | d12 b=16, R50-best graph | device 57.63 ms = head 13.93 + MoE 9.88 + mHC 9.48 + attention v2 5.70 + small GEMMs 6.47 + elementwise 5.80 + rms/rope 1.52 + misc 4.85; official mHC slower; official sparse_attn fwd-only (no bwd); head tiles default best |

Interpretation and caveats: [`../docs/benchmarks.md`](../docs/benchmarks.md). Other JSON files
that may exist in this directory locally are experiment output and are not part of the repository.

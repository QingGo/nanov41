# runs/ — curated benchmark evidence

Only a small, hand-picked subset of benchmark output is version-controlled (see `.gitignore`).
Each JSON file is produced by `bench_mfu.py --json <path>` on a single RTX 5090
(torch 2.12.1+cu130) and records the per-component timing / MFU breakdown plus the top kernels.

| File | Config | Shape | Key numbers |
|---|---|---|---|
| `p0_b8.json` | d12, bf16, fused head + shift CE, MoE loop | b=8, seq=1024 | 236.93 ms/step, 34,576 tok/s, 7.0% bf16 MFU, device-busy 85.1% |
| `p1_b16_s512.json` | d12, bf16, fused head + shift CE, MoE loop | b=16, seq=512 | 199.65 ms/step, 41,031 tok/s, 8.3% bf16 MFU, device-busy 84.3% |
| `p0_b8_5090_rerun_2026-10-28.json` | same command as `p0_b8.json`, fresh 5090 host | b=8, seq=1024 | 275.64 ms/step, 29,720 tok/s, 6.0% bf16 MFU, device-busy 54.6% — host-CPU-bound step, hence slower than the original run |
| `tl_kernels_5090_2026-10-28.json` | `experiments/tl_kernels/bench.py` (NVFP4/FP8 kernels + correctness) | head 8192×129280×384 | kernel-only 1.66–1.72 ms / 632–655 TFLOPS, quantizer 0.53 ms (17.3–17.7×), FP8 cast 2.10 ms (2.3×); FP8 bytes 100% equal |
| `parity_cuda_fp32_5090.log` | `parity.py --dtype float32` on CUDA | tiny_all_topk | ALL PASS (T1 2.384e-07, T2 0.0, T3 PASS) |
| `parity_cuda_bf16_5090.log` | `parity.py --dtype bf16` on CUDA | tiny_all_topk | ALL PASS (T1 1.788e-07, T2 0.0, T3 PASS) |

Interpretation and caveats: [`../docs/benchmarks.md`](../docs/benchmarks.md). Other JSON files
that may exist in this directory locally are experiment output and are not part of the repository.

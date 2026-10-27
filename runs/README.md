# runs/ — curated benchmark evidence

Only a small, hand-picked subset of benchmark output is version-controlled (see `.gitignore`).
Each JSON file is produced by `bench_mfu.py --json <path>` on a single RTX 5090
(torch 2.12.1+cu130) and records the per-component timing / MFU breakdown plus the top kernels.

| File | Config | Shape | Key numbers |
|---|---|---|---|
| `p0_b8.json` | d12, bf16, fused head + shift CE, MoE loop | b=8, seq=1024 | 236.93 ms/step, 34,576 tok/s, 7.0% bf16 MFU, device-busy 85.1% |
| `p1_b16_s512.json` | d12, bf16, fused head + shift CE, MoE loop | b=16, seq=512 | 199.65 ms/step, 41,031 tok/s, 8.3% bf16 MFU, device-busy 84.3% |

Interpretation and caveats: [`../docs/benchmarks.md`](../docs/benchmarks.md). Other JSON files
that may exist in this directory locally are experiment output and are not part of the repository.

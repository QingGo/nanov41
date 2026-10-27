# Benchmarks

All numbers below were produced by this repository's own scripts. Raw JSON evidence for the
GPU runs is checked in under [`runs/`](../runs/). Absolute throughput is sensitive to
torch/CUDA/driver versions and host CPU contention — compare within a run, not across machines.

## 1. CPU / Apple MPS parity and smoke tests

Measured 2026-10-27 on macOS arm64: Python 3.13.2, torch 2.9.0, 4 CPU threads, MPS available,
no CUDA.

| Command | Result | Wall time |
|---|---|---|
| `parity.py --dtype float32` | ALL PASS (T1 2.123e-07, T2 0.0, T3 680040 grads / 95.6% non-zero / 0 NaN) | ~6 s |
| `parity.py --dtype bf16` | ALL PASS (T1/T2 = 0.0) | ~16 s |
| `parity.py --dtype float32 --device mps` | ALL PASS (T1 2.384e-07) | ~117 s (first run) |
| `train.py --config tiny_all_topk --steps 100 ...` | loss 6.2230 → 5.3106 | 4.8 s (~2,650 tok/s) |
| `train.py --config d8 --steps 0` | 81.62M params (15.43M non-embedding) | ~6 s |
| `train.py --config d12 --steps 0` | 193.41M params (94.12M non-embedding) | ~7 s |

The CPU training smoke test is deterministic: two runs produce bit-identical loss curves.

## 2. RTX 5090 training breakdown (d12, bf16, torch 2.12.1+cu130)

`bench_mfu.py` with the default fused head + shift CE, per-component timing and MFU against
bf16 peak. Raw results: [`runs/p0_b8.json`](../runs/p0_b8.json),
[`runs/p1_b16_s512.json`](../runs/p1_b16_s512.json).

| Config | b × seq | tokens/step | step time | tokens/s | achieved TFLOPS | bf16 MFU | device-busy |
|---|---|---|---|---|---|---|---|
| d12, attention ref / MoE loop | 8 × 1024 | 8,192 | 236.93 ms | 34,576 | 14.66 | 7.0% | 85.1% |
| d12, attention ref / MoE loop | 16 × 512 | 8,192 | 199.65 ms | 41,031 | 17.39 | 8.3% | 84.3% |

Each step launches ~8.5k–9.9k kernels, so these runs are partly **CPU-launch bound**, not tensor-core
bound. That is the motivation for the fused/chunked head, grouped MoE GEMM and CUDA-graph work
tracked for W2; those experiments live outside this repository and are not part of the public
release (W1/W2 kernel-parity checks that are CPU-runnable are in [`tests/`](../tests/)).

Absolute peak used for the MFU denominator: **bf16, 5090 vendor peak (165.2 TFLOPS)**. Throughput
figures should be compared between variants measured in the same interleaved run
(`tools/ab_bench.py`), not across sessions.

**Host-CPU sensitivity (re-run 2026-10-28).** A fresh run of the same command on a different 5090
host gave **275.6 ms/step, 29,720 tok/s, device-busy 54.6%** (vs 236.93 ms / 34,576 tok/s /
85.1% above): the step is CPU-launch bound, so the host CPU (and its contention) changes the result
by ~15%. Both runs are kept as evidence — [`runs/p0_b8.json`](../runs/p0_b8.json) (original) and
[`runs/p0_b8_5090_rerun_2026-10-28.json`](../runs/p0_b8_5090_rerun_2026-10-28.json) (re-run) —
and numbers should only be compared within one session.

## 3. sm_120 NVFP4 / FP8 experiments (optional, experimental)

[`experiments/tl_kernels/`](../experiments/tl_kernels/) contains optional TileLang kernels for
NVFP4 block-scaled GEMM and fused FP8 casts on sm_120 (RTX 5090). They are **not** used by
`parity.py` / `train.py` and are not covered by the performance CI. Correctness tests for the
pure-torch host quantizer run on CPU (`pytest experiments/tl_kernels/tests`); GPU tests skip
without sm_120 + TileLang.

**Re-validated on 2026-10-28** on a fresh RTX 5090 (torch 2.12.1+cu130, tilelang 0.1.15):
all 13 GPU tests passed, and `python experiments/tl_kernels/bench.py` produced the numbers below
(raw: [`runs/tl_kernels_5090_2026-10-28.json`](../runs/tl_kernels_5090_2026-10-28.json)). Absolute
timings vary by a few percent between runs.

| Item | Re-validated result (2026-10-28) |
|---|---|
| NVFP4 GEMM kernel-only (head shape 8192×129280×384→pad 512) | **1.66–1.72 ms, 632–655 TFLOPS**, ≈2.2–2.3× vs bf16 (`a @ b.T`, 3.73 ms) |
| Fused TileLang NVFP4 quantizer ([129280, 512] weight) | **0.53–0.54 ms** vs 9.42 ms for the PyTorch quantizer (**17.3–17.7×**) |
| Quantizer agreement vs PyTorch | scale words **100% identical**, packed bytes **93.97%** identical (E2M1 rounding ties), dequant max abs diff **1.37e-2** |
| NVFP4 GEMM vs dequantized reference (small shape) | max abs diff **0.0** (bit-identical to the dequantized-operand matmul) |
| Fused FP8 cast ([8192, 129280]) | **2.10 ms** vs 4.84 ms for the bf16-math PyTorch path (**2.3×**; strict fp32 math path: 13.2 ms, 6.3×); bytes **100% equal** |
| FP4 forward + FP8 backward (head) | ≈2.2× vs bf16 — this end-to-end number is from the internal log and was **not** re-measured here |

CUDA parity was also re-checked on the same instance: `parity.py --dtype float32` and
`--dtype bf16` both print **ALL PASS** (T1 max diff 2.384e-07 / 1.788e-07; T2 0.0). The
informational T1b indexer tie probe shows 6.6e-02 on CUDA (device-dependent tie order; see the
README limitations). Logs: [`runs/parity_cuda_fp32_5090.log`](../runs/parity_cuda_fp32_5090.log),
[`runs/parity_cuda_bf16_5090.log`](../runs/parity_cuda_bf16_5090.log).

## 4. Reproducing

```bash
# CPU/Mac parity + smoke numbers
python3 parity.py --dtype float32
python3 train.py --config tiny_all_topk --steps 100 --batch-size 4 --seq-len 32 --lr 1e-3
python3 train.py --config d8 --steps 0

# CPU kernel-parity regressions (fast subset; slow subset ~133 s)
python3 -m pytest -q -m "not slow"
python3 -m pytest -q -m slow

# optional sm_120 experiments (needs CUDA + TileLang; GPU tests skip otherwise)
python3 -m pytest experiments/tl_kernels/tests -q

# GPU breakdown (needs CUDA; e.g. RTX 5090)
python3 bench_mfu.py --b 8 --seq 1024 --json /tmp/p0_b8.json
python3 bench_mfu.py --b 16 --seq 512 --json /tmp/p1_b16_s512.json
```

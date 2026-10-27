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
release.

Absolute peak used for the MFU denominator: **bf16, 5090 vendor peak (165.2 TFLOPS)**. Throughput
figures should be compared between variants measured in the same interleaved run
(`tools/ab_bench.py`), not across sessions.

## 3. Reproducing

```bash
# CPU/Mac parity + smoke numbers
python3 parity.py --dtype float32
python3 train.py --config tiny_all_topk --steps 100 --batch-size 4 --seq-len 32 --lr 1e-3
python3 train.py --config d8 --steps 0

# GPU breakdown (needs CUDA; e.g. RTX 5090)
python3 bench_mfu.py --b 8 --seq 1024 --json /tmp/p0_b8.json
python3 bench_mfu.py --b 16 --seq 512 --json /tmp/p1_b16_s512.json
```

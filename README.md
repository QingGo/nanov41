# nanov41 — a nano-scale, trainable port of the DeepSeek-V4.1-Flash architecture

A research-scale language model built **directly on DeepSeek's official
[DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) inference reference**.
The W1 goal is narrow and testable: prove that the training port is *numerically identical* to the
official implementation on the same weights — on a laptop CPU — before doing any speed work.

> **Status:** W1 complete (parity harness passing). Research code, not a production trainer.
> No model weights, tokenizer or pretrained-data pipeline are included.
> This project is independent and **not affiliated with or endorsed by DeepSeek**.
>
> **W2 best measured (RTX 5090):** 60.43 ms/step at b=16×1024 = 114.9 TFLOP/s =
> **54.9% bf16-MFU** (3-step parameter parity 0.0; graph replay + optimizer). See
> [W2 training-step results](#w2-training-step-results-best-measured-rtx-5090).

## What is in the box

| Path | What |
|---|---|
| `ref/` | Untouched **subset** of the official inference reference (`model.py`, `kernel.py`, `engram.py`, `convert.py`, `generate.py`, `config.json`), byte-identical to upstream |
| `tools/make_patched_ref.py` | Generates `nanov41/_ref_patched_model.py` from `ref/model.py`; `--check` verifies the generated file has not drifted |
| `nanov41/patched_env.py` | Torch fallbacks for the official TileLang kernels (`sparse_attn` / `hc_split_sinkhorn` / quant no-ops) + stubs for `vision` / `image_processor` |
| `nanov41/configs.py` | Nano configs: `tiny` / `tiny_all_topk` (0.68M), `d8` (81.6M), `d12` (193.4M) |
| `nanov41/model.py` | `NanoV41` (forward / logits / loss / generate), `init_random_weights`, `build_pristine` |
| `parity.py` | W1 gate: T1 full-sequence vs incremental prefill, T2 patched vs official, T3 backward, T1b indexer tie probe |
| `train.py` | Speedrun-style training loop skeleton (AdamW + cosine + clip, synthetic learnable data) |
| `bench_mfu.py` | Per-component MFU / time breakdown (needs CUDA; benchmarked on RTX 5090) |
| `tools/ab_bench.py` | Interleaved A/B benchmark of head × MoE implementations (needs CUDA) |
| `experiments/tl_kernels/` | Optional sm_120 (RTX 5090) NVFP4/FP8 kernels (TileLang): host quantizer, FP4 GEMM, fused FP8 cast — not needed for parity/training |
| `docs/` | [Architecture notes](docs/architecture.md), [benchmarks](docs/benchmarks.md) |
| `runs/` | Curated benchmark evidence (RTX 5090 JSON) |
| `tests/` | CPU tests: parity smoke, config counts, generated-file check, kernel-parity regressions |

Internal development notes and one-off experiment scripts are intentionally **not part of this
repository** (see `.gitignore`); only the material needed to reproduce the W1 claims is versioned.

## How it works

DeepSeek-V4.1-Flash is a **Causal Encoder-Decoder (CED)** transformer: every layer attends a
sliding window of raw KV, layer groups additionally share a **compressed KV** latent produced by
the first layer of the group (`kv_source_layers` / `compress_ratios`), and a small **lightning
indexer** selects `index_topk` compressed positions per query. See
[`docs/architecture.md`](docs/architecture.md) for the component list, config invariants and the
five patches used by the training port.

The training port is **not a re-implementation**. `tools/make_patched_ref.py` copies
`ref/model.py` and applies five minimal, documented patches (grad-carrying prefill caches plus a
`full_logits` switch). Everything else stays byte-identical, so the patched model and the pristine
official code can be compared on the same `state_dict` — that is the T2 parity check.

The official TileLang kernels are replaced by pure-torch fallbacks with identical math
(`nanov41/patched_env.py`), which is what makes CPU/Mac parity runs possible. `import nanov41` and
`parity.py` do not require TileLang, Triton or CUDA.

> `nanov41/_ref_patched_model.py` is generated. Do **not** import it directly — load it through
> `nanov41.patched_env.import_patched()` (it relies on the official code's sibling imports).

## Requirements

- Python >= 3.10, PyTorch >= 2.9 (CPU, Apple MPS, or CUDA).
- **CPU / Apple MPS:** `parity.py` and the `train.py` smoke test (no CUDA needed).
- **NVIDIA GPU (optional):** `bench_mfu.py`, `tools/ab_bench.py`, `experiments/tl_kernels/` and
  the fused-kernel modules (`nanov41/fused_*`, `nanov41/triton_*`, `nanov41/tl_*`) need CUDA;
  `bench_mfu.py` and `tools/ab_bench.py` exit early without it. The experiments package targets
  sm_120 (RTX 5090) with TileLang; its pure-torch host quantizer also runs on CPU.

## Install

```bash
git clone <repo-url> nanov41 && cd nanov41
python3 -m venv .venv && . .venv/bin/activate
python3 -m pip install -r requirements.txt        # CPU/Mac core (torch, numpy, sympy)
# optional GPU extras: python3 -m pip install -r requirements-cuda.txt
python3 -m pip install -e .                       # optional; or simply run from the repo root
```

## Quick start (run from the repository root)

```bash
# W1 parity gate: full-sequence vs incremental prefill, patched vs official, backward smoke
python3 parity.py --dtype float32                # strict, CPU (~6 s)
python3 parity.py --dtype bf16                   # bf16 path (~16 s)
python3 parity.py --dtype float32 --device mps   # macOS MPS (~2 min)

# CPU training smoke (0.68M params, learnable synthetic pattern)
python3 train.py --config tiny_all_topk --steps 100 --batch-size 4 --seq-len 32 --lr 1e-3

# model stats for the real target configs (no training)
python3 train.py --config d8  --steps 0           # 81.62M params (15.43M non-embedding)
python3 train.py --config d12 --steps 0           # 193.41M params (94.12M non-embedding)

# generated-file integrity check (also run by CI)
python3 tools/make_patched_ref.py --check
```

`parity.py` exits 0 only when all checks pass; it prints `VERDICT ... ALL PASS`.

## W1 results

Measured on macOS (Python 3.13.2, torch 2.9.0, CPU unless noted):

| Check | Result |
|---|---|
| T1 full-sequence == incremental prefill (fp32) | **PASS**, max diff **2.123e-07** |
| T1 (bf16, CPU) | **PASS**, max diff **0.0** |
| T2 patched training model == pristine official code | **PASS**, max diff **0.0** (fp32 and bf16) |
| T3 backward: loss + gradients | **PASS**: 680,040 grad elements, 95.6% non-zero, 0 NaN |
| T1b indexer tie probe (`index_topk=8`) | max **2.382e-02**, 1 position above tolerance (documented limitation) |
| CPU training smoke (tiny_all_topk, 100 steps) | **PASS**: loss **6.2230 → 5.3106** in 4.8 s |
| `d8` / `d12` configs | build and step on CPU; parameter counts match the docs |
| MPS device path | `parity.py --device mps` → **ALL PASS** |
| CUDA parity (re-checked 2026-10-28, RTX 5090, torch 2.12.1+cu130) | **ALL PASS** for fp32 and bf16: T1 max diff **2.384e-07** / **1.788e-07**, T2 **0.0**; logs in [`runs/`](runs/) |

Benchmark numbers on an RTX 5090 — W1 step breakdown, W2 best training steps (above) and the
re-validated NVFP4/FP8 experiment numbers — are in [`docs/benchmarks.md`](docs/benchmarks.md)
(raw evidence in [`runs/`](runs/)).

## Development checks (what CI runs)

CI runs on Python 3.10 and 3.12 with a pinned CPU wheel (`torch==2.14.1`); run the same
checks locally before pushing:

```bash
python3 parity.py --dtype float32
python3 tools/make_patched_ref.py --check
python3 -m pytest -q -m "not slow"                      # fast subset (~40 s)
python3 -m pytest -q -m slow                            # kernel-parity slow subset (~2 min)
python3 -m pytest experiments/tl_kernels/tests -q       # optional-kernel CPU tests
```

Some kernel-parity gates compare aten op counts, which are torch-version dependent (the
op decomposition changed between 2.14.1 and 2.15.0); CI therefore pins one torch version
for the whole matrix. If you suspect a version-sensitive failure, reproduce with the
pinned wheel in a venv, e.g. `uv venv --python 3.10 && uv pip install torch==2.14.1 pytest`.

## W2 training-step results (best measured, RTX 5090)

After W1, the training path was optimized outside this repository (fused kernels, grouped MoE,
CUDA-graph capture) and measured with the internal graph-training driver. Best measured on
2026-10-22: d12, bf16, **b=16 × seq=1024** (16,384 tokens/step), 3-step parameter parity
(`parity_param_rel = 0.0`), graph replay + optimizer, torch 2.14.1+cu130, RTX 5090.

| Variant (b=16, seq=1024) | step (ms) | TFLOP/s | bf16-MFU | note |
|---|---:|---:|---:|---|
| R42 — first graph-trained best | 92.08 | 75.43 | 36.0% | trajectory baseline |
| R48 — best | 60.92 | 114.01 | 54.42% | |
| R49 — best (re-measured as R50 base) | 60.94 | 113.96 | 54.40% | |
| **R50 — best (+ MoE NOPAD)** | **60.43–60.48** | **114.83–114.93** | **54.81–54.86%** | new absolute best, two repeats |
| R50 — MoE fp8 v1 | 60.88–60.94 | 114.07–113.96 | 54.40–54.45% | **no gain** (see below) |
| R50 — best at b=8, seq=1024 | 34.87 | 99.58 | 47.53% | 8,192 tokens/step |

- **bf16-MFU** is the cross-round comparable metric: `TFLOP/s ÷ 209.5 TF` (bf16 peak, fp32
  accumulate, 5090 vendor spec). At b=16 the 60% wall is **55.25 ms**, so the R50 best is
  **5.2 ms** away (114.9 vs 125.7 TFLOP/s).
- **Discounted MFU (V4.1-Flash report convention):** weight bf16 / fp8 / fp4 operations by
  **1 / 0.5 / 0.25** (report Figure 2) and divide by the fixed 209.5 TF bf16 peak — no per-run
  peak swapping. R50 best (fp8 head + bf16 rest) = **35.6%** discounted MFU; its 60% wall is
  35.83 ms (all-fp8 MoE what-if: 29.56 ms). The raw-FLOPs bf16-MFU above is the cross-round
  throughput metric and is not comparable across precision mixes. See
  [`docs/v41-scaling-check.md`](docs/v41-scaling-check.md) and
  [`tools/mfu_accounting.py`](tools/mfu_accounting.py).
- **Scaling check vs V4.1-Flash:** d12 matches the full model at **226.5×** down-scaling in both
  activated parameters (16B vs 70.65M) and raw decode FLOPs (32 GFLOP vs 141.3 MFLOP); total
  backbone params (552B, 2854×) are not a compute metric because the vocab/head are not scaled
  and the expert activation rate differs (2/16 vs 6/384). Details in the same doc.
- **MoE fp8 v1 did not pay off**: 60.88–60.94 ms vs 60.94 ms bf16. The MoE grouped GEMMs have
  K=384 (only 6 k-steps), so they are launch/latency-bound, and per-tile activation quantization
  inside the kernel eats the dot-product savings. The next direction (planned, **not yet
  measured**) is fp8 on the large-K dW segment with pre-quantized activations/weights, targeting
  ≈57.5 ms / ≈57.6% bf16-MFU.
- Evidence: [`runs/r50_best_b16.json`](runs/r50_best_b16.json),
  [`runs/r50_best_b8.json`](runs/r50_best_b8.json), [`runs/r50_base.json`](runs/r50_base.json),
  [`runs/r50_fp8.json`](runs/r50_fp8.json), [`runs/r48_best_b16.json`](runs/r48_best_b16.json),
  [`runs/r42_best_final.json`](runs/r42_best_final.json). The graph-training driver is internal
  tooling and is not part of this repository; these numbers come from a single 5090 session and
  are not directly comparable across hosts (launch-bound steps depend on host CPU).

## Known limitations (W2+)

- **Engram is disabled** in all configs (it needs the DeepSeek tokenizer to build the compressed
  token map); the vision encoder and the DSpark/MTP losses are out of scope for v0.
- **Quantization is a no-op** (`act_quant` / `fp4_act_quant`): training runs bf16.
- **Indexer tie-breaking:** the official indexer ReLUs its scores, so random-init models can have
  many exact zeros; `topk` tie order then depends on the candidate count (prefix length). This can
  make full-sequence vs incremental prefill differ at those ties (`T1b` probes it: 1 position above
  tolerance). A canonical tie-break is planned for W2.
- **CUDA bf16 parity:** bf16 T1 differences up to ~7e-2 were observed on a 4090 during W2
  (Hyper-Connections amplify GEMM reduction-order noise). Re-checked on 2026-10-28 on an RTX 5090
  (torch 2.12.1+cu130): **T1/T2/T3 pass in both fp32 and bf16**. The informational T1b indexer
  tie probe still shows 6.6e-2 on CUDA because top-k tie order is device-dependent; a canonical
  tie-break is still a W2 item.
- `original_seq_len=0` in the nano configs → YaRN is disabled (short-context training only).
- If `tilelang`/`triton`/TileKernels are absent, fused-kernel modules silently fall back to torch
  (or stay unavailable); `import nanov41` never requires them.

## Data, weights and tokenizer

This repository ships **no model weights, no tokenizer and no training data**. W1/W2 numbers come
from randomly initialised weights and a deterministic synthetic batch (`train.py --data synthetic`).
The DeepSeek tokenizer and pretokenized shards are not included; use the official weights/tokenizer
under their own upstream terms.

## License & attribution

- This project: **MIT** — see [`LICENSE`](LICENSE) (Copyright (c) 2026 QingGo).
- `ref/` and the generated `nanov41/_ref_patched_model.py` are derived from
  [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
  (inference reference code), licensed under MIT — see [`NOTICE`](NOTICE) for the retained
  copyright and permission notice.
- "DeepSeek" and "DeepSeek-V4.1" are trademarks of DeepSeek. This is an independent research
  project, not affiliated with, endorsed by, or sponsored by DeepSeek.

## Repository layout

```
.
├── nanov41/            core package (model, configs, patched_env, training utils, optional kernels)
├── ref/                untouched official inference reference subset
├── parity.py           W1 parity harness (T1/T2/T3 + tie probe)
├── train.py            speedrun-style training skeleton
├── bench_mfu.py        per-component MFU breakdown (needs CUDA)
├── docs/               architecture notes and benchmarks
├── experiments/        optional sm_120 NVFP4/FP8 kernels (TileLang; needs CUDA)
├── runs/               curated benchmark evidence (JSON)
├── tests/              CPU tests (CI), incl. kernel-parity regressions
└── tools/              make_patched_ref.py, ab_bench.py (needs CUDA)
```

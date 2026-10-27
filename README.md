# nanov41 — a nano-scale, trainable port of the DeepSeek-V4.1-Flash architecture

A research-scale language model built **directly on DeepSeek's official
[DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) inference reference**.
The W1 goal is narrow and testable: prove that the training port is *numerically identical* to the
official implementation on the same weights — on a laptop CPU — before doing any speed work.

> **Status:** W1 complete (parity harness passing). Research code, not a production trainer.
> No model weights, tokenizer or pretrained-data pipeline are included.
> This project is independent and **not affiliated with or endorsed by DeepSeek**.

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

Benchmark numbers on an RTX 5090 are in [`docs/benchmarks.md`](docs/benchmarks.md)
(raw evidence in [`runs/`](runs/)).

## Known limitations (W2+)

- **Engram is disabled** in all configs (it needs the DeepSeek tokenizer to build the compressed
  token map); the vision encoder and the DSpark/MTP losses are out of scope for v0.
- **Quantization is a no-op** (`act_quant` / `fp4_act_quant`): training runs bf16.
- **Indexer tie-breaking:** the official indexer ReLUs its scores, so random-init models can have
  many exact zeros; `topk` tie order then depends on the candidate count (prefix length). This can
  make full-sequence vs incremental prefill differ at those ties (`T1b` probes it: 1 position above
  tolerance). A canonical tie-break is planned for W2.
- **CUDA bf16 parity:** bf16 T1 differences up to ~7e-2 were observed on a 4090 (Hyper-Connections
  amplify GEMM reduction-order noise). Strict causality checks are done in fp32; a HC-aware
  bf16 tolerance or `hc_mult=1` control is planned for W2.
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

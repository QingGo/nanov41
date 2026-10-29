# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
# experiments/tl_kernels — optional sm_120 NVFP4 / FP8 kernels (not installed by pip)

This directory holds a curated subset of the internal W2 kernel experiments: sm_120
(RTX 5090) NVFP4 block-scaled GEMM plus a fused quantizer, and fused FP8 cast kernels.
They are **optional** and **experimental**:

- `nanov41` itself never imports them; `parity.py`, `train.py` and the CPU test suite do
  not depend on TileLang or CUDA.
- The kernels require **TileLang** and an **sm_120** GPU (tested on an RTX 5090); other
  Blackwell/Ada parts are untested.
- The pure-torch **host quantizer/dequantizer** in `nvfp4_gemm.py` works on CPU and is
  covered by CI (`tests/test_host_quant_cpu.py`).

## Format note: NVFP4 vs the official MXFP4

The DeepSeek-V4.1-Flash report uses two different FP4 formats:

| Where | Format | Scale |
|---|---|---|
| V4/V4.1 **expert weights** (and indexer QK) | **MXFP4** | E2M1 + power-of-two (**E8M0**) scale, one per **1×32** block |
| V4.1 **main KV cache** | NVFP4-class | E2M1 + **E4M3** scale, one per **1×16** block |

The kernels in this directory are **NVFP4** (E4M3 scale per 1×16): they match the V4.1 *KV-cache*
format, **not** the expert-weight MXFP4 format, and they are therefore **not bit-compatible** with
the official MXFP4 experts. They were built as an sm_120 (RTX 5090) FP4 tensor-core experiment —
TileLang's sm120 block-scaled GEMM example is NVFP4, and the official MXFP4 GEMM kernels
(DeepGEMM) only support SM90/SM100. NVFP4's finer scales also give better reconstruction accuracy,
which is what we wanted for a from-scratch QAT probe.

For the **official training recipe** (FP4 weight QAT storage, dequantised back to FP8 for compute
and backward, FP32 master + STE — V4 report §5.2.1), the weight *storage* side is MXFP4-compatible:
`tile_kernels.quant.per_token_cast(..., "e2m1", num_per_channels=32, round_sf=True)` produces
MXFP4-style words, and the compute path is FP8. That recipe lives in the internal W2 notes; this
public package only ships the NVFP4 tensor-core probe and the fused quantizer/cast kernels.


## Files

| File | Contents | Needs |
|---|---|---|
| `nvfp4_gemm.py` | NVFP4 host quantizer/dequantizer, scale-word swizzle, sm120 block-scaled GEMM, `NVFP4Linear` (FP4 forward + BF16 STE backward) | host quantizer: torch only; GEMM: TileLang + sm120 |
| `nvfp4_quant.py` | fused TileLang NVFP4 quantizer (bf16 → packed E2M1 + per-1x16 E4M3 scales) | TileLang + sm120 |
| `fp8_cast.py` | fused TileLang FP8 cast (bf16 → e4m3), scalar and per-row scale | TileLang + CUDA |

## Usage

```python
from experiments.tl_kernels import gemm_nvfp4, quantize_nvfp4, dequantize_nvfp4
from experiments.tl_kernels import NVFP4Linear, cast_fp8

# pure-torch host quantization (CPU works):
packed, words = quantize_nvfp4(x_bf16)          # x: [rows, K] -> packed FP4 + E4M3 scales
deq = dequantize_nvfp4(packed, words, rows=rows, kpad=512)

# kernels (require TileLang + sm_120):
y = gemm_nvfp4(a_bf16, b_bf16)                  # C = A @ B^T, FP4 operands
layer = NVFP4Linear(in_features=384, out_features=384)
```

Constraints (inherited from the TileLang example): `K` is padded to a multiple of 256;
`M` and `N` must be multiples of 128; the FP4 GEMM uses 128×128×256 tiles.

## Measured numbers (re-validated 2026-10-28)

Environment: fresh RTX 5090 (sm_120), torch 2.12.1+cu130, tilelang 0.1.15. All 13 GPU tests in
this directory passed, and `python experiments/tl_kernels/bench.py` produced the numbers below
(raw JSON: [`runs/tl_kernels_5090_2026-10-28.json`](../../runs/tl_kernels_5090_2026-10-28.json)).
Absolute timings vary by a few percent between runs; re-run the script for your own hardware.

| Item | Result |
|---|---|
| NVFP4 GEMM kernel-only, head shape 8192×129280×384→pad512 | **1.66–1.72 ms, 632–655 TFLOPS**, ≈2.2–2.3× vs bf16 (`a @ b.T`, 3.73 ms) |
| Fused quantizer, [129280, 512] weight | **0.53–0.54 ms** vs 9.42 ms for the PyTorch quantizer (**17.3–17.7×**) |
| Quantizer agreement vs PyTorch | scale words **100% identical**, packed bytes **93.97%** identical (E2M1 ties), dequant max abs diff **1.37e-2** |
| NVFP4 GEMM vs dequantized-operand reference (small shape) | max abs diff **0.0** (bit-identical) |
| FP8 cast, [8192, 129280] | **2.10 ms** vs 4.84 ms for the bf16-math PyTorch path (**2.3×**; strict fp32 path 13.2 ms → 6.3×); bytes **100% equal** |
| FP4 forward + FP8 backward (head) | ≈2.2× vs bf16 — from the internal log, **not** re-measured in this validation run |

## Attribution

- `nvfp4_gemm.py` adapts the TileLang example
  [`examples/gemm_sm120/sm120_nvfp4_blockscaled_gemm.py`](https://github.com/tile-ai/tilelang/blob/v0.1.15/examples/gemm_sm120/sm120_nvfp4_blockscaled_gemm.py)
  (MIT, Copyright (c) Tile-AI; pin: tilelang v0.1.15). The host quantizer/dequantizer and
  `NVFP4Linear` were written for this project.
- `nvfp4_quant.py` is adapted from the DeepSeek-V4.1-Flash inference reference
  `ref/kernel.py: fp4_quant_kernel` (MIT, Copyright (c) 2023 DeepSeek).
- `fp8_cast.py` is original code.

See the repository [`NOTICE`](../../NOTICE) for the full third-party license texts. This
package is not affiliated with TileLang, DeepSeek or NVIDIA; "TileLang", "DeepSeek",
"RTX 5090" are used descriptively only.

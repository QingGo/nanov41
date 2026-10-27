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

## Measured numbers (internal run — not re-validated by this public test suite)

Environment: RTX 5090 (sm_120), torch 2.12.1+cu130, tilelang 0.1.15, 2026-10-21.
These numbers come from the internal development log; the GPU tests here are skipped
without an sm_120 GPU, and no performance claim is made for other versions/hardware.
Re-run `pytest experiments/tl_kernels/tests` on your GPU to validate correctness.

| Item | Result |
|---|---|
| NVFP4 GEMM kernel-only, head shape 8192×129280×384→pad512 | ≈1.66 ms, ≈652 TFLOPS (bf16 baseline ≈215.7 TFLOPS, ≈3.0×) |
| Fused quantizer, [129280, 512] weight | ≈0.54 ms vs ≈9.43 ms for the PyTorch quantizer (≈17.4×); scale words 100% identical, packed bytes ≈94% identical (E2M1 ties) |
| Quantize/dequantize agreement | max abs diff ≈1.4e-2 (expected FP4 rounding) |
| FP8 cast, [8192, 129280] | ≈2.12 ms vs ≈4.90 ms for `(x*inv).to(fp8)` (≈2.3×); bytes 100% equal |
| FP4 forward + FP8 backward (head) | ≈2.2× vs bf16 (≈11.3 ms vs ≈24.6 ms) |

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

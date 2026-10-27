# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""SM120 NVFP4 block-scaled GEMM for nanov41 training (TileLang).

Adapted from the official TileLang example
``examples/gemm_sm120/sm120_nvfp4_blockscaled_gemm.py`` (MIT, Copyright (c) Tile-AI;
pin: tilelang v0.1.15) — see ``NOTICE`` and ``README.md``. The host quantizer /
dequantizer and the ``NVFP4Linear`` training wrapper are additions from this project.

Why this exists: PyTorch 2.12/2.14 gates ``_scaled_mm`` FP4 to B200/B300
(``ScaledBlas.cpp: _check_mxfp4_support -> dprops->major == 10``), and TransformerEngine
still rejects RTX 50 NVFP4 (issue #2255). TileLang has the sm_120 block-scaled MMA
lowering (``T.mma_gemm_blockscaled``, ``sf_layout="blockscaled_chunk_kmajor"``), so this
module can run NVFP4 GEMMs on consumer Blackwell.

Constraints inherited from the example (important for the model):
  * K must be a multiple of 256 (scale layout blocks are 256-wide); the d12 head's
    K=384 is padded to 512 by the quantizer (+33% FLOPs on that GEMM).
  * M and N are padded to multiples of 128 by the caller (head N=129280 = 1010*128,
    expert N=384 = 3*128, so only K padding is needed in practice).
  * scales: per 1x16 block, E4M3 bytes packed 4-per-uint32 word, then swizzled in
    ``blockscaled_chunk_kmajor`` order.

Training use: FP4 forward + BF16 STE backward (one FP4 GEMM of three matmuls for a
linear layer). All-FP4 backward (also quantizing dC) is a later experiment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # TileLang is optional: the host quantizer below is pure torch.
    import tilelang
    import tilelang.language as T

    TILELANG_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the environment
    tilelang = None
    T = None
    TILELANG_AVAILABLE = False

# E2M1 code -> value (matches the official example's decode LUT)
FP4_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
FP4_MAX = 6.0
SF_GRANULARITY_K = 16


# ------------------------------------------------------------------ scale swizzle
def swizzle_blockscaled_chunk_kmajor_scale_words(
    words: torch.Tensor, block_rows: int = 128, block_words: int = 4
) -> torch.Tensor:
    """Pack semantic scale words in SM120 BlockScaledBasicChunk K-major order.

    Same contract as the upstream TileLang example (helper adapted from it)."""
    if block_rows != 128 or block_words != 4:
        raise ValueError("SM120 BlockScaledBasicChunk K-major scale packing requires 128/4")
    if not isinstance(words, torch.Tensor) or words.dtype != torch.uint32:
        raise TypeError("words must be a uint32 tensor")
    rows, cols = words.shape
    if cols % block_words != 0:
        raise ValueError("blockscaled_chunk_kmajor requires K/64 words multiple of 4")
    if rows % block_rows != 0:
        padded_rows = (rows + block_rows - 1) // block_rows * block_rows
        padded = torch.zeros((padded_rows, cols), dtype=words.dtype, device=words.device)
        padded[:rows] = words
        words, rows = padded, padded_rows
    row_blocks = rows // block_rows
    source = words.contiguous().reshape(row_blocks, 4, 32, cols)
    return source.permute(0, 3, 2, 1).contiguous().reshape(rows, cols)


# ------------------------------------------------------------------ host quantizer
def _round_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round a float tensor to the nearest E2M1 value (magnitude LUT, nearest-even at midpoints)."""
    mags = torch.tensor(FP4_LUT, device=x.device, dtype=torch.float32)
    boundaries = (mags[:-1] + mags[1:]) / 2
    absx = x.abs().clamp(max=FP4_MAX)
    idx = torch.bucketize(absx, boundaries).clamp(0, 7)
    sign = torch.where(x < 0, -1.0, 1.0)
    return sign * mags[idx]


def _e2m1_code(x: torch.Tensor) -> torch.Tensor:
    """Value -> 4-bit code (0..7 positive, 8..15 negative), matching the official decode LUT."""
    mags = torch.tensor(FP4_LUT, device=x.device, dtype=torch.float32)
    boundaries = (mags[:-1] + mags[1:]) / 2
    absx = x.abs().clamp(max=FP4_MAX)
    idx = torch.bucketize(absx, boundaries).clamp(0, 7).to(torch.uint8)
    return (idx.to(torch.int16) + torch.where(x < 0, 8, 0)).to(torch.uint8)


def quantize_nvfp4(x: torch.Tensor, k_pad_to: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """bf16 [rows, K] -> (packed FP4 [rows, Kpad//2] int8-view, scale words [N, 4] uint32).

    Per-16-channel E4M3 scale = amax / 6.0 (rounded to E4M3); K is right-padded with
    zeros to a multiple of ``k_pad_to`` (must be a multiple of 256 for the kernel).
    """
    assert x.dim() == 2
    rows, k = x.shape
    kpad = (k + k_pad_to - 1) // k_pad_to * k_pad_to
    assert kpad % 256 == 0, "kernel requires Kpad % 256 == 0"
    if kpad != k:
        x = F.pad(x, (0, kpad - k))
    xf = x.float()
    groups = xf.reshape(rows, kpad // SF_GRANULARITY_K, SF_GRANULARITY_K)
    amax = groups.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
    scale = (amax / FP4_MAX).to(torch.float8_e4m3fn).float()
    scale = scale.clamp_min(2.0 ** -9)  # E4M3 min subnormal; avoids div-by-zero
    q = _round_e2m1(groups / scale)
    codes = _e2m1_code(q).reshape(rows, kpad)
    # pack two codes per byte: low nibble = even column, high nibble = odd column
    lo = codes[:, 0::2]
    hi = codes[:, 1::2]
    packed = (lo | (hi << 4)).to(torch.uint8).contiguous()
    # pack 4 E4M3 scale bytes into one uint32 word (little-endian, as in the example)
    scale_bytes = scale.squeeze(-1).to(torch.float8_e4m3fn).view(torch.uint8).reshape(rows, kpad // 16)
    sb = scale_bytes.to(torch.int64).reshape(rows, kpad // 64, 4)
    words = sb[:, :, 0] | (sb[:, :, 1] << 8) | (sb[:, :, 2] << 16) | (sb[:, :, 3] << 24)
    words = words.to(torch.uint32)
    sw = swizzle_blockscaled_chunk_kmajor_scale_words(words, 128, 4)
    return packed.view(torch.int8), sw.reshape(-1, 4)


def dequantize_nvfp4(packed: torch.Tensor, words: torch.Tensor, rows: int, kpad: int) -> torch.Tensor:
    """Inverse of ``quantize_nvfp4`` (for tests; returns fp32 [rows, kpad]).

    ``quantize_nvfp4`` pads the scale-word rows up to a multiple of 128 (swizzle
    requirement), so the word count implies a padded row count; the padded rows are
    trimmed here to return exactly ``rows`` rows. ``kpad`` must equal the padded K.
    """
    u = packed.view(torch.uint8)
    lut = torch.tensor(list(FP4_LUT) + [-v for v in FP4_LUT], device=packed.device, dtype=torch.float32)
    codes = torch.empty((rows, kpad), device=packed.device, dtype=torch.uint8)
    codes[:, 0::2] = u & 0x0F
    codes[:, 1::2] = (u >> 4) & 0x0F
    vals = lut[codes.long()]
    cols = kpad // 64
    if words.numel() % cols != 0:
        raise ValueError(f"words.numel()={words.numel()} is not a multiple of kpad//64={cols}")
    padded_rows = words.numel() // cols
    if padded_rows < rows:
        raise ValueError(f"words describe {padded_rows} rows but {rows} were requested")
    w = words.reshape(padded_rows, cols)
    # inverse of swizzle: [rows_pad, cols] -> [row_blocks, 4, 32, cols]
    w = w.reshape(-1, cols, 32, 4).permute(0, 3, 2, 1).reshape(padded_rows, cols)
    wi = w.to(torch.int64)
    sb = torch.empty((padded_rows, cols, 4), device=packed.device, dtype=torch.int64)
    sb[..., 0] = wi & 0xFF
    sb[..., 1] = (wi >> 8) & 0xFF
    sb[..., 2] = (wi >> 16) & 0xFF
    sb[..., 3] = (wi >> 24) & 0xFF
    scale_bytes = sb.reshape(padded_rows, cols * 4).to(torch.uint8)  # [rows_pad, kpad//16]
    scales = scale_bytes.contiguous().view(torch.float8_e4m3fn).float()[:rows]
    return vals * scales.repeat_interleave(SF_GRANULARITY_K, dim=-1)


# ------------------------------------------------------------------ TileLang kernel
if TILELANG_AVAILABLE:

    @tilelang.jit
    def _make_nvfp4_gemm_kernel(
        M: int,
        N: int,
        K: int,
        block_M: int = 128,
        block_N: int = 128,
        block_K: int = 256,
        num_stages: int = 2,
        out_dtype=T.bfloat16,
    ):
        assert N % 8 == 0
        assert K % block_K == 0 and block_K % 64 == 0
        assert block_M == 128 and block_N == 128
        assert M % 128 == 0 and N % 128 == 0, "M and N must be padded to multiples of 128"
        in_dtype = T.float4_e2m1fn
        sf_words_per_block_k = block_K // 64
        M_pad = -(-M // block_M) * block_M
        N_pad = -(-N // block_N) * block_N
        k_blocks = K // block_K

        @T.prim_func
        def main(
            A: T.Tensor((M, K), in_dtype),
            B: T.Tensor((N, K), in_dtype),
            SFA: T.Tensor((M_pad * k_blocks, sf_words_per_block_k), T.uint32),
            SFB: T.Tensor((N_pad * k_blocks, sf_words_per_block_k), T.uint32),
            C: T.Tensor((M, N), out_dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), in_dtype)
                B_shared = T.alloc_shared((block_N, block_K), in_dtype)
                SFA_shared = T.alloc_shared((block_M, sf_words_per_block_k), T.uint32)
                SFB_shared = T.alloc_shared((block_N, sf_words_per_block_k), T.uint32)
                C_local = T.alloc_fragment((block_M, block_N), T.float32)
                T.clear(C_local)
                for ko in T.Pipelined(K // block_K, num_stages=num_stages):
                    T.copy(A[by * block_M, ko * block_K], A_shared)
                    T.copy(B[bx * block_N, ko * block_K], B_shared)
                    for r, w in T.Parallel(block_M, sf_words_per_block_k):
                        SFA_shared[r, w] = SFA[(by * k_blocks + ko) * block_M + r, w]
                    for r, w in T.Parallel(block_N, sf_words_per_block_k):
                        SFB_shared[r, w] = SFB[(bx * k_blocks + ko) * block_N + r, w]
                    T.mma_gemm_blockscaled(
                        A_shared,
                        B_shared,
                        C_local,
                        SFA_shared,
                        SFB_shared,
                        transpose_B=True,
                        clear_accum=False,
                        k_start=ko * block_K,
                        sf_a_granularity_k=SF_GRANULARITY_K,
                        sf_b_granularity_k=SF_GRANULARITY_K,
                        sf_layout="blockscaled_chunk_kmajor",
                    )
                T.copy(C_local, C[by * block_M, bx * block_N])

        return main

else:  # pragma: no cover
    _make_nvfp4_gemm_kernel = None


def nvfp4_gemm(a_bf16: torch.Tensor, b_bf16: torch.Tensor) -> torch.Tensor:
    """C = A @ B^T with NVFP4 operands (A [M,K], B [N,K]) -> bf16 [M,N]. Pads K to 256 multiple."""
    if not TILELANG_AVAILABLE:
        raise RuntimeError("nvfp4_gemm requires tilelang (see experiments/tl_kernels/README.md)")
    M, K = a_bf16.shape
    N = b_bf16.shape[0]
    kpad = (K + 255) // 256 * 256
    Aq, SFA = quantize_nvfp4(a_bf16, k_pad_to=kpad)
    Bq, SFB = quantize_nvfp4(b_bf16, k_pad_to=kpad)
    kernel = _make_nvfp4_gemm_kernel(M, N, kpad)
    C = torch.empty((M, N), device=a_bf16.device, dtype=torch.bfloat16)
    kernel(Aq, Bq, SFA, SFB, C)
    return C


class _NVFP4MatmulSTE(torch.autograd.Function):
    """FP4 forward, BF16 STE backward (dA = dC @ B, dB = dC^T @ A)."""

    @staticmethod
    def forward(ctx, a, b):
        out = nvfp4_gemm(a.detach(), b.detach())
        ctx.save_for_backward(a.detach(), b.detach())
        return out

    @staticmethod
    def backward(ctx, dc):
        a, b = ctx.saved_tensors
        dc = dc.to(torch.bfloat16) if dc.dtype != torch.bfloat16 else dc
        da = dc @ b
        db = dc.t().contiguous() @ a
        return da, db


class NVFP4Linear(nn.Module):
    """Drop-in ``nn.Linear`` replacement: FP4 forward GEMM, BF16 STE backward (experimental)."""

    def __init__(self, in_features: int, out_features: int, device=None, dtype=torch.bfloat16):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        y = _NVFP4MatmulSTE.apply(x2, self.weight)
        return y.reshape(*shape[:-1], self.out_features)


__all__ = [
    "FP4_LUT",
    "FP4_MAX",
    "NVFP4Linear",
    "SF_GRANULARITY_K",
    "TILELANG_AVAILABLE",
    "dequantize_nvfp4",
    "nvfp4_gemm",
    "quantize_nvfp4",
    "swizzle_blockscaled_chunk_kmajor_scale_words",
]

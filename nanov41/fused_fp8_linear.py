"""Training-oriented fused FP8 linear family (R82) — "producer-quant + cuBLASLt GEMM".

Why this design (docs/fp8-first-plan.md §4):
  * the official *inference* kernel (``ref/kernel.py::fp8_gemm_kernel``) uses block_M=32, a
    decode tile — at our training shape M=16k it is far from fp8 peak;
  * cuBLASLt (``torch._scaled_mm``) already measured **2.27x** over bf16 on the big d12 shapes
    (graph-timed: wo_b fwd 0.0925 -> 0.0407 ms; engram_wkv 1.63x incl quant), so we do not
    rewrite the GEMM — we remove everything around it:
      1. activation quantisation = one fused Triton pass (rowwise amax + cast, two reads of the
         row / one write) — and the plan's next step replaces it with casts fused into the
         producer kernels (V4 Fig.5 "SwiGLU + FP8 Cast" pattern);
      2. the weight is quantised **once per optimizer step** into static, graph-safe buffers;
         the same cached ``w8`` serves fwd (``x @ w8^T``) and dgrad (``dy @ w8``);
      3. wgrad stays bf16 ``dy.T @ x`` (cuBLAS handles the transpose natively; the fp8
         transpose-B path was 8x slower in R78 — revisit with a custom split-K kernel later);
      4. no weight-buffer allocation inside the graph (WCACHE pattern from ``fp8_head_v2``).

Env gates (default OFF; unset path byte-identical):
  NANOV41_LINEAR_FP8_V2=1            master
  NANOV41_LINEAR_FP8_V2_SCOPE        comma list of name substrings (default all eligible)
  NANOV41_LINEAR_FP8_V2_MIN_FLOPS    skip small GEMMs (default 0)
  NANOV41_LINEAR_FP8_V2_QUANT        triton | torch   (default triton)
  NANOV41_LINEAR_FP8_V2_DGRAD        1 (default) | 0 (bf16 dgrad)
  NANOV41_LINEAR_FP8_V2_WGRAD_FP8    0 (default, reserved)
"""
from __future__ import annotations

import os

import torch

from . import fp8_cast_registry as _castreg

try:  # GPU-only dependency
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None  # type: ignore
    tl = None  # type: ignore
    _HAS_TRITON = False

E4M3 = 448.0
_EPS = 1e-12


# ----------------------------------------------------------------------------- env helpers
def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in ("", "0", "false", "off", "no")


def enabled() -> bool:
    return _flag("NANOV41_LINEAR_FP8_V2", "0")


def _scope() -> tuple[str, ...]:
    raw = os.environ.get("NANOV41_LINEAR_FP8_V2_SCOPE", "")
    return tuple(s.strip() for s in raw.split(",") if s.strip())


def _min_flops() -> float:
    try:
        return float(os.environ.get("NANOV41_LINEAR_FP8_V2_MIN_FLOPS", "0") or 0)
    except ValueError:
        return 0.0


def _quant_impl() -> str:
    return os.environ.get("NANOV41_LINEAR_FP8_V2_QUANT", "triton").strip().lower()


def dgrad_enabled() -> bool:
    return _flag("NANOV41_LINEAR_FP8_V2_DGRAD", "1")


def _sm_ok() -> bool:
    return hasattr(torch, "_scaled_mm") and torch.cuda.is_available()


_WARNED: set[str] = set()


def _warn_once(tag: str, msg: str) -> None:
    if tag not in _WARNED:
        _WARNED.add(tag)
        print(f"[linear-fp8-v2] {msg}", flush=True)


# ----------------------------------------------------------------------------- quantisation
if _HAS_TRITON:

    @triton.jit
    def _rowwise_quant_kernel(X, Y, S, K, BLOCK_K: tl.constexpr, EPS: tl.constexpr, MAX: tl.constexpr):
        row = tl.program_id(0)
        base = row * K
        offs = tl.arange(0, BLOCK_K)
        amax = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            idx = k0 + offs
            v = tl.load(X + base + idx, mask=idx < K, other=0.0).to(tl.float32)
            amax = tl.maximum(amax, tl.abs(v))
        a = tl.max(amax, axis=0)
        s = tl.maximum(a, EPS) / MAX
        tl.store(S + row, s)
        inv = 1.0 / s
        for k0 in range(0, K, BLOCK_K):
            idx = k0 + offs
            m = idx < K
            v = tl.load(X + base + idx, mask=m, other=0.0).to(tl.float32) * inv
            q = tl.minimum(tl.maximum(v, -MAX), MAX).to(tl.float8e4nv)
            tl.store(Y + base + idx, q, mask=m)


if _HAS_TRITON:

    @triton.jit
    def _row_amax_kernel(X, AMAX, K, BLOCK_K: tl.constexpr):
        row = tl.program_id(0)
        base = row * K
        offs = tl.arange(0, BLOCK_K)
        amax = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            idx = k0 + offs
            v = tl.load(X + base + idx, mask=idx < K, other=0.0).to(tl.float32)
            amax = tl.maximum(amax, tl.abs(v))
        tl.store(AMAX + row, tl.max(amax, axis=0))

    @triton.jit
    def _cast_block_kernel(X, SROW, Y, K, BLOCK_K: tl.constexpr, MAX: tl.constexpr):
        """Cast bf16 [M,K] -> fp8 with a per-row scale broadcast from SROW[row]."""
        row = tl.program_id(0)
        base = row * K
        s = tl.load(SROW + row)
        inv = 1.0 / s
        offs = tl.arange(0, BLOCK_K)
        for k0 in range(0, K, BLOCK_K):
            idx = k0 + offs
            m = idx < K
            v = tl.load(X + base + idx, mask=m, other=0.0).to(tl.float32) * inv
            q = tl.minimum(tl.maximum(v, -MAX), MAX).to(tl.float8e4nv)
            tl.store(Y + base + idx, q, mask=m)

    @triton.jit
    def _wgrad_fp8_kernel(DY, S_DY, X, S_X, DW, M, N, K,
                          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                          BLK: tl.constexpr):
        """dW[N,K] = sum_m dY[m,N]^T @ X[m,K]; per-BLK-token scales factor out per m-block."""
        n0 = tl.program_id(0) * BN
        k0 = tl.program_id(1) * BK
        offs_n = n0 + tl.arange(0, BN)
        offs_k = k0 + tl.arange(0, BK)
        acc = tl.zeros((BN, BK), dtype=tl.float32)
        offs_m = tl.arange(0, BM)
        for m0 in range(0, M, BM):
            dy = tl.load(DY + (m0 + offs_m)[:, None] * N + offs_n[None, :],
                         mask=(m0 + offs_m)[:, None] < M, other=0.0)
            x = tl.load(X + (m0 + offs_m)[:, None] * K + offs_k[None, :],
                        mask=(m0 + offs_m)[:, None] < M, other=0.0)
            part = tl.dot(tl.trans(dy), x)  # [BN, BK] fp32
            s = tl.load(S_DY + m0 // BLK) * tl.load(S_X + m0 // BLK)
            acc += part * s
        tl.store(DW + offs_n[:, None] * K + offs_k[None, :], acc.to(DW.dtype.element_ty),
                 mask=(offs_n[:, None] < N) & (offs_k[None, :] < K))


def quant_rowwise_block(x: torch.Tensor, block: int = 128):
    """[M,K] bf16 -> (fp8 [M,K], s_row [M,1] fp32, s_blk [M/block] fp32).

    One amax per ``block`` consecutive tokens (used by the fp8 wgrad kernel: the scale factor
    ``s_dy[m]*s_x[m]`` is then constant per m-block and can be applied to the m-block partial
    product).  ``s_row`` broadcasts that block scale to the RowWise ``torch._scaled_mm`` contract.
    """
    assert x.is_cuda and x.dim() == 2
    m, k = x.shape
    assert m % block == 0, f"quant_rowwise_block: M={m} not divisible by block={block}"
    y = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    row_amax = torch.empty((m,), dtype=torch.float32, device=x.device)
    if _HAS_TRITON and _quant_impl() != "torch":
        try:
            _row_amax_kernel[(m,)](x, row_amax, k, BLOCK_K=256, num_warps=4)
        except Exception as exc:  # noqa: BLE001
            _warn_once("quant_block", f"triton row-amax failed ({type(exc).__name__}): {exc}")
            row_amax.copy_(x.float().abs().amax(dim=1))
    else:
        row_amax.copy_(x.float().abs().amax(dim=1))
    s_blk = (row_amax.view(-1, block).amax(dim=1).clamp_min(_EPS) / E4M3).contiguous()
    s_row = s_blk.repeat_interleave(block).view(m, 1).contiguous()
    if _HAS_TRITON and _quant_impl() != "torch":
        try:
            _cast_block_kernel[(m,)](x, s_row, y, k, BLOCK_K=256, MAX=E4M3, num_warps=4)
        except Exception as exc:  # noqa: BLE001
            _warn_once("cast_block", f"triton block cast failed ({type(exc).__name__}): {exc}")
            y.copy_((x.float() / s_row).clamp(-E4M3, E4M3))
    else:
        y.copy_((x.float() / s_row).clamp(-E4M3, E4M3))
    return y, s_row, s_blk


if _HAS_TRITON:

    @triton.jit
    def _wgrad_fp8_split_kernel(DY, S_DY, X, S_X, WORK, M, N, K,
                                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                                BLK: tl.constexpr, SPLITS: tl.constexpr):
        """Split-K over the token dim: each program reduces one (N tile, K tile, M chunk).

        dW[n,k] = sum_m s_dy[m]*s_x[m] * dY[m,n]*X[m,k]; scales are constant per BLK-token
        block (BLK == scale block == BM), so they factor out of the tl.dot.
        """
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        pid_s = tl.program_id(2)
        n0 = pid_n * BN
        k0 = pid_k * BK
        chunk = (tl.cdiv(tl.cdiv(M, SPLITS), BLK)) * BLK
        m_start = pid_s * chunk
        m_end = tl.minimum(m_start + chunk, M)
        offs_n = n0 + tl.arange(0, BN)
        offs_k = k0 + tl.arange(0, BK)
        offs_m = tl.arange(0, BM)
        acc = tl.zeros((BN, BK), dtype=tl.float32)
        for m0 in range(m_start, m_end, BM):
            rows = m0 + offs_m
            rmask = rows < m_end
            nmask = offs_n < N
            kmask = offs_k < K
            dy = tl.load(DY + rows[:, None] * N + offs_n[None, :],
                         mask=rmask[:, None] & nmask[None, :], other=0.0)
            x = tl.load(X + rows[:, None] * K + offs_k[None, :],
                        mask=rmask[:, None] & kmask[None, :], other=0.0)
            part = tl.dot(tl.trans(dy), x)  # [BN, BK] fp32
            s = tl.load(S_DY + m0 // BLK) * tl.load(S_X + m0 // BLK)
            acc += part * s
        base = WORK + pid_s * (N * K)
        tl.store(base + offs_n[:, None] * K + offs_k[None, :], acc,
                 mask=(offs_n[:, None] < N) & (offs_k[None, :] < K))

    @triton.jit
    def _wgrad_reduce_kernel(WORK, DW, N, K, BN: tl.constexpr, BK: tl.constexpr,
                             SPLITS: tl.constexpr):
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = pid_k * BK + tl.arange(0, BK)
        mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        acc = tl.zeros((BN, BK), dtype=tl.float32)
        for sp in range(SPLITS):
            acc += tl.load(WORK + sp * (N * K) + offs_n[:, None] * K + offs_k[None, :],
                           mask=mask, other=0.0)
        tl.store(DW + offs_n[:, None] * K + offs_k[None, :], acc.to(DW.dtype.element_ty),
                 mask=mask)


def wgrad_fp8_split(dy8: torch.Tensor, sdy_blk: torch.Tensor, x8: torch.Tensor,
                    sx_blk: torch.Tensor, n: int, k: int,
                    splits: int | None = None, target_ctas: int = 384) -> torch.Tensor:
    """dW [N,K] = dY^T @ X in fp8 with split-K over M (fp32 workspace + two-stage reduction).

    ``sdy_blk``/``sx_blk`` are per-``BLK=128``-token scale vectors (``quant_rowwise_block``).
    """
    m = dy8.shape[0]
    bn = 128 if n % 128 == 0 else (64 if n % 64 == 0 else 32)
    bk = 128 if k % 128 == 0 else (64 if k % 64 == 0 else 32)
    if splits is None:
        base = triton.cdiv(n, bn) * triton.cdiv(k, bk)
        splits = max(1, min(16, -(-target_ctas // base)))
    work = torch.empty((splits, n, k), dtype=torch.float32, device=dy8.device)
    dw = torch.empty((n, k), dtype=torch.bfloat16, device=dy8.device)
    _wgrad_fp8_split_kernel[(triton.cdiv(n, bn), triton.cdiv(k, bk), splits)](
        dy8, sdy_blk, x8, sx_blk, work, m, n, k,
        BM=128, BN=bn, BK=bk, BLK=128, SPLITS=splits, num_warps=4, num_stages=2)
    _wgrad_reduce_kernel[(triton.cdiv(n, bn), triton.cdiv(k, bk))](
        work, dw, n, k, BN=bn, BK=bk, SPLITS=splits, num_warps=4)
    return dw


# ------------------------------------------------------------------ R85: vendor (cuBLASLt) wgrad
# The Triton/TileLang kernels above plateau at 0.5-1.2x bf16 on sm_120 because transposed-A FP8
# MMA lowering is poor.  cuBLASLt *does* have fast transposed-A FP8 kernels on this part
# (measured 480-600 TF/s raw), but requires the operands pre-transposed to its "A row-major /
# B column-major" contract, and only supports per-tensor scales.  So: rescale the existing
# per-block-quantised fp8 activations to a tensor-wide scale while transposing them (one fused
# Triton kernel, 1B read + 1B write per element), then call cuBLASLt.  Env:
#   NANOV41_LINEAR_FP8_V2_WGRAD=2  (vendor path; 1 = Triton split-K; 0 = bf16 cuBLAS, default)
_LT_EXT = [None, False]  # [callable, tried]


def _load_wgrad_lt():
    if _LT_EXT[1]:
        return _LT_EXT[0]
    _LT_EXT[1] = True
    try:
        import os
        try:
            import ninja
            os.environ.setdefault("PATH", "")
            if ninja.BIN_DIR not in os.environ["PATH"]:
                os.environ["PATH"] = ninja.BIN_DIR + ":" + os.environ["PATH"]
        except Exception:  # noqa: BLE001
            pass
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
        from pathlib import Path
        from torch.utils.cpp_extension import load
        root = Path(__file__).resolve().parents[1]
        src = root / "csrc" / "wgrad_lt.cu"
        if not src.exists():
            return None
        build = root / "build_r85"
        build.mkdir(exist_ok=True)
        fn = load(name="wgrad_lt_r85", sources=[str(src)], build_directory=str(build),
                  extra_ldflags=["-lcublasLt", "-lcublas"], extra_cflags=["-O3"], verbose=False)
        _LT_EXT[0] = fn.wgrad_lt
        print("[fp8v2] cuBLASLt fp8 wgrad extension loaded (transposed-A, no copies)", flush=True)
        return _LT_EXT[0]
    except Exception as exc:  # noqa: BLE001
        print(f"[fp8v2] cuBLASLt wgrad extension unavailable: {type(exc).__name__}: {exc}",
              flush=True)
        return None


if _HAS_TRITON:

    @triton.jit
    def _rescale_transpose_kernel(X, R, Y, M, N, BM: tl.constexpr, BN: tl.constexpr,
                                  MAX: tl.constexpr):
        """[M,N] fp8 * per-128-row ratio -> transposed [N,M] fp8 (rows then scaled coherently)."""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        rmask = offs_m < M
        nmask = offs_n < N
        r = tl.load(R + pid_m)
        v = tl.load(X + offs_m[:, None] * N + offs_n[None, :],
                    mask=rmask[:, None] & nmask[None, :], other=0.0).to(tl.float32) * r
        q = tl.minimum(tl.maximum(v, -MAX), MAX).to(tl.float8e4nv)
        tl.store(Y + offs_n[:, None] * M + offs_m[None, :], tl.trans(q),
                 mask=nmask[:, None] & rmask[None, :])


def rescale_transpose_fp8(x8: torch.Tensor, s_blk: torch.Tensor):
    """[M,N] fp8 (per-128-token block scales) -> ([N,M] fp8 tensor-wide scale, scale[1])."""
    m, n = x8.shape
    s_glob = s_blk.max().clamp_min(_EPS)
    ratio = (s_blk / s_glob).float().contiguous()
    y = torch.empty((n, m), dtype=torch.float8_e4m3fn, device=x8.device)
    _rescale_transpose_kernel[(triton.cdiv(m, 128), triton.cdiv(n, 128))](
        x8, ratio, y, m, n, BM=128, BN=128, MAX=E4M3, num_warps=4)
    return y, s_glob.reshape(1).contiguous()


def wgrad_lt(dy8: torch.Tensor, sdy_blk: torch.Tensor, x8: torch.Tensor,
             sx_blk: torch.Tensor, n: int, k: int):
    """dW [N,K] bf16 via cuBLASLt transposed-A fp8 GEMM (per-tensor scales, fused transpose)."""
    fn = _load_wgrad_lt()
    if fn is None:
        return None
    xt, sx = rescale_transpose_fp8(x8, sx_blk)      # [K,M], scale
    dyt, sdy = rescale_transpose_fp8(dy8, sdy_blk)  # [N,M], scale
    return fn(xt, dyt, sx, sdy, 1)


def wgrad_mode() -> int:
    raw = os.environ.get("NANOV41_LINEAR_FP8_V2_WGRAD", "0").strip().lower()
    try:
        return int(raw)
    except ValueError:
        return 0


def wgrad_fp8(dy8: torch.Tensor, sdy_blk: torch.Tensor, x8: torch.Tensor,
              sx_blk: torch.Tensor, n: int, k: int) -> torch.Tensor:
    """dW [N,K] = dY^T @ X in fp8 (requires per-block token scales from ``quant_rowwise_block``)."""
    m = dy8.shape[0]
    dw = torch.empty((n, k), dtype=torch.bfloat16, device=dy8.device)
    bn = 128 if n % 128 == 0 else (64 if n % 64 == 0 else 32)
    bk = 128 if k % 128 == 0 else 64
    _wgrad_fp8_kernel[(triton.cdiv(n, bn), triton.cdiv(k, bk))](
        dy8, sdy_blk, x8, sx_blk, dw, m, n, k,
        BM=128, BN=bn, BK=bk, BLK=128, num_warps=4, num_stages=2)
    return dw


def quant_rowwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[M,K] bf16 -> (fp8 [M,K], per-row scale [M,1] fp32)."""
    assert x.is_cuda and x.dim() == 2, "quant_rowwise expects a CUDA 2-D tensor"
    m, k = x.shape
    y = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty((m,), dtype=torch.float32, device=x.device)
    if _quant_impl() == "torch" or not _HAS_TRITON:
        xf = x.float()
        amax = xf.abs().amax(dim=1).clamp_min(_EPS)
        s.copy_(amax / E4M3)
        y.copy_((xf / s[:, None]).clamp(-E4M3, E4M3))
    else:
        try:
            _rowwise_quant_kernel[(m,)](x, y, s, k, BLOCK_K=256, EPS=_EPS, MAX=E4M3, num_warps=4)
        except Exception as exc:  # noqa: BLE001 - fall back to the torch path on any Triton issue
            _warn_once("quant", f"triton rowwise quant failed ({type(exc).__name__}: {exc}); "
                                "falling back to torch")
            xf = x.float()
            amax = xf.abs().amax(dim=1).clamp_min(_EPS)
            s.copy_(amax / E4M3)
            y.copy_((xf / s[:, None]).clamp(-E4M3, E4M3))
    return y, s.view(m, 1)


def quant_per_tensor(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[N,K] -> (fp8 [N,K], scalar scale); runs out-of-graph once per optimizer step."""
    amax = w.detach().float().abs().amax().clamp_min(_EPS)
    s = (amax / E4M3).reshape(())
    q = (w.detach().float() / s).clamp(-E4M3, E4M3).to(torch.float8_e4m3fn)
    return q, s


# ----------------------------------------------------------------------------- weight cache
def quant_per_channel(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[N,K] -> ([N,K] fp8, [1,N] fp32 scale): one scale per output channel (per row of W).

    ``torch._scaled_mm`` RowWise needs ``scale_b`` shaped (1, N); per-output-channel weight
    scaling is the natural (and more accurate than per-tensor) choice for the forward GEMM.
    """
    wd = w.detach().float()
    s = (wd.abs().amax(dim=1).clamp_min(_EPS) / E4M3).contiguous()
    q = (wd / s[:, None]).clamp(-E4M3, E4M3).to(torch.float8_e4m3fn)
    return q.contiguous(), s.view(1, -1)


class FP8WeightSlot:
    """Static fp8 weight buffers, refreshed in place after each optimizer step (WCACHE pattern).

    RowWise scaling contract of ``torch._scaled_mm``:
      * fwd   : A=x [M,K] rows, B=W^T [K,N] -> scale_a (M,1), scale_b (1,N)
                storage: ``w8_f`` [N,K] fp8 (per-output-channel scales ``ws_n`` [1,N]), passed as
                ``w8_f.t()`` (stride(0)==1, column-major view — no copy).
      * dgrad : A=dy [M,N] rows, B=W [N,K] -> scale_a (M,1), scale_b (1,K)
                storage: ``w8_d`` [K,N] fp8 (per-input-channel scales ``ws_k`` [1,K] = W^T rows),
                passed as ``w8_d.t()``.
    """

    __slots__ = ("w8_f", "ws_n", "w8_d", "ws_k")

    def __init__(self, w: torch.Tensor):
        n, k = w.shape
        self.w8_f = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=w.device)
        self.ws_n = torch.empty((1, n), dtype=torch.float32, device=w.device)
        self.w8_d = torch.empty((k, n), dtype=torch.float8_e4m3fn, device=w.device)
        self.ws_k = torch.empty((1, k), dtype=torch.float32, device=w.device)
        self.refresh(w)

    def refresh(self, w: torch.Tensor) -> None:
        with torch.no_grad():
            w8f, s_n = quant_per_channel(w)
            w8d, s_k = quant_per_channel(w.t())
            self.w8_f.copy_(w8f)
            self.ws_n.copy_(s_n)
            self.w8_d.copy_(w8d)
            self.ws_k.copy_(s_k)

    def bytes(self) -> int:
        return self.w8_f.numel() + self.w8_d.numel() + self.ws_n.numel() * 4 + self.ws_k.numel() * 4


# ----------------------------------------------------------------------------- functional
def linear_fwd(x: torch.Tensor, slot: FP8WeightSlot) -> torch.Tensor:
    """y = x @ W^T with FP8 A/B. x: [..., K] bf16; W cached [N, K]."""
    k = x.shape[-1]
    x2 = x.reshape(-1, k)
    assert x2.dtype == torch.bfloat16, "linear_fwd expects bf16 activations"
    x8, sx = quant_rowwise(x2)
    y = torch._scaled_mm(x8, slot.w8_f.t(), scale_a=sx, scale_b=slot.ws_n, out_dtype=torch.bfloat16)
    return y.view(*x.shape[:-1], slot.w8_f.shape[0])


def linear_dgrad(dy: torch.Tensor, slot: FP8WeightSlot) -> torch.Tensor:
    """dx = dy @ W with FP8 A/B. dy: [..., N]; cached transposed weight gives [K, N]."""
    n = dy.shape[-1]
    dy2 = dy.reshape(-1, n)
    dy8, sdy = quant_rowwise(dy2)
    dx = torch._scaled_mm(dy8, slot.w8_d.t(), scale_a=sdy, scale_b=slot.ws_k, out_dtype=torch.bfloat16)
    return dx.view(*dy.shape[:-1], slot.w8_d.shape[0])


def wgrad_enabled() -> bool:
    # OFF by default: the current Triton trans-dot wgrad kernel is ~7x slower than cuBLAS
    # bf16 at training shapes (grid too small; needs split-K over M).  See docs/fp8-first-plan.md §5.
    return _flag("NANOV41_LINEAR_FP8_V2_WGRAD", "0")


class FusedFP8Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, slot: FP8WeightSlot) -> torch.Tensor:
        ctx.slot = slot
        ctx.x_shape = x.shape
        k = x.shape[-1]
        x2 = x.reshape(-1, k)
        assert x2.dtype == torch.bfloat16
        # Producer-fused cast (R83): if the upstream kernel (RMSNorm/SwiGLU/wo_a) already
        # emitted an FP8 copy + per-row scales for *this exact tensor*, skip the quant pass.
        hit = _castreg.lookup(x) if _castreg.enabled() else None
        if hit is not None:
            x8, sx_row = hit
            sx_blk = None
        else:
            x8, sx_row, sx_blk = quant_rowwise_block(x2)
        y = torch._scaled_mm(x8, slot.w8_f.t(), scale_a=sx_row, scale_b=slot.ws_n,
                             out_dtype=torch.bfloat16)
        ctx.save_for_backward(x2, x8, sx_blk, weight)
        return y.view(*x.shape[:-1], slot.w8_f.shape[0])

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x2, x8, sx_blk, _weight = ctx.saved_tensors
        slot: FP8WeightSlot = ctx.slot
        dy_dtype = dy.dtype
        dy = dy.to(torch.bfloat16)
        dy2 = dy.reshape(-1, dy.shape[-1])
        dy8, sdy_row, sdy_blk = quant_rowwise_block(dy2)
        n, k = slot.w8_f.shape
        x_shape = ctx.x_shape
        if dgrad_enabled():
            dx = torch._scaled_mm(dy8, slot.w8_d.t(), scale_a=sdy_row, scale_b=slot.ws_k,
                                  out_dtype=torch.bfloat16).view(x_shape)
        else:
            dx = torch.mm(dy2, _weight.t()).view(x_shape)
        _mode = wgrad_mode()
        dw = None
        if _mode == 2 and sx_blk is not None:
            dw = wgrad_lt(dy8, sdy_blk, x8, sx_blk, n, k)  # R85 cuBLASLt transposed-A fp8
            if dw is not None:
                _ST.wgrad_fp8_calls += 1
        if dw is None and _mode == 1 and sx_blk is not None:
            # R84: split-K over M + fp32 workspace two-stage reduction.
            dw = wgrad_fp8_split(dy8, sdy_blk, x8, sx_blk, n, k)
            _ST.wgrad_fp8_calls += 1
        if dw is None:
            dw = torch.mm(dy2.t(), x2)
            _ST.wgrad_bf16_calls += 1
        if dx.dtype != dy_dtype:
            dx = dx.to(dy_dtype)
        return dx, dw, None


# ----------------------------------------------------------------------------- install / state
class _State:
    slots: dict[int, FP8WeightSlot]
    weights: dict[int, torch.Tensor]
    names: dict[int, str]
    orig_linear: object | None

    def __init__(self) -> None:
        self.slots = {}
        self.weights = {}
        self.names = {}
        self.orig_linear = None
        self.wgrad_fp8_calls = 0
        self.wgrad_bf16_calls = 0


_ST = _State()


# routed expert weights: names are "....ffn.experts.<idx>.w*" -> ".experts." excludes them
# while keeping "....ffn.shared_experts.w*" eligible.
_EXCLUDE = ("indexer", "gate", "head", "norm", "embed", "sinkhorn", ".experts.")


def eligible_name(name: str) -> bool:
    """Default policy: all eligible linears except precision-sensitive modules."""
    if any(x in name for x in _EXCLUDE):
        return False
    scope = _scope()
    return (not scope) or any(s in name for s in scope)


def register_weights(model) -> int:
    """Allocate fp8 weight slots for every eligible 2-D bf16 Linear weight (call before capture)."""
    _ST.slots.clear()
    _ST.weights.clear()
    _ST.names.clear()
    n = 0
    names: list[str] = []
    for name, p in model.named_parameters():
        if not name.endswith(".weight") or p.dim() != 2 or p.dtype != torch.bfloat16:
            continue
        kk, nn = p.shape[1], p.shape[0]
        if kk % 16 or nn % 16 or p.numel() == 0:
            continue
        if not eligible_name(name):
            continue
        _ST.weights[id(p)] = p
        _ST.names[id(p)] = name
        _ST.slots[id(p)] = FP8WeightSlot(p.detach())
        n += 1
        names.append(name)
    return n


def install(M) -> int:
    """Patch the module-global ``M.linear`` dispatcher; returns the number of weight slots."""
    if _ST.orig_linear is None:
        _ST.orig_linear = M.linear
    orig = _ST.orig_linear

    def linear(x, weight, bias=None):
        slot = _ST.slots.get(id(weight)) if isinstance(weight, torch.Tensor) else None
        if slot is None or bias is not None or not x.is_cuda or x.dtype != torch.bfloat16:
            return orig(x, weight, bias)
        if x.dim() not in (2, 3):
            return orig(x, weight, bias)
        if _min_flops() and 2.0 * x.reshape(-1, x.shape[-1]).shape[0] * weight.shape[0] * weight.shape[1] < _min_flops():
            return orig(x, weight, bias)
        return FusedFP8Linear.apply(x, weight, slot)

    M.linear = linear
    return len(_ST.slots)


def maybe_refresh() -> int:
    """Refresh all cached fp8 weights in place (call after optimizer step, outside graph)."""
    if not _ST.slots:
        return 0
    with torch.no_grad():
        for wid, slot in _ST.slots.items():
            w = _ST.weights.get(wid)
            if w is not None:
                slot.refresh(w)
    return len(_ST.slots)


def stats() -> dict:
    return {
        "n_weights": len(_ST.slots),
        "weight_fp8_bytes": sum(s.bytes() for s in _ST.slots.values()),
        "names": sorted(_ST.names.values()),
        "wgrad_fp8_calls": _ST.wgrad_fp8_calls,
        "wgrad_bf16_calls": _ST.wgrad_bf16_calls,
    }

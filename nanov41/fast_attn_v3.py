"""fast_attn_v3 — attention v2 with the compress-dkv contribution buffer in bf16 (R63).

## Problem

`tl_sparse_attn_v2._V2.backward` materialises the per-query compress-key dkv contributions
as `dkv_c[b, m, nc, d]` **fp32**, then a deterministic dest-major reduce pass reads it back:

    write  16*1024*32*64*4 B = 134 MB / layer      (b=16, m=1024, nc=32, d=64)
    read   134 MB / layer  ( `_dkv_comp_reduce_kernel` )
    -> ~268 MB/layer, x11 compress layers (compress_ratios[0]==0) = ~2.9 GB/step

The window part is already handled without any such buffer (key-stationary, R8), so this is
the single largest byte item inside the 5.70 ms attention segment.  The values are a sum of
two `T.gemm`s over `g*nc = 64` rows whose inputs are already fp16 (p̃ / ds) and bf16 (q / do);
storing the fp32 accumulator as bf16 costs ~2^-9 relative per contribution, and the reduce
still accumulates in fp32.

## What this module does

Re-installs `v2._bwd` / `v2._scatter_compressed` with a bf16 `dkv_c` (same kernels, same
deterministic sort+dest-major reduce, same numerics class) — opt-in via
`NANOV41_ATTN_DKV_BF16=1`, default path untouched.  Halves the buffer traffic (~-0.9 ms).

`install()` is idempotent; `uninstall()` restores the fp32 path.
"""
import os

import torch

from . import tl_sparse_attn_v2 as v2

try:  # pragma: no cover - GPU only
    import tilelang
    import tilelang.language as T

    HAS_TILELANG = True
except Exception:  # noqa: BLE001
    HAS_TILELANG = False
    tilelang = None
    T = None

BF16 = "bfloat16"
FP32 = "float32"
INT32 = "int32"

_STATE = {"installed": False, "prev_bwd": None, "prev_scatter": None,
          "kernel_failed": False, "calls": 0}


if HAS_TILELANG:

    @tilelang.jit(pass_configs=v2.pass_configs)
    def _comp_reduce_bf16_kernel(d: int, nc: int, threads: int):
        """Same dest-major serial accumulate as v2's kernel, contrib in bf16."""
        b = T.symbolic("b")
        m = T.symbolic("m")
        n = T.symbolic("n")
        K = T.symbolic("K")
        bn = T.symbolic("bn")

        @T.prim_func
        def kernel(
            contrib: T.Tensor[(b, m, nc, d), BF16],
            perm: T.Tensor[(K,), INT32],
            off: T.Tensor[(bn + 1,), INT32],
            dkv: T.Tensor[(b, n, d), FP32],
        ):
            with T.Kernel(n - m, b, threads=threads) as (bx, by):
                acc = T.alloc_fragment((d,), FP32)
                for j in T.Parallel(d):
                    acc[j] = 0.0
                for i in T.serial(off[by * n + m + bx + 1] - off[by * n + m + bx]):
                    r = perm[off[by * n + m + bx] + i] - by * (m * nc)
                    for j in T.Parallel(d):
                        acc[j] += contrib[by, r // nc, r % nc, j]
                for j in T.Parallel(d):
                    dkv[by, m + bx, j] += acc[j]

        return kernel

    def _scatter_compressed_bf16(dkv_c, comp, dkv_flat, dkv3, n):
        b, m, nc, d = dkv_c.shape
        perm, off, _ = v2._comp_scatter_layout(comp, n)
        if _STATE["kernel_failed"]:
            return v2._scatter_compressed_eb(dkv_c.float(), comp, dkv3, n)
        try:
            k = _comp_reduce_bf16_kernel(d, nc, min(d, v2.THREADS))
            with v2._TorchStream():
                k(dkv_c, perm, off, dkv3)
        except Exception as exc:  # noqa: BLE001
            if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
                raise
            _STATE["kernel_failed"] = True
            print(f"[fast_attn_v3] bf16 compress reduce kernel 不可用 "
                  f"({type(exc).__name__}: {exc}); 回退 eb（bf16->fp32 拷贝）", flush=True)
            v2._scatter_compressed_eb(dkv_c.float(), comp, dkv3, n)
else:  # pragma: no cover
    def _scatter_compressed_bf16(dkv_c, comp, dkv_flat, dkv3, n):
        raise RuntimeError("tilelang unavailable")


def _bwd_bf16(q, kv, do, lse, sink, idxs, win, nc, rows, scale, dkv_win_kernel,
              dkv_flat, dkv3, n):
    """`tl_sparse_attn_v2._bwd` with `dkv_c` in bf16 (everything else identical)."""
    b, m, h, d = q.shape
    _STATE["calls"] += 1
    dq = torch.empty_like(q)
    rowsum = torch.empty(b, m, h, dtype=torch.float32, device=q.device)
    dsink_rows = torch.empty(b, m, h, dtype=torch.float32, device=q.device)
    dkv_c = torch.empty(b, m, max(nc, 1), d, dtype=torch.bfloat16, device=q.device)
    lb = v2._lb(win)
    k = v2._bwd_kernel(h, d, win, nc, scale, v2.G, lb, v2.THREADS, bf16_dkv=True)
    with v2._TorchStream():
        k(q, kv, do, lse, sink, idxs, dq, rowsum, dsink_rows, dkv_c)
        dkv_win_kernel(q, kv, do, lse, rowsum, dkv3)
    if nc:
        _scatter_compressed_bf16(dkv_c, idxs[:, :, win:win + nc], dkv_flat, dkv3, n)
    return dq, dkv_flat, dsink_rows


def install() -> dict:
    """Patch `tl_sparse_attn_v2` so the compress dkv buffer is bf16.  Idempotent."""
    if not available():
        return {"installed": False, "reason": "tilelang/cuda unavailable"}
    if _STATE["installed"]:
        return dict(_STATE)
    _STATE["prev_bwd"] = v2._bwd
    _STATE["prev_scatter"] = v2._scatter_compressed
    v2._bwd = _bwd_bf16
    v2._scatter_compressed = _scatter_compressed_bf16
    _STATE["installed"] = True
    return dict(_STATE)


def uninstall() -> None:
    if not _STATE["installed"]:
        return
    v2._bwd = _STATE["prev_bwd"]
    v2._scatter_compressed = _STATE["prev_scatter"]
    _STATE["installed"] = False


def available() -> bool:
    return bool(HAS_TILELANG and torch.cuda.is_available())


def get_stats() -> dict:
    return dict(_STATE)

"""官方结构的 sparse attention（训练版）：TileLang fwd + bwd。

## 设计来源
`ref/kernel.py::sparse_attn_kernel`（V4.1 官方 inference kernel，随仓库收录于 `ref/`）：
- **grid=(m,b)：一个 program 处理一个 query 位置**（不是 query tile）；
- 该位置**所有 head 的 q 一次载入**（h pad 到 16）⇒ KV gather 被 16 个 head 共享；
- `window ∪ compress` 是**一个 topk 索引列表**（`topk_idxs [b,m,topk]`），没有掩码/分段逻辑；
- 每个 topk block（64）：gather `block×d` KV → `T.gemm(q, kvᵀ)` → online softmax → `T.gemm(acc_s, kv)`；
- sink 折入 `sum_exp += exp(sink − max)`。

## 本文件新增（训练必需）
1. fwd 额外输出 **lse [b,m,h]**（= max + log(sum_exp)），bwd 用它重算 p̃；
2. bwd 同结构：重算 s/p，`dp = do·kvᵀ`，`ds = p̃·(dp − Σp̃dp)`，
   `dq = scale·ds·kv`、`dkv = scale·dsᵀ·q`（**按 topk 位置顺序写成 [b,m,topk,d] 贡献**，
   再由 torch `index_add_` 聚到 dkv —— 不用原子加，保证确定性）、`dsink = −p̃_sink·Σp̃dp`。

## 为什么 dkv 写成贡献缓冲
直接对 dkv[b,n,d] 做原子加会引入非确定顺序（破坏"graph vs eager 逐位一致"的项目约定）。
写成 [b,m,topk,d] 的贡献（bf16/fp32）+ torch index_add 是**确定**的；代价是一块显存。
"""
import os
import sys

import torch

try:
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

pass_configs = None
if HAS_TILELANG:
    pass_configs = {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    }

BLOCK = 64
THREADS = 256

# TVM-FFI 默认在自己记录的 stream 上启动 kernel；CUDA graph capture 时必须显式切到 torch 的当前流，
# 否则 capture 会被打断（cudaErrorStreamCaptureInvalidated）。`use_torch_stream()` 就是官方为此提供的。
try:
    import tvm_ffi as _tvm_ffi
    _HAS_TVM_FFI = True
except Exception:  # noqa: BLE001
    _tvm_ffi = None
    _HAS_TVM_FFI = False


class _TorchStream:
    def __enter__(self):
        if _HAS_TVM_FFI:
            self._ctx = _tvm_ffi.use_torch_stream()
            self._ctx.__enter__()
        return self

    def __exit__(self, *a):
        if _HAS_TVM_FFI:
            self._ctx.__exit__(*a)
        return False


if HAS_TILELANG:

    @tilelang.jit(pass_configs=pass_configs)
    def sparse_attn_fwd_kernel(h: int, d: int, topk: int, scale: float):
        b = T.symbolic("b")
        m = T.symbolic("m")
        n = T.symbolic("n")
        nb = (topk + BLOCK - 1) // BLOCK

        @T.prim_func
        def kernel(
            q: T.Tensor[(b, m, h, d), BF16],
            kv: T.Tensor[(b, n, d), BF16],
            o: T.Tensor[(b, m, h, d), BF16],
            lse: T.Tensor[(b, m, h), FP32],
            attn_sink: T.Tensor[(h,), FP32],
            topk_idxs: T.Tensor[(b, m, topk), INT32],
        ):
            with T.Kernel(m, b, threads=THREADS) as (bx, by):
                q_shared = T.alloc_shared((h, d), BF16)
                kv_shared = T.alloc_shared((BLOCK, d), BF16)
                o_shared = T.alloc_shared((h, d), BF16)
                acc_s_cast = T.alloc_shared((h, BLOCK), BF16)
                idxs = T.alloc_fragment(BLOCK, INT32)
                acc_s = T.alloc_fragment((h, BLOCK), FP32)
                acc_o = T.alloc_fragment((h, d), FP32)
                scores_max = T.alloc_fragment(h, FP32)
                scores_max_prev = T.alloc_fragment(h, FP32)
                scores_scale = T.alloc_fragment(h, FP32)
                scores_sum = T.alloc_fragment(h, FP32)
                sum_exp = T.alloc_fragment(h, FP32)
                lse_f = T.alloc_fragment(h, FP32)

                T.clear(acc_o)
                T.clear(sum_exp)
                T.fill(scores_max, -1e30)
                T.copy(q[by, bx, :, :], q_shared)
                for t in T.Pipelined(nb, num_stages=2):
                    for i in T.Parallel(BLOCK):
                        idxs[i] = T.if_then_else(t * BLOCK + i < topk,
                                                 topk_idxs[by, bx, t * BLOCK + i], -1)
                    for i, j in T.Parallel(BLOCK, d):
                        kv_shared[i, j] = T.if_then_else(idxs[i] != -1, kv[by, idxs[i], j], 0)
                    for i, j in T.Parallel(h, BLOCK):
                        acc_s[i, j] = T.if_then_else(idxs[j] != -1, 0, -T.infinity(FP32))
                    T.gemm(q_shared, kv_shared, acc_s, transpose_B=True,
                           policy=T.GemmWarpPolicy.FullRow)
                    for i, j in T.Parallel(h, BLOCK):
                        acc_s[i, j] *= scale
                    T.copy(scores_max, scores_max_prev)
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(h):
                        scores_scale[i] = T.exp(scores_max_prev[i] - scores_max[i])
                    for i, j in T.Parallel(h, BLOCK):
                        acc_s[i, j] = T.exp(acc_s[i, j] - scores_max[i])
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(h):
                        sum_exp[i] = sum_exp[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)
                    for i, j in T.Parallel(h, d):
                        acc_o[i, j] *= scores_scale[i]
                    T.gemm(acc_s_cast, kv_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                for i in T.Parallel(h):
                    sum_exp[i] += T.exp(attn_sink[i] - scores_max[i])
                    lse_f[i] = scores_max[i] + T.log(sum_exp[i])
                for i, j in T.Parallel(h, d):
                    acc_o[i, j] /= sum_exp[i]
                T.copy(acc_o, o_shared)
                T.copy(o_shared, o[by, bx, :, :])
                T.copy(lse_f, lse[by, bx, :])

        return kernel

    @tilelang.jit(pass_configs=pass_configs)
    def sparse_attn_bwd_kernel(h: int, d: int, topk: int, scale: float):
        b = T.symbolic("b")
        m = T.symbolic("m")
        n = T.symbolic("n")
        nb = (topk + BLOCK - 1) // BLOCK

        @T.prim_func
        def kernel(
            q: T.Tensor[(b, m, h, d), BF16],
            kv: T.Tensor[(b, n, d), BF16],
            do: T.Tensor[(b, m, h, d), BF16],
            lse: T.Tensor[(b, m, h), FP32],
            attn_sink: T.Tensor[(h,), FP32],
            topk_idxs: T.Tensor[(b, m, topk), INT32],
            dq: T.Tensor[(b, m, h, d), BF16],
            dkv_contrib: T.Tensor[(b, m, nb, BLOCK, d), BF16],
            dsink_rows: T.Tensor[(b, m, h), FP32],
        ):
            with T.Kernel(m, b, threads=THREADS) as (bx, by):
                # 反向统一用 fp16 共享内存：ds 用 fp16 尾数更准，且 gemm 不允许 fp16×bf16 混用
                q_shared = T.alloc_shared((h, d), "float16")
                do_shared = T.alloc_shared((h, d), "float16")
                kv_shared = T.alloc_shared((BLOCK, d), "float16")
                ds_shared = T.alloc_shared((h, BLOCK), "float16")
                p_shared = T.alloc_shared((h, BLOCK), "float16")
                dq_shared = T.alloc_shared((h, d), BF16)
                idxs = T.alloc_fragment(BLOCK, INT32)
                acc_s = T.alloc_fragment((h, BLOCK), FP32)
                dp = T.alloc_fragment((h, BLOCK), FP32)
                tmp = T.alloc_fragment((h, BLOCK), FP32)
                row_sum = T.alloc_fragment(h, FP32)
                dq_f = T.alloc_fragment((h, d), FP32)
                dkv_f = T.alloc_fragment((BLOCK, d), FP32)
                dkv_cur = T.alloc_shared((BLOCK, d), BF16)
                lse_f = T.alloc_fragment(h, FP32)
                sink_f = T.alloc_fragment(h, FP32)
                sink_term = T.alloc_fragment(h, FP32)

                T.clear(dq_f)
                T.clear(sink_term)
                T.copy(q[by, bx, :, :], q_shared)
                T.copy(do[by, bx, :, :], do_shared)
                for i in T.Parallel(h):
                    lse_f[i] = lse[by, bx, i]
                    sink_f[i] = attn_sink[i]
                for t in T.Pipelined(nb, num_stages=1):
                    for i in T.Parallel(BLOCK):
                        idxs[i] = T.if_then_else(t * BLOCK + i < topk,
                                                 topk_idxs[by, bx, t * BLOCK + i], -1)
                    for i, j in T.Parallel(BLOCK, d):
                        kv_shared[i, j] = T.if_then_else(idxs[i] != -1, kv[by, idxs[i], j], 0)
                    # s = scale·q·kvᵀ；p̃ = exp(s − lse)（无效列 = 0）
                    for i, j in T.Parallel(h, BLOCK):
                        acc_s[i, j] = T.if_then_else(idxs[j] != -1, 0, -T.infinity(FP32))
                    T.gemm(q_shared, kv_shared, acc_s, transpose_B=True,
                           policy=T.GemmWarpPolicy.FullRow)
                    for i, j in T.Parallel(h, BLOCK):
                        acc_s[i, j] = T.exp(acc_s[i, j] * scale - lse_f[i])
                    T.copy(acc_s, p_shared)   # p̃：dkv 的 value path 需要
                    # dp = do·kvᵀ
                    T.clear(dp)
                    T.gemm(do_shared, kv_shared, dp, transpose_B=True,
                           policy=T.GemmWarpPolicy.FullRow)
                    for i, j in T.Parallel(h, BLOCK):
                        tmp[i, j] = acc_s[i, j] * dp[i, j]
                    T.reduce_sum(tmp, row_sum, dim=1)
                    # ds = p̃·(dp − Σp̃dp)；dqv = scale·ds·kv；dkv = scale·dsᵀ·q
                    for i in T.Parallel(h):
                        sink_term[i] += T.exp(sink_f[i] - lse_f[i]) * row_sum[i]
                    for i, j in T.Parallel(h, BLOCK):
                        acc_s[i, j] = acc_s[i, j] * (dp[i, j] - row_sum[i])
                    T.copy(acc_s, ds_shared)   # fp32 fragment → fp16 shared（精度更好）
                    T.gemm(ds_shared, kv_shared, dq_f, policy=T.GemmWarpPolicy.FullRow)
                    T.clear(dkv_f)
                    T.gemm(ds_shared, q_shared, dkv_f, transpose_A=True,
                           policy=T.GemmWarpPolicy.FullRow)
                    for i, j in T.Parallel(BLOCK, d):
                        dkv_f[i, j] = dkv_f[i, j] * scale
                    T.gemm(p_shared, do_shared, dkv_f, transpose_A=True,
                           policy=T.GemmWarpPolicy.FullRow)
                    T.copy(dkv_f, dkv_cur)
                    # 注意：score 路径已在上面乘过 scale，value 路径不需要 ⇒ 这里**不能**再乘 scale
                    for i, j in T.Parallel(BLOCK, d):
                        dkv_contrib[by, bx, t, i, j] = T.if_then_else(
                            idxs[i] != -1, dkv_cur[i, j], 0.0)
                for i, j in T.Parallel(h, d):
                    dq_f[i, j] = dq_f[i, j] * scale
                T.copy(dq_f, dq_shared)
                T.copy(dq_shared, dq[by, bx, :, :])
                for i in T.Parallel(h):
                    dsink_rows[by, bx, i] = -sink_term[i]

        return kernel

    @tilelang.jit(pass_configs=pass_configs)
    def sparse_attn_bwd_atomic_kernel(h: int, d: int, topk: int, scale: float):
        b = T.symbolic("b")
        m = T.symbolic("m")
        n = T.symbolic("n")
        nb = (topk + BLOCK - 1) // BLOCK

        @T.prim_func
        def kernel(
            q: T.Tensor[(b, m, h, d), BF16],
            kv: T.Tensor[(b, n, d), BF16],
            do: T.Tensor[(b, m, h, d), BF16],
            lse: T.Tensor[(b, m, h), FP32],
            attn_sink: T.Tensor[(h,), FP32],
            topk_idxs: T.Tensor[(b, m, topk), INT32],
            dq: T.Tensor[(b, m, h, d), BF16],
            dkv: T.Tensor[(b, n + 1, d), FP32],
            dsink_rows: T.Tensor[(b, m, h), FP32],
        ):
            with T.Kernel(m, b, threads=THREADS) as (bx, by):
                # 反向统一用 fp16 共享内存：ds 用 fp16 尾数更准，且 gemm 不允许 fp16×bf16 混用
                q_shared = T.alloc_shared((h, d), "float16")
                do_shared = T.alloc_shared((h, d), "float16")
                kv_shared = T.alloc_shared((BLOCK, d), "float16")
                ds_shared = T.alloc_shared((h, BLOCK), "float16")
                p_shared = T.alloc_shared((h, BLOCK), "float16")
                dq_shared = T.alloc_shared((h, d), BF16)
                idxs = T.alloc_fragment(BLOCK, INT32)
                acc_s = T.alloc_fragment((h, BLOCK), FP32)
                dp = T.alloc_fragment((h, BLOCK), FP32)
                tmp = T.alloc_fragment((h, BLOCK), FP32)
                row_sum = T.alloc_fragment(h, FP32)
                dq_f = T.alloc_fragment((h, d), FP32)
                dkv_f = T.alloc_fragment((BLOCK, d), FP32)
                dkv_cur = T.alloc_shared((BLOCK, d), BF16)
                lse_f = T.alloc_fragment(h, FP32)
                sink_f = T.alloc_fragment(h, FP32)
                sink_term = T.alloc_fragment(h, FP32)

                T.clear(dq_f)
                T.clear(sink_term)
                T.copy(q[by, bx, :, :], q_shared)
                T.copy(do[by, bx, :, :], do_shared)
                for i in T.Parallel(h):
                    lse_f[i] = lse[by, bx, i]
                    sink_f[i] = attn_sink[i]
                for t in T.Pipelined(nb, num_stages=1):
                    for i in T.Parallel(BLOCK):
                        idxs[i] = T.if_then_else(t * BLOCK + i < topk,
                                                 topk_idxs[by, bx, t * BLOCK + i], -1)
                    for i, j in T.Parallel(BLOCK, d):
                        kv_shared[i, j] = T.if_then_else(idxs[i] != -1, kv[by, idxs[i], j], 0)
                    # s = scale·q·kvᵀ；p̃ = exp(s − lse)（无效列 = 0）
                    for i, j in T.Parallel(h, BLOCK):
                        acc_s[i, j] = T.if_then_else(idxs[j] != -1, 0, -T.infinity(FP32))
                    T.gemm(q_shared, kv_shared, acc_s, transpose_B=True,
                           policy=T.GemmWarpPolicy.FullRow)
                    for i, j in T.Parallel(h, BLOCK):
                        acc_s[i, j] = T.exp(acc_s[i, j] * scale - lse_f[i])
                    T.copy(acc_s, p_shared)   # p̃：dkv 的 value path 需要
                    # dp = do·kvᵀ
                    T.clear(dp)
                    T.gemm(do_shared, kv_shared, dp, transpose_B=True,
                           policy=T.GemmWarpPolicy.FullRow)
                    for i, j in T.Parallel(h, BLOCK):
                        tmp[i, j] = acc_s[i, j] * dp[i, j]
                    T.reduce_sum(tmp, row_sum, dim=1)
                    # ds = p̃·(dp − Σp̃dp)；dqv = scale·ds·kv；dkv = scale·dsᵀ·q
                    for i in T.Parallel(h):
                        sink_term[i] += T.exp(sink_f[i] - lse_f[i]) * row_sum[i]
                    for i, j in T.Parallel(h, BLOCK):
                        acc_s[i, j] = acc_s[i, j] * (dp[i, j] - row_sum[i])
                    T.copy(acc_s, ds_shared)   # fp32 fragment → fp16 shared（精度更好）
                    T.gemm(ds_shared, kv_shared, dq_f, policy=T.GemmWarpPolicy.FullRow)
                    T.clear(dkv_f)
                    T.gemm(ds_shared, q_shared, dkv_f, transpose_A=True,
                           policy=T.GemmWarpPolicy.FullRow)
                    for i, j in T.Parallel(BLOCK, d):
                        dkv_f[i, j] = dkv_f[i, j] * scale
                    T.gemm(p_shared, do_shared, dkv_f, transpose_A=True,
                           policy=T.GemmWarpPolicy.FullRow)
                    # dkv 直接原子加（无效位置落到垃圾桶行 n）。这里直接读全局 topk_idxs，避免
                    # 与 dkv_f 的 fragment layout 冲突（TileLang 的 layout 推断限制）。
                    for i, j in T.Parallel(BLOCK, d):
                        idx_i = T.if_then_else(t * BLOCK + i < topk,
                                               topk_idxs[by, bx, t * BLOCK + i], -1)
                        T.atomic_add(dkv[by, T.if_then_else(idx_i != -1, idx_i, n), j],
                                     dkv_f[i, j])
                for i, j in T.Parallel(h, d):
                    dq_f[i, j] = dq_f[i, j] * scale
                T.copy(dq_f, dq_shared)
                T.copy(dq_shared, dq[by, bx, :, :])
                for i in T.Parallel(h):
                    dsink_rows[by, bx, i] = -sink_term[i]

        return kernel

    def sparse_attn_fwd(q, kv, sink, idxs, scale):
        b, m, h, d = q.shape
        topk = idxs.size(-1)
        o = torch.empty_like(q)
        lse = torch.empty(b, m, h, dtype=torch.float32, device=q.device)
        kernel = sparse_attn_fwd_kernel(h, d, topk, scale)
        with _TorchStream():
            kernel(q, kv, o, lse, sink, idxs)
        return o, lse

    def sparse_attn_bwd_atomic(q, kv, do, lse, sink, idxs, scale):
        b, m, h, d = q.shape
        topk = idxs.size(-1)
        n = kv.size(1)
        dq = torch.empty_like(q)
        dkv = torch.zeros(b, n + 1, d, dtype=torch.float32, device=q.device)
        dsink_rows = torch.empty(b, m, h, dtype=torch.float32, device=q.device)
        kernel = sparse_attn_bwd_atomic_kernel(h, d, topk, scale)
        with _TorchStream():
            kernel(q, kv, do, lse, sink, idxs, dq, dkv, dsink_rows)
        return dq, dkv, dsink_rows

    def sparse_attn_bwd(q, kv, do, lse, sink, idxs, scale):
        b, m, h, d = q.shape
        topk = idxs.size(-1)
        n = kv.size(1)
        dq = torch.empty_like(q)
        nb = (topk + BLOCK - 1) // BLOCK
        dkv_contrib = torch.empty(b, m, nb, BLOCK, d, dtype=torch.bfloat16, device=q.device)
        dsink_rows = torch.empty(b, m, h, dtype=torch.float32, device=q.device)
        kernel = sparse_attn_bwd_kernel(h, d, topk, scale)
        with _TorchStream():
            kernel(q, kv, do, lse, sink, idxs, dq, dkv_contrib, dsink_rows)
        return dq, dkv_contrib.view(b, m, nb * BLOCK, d)[:, :, :topk], dsink_rows


class _TLSparseAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, sink, idxs, scale):
        h = q.size(2)
        pad = (16 - h) if h < 16 else 0
        qp = q
        sp = sink
        if pad:
            qp = torch.cat([q, q.new_zeros(*q.shape[:2], pad, q.shape[3])], dim=2)
            sp = torch.cat([sink, sink.new_zeros(pad)])
        qp = qp.contiguous()
        kv = kv.contiguous()
        idxs = idxs.contiguous()
        o, lse = sparse_attn_fwd(qp, kv, sp.contiguous(), idxs, scale)
        if pad:
            o = o[:, :, :h].contiguous()
        ctx.save_for_backward(qp, kv, sp.contiguous(), idxs, lse)
        ctx.scale, ctx.h, ctx.pad = scale, h, pad
        ctx.dkv_mode = os.environ.get("TL_ATTN_DKV", "atomic").lower()
        return o

    @staticmethod
    def _aggregate_contrib(contrib, idxs, kv):
        """确定性聚合（contrib 模式）：静态形状 + 垃圾桶行，避免 nonzero() 破坏 capture。"""
        b, m, topk, d = contrib.shape
        n = kv.size(1)
        flat_idx = idxs.reshape(-1).to(torch.int64)
        valid = flat_idx >= 0
        dkv = torch.zeros(b * n + 1, d, dtype=torch.bfloat16, device=kv.device)
        batch_off = (torch.arange(b, device=kv.device).view(b, 1, 1) * n).expand(b, m, topk)
        target = torch.where(valid, batch_off.reshape(-1) + flat_idx.clamp_min(0),
                             torch.full_like(flat_idx, b * n))
        dkv.index_add_(0, target, contrib.reshape(-1, d))
        return dkv[:b * n].view(b, n, d).to(kv.dtype)

    @staticmethod
    def backward(ctx, do):
        _STATS["bwd_calls"] += 1
        qp, kv, sp, idxs, lse = ctx.saved_tensors
        h, pad = ctx.h, ctx.pad
        dop = do
        if pad:
            dop = torch.cat([do, do.new_zeros(*do.shape[:2], pad, do.shape[3])], dim=2)
        if ctx.dkv_mode == "contrib":
            dqp, contrib, dsink_rows = sparse_attn_bwd(qp, kv, dop.contiguous(), lse, sp, idxs, ctx.scale)
            dkv = _TLSparseAttn._aggregate_contrib(contrib, idxs, kv)
        else:
            dqp, dkv_raw, dsink_rows = sparse_attn_bwd_atomic(qp, kv, dop.contiguous(), lse, sp, idxs, ctx.scale)
            dkv = dkv_raw[:, :kv.size(1)].to(kv.dtype)
        if pad:
            dqp = dqp[:, :, :h].contiguous()
        dsink = dsink_rows.sum(dim=(0, 1)).to(torch.float32)
        if pad:
            dsink = dsink[:h].contiguous()
        return dqp, dkv, dsink, None, None


_STATS = {"fwd_calls": 0, "bwd_calls": 0}


def stats():
    return dict(_STATS)


def reset_stats():
    _STATS["fwd_calls"] = 0
    _STATS["bwd_calls"] = 0


def sparse_attn_tl(q, kv, attn_sink, topk_idxs, softmax_scale):
    _STATS["fwd_calls"] += 1
    return _TLSparseAttn.apply(q, kv, attn_sink, topk_idxs, softmax_scale)


def available() -> bool:
    return HAS_TILELANG and torch.cuda.is_available()


_PREV = {}


def install(M):
    prev = getattr(M, "sparse_attn", None)
    _PREV[id(M)] = prev
    M.sparse_attn = sparse_attn_tl
    return prev

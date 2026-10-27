"""训练版 sparse attention v2（TileLang）：按 query 组（G=2）单 tile 融合 + 无 pad。

## 相对 v1（`tl_sparse_attn.py`）的三处结构性改动

1. **fwd：单 tile，没有 online-softmax 循环。** v1 沿用官方 inference 结构（每 query 一个
   program、topk 切成 64 的 block、逐块 rescale）。但训练场景的 key 数固定（window 64 +
   compress 32），可以一次 gather 成 [W, d] 单 tile、一次 gemm、一次 softmax。实测
   b=16 fwd 0.255 → 0.054 ms（4.7×），因为不再有 block 循环带来的 barrier/流水开销。

2. **h 不再 pad 到 16，改成 G=2 个相邻 query 共享一个 CTA。** v1 把 8 个 head pad 到 16
   去做 M=16 的 gemm（50% 空转）。这里把**两个相邻 query**的 8×2=16 行拼成一个真正的
   M=16 tile；相邻 query 的窗口只差一列，取 union（win+G-1=65 列，装进 96 列 tile）后
   gather 量从 2×96 降到 65+64。
   （TileLang 约束：T.gemm 的 M 必须是 16 的倍数，所以 h=8 单 query 无法直接做。）

3. **bwd 的 dkv 拆成两条路，避开 1.1 ms 的 torch `index_add_`。** v1 把所有 topk 槽位
   （含 window）的贡献写成 [b,m,128,d] bf16（268 MB/层）再 `index_add_`；实测这一步
   1.10 ms/层，是 attention 段最大的单项。问题在 window 部分：每个 key 有 64 个
   contributor（contention 极高），index_add 只有 ~350 GB/s。v2 拆成：
     - **window dkv**：key-stationary（每个 CTA 拥有 64 个 key，重算 s/dp），
       输出不相交、无需原子加/无 contrib 缓冲，确定性；
     - **compress dkv**：每 query 只 32 槽，贡献缓冲缩到 [b,m,32,d]（67 MB），
       `index_add_` 0.19 ms。

## 数值口径
与 `fast_attn` 一致：scores/probs 存 fp16、tensor-core 累加 fp32、softmax 归约 fp32，
sink 只进分母（`sum_exp += exp(sink − max)`），无效槽位 = −inf/−1。

## 确定性（R8 修订）
- fwd / bwd 主 kernel：每个输出元素由固定 CTA + 固定归约顺序得到；
- window dkv：key-stationary，key tile 互不相交，tile 内按 query chunk 固定顺序累加；
- compress dkv：**dest-major 确定性 reduce（默认）**。先用 `torch.sort(stable=True)` 把
  [b,m,nc] 个贡献按目标行排序（每行 = batch*n + key，无效槽位统一落到哨兵行 b*n），
  再用一个 program-per-dest 的 kernel 按**升序贡献 id（= 升序 query、升序槽位）**串行累加。
  排序是确定性的（CUB radix sort，无原子），累加顺序只取决于索引内容 ⇒ 同一输入逐位一致。
  成本见 `research/R8-attn-v2-determinism.md`（目标 b=16 单层 ≤0.3 ms）。
  另有两种等价实现（同一份 perm/off、同一累加顺序）：`eb` = 把 reduce 交给 torch
  `F.embedding_bag(mode="sum")`（每个 bag 一个 block 按序累加，纯 torch 回退）；
  `index_add` = 旧实现（PyTorch CUDA 的 `index_add_` 用 `atomicAdd`，**非确定**，保留做 A/B）。
- 没有 float 原子加 ⇒ 同一输入下 graph replay 与 eager 逐位一致（项目约定）。

## 启用
`import nanov41.tl_sparse_attn_v2 as v2; v2.available(); v2.install(M)`（幂等，可重复调用），
`v2.uninstall(M)` 还原；`v2.scatter_mode()` / `v2.stats()` 可查当前聚合实现与调用计数。
环境变量：`NANOV41_ATTN_V2_SCATTER`（`sort` 默认 / `eb` / `index_add` 旧路径）、
`NANOV41_ATTN_V2_OLD_SCATTER=1`（等价于 `index_add`）。

## 布局前提
kernel 用**闭式 band**公式生成 window 索引（`clamp(i-win+1,0)+c`，超过 i 的置 −1），
不读 `topk_idxs[..., :win]`。入口用 `fast_attn.detect_layout` 逐元素校验，不匹配就回退
到 v1（`tl_sparse_attn`）或参考实现，因此不会在未验证的布局上走快路。
"""
import os

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
FP16 = "float16"
FP32 = "float32"
INT32 = "int32"

pass_configs = None
if HAS_TILELANG:
    pass_configs = {
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    }

try:  # TVM-FFI 默认在自己的 stream 上启动 kernel，CUDA graph capture 必须切到 torch 当前流
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


# R47：G/THREADS 可扫（G=2 是 R41 的默认：h=8 ⇒ M=16 正好一个 tensor-core tile；
# 加大 G 让 CTA 复用同一份 window union KV，但 W=lb+g*nc 会推高 smem，可能超 99 KB 上限）
G = int(os.environ.get("NANOV41_ATTN_V2_G", "2"))
THREADS = int(os.environ.get("NANOV41_ATTN_V2_THREADS", "128"))


def _lb(win, g=G):
    """窗口 union 的 tile 宽度：>= win+g-1，且 32 对齐（4 warps × 8 列）。"""
    need = win + g - 1
    return ((need + 31) // 32) * 32


if HAS_TILELANG:

    # ------------------------------------------------------------------ fwd
    @tilelang.jit(pass_configs=pass_configs)
    def _fwd_kernel(h: int, d: int, win: int, nc: int, scale: float, g: int, lb: int, threads: int):
        b = T.symbolic("b")
        m = T.symbolic("m")
        n = T.symbolic("n")
        GH = g * h
        W = lb + g * nc
        topk = win + nc

        @T.prim_func
        def kernel(
            q: T.Tensor[(b, m, h, d), BF16],
            kv: T.Tensor[(b, n, d), BF16],
            o: T.Tensor[(b, m, h, d), BF16],
            lse: T.Tensor[(b, m, h), FP32],
            attn_sink: T.Tensor[(h,), FP32],
            idxs: T.Tensor[(b, m, topk), INT32],
        ):
            with T.Kernel(T.ceildiv(m, g), b, threads=threads) as (bx, by):
                q_s = T.alloc_shared((GH, d), BF16)
                kv_s = T.alloc_shared((W, d), BF16)
                o_s = T.alloc_shared((GH, d), BF16)
                p_s = T.alloc_shared((GH, W), BF16)
                acc_s = T.alloc_fragment((GH, W), FP32)
                acc_o = T.alloc_fragment((GH, d), FP32)
                rmax = T.alloc_fragment(GH, FP32)
                rsum = T.alloc_fragment(GH, FP32)

                for r, j in T.Parallel(GH, d):
                    q_s[r, j] = q[by, bx * g + r // h, r % h, j]
                # window union: 绝对位置 [i0-win+1, i0-win+1+lb)
                for j, c in T.Parallel(lb, d):
                    key = bx * g - win + 1 + j
                    kv_s[j, c] = T.if_then_else((key >= 0) & (key < m), kv[by, key, c], 0)
                # compress: query g 的 nc 列放在 [lb + g*nc, lb + (g+1)*nc)
                if nc:
                    for j, c in T.Parallel(g * nc, d):
                        ix = idxs[by, bx * g + j // nc, win + j % nc]
                        kv_s[lb + j, c] = T.if_then_else(ix >= 0, kv[by, ix, c], 0)
                # 有效位：window 用闭式 band；compress 只对本 query 自己的 nc 列有效
                if nc:
                    for r, j in T.Parallel(GH, W):
                        qi = bx * g + r // h
                        key = bx * g - win + 1 + j
                        okw = (key >= 0) & (key < m) & (key <= qi) & (key >= qi - win + 1)
                        okc = ((j - lb) // nc == r // h) & (idxs[by, qi, win + (j - lb) % nc] >= 0)
                        acc_s[r, j] = T.if_then_else(T.if_then_else(j < lb, okw, okc), 0.0, -T.infinity(FP32))
                else:
                    for r, j in T.Parallel(GH, W):
                        qi = bx * g + r // h
                        key = bx * g - win + 1 + j
                        okw = (key >= 0) & (key < m) & (key <= qi) & (key >= qi - win + 1)
                        acc_s[r, j] = T.if_then_else(okw, 0.0, -T.infinity(FP32))
                T.gemm(q_s, kv_s, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for r, j in T.Parallel(GH, W):
                    acc_s[r, j] = acc_s[r, j] * scale
                T.reduce_max(acc_s, rmax, dim=1, clear=True)
                for r, j in T.Parallel(GH, W):
                    acc_s[r, j] = T.exp(acc_s[r, j] - rmax[r])
                T.reduce_sum(acc_s, rsum, dim=1)
                T.copy(acc_s, p_s)
                T.clear(acc_o)
                T.gemm(p_s, kv_s, acc_o, policy=T.GemmWarpPolicy.FullRow)
                for r, j in T.Parallel(GH, d):
                    acc_o[r, j] = acc_o[r, j] / (rsum[r] + T.exp(attn_sink[r % h] - rmax[r]))
                T.copy(acc_o, o_s)
                for r, j in T.Parallel(GH, d):
                    o[by, bx * g + r // h, r % h, j] = o_s[r, j]
                for r in T.Parallel(GH):
                    lse[by, bx * g + r // h, r % h] = rmax[r] + T.log(
                        rsum[r] + T.exp(attn_sink[r % h] - rmax[r]))
        return kernel

    # ------------------------------------------------------------------ bwd 主 kernel
    @tilelang.jit(pass_configs=pass_configs)
    def _bwd_kernel(h: int, d: int, win: int, nc: int, scale: float, g: int, lb: int, threads: int):
        b = T.symbolic("b")
        m = T.symbolic("m")
        n = T.symbolic("n")
        GH = g * h
        W = lb + g * nc
        topk = win + nc

        @T.prim_func
        def kernel(
            q: T.Tensor[(b, m, h, d), BF16],
            kv: T.Tensor[(b, n, d), BF16],
            do: T.Tensor[(b, m, h, d), BF16],
            lse: T.Tensor[(b, m, h), FP32],
            attn_sink: T.Tensor[(h,), FP32],
            idxs: T.Tensor[(b, m, topk), INT32],
            dq: T.Tensor[(b, m, h, d), BF16],
            rowsum: T.Tensor[(b, m, h), FP32],
            dsink_rows: T.Tensor[(b, m, h), FP32],
            dkv_c: T.Tensor[(b, m, max(nc, 1), d), FP32],
        ):
            with T.Kernel(T.ceildiv(m, g), b, threads=threads) as (bx, by):
                q_s = T.alloc_shared((GH, d), FP16)
                do_s = T.alloc_shared((GH, d), FP16)
                kv_s = T.alloc_shared((W, d), FP16)
                p_s = T.alloc_shared((GH, W), FP16)
                ds_s = T.alloc_shared((GH, W), FP16)
                p_c = T.alloc_shared((GH, max(g * nc, 1)), FP16)
                ds_c = T.alloc_shared((GH, max(g * nc, 1)), FP16)
                dq_s = T.alloc_shared((GH, d), BF16)
                dkc_s = T.alloc_shared((max(g * nc, 1), d), FP32)
                acc_s = T.alloc_fragment((GH, W), FP32)
                dp = T.alloc_fragment((GH, W), FP32)
                tmp = T.alloc_fragment((GH, W), FP32)
                dq_f = T.alloc_fragment((GH, d), FP32)
                dkc_f = T.alloc_fragment((g * nc, d), FP32)
                rs = T.alloc_fragment(GH, FP32)
                sink_term = T.alloc_fragment(GH, FP32)
                lse_f = T.alloc_fragment(GH, FP32)

                if nc:
                    T.clear(dkc_f)
                for r, j in T.Parallel(GH, d):
                    q_s[r, j] = q[by, bx * g + r // h, r % h, j]
                    do_s[r, j] = do[by, bx * g + r // h, r % h, j]
                for r in T.Parallel(GH):
                    lse_f[r] = lse[by, bx * g + r // h, r % h]
                for j, c in T.Parallel(lb, d):
                    key = bx * g - win + 1 + j
                    kv_s[j, c] = T.if_then_else((key >= 0) & (key < m), kv[by, key, c], 0)
                if nc:
                    for j, c in T.Parallel(g * nc, d):
                        ix = idxs[by, bx * g + j // nc, win + j % nc]
                        kv_s[lb + j, c] = T.if_then_else(ix >= 0, kv[by, ix, c], 0)
                if nc:
                    for r, j in T.Parallel(GH, W):
                        qi = bx * g + r // h
                        key = bx * g - win + 1 + j
                        okw = (key >= 0) & (key < m) & (key <= qi) & (key >= qi - win + 1)
                        okc = ((j - lb) // nc == r // h) & (idxs[by, qi, win + (j - lb) % nc] >= 0)
                        acc_s[r, j] = T.if_then_else(T.if_then_else(j < lb, okw, okc), 0.0, -T.infinity(FP32))
                else:
                    for r, j in T.Parallel(GH, W):
                        qi = bx * g + r // h
                        key = bx * g - win + 1 + j
                        okw = (key >= 0) & (key < m) & (key <= qi) & (key >= qi - win + 1)
                        acc_s[r, j] = T.if_then_else(okw, 0.0, -T.infinity(FP32))
                T.gemm(q_s, kv_s, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                # p̃ = exp(s·scale − lse)，无效列 = 0
                for r, j in T.Parallel(GH, W):
                    acc_s[r, j] = T.exp(acc_s[r, j] * scale - lse_f[r])
                T.copy(acc_s, p_s)
                T.clear(dp)
                T.gemm(do_s, kv_s, dp, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for r, j in T.Parallel(GH, W):
                    tmp[r, j] = acc_s[r, j] * dp[r, j]
                T.reduce_sum(tmp, rs, dim=1)   # rowsum = Σ p̃·dp（整行，含 sink 项之外的项）
                for r in T.Parallel(GH):
                    sink_term[r] = T.exp(attn_sink[r % h] - lse_f[r]) * rs[r]
                for r, j in T.Parallel(GH, W):
                    acc_s[r, j] = acc_s[r, j] * (dp[r, j] - rs[r])
                T.copy(acc_s, ds_s)
                T.clear(dq_f)
                T.gemm(ds_s, kv_s, dq_f, policy=T.GemmWarpPolicy.FullRow)
                for r, j in T.Parallel(GH, d):
                    dq_f[r, j] = dq_f[r, j] * scale
                T.copy(dq_f, dq_s)
                for r, j in T.Parallel(GH, d):
                    dq[by, bx * g + r // h, r % h, j] = dq_s[r, j]
                for r in T.Parallel(GH):
                    rowsum[by, bx * g + r // h, r % h] = rs[r]
                    dsink_rows[by, bx * g + r // h, r % h] = -sink_term[r]
                # compress 的 dkv 贡献：dsᵀ·q·scale + p̃ᵀ·do（只算 compress 的 g*nc 列）
                if nc:
                    for r, j in T.Parallel(GH, g * nc):
                        ds_c[r, j] = ds_s[r, lb + j] * scale   # score 路径乘 scale
                        p_c[r, j] = p_s[r, lb + j]            # value 路径不乘
                    T.gemm(ds_c, q_s, dkc_f, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)
                    T.gemm(p_c, do_s, dkc_f, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(dkc_f, dkc_s)
                    for j, c in T.Parallel(g * nc, d):
                        dkv_c[by, bx * g + j // nc, j % nc, c] = dkc_s[j, c]
        return kernel

    # ------------------------------------------------------------------ window dkv
    @tilelang.jit(pass_configs=pass_configs)
    def _dkv_win_kernel(h: int, d: int, win: int, scale: float, bk: int, bq: int, threads: int):
        """key-stationary：每个 CTA 拥有 bk 个 window key，重算 s/dp 后累加 dkv。"""
        b = T.symbolic("b")
        m = T.symbolic("m")
        n = T.symbolic("n")
        nn = T.symbolic("nn")
        BR = bq * h              # 每个 query chunk 的行数
        # 因果窗口：key n 被 query [n, n+win-1] 看到 ⇒ 一个 key tile 需要的 query 范围是
        # [k0, k0+bk-1+win-1]，宽度 bk+win-1（下一步的 tile 会接上）。
        R = bk + win - 1

        @T.prim_func
        def kernel(
            q: T.Tensor[(b, m, h, d), BF16],
            kv: T.Tensor[(b, n, d), BF16],
            do: T.Tensor[(b, m, h, d), BF16],
            lse: T.Tensor[(b, m, h), FP32],
            rowsum: T.Tensor[(b, m, h), FP32],
            dkv: T.Tensor[(b, nn, d), FP32],
        ):
            with T.Kernel(T.ceildiv(m, bk), b, threads=threads) as (bx, by):
                k_s = T.alloc_shared((bk, d), FP16)
                q_s = T.alloc_shared((BR, d), FP16)
                do_s = T.alloc_shared((BR, d), FP16)
                p_s = T.alloc_shared((BR, bk), FP16)
                ds_s = T.alloc_shared((BR, bk), FP16)
                s_f = T.alloc_fragment((BR, bk), FP32)
                dp_f = T.alloc_fragment((BR, bk), FP32)
                dkv_f = T.alloc_fragment((bk, d), FP32)
                dkv_s = T.alloc_shared((bk, d), FP32)
                lse_f = T.alloc_fragment(BR, FP32)
                rs_f = T.alloc_fragment(BR, FP32)

                T.clear(dkv_f)
                for j, c in T.Parallel(bk, d):
                    k_s[j, c] = T.if_then_else(bx * bk + j < m, kv[by, bx * bk + j, c], 0)
                for it in T.serial(T.ceildiv(R, bq)):
                    i0 = bx * bk + it * bq
                    for r, c in T.Parallel(BR, d):
                        q_s[r, c] = T.if_then_else((i0 + r // h >= 0) & (i0 + r // h < m),
                                                   q[by, i0 + r // h, r % h, c], 0)
                        do_s[r, c] = T.if_then_else((i0 + r // h >= 0) & (i0 + r // h < m),
                                                    do[by, i0 + r // h, r % h, c], 0)
                    for r in T.Parallel(BR):
                        lse_f[r] = lse[by, T.max(T.min(i0 + r // h, m - 1), 0), r % h]
                        rs_f[r] = rowsum[by, T.max(T.min(i0 + r // h, m - 1), 0), r % h]
                    for r, j in T.Parallel(BR, bk):
                        qi = i0 + r // h
                        key = bx * bk + j
                        ok = (qi >= 0) & (qi < m) & (key <= qi) & (key >= qi - win + 1)
                        s_f[r, j] = T.if_then_else(ok, 0.0, -T.infinity(FP32))
                    T.gemm(q_s, k_s, s_f, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    for r, j in T.Parallel(BR, bk):
                        s_f[r, j] = T.exp(s_f[r, j] * scale - lse_f[r])
                    T.copy(s_f, p_s)
                    T.clear(dp_f)
                    T.gemm(do_s, k_s, dp_f, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    for r, j in T.Parallel(BR, bk):
                        s_f[r, j] = s_f[r, j] * (dp_f[r, j] - rs_f[r]) * scale
                    T.copy(s_f, ds_s)
                    T.gemm(ds_s, q_s, dkv_f, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)
                    T.gemm(p_s, do_s, dkv_f, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)
                T.copy(dkv_f, dkv_s)
                for j, c in T.Parallel(bk, d):
                    dkv[by, bx * bk + j, c] = dkv_s[j, c]
        return kernel

    # ------------------------------------------------------------------ compress dkv 确定性 reduce
    @tilelang.jit(pass_configs=pass_configs)
    def _dkv_comp_reduce_kernel(d: int, nc: int, threads: int):
        """dest-major：每个 program 拥有一个 compress 目标行，按排好序的贡献 id 串行累加。

        输入 `perm` 是 `torch.sort(target, stable=True)` 的索引（target 见 `_scatter_compressed_sorted`），
        所以同一行的贡献在 `perm` 里连续，且按 (query, slot) 升序 ⇒ 固定顺序、无原子加。
        `off` 是每行的起止偏移（`searchsorted(sorted_target, arange(b*n+1))`）。
        """
        b = T.symbolic("b")
        m = T.symbolic("m")
        n = T.symbolic("n")
        K = T.symbolic("K")
        bn = T.symbolic("bn")

        @T.prim_func
        def kernel(
            contrib: T.Tensor[(b, m, nc, d), FP32],   # 贡献 [b,m,nc,d]（fp32，已按 query/slot 展开）
            perm: T.Tensor[(K,), INT32],              # 按目标行排序后的贡献 id（flat id）
            off: T.Tensor[(bn + 1,), INT32],          # 每行在 perm 里的起始偏移（含哨兵行）
            dkv: T.Tensor[(b, n, d), FP32],           # 输出（compress 段行 m..n-1，window 段由 K2 写）
        ):
            with T.Kernel(n - m, b, threads=threads) as (bx, by):
                acc = T.alloc_fragment((d,), FP32)
                for j in T.Parallel(d):
                    acc[j] = 0.0
                # 该目标行的贡献在 perm 里连续：[off[row], off[row+1])；按此顺序串行累加
                for i in T.serial(off[by * n + m + bx + 1] - off[by * n + m + bx]):
                    r = perm[off[by * n + m + bx] + i] - by * (m * nc)   # 本 batch 内的 flat id
                    for j in T.Parallel(d):
                        acc[j] += contrib[by, r // nc, r % nc, j]
                for j in T.Parallel(d):
                    dkv[by, m + bx, j] += acc[j]
        return kernel

    # ------------------------------------------------------------------ 入口
    def _fwd(q, kv, sink, idxs, win, nc, scale):
        b, m, h, d = q.shape
        o = torch.empty_like(q)
        lse = torch.empty(b, m, h, dtype=torch.float32, device=q.device)
        lb = _lb(win)
        k = _fwd_kernel(h, d, win, nc, scale, G, lb, THREADS)
        with _TorchStream():
            k(q, kv, o, lse, sink, idxs)
        return o, lse

    def _bwd(q, kv, do, lse, sink, idxs, win, nc, rows, scale, dkv_win_kernel, dkv_flat, dkv3, n):
        b, m, h, d = q.shape
        dq = torch.empty_like(q)
        rowsum = torch.empty(b, m, h, dtype=torch.float32, device=q.device)
        dsink_rows = torch.empty(b, m, h, dtype=torch.float32, device=q.device)
        dkv_c = torch.empty(b, m, max(nc, 1), d, dtype=torch.float32, device=q.device)
        lb = _lb(win)
        k = _bwd_kernel(h, d, win, nc, scale, G, lb, THREADS)
        with _TorchStream():
            k(q, kv, do, lse, sink, idxs, dq, rowsum, dsink_rows, dkv_c)
            dkv_win_kernel(q, kv, do, lse, rowsum, dkv3)
        if nc:
            _scatter_compressed(dkv_c, idxs[:, :, win:win + nc], dkv_flat, dkv3, n)
        return dq, dkv_flat, dsink_rows


NTRASH = 256   # 旧 index_add 路径：无效槽位（-1）落到 b*n + (flat % NTRASH) 的垃圾桶行

_SCATTER_MODES = ("sort", "eb", "index_add")
_SCATTER = {"mode": "sort"}
if os.environ.get("NANOV41_ATTN_V2_OLD_SCATTER", "0").lower() not in ("0", "false", "no", "off"):
    _SCATTER["mode"] = "index_add"
if os.environ.get("NANOV41_ATTN_V2_SCATTER", "").strip().lower() in _SCATTER_MODES:
    _SCATTER["mode"] = os.environ["NANOV41_ATTN_V2_SCATTER"].strip().lower()


def scatter_mode():
    """当前 compress dkv 聚合实现：`sort`（默认，确定性）/ `eb`（确定性，纯 torch）/ `index_add`（旧，非确定）。"""
    return _SCATTER["mode"]


def set_scatter_mode(mode):
    """切换 compress dkv 聚合实现（测试/A-B 用；`sort` / `eb` 都是确定性路径）。"""
    if mode not in _SCATTER_MODES:
        raise ValueError(f"scatter mode must be one of {_SCATTER_MODES}, got {mode!r}")
    _SCATTER["mode"] = mode


def _scatter_compressed(dkv_c, comp, dkv_flat, dkv3, n):
    mode = _SCATTER["mode"]
    _STATS["scatter_n"] += 1
    if mode == "index_add":
        _scatter_compressed_index_add(dkv_c, comp, dkv_flat, n)
    elif mode == "eb":
        _scatter_compressed_eb(dkv_c, comp, dkv3, n)
    else:
        _scatter_compressed_sorted(dkv_c, comp, dkv3, n)


def _comp_scatter_layout(comp, n):
    """把 [b,m,nc] 的 compress 索引变成 (perm, off, bn)：按目标行稳定排序的反查表。

    目标行 = `batch*n + key`（key ∈ [m,n)）；无效槽位（−1）统一映射到哨兵行 `b*n`
    （贡献恒 0，排序后落在 perm 尾部，不参与任何输出行）。返回：
      * `perm`：按目标行升序、行内按贡献 id（= query/slot）升序的贡献 id 表（int32）；
      * `off`：长度 `b*n+1`，`off[row]` = 该行在 perm 里的起点（`searchsorted`）；
      * `bn = b*n`：哨兵行的行号（也是 off 的最后一个下标）。
    `torch.sort(..., stable=True)` 在 CUDA 上走 CUB radix sort（无原子）⇒ 结果只取决于
    索引内容，与调度/地址无关。
    """
    b, m, nc = comp.shape
    bn = b * n
    # comp 是 topk_idxs[..., win:win+nc] 的**非连续视图**：直接在 3D 上算，避免 reshape 触发 copy
    boff = (torch.arange(b, dtype=torch.int32, device=comp.device) * n).view(b, 1, 1)
    target = (comp + boff).masked_fill_(comp < 0, bn)          # [b, m, nc] int32（连续）
    sorted_t, perm64 = torch.sort(target.reshape(-1), stable=True)
    off = torch.searchsorted(sorted_t, torch.arange(bn + 1, dtype=torch.int32,
                                                    device=comp.device))
    return perm64.to(torch.int32), off.to(torch.int32), bn


_SCATTER_BROKEN = {"sort": False}


def _scatter_compressed_sorted(dkv_c, comp, dkv3, n):
    """确定性 dest-major reduce：`sort` 建反查表 + 每目标行一个 program 固定顺序累加。

    排序给每个目标行的贡献一个**只由索引内容决定**的顺序（(query, slot) 升序），
    再由 `_dkv_comp_reduce_kernel` 一个 CTA 拥有一个行、按该顺序串行累加 ⇒ 无原子、
    无跨 CTA 归约，eager 与 graph replay 逐位一致。

    kernel 首次使用若编译/启动失败（例如某个 TileLang 版本不接受动态 loop extent），
    在**非捕获**状态下自动降级到 `eb`（同一份 perm/off、同一累加顺序的纯 torch 实现），
    保证默认路径仍然确定性；打印一次告警。
    """
    b, m, nc, d = dkv_c.shape
    perm, off, _ = _comp_scatter_layout(comp, n)
    if _SCATTER_BROKEN["sort"]:
        _STATS["eb_fallback"] += 1
        return _scatter_compressed_eb(dkv_c, comp, dkv3, n)
    try:
        k = _dkv_comp_reduce_kernel(d, nc, min(d, THREADS))
        with _TorchStream():
            k(dkv_c, perm, off, dkv3)
    except Exception as exc:  # noqa: BLE001
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise
        _SCATTER_BROKEN["sort"] = True
        print(f"[tl_sparse_attn_v2] compress reduce kernel 不可用 "
              f"({type(exc).__name__}: {exc}); 降级到确定性 eb 路径", flush=True)
        _STATS["eb_fallback"] += 1
        _scatter_compressed_eb(dkv_c, comp, dkv3, n)


def _scatter_compressed_eb(dkv_c, comp, dkv3, n):
    """同一份反查表交给 `F.embedding_bag(mode='sum')`：每个 bag 一个 block 按序累加。

    纯 torch 的确定性等价实现（不依赖自定义 kernel），留作 kernel 不可用/排障时的回退：
    语义与 `sort` 完全相同（同一 perm/off，同一累加顺序），只是把 reduce 借给 PyTorch 的
    embedding_bag kernel 做。
    """
    b, m, nc, d = dkv_c.shape
    perm, off, _ = _comp_scatter_layout(comp, n)
    out = torch.nn.functional.embedding_bag(perm, dkv_c.reshape(-1, d), off, mode="sum",
                                            include_last_offset=True)
    dkv3[:, m:, :].add_(out.view(b, n, d)[:, m:, :])


def _scatter_compressed_index_add(dkv_c, comp, dkv, n):
    """旧实现（保留做 A/B / 复现历史 parity）：`index_add_` 在 CUDA 上走 atomicAdd，非确定。

    无效槽位（−1）的贡献恒为 0，落到 b*n + (id % NTRASH) 的垃圾桶行以降低争用。
    """
    b, m, nc, d = dkv_c.shape
    flat = comp.reshape(-1).to(torch.int64)
    batch_off = (torch.arange(b, device=dkv.device).view(b, 1, 1) * n).expand(b, m, nc)
    trash = b * n + torch.arange(flat.numel(), device=dkv.device) % NTRASH
    target = torch.where(flat >= 0, batch_off.reshape(-1) + flat.clamp_min(0), trash)
    dkv.index_add_(0, target, dkv_c.reshape(-1, d))   # dkv: [b*n+NTRASH, d]


_STATS = {"fwd_calls": 0, "bwd_calls": 0, "fallback": 0, "scatter_n": 0,
          "eb_fallback": 0}


def stats():
    return dict(_STATS)


class _V2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, sink, idxs, scale, win, nc, rows):
        _STATS["fwd_calls"] += 1
        o, lse = _fwd(q, kv, sink, idxs, win, nc, scale)
        ctx.save_for_backward(q, kv, sink, idxs, lse)
        ctx.cfg = (scale, win, nc, rows)
        return o

    @staticmethod
    def backward(ctx, do):
        _STATS["bwd_calls"] += 1
        q, kv, sink, idxs, lse = ctx.saved_tensors
        scale, win, nc, rows = ctx.cfg
        b, m, h, d = q.shape
        n = kv.size(1)
        # 多一行垃圾桶：无效 key（-1）落到 b*n
        dkv_flat = torch.zeros(b * n + NTRASH, d, dtype=torch.float32, device=q.device)
        dkv3 = dkv_flat[:b * n].view(b, n, d)
        kwin = _dkv_win_kernel(h, d, win, scale, _BKWIN["bk"], _BKWIN["bq"], _BKWIN["threads"])
        dq, _, dsink_rows = _bwd(q, kv, do.contiguous(), lse, sink, idxs, win, nc, rows,
                                 scale, kwin, dkv_flat, dkv3, n)
        dsink = dsink_rows.sum(dim=(0, 1))
        return dq, dkv3.to(kv.dtype), dsink, None, None, None, None, None


_BKWIN = {"bk": 64, "bq": 16, "threads": 128}

_REQUIRE_CACHE = {}


def _supported(topk_idxs, q, kv):
    """校验 prefill band 布局（复用 fast_attn.detect_layout，带缓存避免每步同步）。"""
    from . import fast_attn
    m = q.size(1)
    key = (q.size(0), m, topk_idxs.shape[-1], kv.size(1), topk_idxs.dtype, topk_idxs.stride())
    hit = _REQUIRE_CACHE.get(key)
    if hit is None:
        hit = fast_attn.detect_layout(topk_idxs, m)
        if len(_REQUIRE_CACHE) > 256:
            _REQUIRE_CACHE.clear()
        _REQUIRE_CACHE[key] = hit
    return hit


def sparse_attn_v2(q, kv, attn_sink, topk_idxs, softmax_scale):
    """与 `patched_env.sparse_attn` 同签名；未验证的布局回退到 v1/fast_attn。"""
    if not (HAS_TILELANG and torch.cuda.is_available() and q.is_cuda):
        from .fast_attn import sparse_attn_fast
        return sparse_attn_fast(q, kv, attn_sink, topk_idxs, softmax_scale)
    b, m, h, d = q.shape
    if h * G > 32 or m % G or m < G:  # 形状不满足 kernel 假设 → 回退
        _STATS["fallback"] += 1
        from .fast_attn import sparse_attn_fast
        return sparse_attn_fast(q, kv, attn_sink, topk_idxs, softmax_scale)
    win, nc, ok = _supported(topk_idxs, q, kv)
    if not ok or win < 1:
        _STATS["fallback"] += 1
        from .fast_attn import sparse_attn_fast
        return sparse_attn_fast(q, kv, attn_sink, topk_idxs, softmax_scale)
    q = q.contiguous()
    kv = kv.contiguous()
    idxs = topk_idxs.contiguous()
    sink = attn_sink.float().contiguous()
    return _V2.apply(q, kv, sink, idxs, softmax_scale, win, nc, None)


def available():
    """是否可启用：TileLang 已 import 且 CUDA 可用（与 `sparse_attn_v2` 的回退条件一致）。"""
    return bool(HAS_TILELANG and torch.cuda.is_available())


_PREV = {}


def install(M):
    """幂等安装：`M.sparse_attn = sparse_attn_v2`，返回被替换的原实现（已装过则原样返回）。

    典型用法（lead 侧接线）：
        import nanov41.tl_sparse_attn_v2 as v2
        if v2.available():
            v2.install(M)          # 可重复调用，不会把 v2 自己记成 "prev"
    """
    cur = getattr(M, "sparse_attn", None)
    if cur is sparse_attn_v2:
        return _PREV.get(id(M))
    _PREV[id(M)] = cur
    M.sparse_attn = sparse_attn_v2
    return cur


def uninstall(M):
    """恢复 `install(M)` 之前的实现（重复调用安全）。"""
    if id(M) not in _PREV:
        return
    M.sparse_attn = _PREV.pop(id(M))

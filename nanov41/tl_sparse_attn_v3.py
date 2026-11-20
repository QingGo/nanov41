"""TileLang sparse attention v3 — v1 的 fwd/bwd 结构 + 三条结构性修正（默认 opt-in）。

对齐形状（aligned d12：h=8, d=64, win=128, index_topk=64 ⇒ topk=192, b=1×seq8192）
下，v1（`tl_sparse_attn.py`）单层 8192 token 的字节账（`n = m + m/ratio`，KV 只有 1.5–2 MB，
gather 基本由 L2 服务）：

| 项 | v1 字节/layer | 备注 |
|---|---:|---|
| fwd q/o（h 被 pad 到 16） | 33.6 MB | 另有 torch `cat`(25 MB) + `o[:, :, :h].contiguous()`(17 MB) |
| fwd KV gather | 201 MB | 192 槽位 × 128 B × 8192 query |
| bwd q/do/dq（pad 16） | 50 MB | 另有 torch `cat`(25 MB) + dq slice(17 MB) |
| bwd KV gather | 201 MB | 同上 |
| **bwd dkv fp32 原子加** | **402 MB** | 192 槽位 × 64 d × 4 B × 8192 query |

其中 **window 槽位占 128/192 = 67%（268 MB/layer）**，而 window 是 band 结构：每个 key 被
~128 个 query 命中，散射争用极高（`tools/bench_r6_dkv_agg.py` 实测同形态 `index_add_` 只有
~250 GB/s）。v1 bwd 实测 0.52 ms/layer ≈ 402 MB @ ~0.8 TB/s，与"原子加主导"一致。

## v3 的三处改动

1. **fwd/bwd 不再用 torch 做 h=8→16 的 padding**：q/do 直接按 h 行读入（kernel 内把 shared
   tile 的填充行清零），o/dq/dsink 直接写未 padding 的张量。每层省 4 个 elementwise kernel
   + ~59 MB 流量，并且 ctx 里不再需要保存 padding 后的 q（−200 MB 图内存）。
2. **bwd 的 dkv 拆两条路**（沿用 v2/R8 的结论）：
   - **window key**：`_dkv_win_kernel` 键驻留（每个 CTA 拥有 `bk` 个 key，重算 s/dp 后按
     query 顺序累加，输出互不相交）⇒ **268 MB/layer 的原子加 → 0**；代价是重算
     ~6.4 GFLOP/layer + 3× q/do 重读（50 MB/layer）。
   - **compress key**：仍在主 kernel 里按 query 做 fp32 原子加（134 MB/layer）。
   主 kernel 的 topk block 循环拆成"window 块（只算 rowsum/dq）"+"compress 块（写 dkv）"。
   因为 win 与 nc 都是 `BLOCK=64` 的整数倍，切分是 block 对齐、编译期常量。
3. **修正 v1 的 rowsum bug（可选，默认开）**：v1 在 block 循环里用
   `R_blk = Σ_{j∈block} p̃·dp` 去算 `ds = p̃·(dp − R)`，而正确值是**整行**的
   `R = Σ_{all j} p̃·dp`。CPU fp64 对拍（`tools/bench_tl_attn_v3.py --check`）显示：
   `dq` 相对误差 **20–37%**、`dkv` **0.6–0.9%**、`dsink` 无误差（它按 block 累加，恰好对）。
   v3 用 `R = rowsum(dO ∘ O)`（FlashAttention 的 `D_i`）修掉：恒等式
   `Σ_j p̃_j (do·k_j) = do·(Σ_j p̃_j k_j) = do·o`（o 是归一化后的 fwd 输出），
   所以 fwd 每层多存一份 `o` 的 bf16 影子（`o_sv`，8.4 MB/layer），bwd 开头做一次
   `(hp, d)` 的 fp32 归约即可，**不需要额外 gather、不需要第二遍**。
   与 fp64 参考对比：`dq` 1.3e-3、`dkv` 4e-5（= o 的 bf16 舍入量级，而 dq/dkv 本身输出就是
   bf16，量化台阶 4e-3）。`NANOV41_ATTN_V3_ROWS=block` 可退回 v1 的逐 block 口径做纯速度 A/B。

顺带的好处：total 模式下 bwd 不再需要 per-block 的 `reduce_sum`（少一次跨线程归约 + 一个
fragment），`dsink` 也直接由 R 得到（省掉累加器）。

## tilelang 0.1.15 布局结论（2026-11-11 GPU 实测）

逐 kernel 探针（`tools/bench_tl_attn_v3.py --probe`）结果：

| kernel | 结果 |
|---|---|
| fwd | 编译通过 |
| bwd `rows=pre` / `rows=block` | 编译通过 |
| bwd `rows=total`（kernel 内 `rbuf` + `T.reduce_sum` 求 R） | **`InternalError: no available layout found`** |
| window 键驻留 kernel（orig 256/128 threads、bk32/bq8、shacc 变体、v2 同构） | 全部编译通过 |

根因：`rows=total` 里 `rbuf` 是**free 布局**的 fragment（只从 shared/global 读入写，不参与任何
gemm），再喂给 `T.reduce_sum`；tilelang 的 free-mode layout inference 把组件里每个 op 都当
root 试过仍没有自洽方案 ⇒ `has_best=false`。window kernel 从没失败过。
修法（默认 `ROWS=pre`）：`R = rowsum(dO∘O)` 用 torch 先算好（fp32，每层 ~2 个 elementwise
kernel）作为额外输入传给 bwd kernel，kernel 内不再有 free fragment + reduce，布局推断全部通过；
数值等价（dq 1.9e-3、dkv 1.8e-3，= bf16 输出台阶量级）。`rows=total` 保留给以后 tilelang 版本修复。

## R77 实测（2026-11-11，RTX 5090，aligned d12：h=8/d=64/win128+nc64/topk192）

单层 ×12 层 = per-8k-replay；`tools/bench_tl_attn_v3.py --bench --graph --m 8192`（真实 indexer
统计下的 idx：只有前 ~nc*ratio 行有 -1；合成数据里 50% -1 会把几百万个 -1 原子加到同一垃圾桶行，
把 bwd 虚高 ~1.5 ms/layer，别再用旧造数）：

| variant | fwd | bwd | fwd+bwd | Δ |
|---|---:|---:|---:|---:|
| v1（生产） | 1.90 | 6.56 | **8.45** | — |
| v3 split/pre, THREADS=256 | 1.77 | 4.40 | 6.17 | −2.28 (−27.0%) |
| v3 split/pre, THREADS=128（默认） | 1.59 | 3.39 | **4.98** | **−3.44 (−40.9%)** |
| v3 all/pre, THREADS=256（不拆 win dkv） | 1.75 | 6.79 | 8.54 | +0.09 |

（THREADS=64 单层 4.87 略好于 128 的 4.98，但 fwd 1.17 / bwd 3.71 结构不同且未进图验证，默认仍 128；
THREADS=512 编译失败→回退 v1。窗口 kernel：Triton 键驻留 0.0399 ms/layer vs TileLang 0.0576。）

in-graph 生产（b2/seq8192/accum32，`tools/r77_profile.sh`）：base replay 3045.9 ms/step →
**v3 默认 2835.25 ms/step（−210.7，−6.92%）、169,175 tok/s（+6.9%）、raw MFU 44.33%**。
对拍（m512 vs fp64）：v1 dq **2.935e-1**（bug），v3 dq **1.881e-3**、dkv 1.788e-3、dsink 1.885e-3。

## 数值口径
与 v1 相同：scores/probs 存 fp16、tensor-core fp32 累加、softmax 归约 fp32、sink 只进分母。
fwd 与 v1 的数学完全一致（填充行在两边都是 0）⇒ `o`/`lse` 应逐位相同。

## 启用

    import nanov41.tl_sparse_attn_v3 as v3
    v3.install(M)       # 幂等；M.sparse_attn = sparse_attn_v3
    v3.uninstall(M)

环境变量（默认值 = 本文件推荐的 aligned 形状配置）：

    NANOV41_ATTN_V3_DKV   split（默认）| all（等价 v1 语义，用于逐位自检）| none（诊断上限）
    NANOV41_ATTN_V3_ROWS  pre（默认，修正 dq：R=rowsum(dO∘O) 由 torch 先算好传入）|
                          total（同样修正 dq，但 R 在 kernel 内 rbuf+reduce_sum 算；
                          **tilelang 0.1.15 上 layout 推断失败 → 会回退 v1**，保留待修）|
                          block（v1 的逐 block 口径，只做纯速度 A/B）
    NANOV41_ATTN_V3_THREADS     主 kernel 线程数（默认 128；实测比 256 快 ~30%）
    NANOV41_ATTN_V3_ROPE_IN     1 = q 的 RoPE 折进 kernel（省 2 kernel + 1 clone/层）
    NANOV41_ATTN_V3_ROWSUM_TRITON 1 = R=rowsum(dO∘O) 用 1 个 Triton kernel（省 4 个 aten kernel/层）
    NANOV41_ATTN_V3_WIN  triton（默认，Triton 键驻留，0.0402ms/layer）| orig | shacc
                         （orig/shacc 是 TileLang 实现，0.0576ms/layer；triton 不可用时自动回退 TL）
    NANOV41_ATTN_V3_TBK/TBQ/TWARPS  Triton window kernel 的 tile（默认 32/4/4）
    NANOV41_ATTN_V3_BK/BQ/BWTHREADS  TileLang window kernel 的 tile（默认 64/16/256）

`v3.plan(q, kv, idxs)` 打印/返回本层的切分与是否可走 v3；`v3.stats()` 是调用计数。
不满足布局/整除条件时**回退 v1**（`tl_sparse_attn.sparse_attn_tl`），所以默认安装不会在
未验证的形状上改变行为。
"""
# ⚠️ 不要在本文件加 `from __future__ import annotations`。
# TileLang/TVMScript 解析 `@T.prim_func` 时，是按内层 `kernel` 的 **co_freevars**（闭包）
# 去解析注解里的符号（b/m/n/hp/topk…）的。PEP 563 会把注解变成字符串，于是只在注解里
# 出现的符号（如 fwd 的 `n`、window kernel 的 `nn`）就不再是 kernel 的自由变量，构造时会
# 抛 `NameError: name 'n' is not defined`。v1/v2 同样没有 future import。

import os

import torch

try:  # pragma: no cover - GPU only
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

# TVM-FFI 默认在自己记录的 stream 上启动 kernel；CUDA graph capture 必须显式切到 torch 当前流。
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


BLOCK = 64  # topk block（与 v1 相同，已验证）
# 主 kernel 线程数。R77 实测：128 比 256 快很多（M=16 的 gemm 用 4 warps 更贴合，fwd 1.90→1.59、
# bwd 6.56→3.39 ms/replay；in-graph replay 2924.6→2835.3ms/step）。生产用默认 128。
THREADS = int(os.environ.get("NANOV41_ATTN_V3_THREADS", "128"))

_DKV_MODES = ("split", "all", "none")
_ROWS_MODES = ("total", "block", "pre")  # "pre" = R 由 torch 预计算（布局兜底）
# window 键驻留 kernel 的实现变体：
#   orig  = 累加器 dkv_f 跨 serial 迭代（gemm C 复用，已在 v2 同构代码里用过）
#   shacc = 每次迭代 gemm 进新清空的 dkv_f，再累加进 shared dkv_s（避开跨迭代 C 复用）
_WIN_MODES = ("orig", "shacc", "triton")


def _env_choice(name, default, choices):
    v = os.environ.get(name, "").strip().lower()
    return v if v in choices else default


_CFG = {
    "dkv": _env_choice("NANOV41_ATTN_V3_DKV", "split", _DKV_MODES),
    # 默认 "pre"：total 模式（kernel 内 rbuf → reduce_sum 求 R）在 tilelang 0.1.15 上
    # 触发 `InternalError: no available layout found`（free 布局的 rbuf + reduce_sum），
    # 改成 torch 先把 R=rowsum(dO∘O) 算好传进 kernel（fp32 归约，成本 ~2 个 elementwise
    # kernel/layer），布局推断全部通过，数值等价（dq/dkv 与 fp64 参考差在 bf16 台阶内）。
    "rows": _env_choice("NANOV41_ATTN_V3_ROWS", "pre", _ROWS_MODES),
    # window dkv 默认走 Triton 键驻留 kernel：生产形状实测 0.0402 ms/layer（bk=32,bq=4）
    # vs TileLang 版 0.0576；且完全不依赖 tilelang 的布局推断（更稳）。TL 版仍可选（orig/shacc）。
    "win": _env_choice("NANOV41_ATTN_V3_WIN", "triton", _WIN_MODES),
    # R79 T2-attn: 把 q/o 的 RoPE 折进 attention kernel（省 4 个 _rope_tail_kernel + 2 个
    # backward clone / 层，~0.43 ms/8k 里可拿 ~0.3；kv 的 rope 仍在 fused_rope_attn 里做）。
    # 逐位一致（bf16 舍入次序完全复刻），默认 OFF。
    "rope_in": os.environ.get("NANOV41_ATTN_V3_ROPE_IN", "0").lower()
    not in ("0", "false", "no", "off"),
    # R79 T2-attn: rows="pre" 的 R=rowsum(dO∘O) 原来用 4 个 aten kernel
    # (do.float()/o_sv.float() 两个 fp32 转换 + mul + sum，实测 1.04 ms/replay = 0.52 ms/8k)。
    # 换成 1 个 Triton kernel 直接 fp32 归约（不物化 fp32 中间张量），默认 OFF 待验证。
    "rows_triton": os.environ.get("NANOV41_ATTN_V3_ROWSUM_TRITON", "0").lower()
    not in ("0", "false", "no", "off"),
    "bk": int(os.environ.get("NANOV41_ATTN_V3_BK", "64")),
    "bq": int(os.environ.get("NANOV41_ATTN_V3_BQ", "16")),
    "bwthreads": int(os.environ.get("NANOV41_ATTN_V3_BWTHREADS", "256")),
    # Triton window kernel 的 tile（见 `_dkv_win_triton`）；实测 bk=32,bq=4 最快
    "tbk": int(os.environ.get("NANOV41_ATTN_V3_TBK", "32")),
    "tbq": int(os.environ.get("NANOV41_ATTN_V3_TBQ", "4")),
    "twarps": int(os.environ.get("NANOV41_ATTN_V3_TWARPS", "4")),
}


def _hp_of(h: int) -> int:
    """T.gemm 的 M 必须是 16 的倍数：把 head 数补到 >= 16 的 16 倍数。"""
    return int(h) if int(h) % 16 == 0 else 16


def _split_plan(win: int, nc: int, m: int, h: int, bk: int):
    """把 (win, nc) 变成 block 对齐的切分方案；不可行时返回 None。"""
    if win < BLOCK or win % BLOCK or nc <= 0 or nc % BLOCK:
        return None
    if h > 16 and h % 16:
        return None
    if bk <= 0 or m % bk:
        return None
    topk = win + nc
    nb = (topk + BLOCK - 1) // BLOCK
    nwin = win // BLOCK
    if nwin <= 0 or nb <= nwin:
        return None
    return {"topk": topk, "nb": nb, "nwin": nwin, "ncblk": nb - nwin, "hp": _hp_of(h)}


if HAS_TILELANG:

    # ------------------------------------------------------------------ fwd
    @tilelang.jit(pass_configs=pass_configs)
    def _fwd_kernel(h: int, hp: int, d: int, topk: int, scale: float, rd: int = 0):
        """v1 的 fwd（online softmax over topk blocks），但 h 不 padding、多写一份 o_sv。

        `rd > 0`（R79 rope-in，只折 q）时额外接收 `freqs [m, rd]` fp32：q 的 tail 在 kernel
        内做 forward RoPE（与 `fused_rope_attn._rope_tail_kernel` 的 bf16 舍入次序逐位一致）；
        o 的 rope 仍由调用方在外部做（那样 bwd 的 do 依然是 raw 空间，dkv/win/rowsum 全不变）。
        """
        b = T.symbolic("b")
        m = T.symbolic("m")
        n = T.symbolic("n")
        rdp = max(rd, 1)
        nb = (topk + BLOCK - 1) // BLOCK

        @T.prim_func
        def kernel(
            q: T.Tensor[(b, m, h, d), BF16],
            kv: T.Tensor[(b, n, d), BF16],
            freqs: T.Tensor[(m, rdp), FP32],
            o: T.Tensor[(b, m, h, d), BF16],
            o_sv: T.Tensor[(b, m, h, d), BF16],
            lse: T.Tensor[(b, m, h), FP32],
            attn_sink: T.Tensor[(h,), FP32],
            topk_idxs: T.Tensor[(b, m, topk), INT32],
        ):
            with T.Kernel(m, b, threads=THREADS) as (bx, by):
                q_shared = T.alloc_shared((hp, d), BF16)
                kv_shared = T.alloc_shared((BLOCK, d), BF16)
                o_shared = T.alloc_shared((hp, d), BF16)
                acc_s_cast = T.alloc_shared((hp, BLOCK), BF16)
                idxs = T.alloc_fragment(BLOCK, INT32)
                acc_s = T.alloc_fragment((hp, BLOCK), FP32)
                acc_o = T.alloc_fragment((hp, d), FP32)
                scores_max = T.alloc_fragment(hp, FP32)
                scores_max_prev = T.alloc_fragment(hp, FP32)
                scores_scale = T.alloc_fragment(hp, FP32)
                scores_sum = T.alloc_fragment(hp, FP32)
                sum_exp = T.alloc_fragment(hp, FP32)
                # lse 暂存 shared（而非 hp 片段）：下面只写前 h 行，对 hp 片段做“部分 extent”
                # 访问需要 TileLang 布局推断额外支持，shared 按显式下标访问没有这个问题。
                lse_s = T.alloc_shared((hp,), FP32)
                sink_f = T.alloc_fragment(hp, FP32)

                T.clear(acc_o)
                T.clear(sum_exp)
                # 有限下界（而非 -inf）：整行无有效槽位时 exp(-1e30 - (-1e30)) 会是 NaN。
                T.fill(scores_max, -1e30)
                for i, j in T.Parallel(h, d):
                    q_shared[i, j] = q[by, bx, i, j]
                if hp > h:
                    for i, j in T.Parallel(hp - h, d):
                        q_shared[h + i, j] = 0
                if rd:  # q tail：forward RoPE（与 _rope_tail_kernel 同序：先生成 bf16 再进 fp16）
                    for i, p in T.Parallel(h, rd // 2):
                        re = T.Cast(FP32, q_shared[i, d - rd + 2 * p])
                        im = T.Cast(FP32, q_shared[i, d - rd + 2 * p + 1])
                        fre = freqs[bx, 2 * p]
                        fim = freqs[bx, 2 * p + 1]
                        q_shared[i, d - rd + 2 * p] = T.Cast(
                            FP16, T.Cast(BF16, re * fre - im * fim))
                        q_shared[i, d - rd + 2 * p + 1] = T.Cast(
                            FP16, T.Cast(BF16, re * fim + im * fre))
                for i in T.Parallel(hp):
                    sink_f[i] = T.if_then_else(i < h, attn_sink[T.min(i, h - 1)], 0.0)
                for t in T.Pipelined(nb, num_stages=2):
                    for i in T.Parallel(BLOCK):
                        idxs[i] = T.if_then_else(t * BLOCK + i < topk,
                                                 topk_idxs[by, bx, t * BLOCK + i], -1)
                    for i, j in T.Parallel(BLOCK, d):
                        kv_shared[i, j] = T.if_then_else(idxs[i] != -1,
                                                         kv[by, T.max(idxs[i], 0), j], 0)
                    for i, j in T.Parallel(hp, BLOCK):
                        acc_s[i, j] = T.if_then_else(idxs[j] != -1, 0, -T.infinity(FP32))
                    T.gemm(q_shared, kv_shared, acc_s, transpose_B=True,
                           policy=T.GemmWarpPolicy.FullRow)
                    for i, j in T.Parallel(hp, BLOCK):
                        acc_s[i, j] *= scale
                    T.copy(scores_max, scores_max_prev)
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(hp):
                        scores_scale[i] = T.exp(scores_max_prev[i] - scores_max[i])
                    for i, j in T.Parallel(hp, BLOCK):
                        acc_s[i, j] = T.exp(acc_s[i, j] - scores_max[i])
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(hp):
                        sum_exp[i] = sum_exp[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)
                    for i, j in T.Parallel(hp, d):
                        acc_o[i, j] *= scores_scale[i]
                    T.gemm(acc_s_cast, kv_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                for i in T.Parallel(hp):
                    sum_exp[i] += T.exp(sink_f[i] - scores_max[i])
                    lse_s[i] = scores_max[i] + T.log(sum_exp[i])
                for i, j in T.Parallel(hp, d):
                    acc_o[i, j] /= sum_exp[i]
                T.copy(acc_o, o_shared)
                for i, j in T.Parallel(h, d):
                    o[by, bx, i, j] = o_shared[i, j]
                    o_sv[by, bx, i, j] = o_shared[i, j]
                for i in T.Parallel(h):
                    lse[by, bx, i] = lse_s[i]

        return kernel

    # ------------------------------------------------------------------ bwd 主 kernel
    @tilelang.jit(pass_configs=pass_configs)
    def _bwd_kernel(h: int, hp: int, d: int, topk: int, nwin: int, scale: float,
                    store_window: bool, rows_mode: str = "total", rd: int = 0):
        """v1 的 bwd：window 块（只算 rowsum/dq）+ compress 块（写 dkv 原子加）。

        `rd > 0`（R79 rope-in，只折 q）时接收 `freqs [m, rd]` fp32：输入的 q 是 raw 空间，
        kernel 内在 q_shared 上重建 forward RoPE；写出的 dq 在 raw 空间（写回前在 dq_shared
        上做 inverse RoPE）。`do` 仍是 raw 空间（o 的 rope 在外部，未折叠）⇒ dp/ds/rowsum/
        window-dkv 全部不变。bwd 的恢复/旋转都与 `_rope_tail_kernel` 的 bf16 舍入逐位一致。
        """
        b = T.symbolic("b")
        m = T.symbolic("m")
        n = T.symbolic("n")
        rdp = max(rd, 1)
        nb = (topk + BLOCK - 1) // BLOCK
        ncb = nb - nwin
        rows_total = rows_mode in ("total", "pre")
        rmax = max(d, BLOCK)      # rowsum 归约缓冲的列数（total/block 两种口径共用一块）

        @T.prim_func
        def kernel(
            q: T.Tensor[(b, m, h, d), BF16],
            kv: T.Tensor[(b, n, d), BF16],
            freqs: T.Tensor[(m, rdp), FP32],
            do: T.Tensor[(b, m, h, d), BF16],
            o_sv: T.Tensor[(b, m, h, d), BF16],
            rowsum_in: T.Tensor[(b, m, h), FP32],   # 仅 rows_mode="pre" 使用
            lse: T.Tensor[(b, m, h), FP32],
            attn_sink: T.Tensor[(h,), FP32],
            topk_idxs: T.Tensor[(b, m, topk), INT32],
            dq: T.Tensor[(b, m, h, d), BF16],
            dkv: T.Tensor[(b, n + 1, d), FP32],
            dsink_rows: T.Tensor[(b, m, h), FP32],
            rowsum_rows: T.Tensor[(b, m, h), FP32],
            q_roped: T.Tensor[(b, m, h, d), BF16],   # 仅 rd>0：roped q 供 window-dkv kernel 复用
        ):
            with T.Kernel(m, b, threads=THREADS) as (bx, by):
                # 反向统一用 fp16 shared：ds 用 fp16 尾数更准，且 gemm 不允许 fp16×bf16 混用
                q_shared = T.alloc_shared((hp, d), FP16)
                do_shared = T.alloc_shared((hp, d), FP16)
                kv_shared = T.alloc_shared((BLOCK, d), FP16)
                ds_shared = T.alloc_shared((hp, BLOCK), FP16)
                p_shared = T.alloc_shared((hp, BLOCK), FP16)
                dq_shared = T.alloc_shared((hp, d), BF16)
                idxs = T.alloc_fragment(BLOCK, INT32)
                acc_s = T.alloc_fragment((hp, BLOCK), FP32)
                dp = T.alloc_fragment((hp, BLOCK), FP32)
                dq_f = T.alloc_fragment((hp, d), FP32)
                dkv_f = T.alloc_fragment((BLOCK, d), FP32)
                lse_f = T.alloc_fragment(hp, FP32)
                sink_f = T.alloc_fragment(hp, FP32)
                rs_f = T.alloc_fragment(hp, FP32)       # 整行 rowsum R
                rbuf = T.alloc_fragment((hp, rmax), FP32)   # rowsum 归约缓冲（两种口径共用）
                row_sum = T.alloc_fragment(hp, FP32)
                # dsink/rowsum 的输出只有 h 列，但 sink_f/lse_f/rs_f 是 hp 片段：先在 hp 全长度
                # 上算进 shared，再取前 h 行写出（避免对 hp 片段做“部分 extent”访问）。
                dsk_s = T.alloc_shared((hp,), FP32)
                rsum_s = T.alloc_shared((hp,), FP32)

                T.clear(dq_f)
                for i, j in T.Parallel(h, d):
                    q_shared[i, j] = q[by, bx, i, j]
                    do_shared[i, j] = do[by, bx, i, j]
                if hp > h:
                    for i, j in T.Parallel(hp - h, d):
                        q_shared[h + i, j] = 0
                        do_shared[h + i, j] = 0
                if rd:  # q tail 在 kernel 内重建 forward RoPE（输入 q 是 raw 空间）
                    for i, p in T.Parallel(h, rd // 2):
                        re = T.Cast(FP32, q_shared[i, d - rd + 2 * p])
                        im = T.Cast(FP32, q_shared[i, d - rd + 2 * p + 1])
                        fre = freqs[bx, 2 * p]
                        fim = freqs[bx, 2 * p + 1]
                        q_shared[i, d - rd + 2 * p] = T.Cast(
                            FP16, T.Cast(BF16, re * fre - im * fim))
                        q_shared[i, d - rd + 2 * p + 1] = T.Cast(
                            FP16, T.Cast(BF16, re * fim + im * fre))
                    # window-dkv kernel 也要 roped q；这里顺手写一份全局 scratch（8.4MB/层，
                    # bf16 与 roped q 逐位相同），比在 win kernel 里重做 rope 便宜也简单。
                    for i, j in T.Parallel(h, d):
                        q_roped[by, bx, i, j] = T.Cast(BF16, q_shared[i, j])
                for i in T.Parallel(hp):
                    lse_f[i] = T.if_then_else(i < h, lse[by, bx, T.min(i, h - 1)], 0.0)
                    sink_f[i] = T.if_then_else(i < h, attn_sink[T.min(i, h - 1)], 0.0)
                if rows_mode == "pre":
                    # R 已由 torch 在图外算好（点积行 + 归约），kernel 不再做 reduce：
                    # 避开 "free 布局的 rbuf + reduce_sum" 在某些 tilelang 版本的布局推断失败。
                    for i in T.Parallel(hp):
                        rs_f[i] = T.if_then_else(i < h, rowsum_in[by, bx, T.min(i, h - 1)], 0.0)
                elif rows_total:
                    # R = Σ_d do·o = Σ_j p̃_j (do·k_j)（恒等式，见模块 docstring）
                    for i, j in T.Parallel(hp, d):
                        rbuf[i, j] = T.Cast(FP32, do_shared[i, j]) * T.if_then_else(
                            i < h, T.Cast(FP32, o_sv[by, bx, T.min(i, h - 1), j]), 0.0)
                    if rmax > d:
                        for i, j in T.Parallel(hp, rmax - d):
                            rbuf[i, d + j] = 0.0
                    T.reduce_sum(rbuf, rs_f, dim=1)
                else:
                    T.clear(rs_f)

                # ---- window 块：只算 rowsum / dq（dkv 由 `_dkv_win_kernel` 键驻留写出）
                for tt in T.Pipelined(nwin, num_stages=1):
                    for i in T.Parallel(BLOCK):
                        idxs[i] = T.if_then_else(tt * BLOCK + i < topk,
                                                 topk_idxs[by, bx, tt * BLOCK + i], -1)
                    for i, j in T.Parallel(BLOCK, d):
                        kv_shared[i, j] = T.if_then_else(idxs[i] != -1,
                                                         kv[by, T.max(idxs[i], 0), j], 0)
                    for i, j in T.Parallel(hp, BLOCK):
                        acc_s[i, j] = T.if_then_else(idxs[j] != -1, 0, -T.infinity(FP32))
                    T.gemm(q_shared, kv_shared, acc_s, transpose_B=True,
                           policy=T.GemmWarpPolicy.FullRow)
                    for i, j in T.Parallel(hp, BLOCK):
                        acc_s[i, j] = T.exp(acc_s[i, j] * scale - lse_f[i])
                    T.copy(acc_s, p_shared)
                    T.clear(dp)
                    T.gemm(do_shared, kv_shared, dp, transpose_B=True,
                           policy=T.GemmWarpPolicy.FullRow)
                    if rows_total:
                        for i, j in T.Parallel(hp, BLOCK):
                            acc_s[i, j] = acc_s[i, j] * (dp[i, j] - rs_f[i])
                    else:
                        for i, j in T.Parallel(hp, BLOCK):
                            rbuf[i, j] = acc_s[i, j] * dp[i, j]
                        if rmax > BLOCK:
                            for i, j in T.Parallel(hp, rmax - BLOCK):
                                rbuf[i, BLOCK + j] = 0.0
                        T.reduce_sum(rbuf, row_sum, dim=1)
                        for i in T.Parallel(hp):
                            rs_f[i] += row_sum[i]
                        for i, j in T.Parallel(hp, BLOCK):
                            acc_s[i, j] = acc_s[i, j] * (dp[i, j] - row_sum[i])
                    T.copy(acc_s, ds_shared)
                    T.gemm(ds_shared, kv_shared, dq_f, policy=T.GemmWarpPolicy.FullRow)
                    if store_window:
                        T.clear(dkv_f)
                        T.gemm(ds_shared, q_shared, dkv_f, transpose_A=True,
                               policy=T.GemmWarpPolicy.FullRow)
                        for i, j in T.Parallel(BLOCK, d):
                            dkv_f[i, j] = dkv_f[i, j] * scale
                        T.gemm(p_shared, do_shared, dkv_f, transpose_A=True,
                               policy=T.GemmWarpPolicy.FullRow)
                        for i, j in T.Parallel(BLOCK, d):
                            idx_i = T.if_then_else(tt * BLOCK + i < topk,
                                                   topk_idxs[by, bx, tt * BLOCK + i], -1)
                            T.atomic_add(dkv[by, T.if_then_else(idx_i != -1, idx_i, n), j],
                                         dkv_f[i, j])

                # ---- compress 块：算 rowsum / dq 并写 dkv（唯一需要原子加的部分）
                for tt in T.Pipelined(ncb, num_stages=1):
                    for i in T.Parallel(BLOCK):
                        idxs[i] = T.if_then_else((nwin + tt) * BLOCK + i < topk,
                                                 topk_idxs[by, bx, (nwin + tt) * BLOCK + i], -1)
                    for i, j in T.Parallel(BLOCK, d):
                        kv_shared[i, j] = T.if_then_else(idxs[i] != -1,
                                                         kv[by, T.max(idxs[i], 0), j], 0)
                    for i, j in T.Parallel(hp, BLOCK):
                        acc_s[i, j] = T.if_then_else(idxs[j] != -1, 0, -T.infinity(FP32))
                    T.gemm(q_shared, kv_shared, acc_s, transpose_B=True,
                           policy=T.GemmWarpPolicy.FullRow)
                    for i, j in T.Parallel(hp, BLOCK):
                        acc_s[i, j] = T.exp(acc_s[i, j] * scale - lse_f[i])
                    T.copy(acc_s, p_shared)
                    T.clear(dp)
                    T.gemm(do_shared, kv_shared, dp, transpose_B=True,
                           policy=T.GemmWarpPolicy.FullRow)
                    if rows_total:
                        for i, j in T.Parallel(hp, BLOCK):
                            acc_s[i, j] = acc_s[i, j] * (dp[i, j] - rs_f[i])
                    else:
                        for i, j in T.Parallel(hp, BLOCK):
                            rbuf[i, j] = acc_s[i, j] * dp[i, j]
                        if rmax > BLOCK:
                            for i, j in T.Parallel(hp, rmax - BLOCK):
                                rbuf[i, BLOCK + j] = 0.0
                        T.reduce_sum(rbuf, row_sum, dim=1)
                        for i in T.Parallel(hp):
                            rs_f[i] += row_sum[i]
                        for i, j in T.Parallel(hp, BLOCK):
                            acc_s[i, j] = acc_s[i, j] * (dp[i, j] - row_sum[i])
                    T.copy(acc_s, ds_shared)
                    T.gemm(ds_shared, kv_shared, dq_f, policy=T.GemmWarpPolicy.FullRow)
                    T.clear(dkv_f)
                    T.gemm(ds_shared, q_shared, dkv_f, transpose_A=True,
                           policy=T.GemmWarpPolicy.FullRow)
                    for i, j in T.Parallel(BLOCK, d):
                        dkv_f[i, j] = dkv_f[i, j] * scale
                    T.gemm(p_shared, do_shared, dkv_f, transpose_A=True,
                           policy=T.GemmWarpPolicy.FullRow)
                    # 直接读全局 topk_idxs（避免与 dkv_f 的 fragment layout 冲突，同 v1）
                    for i, j in T.Parallel(BLOCK, d):
                        idx_i = T.if_then_else((nwin + tt) * BLOCK + i < topk,
                                               topk_idxs[by, bx, (nwin + tt) * BLOCK + i], -1)
                        T.atomic_add(dkv[by, T.if_then_else(idx_i != -1, idx_i, n), j],
                                     dkv_f[i, j])

                for i, j in T.Parallel(hp, d):
                    dq_f[i, j] = dq_f[i, j] * scale
                T.copy(dq_f, dq_shared)
                if rd:  # dq tail：roped 空间 -> raw 空间梯度（inverse RoPE，bf16 舍入入/出）
                    for i, p in T.Parallel(h, rd // 2):
                        re = T.Cast(FP32, dq_shared[i, d - rd + 2 * p])
                        im = T.Cast(FP32, dq_shared[i, d - rd + 2 * p + 1])
                        fre = freqs[bx, 2 * p]
                        fim = -freqs[bx, 2 * p + 1]
                        dq_shared[i, d - rd + 2 * p] = T.Cast(BF16, re * fre - im * fim)
                        dq_shared[i, d - rd + 2 * p + 1] = T.Cast(BF16, re * fim + im * fre)
                for i, j in T.Parallel(h, d):
                    dq[by, bx, i, j] = dq_shared[i, j]
                for i in T.Parallel(hp):
                    dsk_s[i] = -T.exp(sink_f[i] - lse_f[i]) * rs_f[i]
                    rsum_s[i] = rs_f[i]
                for i in T.Parallel(h):
                    dsink_rows[by, bx, i] = dsk_s[i]
                    rowsum_rows[by, bx, i] = rsum_s[i]

        return kernel

    # ------------------------------------------------------------------ window dkv（键驻留）
    @tilelang.jit(pass_configs=pass_configs)
    def _dkv_win_kernel(h: int, d: int, win: int, scale: float, bk: int, bq: int, threads: int,
                        mode: str = ""):
        """键驻留：每个 CTA 拥有 `bk` 个 window key，重算 s/dp 后按 query 顺序累加 dkv。

        结构同 `tl_sparse_attn_v2._dkv_win_kernel`（已在该形状上验证可编译/可捕获），
        差别只有：全局索引一律 clamp（v2 用 if_then_else 包住越界读，理论上会读到张量外）、
        并且 rowsum 由主 bwd kernel 写好（不再需要自己算）。
        `mode` 见 `_WIN_MODES`：orig = dkv_f 跨迭代累加；shacc = shared 累加器（布局诊断用）。
        """
        mode = mode or _CFG["win"]
        b = T.symbolic("b")
        m = T.symbolic("m")
        n = T.symbolic("n")
        nn = T.symbolic("nn")
        BR = bq * h              # 每个 query chunk 的行数（query × head）
        # 因果窗口：key k 被 query [k, k+win-1] 看到 ⇒ 一个 key tile 需要 query 范围
        # [k0, k0+bk-1+win-1]，宽度 bk+win-1。
        R = bk + win - 1
        NIT = (R + bq - 1) // bq

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
                # 与 v2 一致用 fp32：dkv 的 window 部分按 fp32 累加后一次写出，
                # 不再多一次 bf16 舍入（否则 --check 的 dkv 误差会从 ~2e-3 抬到 ~4e-3）。
                dkv_s = T.alloc_shared((bk, d), FP32)
                s_f = T.alloc_fragment((BR, bk), FP32)
                dp_f = T.alloc_fragment((BR, bk), FP32)
                dkv_f = T.alloc_fragment((bk, d), FP32)
                lse_f = T.alloc_fragment(BR, FP32)
                rs_f = T.alloc_fragment(BR, FP32)

                if mode == "shacc":
                    # shared 累加器：dkv_f 每次迭代清零（见 `_WIN_MODES`）
                    for j, c in T.Parallel(bk, d):
                        dkv_s[j, c] = 0.0
                else:
                    T.clear(dkv_f)
                for j, c in T.Parallel(bk, d):
                    k_s[j, c] = T.if_then_else(bx * bk + j < m,
                                               kv[by, T.min(bx * bk + j, m - 1), c], 0)
                for it in T.serial(NIT):
                    i0 = bx * bk + it * bq
                    for r, c in T.Parallel(BR, d):
                        q_s[r, c] = T.if_then_else(
                            (i0 + r // h >= 0) & (i0 + r // h < m),
                            q[by, T.min(T.max(i0 + r // h, 0), m - 1), r % h, c], 0)
                        do_s[r, c] = T.if_then_else(
                            (i0 + r // h >= 0) & (i0 + r // h < m),
                            do[by, T.min(T.max(i0 + r // h, 0), m - 1), r % h, c], 0)
                    for r in T.Parallel(BR):
                        lse_f[r] = lse[by, T.min(T.max(i0 + r // h, 0), m - 1), r % h]
                        rs_f[r] = rowsum[by, T.min(T.max(i0 + r // h, 0), m - 1), r % h]
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
                    if mode == "shacc":
                        T.clear(dkv_f)
                    T.gemm(ds_s, q_s, dkv_f, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)
                    T.gemm(p_s, do_s, dkv_f, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)
                    if mode == "shacc":
                        for j, c in T.Parallel(bk, d):
                            dkv_s[j, c] += dkv_f[j, c]
                if mode != "shacc":
                    T.copy(dkv_f, dkv_s)
                # 每个 CTA 独占 [bx*bk, bx*bk+bk) 这些行（调用方保证 m % bk == 0）
                for j, c in T.Parallel(bk, d):
                    dkv[by, bx * bk + j, c] = dkv_s[j, c]

        return kernel

    # ------------------------------------------------------------------ 入口
    def _fwd_v3(q, kv, sink, idxs, scale, freqs=None, rd=0):
        b, m, h, d = q.shape
        topk = idxs.size(-1)
        o = torch.empty(b, m, h, d, dtype=q.dtype, device=q.device)
        o_sv = torch.empty(b, m, h, d, dtype=q.dtype, device=q.device)
        lse = torch.empty(b, m, h, dtype=torch.float32, device=q.device)
        if rd:
            fr = freqs
        else:
            fr = lse.new_empty((m, 1))     # 占位：rd=0 时 kernel 不会读
        k = _fwd_kernel(h, _hp_of(h), d, topk, scale, rd)
        with _TorchStream():
            k(q, kv, fr, o, o_sv, lse, sink, idxs)
        return o, o_sv, lse

    def _bwd_v3(q, kv, do, o_sv, lse, sink, idxs, win, nc, scale, plan, dkv_mode, rows_mode,
                freqs=None, rd=0):
        b, m, h, d = q.shape
        n = kv.size(1)
        hp = plan["hp"]
        dq = torch.empty(b, m, h, d, dtype=q.dtype, device=q.device)
        dkv = torch.zeros(b, n + 1, d, dtype=torch.float32, device=q.device)
        dsink_rows = torch.empty(b, m, h, dtype=torch.float32, device=q.device)
        rowsum = torch.empty(b, m, h, dtype=torch.float32, device=q.device)
        if rows_mode == "pre":
            # R = rowsum(dO ∘ O)，fp32 归约（等价于 kernel 内 rbuf+reduce_sum 的口径）
            if _CFG["rows_triton"] and HAS_TRITON:
                rowsum_in = _rowsum_dprod(do, o_sv)
            else:
                rowsum_in = (do.float() * o_sv.float()).sum(dim=-1).contiguous()
        else:
            rowsum_in = rowsum       # 未使用的占位（kernel 内不会读）
        store_window = dkv_mode == "all"
        fr = freqs if rd else rowsum.new_empty((m, 1))
        # rope-in 模式：主 bwd kernel 把 roped q 写到 scratch，供 window-dkv kernel 复用；
        # rd=0 时把 dq 当占位（kernel 不写它）。
        q_roped = torch.empty_like(q) if rd else dq
        k = _bwd_kernel(h, hp, d, plan["topk"], plan["nwin"], scale,
                        store_window, rows_mode, rd)
        q_win = q_roped if rd else q
        with _TorchStream():
            k(q, kv, fr, do, o_sv, rowsum_in, lse, sink, idxs, dq, dkv, dsink_rows, rowsum,
              q_roped)
            if dkv_mode == "split":
                if _CFG["win"] == "triton" and HAS_TRITON:
                    _dkv_win_triton(q_win, kv, do, lse, rowsum, dkv, win, scale,
                                    _CFG["tbk"], _CFG["tbq"], _CFG["twarps"])
                else:
                    kw = _dkv_win_kernel(h, d, win, scale, _CFG["bk"], _CFG["bq"],
                                         _CFG["bwthreads"], _CFG["win"])
                    kw(q_win, kv, do, lse, rowsum, dkv)
        return dq, dkv, dsink_rows


# ---------------------------------------------------------------------------- window dkv（Triton 兜底）
# TileLang 的 window 键驻留 kernel 在部分 tilelang 版本上会触发 layout 推断失败
# （`InternalError: Check failed: (has_best) is false: no available layout found`）。
# 这里保留一个同数学、同数据布局（q bf16 / kv bf16 / lse+rowsum fp32 / dkv fp32，输出行不相交）
# 的 Triton 实现：`NANOV41_ATTN_V3_WIN=triton`。它**不依赖 TileLang**，作为最后兜底；
# 数学与 `_dkv_win_kernel` 完全一致（p̃=exp(s·scale−lse)、ds=p̃·(dp−R)·scale、
# dkv += dsᵀ·q + p̃ᵀ·do）。
try:  # pragma: no cover - GPU only
    import triton as _triton
    import triton.language as _tl

    HAS_TRITON = True
except Exception:  # noqa: BLE001
    _triton = None
    _tl = None
    HAS_TRITON = False


if HAS_TRITON:

    @_triton.jit
    def _dkv_win_tri_kernel(Q, KV, DOUT, LSE, ROWSUM, DKV,
                            NN, M, N,
                            SCALE: _tl.constexpr, WIN: _tl.constexpr,
                            H: _tl.constexpr, D: _tl.constexpr,
                            BK: _tl.constexpr, BQ: _tl.constexpr):
        bx = _tl.program_id(0)
        by = _tl.program_id(1)
        BR: _tl.constexpr = BQ * H
        R: _tl.constexpr = BK + WIN - 1
        NIT: _tl.constexpr = (R + BQ - 1) // BQ

        rk = _tl.arange(0, BK)
        rd = _tl.arange(0, D)
        k0 = bx * BK
        kj = k0 + rk                                        # 本 program 的 window key 行
        k = _tl.load(KV + (by * N + kj[:, None]) * D + rd[None, :],
                     mask=(kj[:, None] < M), other=0.0).to(_tl.float16)          # [BK,D]

        rr = _tl.arange(0, BR)
        rq = rr // H
        rh = rr % H
        acc = _tl.zeros((BK, D), dtype=_tl.float32)
        for it in range(0, NIT):                           # query chunk 顺序固定 ⇒ 确定性
            i0 = k0 + it * BQ
            qi = i0 + rq                                   # [BR] 全局 query 行
            qok = (qi >= 0) & (qi < M)
            qsafe = _tl.minimum(_tl.maximum(qi, 0), M - 1)
            off = ((by * M + qsafe) * H + rh) * D          # [BR] 每行的 element 偏移
            lse = _tl.load(LSE + (by * M + qsafe) * H + rh, mask=qok, other=0.0)
            rs = _tl.load(ROWSUM + (by * M + qsafe) * H + rh, mask=qok, other=0.0)
            qT = _tl.load(Q + off[None, :] + rd[:, None], mask=qok[None, :],
                          other=0.0).to(_tl.float16)       # [D,BR]（转置读，省一次 trans）
            doT = _tl.load(DOUT + off[None, :] + rd[:, None], mask=qok[None, :],
                           other=0.0).to(_tl.float16)      # [D,BR]
            sT = _tl.dot(k, qT) * SCALE                     # [BK,BR] fp32
            kiv = kj[:, None]
            qiv = qi[None, :]
            valid = (kiv <= qiv) & (kiv > (qiv - WIN)) & qok[None, :] & (kiv < M)
            sT = _tl.where(valid, sT, float("-inf"))
            pT = _tl.exp(sT - lse[None, :])                 # 无效列 exp(-inf) = 0
            dpT = _tl.dot(k, doT)                           # [BK,BR]
            dsT = pT * (dpT - rs[None, :]) * SCALE
            qn = _tl.load(Q + off[:, None] + rd[None, :], mask=qok[:, None],
                          other=0.0).to(_tl.float16)       # [BR,D]
            don = _tl.load(DOUT + off[:, None] + rd[None, :], mask=qok[:, None],
                           other=0.0).to(_tl.float16)      # [BR,D]
            acc += _tl.dot(dsT.to(_tl.float16), qn)         # dsᵀ·q（score 路径含 scale）
            acc += _tl.dot(pT.to(_tl.float16), don)         # p̃ᵀ·do（value 路径）
        _tl.store(DKV + (by * NN + kj[:, None]) * D + rd[None, :], acc,
                  mask=(kj[:, None] < M))


    @_triton.jit
    def _rowsum_dprod_kernel(DO, O, RS, NROW,
                             HD: _tl.constexpr, D: _tl.constexpr, BR: _tl.constexpr):
        """R[row, h] = Σ_d do[row,h,d]·o[row,h,d]（fp32 累加，不物化 fp32 中间张量）。

        DO/O 是 [b*m, h*d] 的连续视图（bf16），RS 是 [b*m, h] fp32。
        """
        pid = _tl.program_id(0)
        rows = pid * BR + _tl.arange(0, BR)
        rmask = rows < NROW
        cols = _tl.arange(0, HD)
        base = rows[:, None] * HD + cols[None, :]
        do = _tl.load(DO + base, mask=rmask[:, None], other=0.0).to(_tl.float32)
        oo = _tl.load(O + base, mask=rmask[:, None], other=0.0).to(_tl.float32)
        prod = _tl.reshape(do * oo, (BR, HD // D, D))
        acc = _tl.sum(prod, axis=2)                       # [BR, H] fp32
        hidx = _tl.arange(0, HD // D)
        _tl.store(RS + rows[:, None] * (HD // D) + hidx[None, :], acc,
                  mask=rmask[:, None])


    def _rowsum_dprod(do, o_sv, br=8):
        """一次 Triton kernel 求 R=rowsum(dO∘O)（替代 4 个 aten kernel）；返回 [b, m, h] fp32。"""
        b, m, h, d = do.shape
        nrow = b * m
        rs = torch.empty(b, m, h, dtype=torch.float32, device=do.device)
        _rowsum_dprod_kernel[(_triton.cdiv(nrow, br),)](
            do, o_sv, rs, nrow, HD=h * d, D=d, BR=br, num_warps=4)
        return rs


    def _dkv_win_triton(q, kv, do, lse, rowsum, dkv, win, scale, bk, bq, num_warps):
        """启动 Triton window-dkv 兜底 kernel（调用方保证 m % bk == 0、布局 = band 窗口）。"""
        b, m, h, d = q.shape
        n = kv.size(1)
        assert m % bk == 0, f"m={m} 不是 bk={bk} 的整数倍"
        grid = ((m + bk - 1) // bk, b)
        _dkv_win_tri_kernel[grid](
            q, kv, do, lse, rowsum, dkv, dkv.size(1), m, n,
            SCALE=float(scale), WIN=int(win), H=int(h), D=int(d),
            BK=int(bk), BQ=int(bq), num_warps=int(num_warps))


_STATS = {"calls": 0, "v3_calls": 0, "fallback": 0}
_BROKEN = {"on": False}
_BUILT = {}


def stats():
    return dict(_STATS)


def reset_stats():
    for k in _STATS:
        _STATS[k] = 0


def set_config(**kw):
    """测试/A-B 用：改 dkv / rows / win / bk / bq / bwthreads（不传的保持不变）。"""
    for k, v in kw.items():
        if k == "dkv":
            if v not in _DKV_MODES:
                raise ValueError(f"dkv must be one of {_DKV_MODES}, got {v!r}")
        elif k == "rows":
            if v not in _ROWS_MODES:
                raise ValueError(f"rows must be one of {_ROWS_MODES}, got {v!r}")
        elif k == "win":
            if v not in _WIN_MODES:
                raise ValueError(f"win must be one of {_WIN_MODES}, got {v!r}")
        elif k == "rope_in":
            if not isinstance(v, bool):
                raise ValueError(f"rope_in must be bool, got {v!r}")
        elif k == "rows_triton":
            if not isinstance(v, bool):
                raise ValueError(f"rows_triton must be bool, got {v!r}")
        elif k not in ("bk", "bq", "bwthreads", "tbk", "tbq", "twarps"):
            raise ValueError(f"unknown knob {k!r}")
        _CFG[k] = v
    return dict(_CFG)


def get_config():
    return dict(_CFG)


_LAYOUT_CACHE = {}


def plan(q, kv, idxs, softmax_scale=None):
    """返回本层能否走 v3 以及切分方案（可缓存，避免每步 device→host 同步）。"""
    from . import fast_attn

    b, m, h, d = q.shape
    key = (b, m, idxs.shape[-1], kv.size(1), idxs.dtype, idxs.stride())
    hit = _LAYOUT_CACHE.get(key)
    if hit is None:
        win, nc, ok = fast_attn.detect_layout(idxs, m)
        hit = (win, nc, ok)
        if len(_LAYOUT_CACHE) > 256:
            _LAYOUT_CACHE.clear()
        _LAYOUT_CACHE[key] = hit
    win, nc, ok = hit
    if not ok or win < 1:
        return None
    sp = _split_plan(win, nc, m, h, _CFG["bk"])
    if sp is None:
        return None
    out = dict(sp)
    out.update({"win": win, "nc": nc, "m": m, "h": h, "d": d, "b": b,
                "kv_n": kv.size(1), "dkv": _CFG["dkv"], "rows": _CFG["rows"],
                "bk": _CFG["bk"], "bq": _CFG["bq"]})
    return out


def rope_in():
    """rope-in（R79）是否启用（fused_rope_attn 用它决定是否把 q 的 RoPE 交给 attention）。"""
    return bool(_CFG["rope_in"])


def _freqs_to_complex(freqs, rd):
    """[m, rd] fp32（re,im 交错）-> complex64 [m, rd/2]（回退到 v1 时用）。"""
    return torch.view_as_complex(freqs.reshape(-1, rd // 2, 2).contiguous()).contiguous()


def _fallback(q, kv, attn_sink, topk_idxs, softmax_scale, freqs=None, rd=0):
    from . import tl_sparse_attn as v1

    qq = q
    fc = None
    if freqs is not None and rd:
        from .fused_rope_attn import rope_tail
        fc = _freqs_to_complex(freqs, rd)
        qq = rope_tail(q, fc)
    if v1.available():
        o = v1.sparse_attn_tl(qq, kv, attn_sink, topk_idxs, softmax_scale)
    else:
        from .fast_attn import sparse_attn_fast
        o = sparse_attn_fast(qq, kv, attn_sink, topk_idxs, softmax_scale)
    if fc is not None:
        from .fused_rope_attn import rope_tail
        o = rope_tail(o, fc, True)
    return o


class _V3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, sink, idxs, scale, win, nc, plan_d, freqs, rd):
        o, o_sv, lse = _fwd_v3(q, kv, sink, idxs, scale, freqs, rd)
        ctx.save_for_backward(q, kv, o_sv, sink, idxs, lse)
        ctx.cfg = (scale, win, nc, plan_d, _CFG["dkv"], _CFG["rows"], rd)
        ctx.freqs = freqs
        return o

    @staticmethod
    def backward(ctx, do):
        q, kv, o_sv, sink, idxs, lse = ctx.saved_tensors
        scale, win, nc, plan_d, dkv_mode, rows_mode, rd = ctx.cfg
        dq, dkv, dsink_rows = _bwd_v3(q, kv, do.contiguous(), o_sv, lse, sink, idxs,
                                      win, nc, scale, plan_d, dkv_mode, rows_mode,
                                      ctx.freqs, rd)
        n = kv.size(1)
        dsink = dsink_rows.sum(dim=(0, 1)).to(torch.float32)
        return (dq, dkv[:, :n].to(kv.dtype).contiguous(), dsink,
                None, None, None, None, None, None, None)


def _ensure_compiled(p, scale, rd=0):
    """首次调用就把三个 kernel 都编译出来（编译异常可在非捕获态被上层捕获并回退）。"""
    key = (p["h"], p["hp"], p["d"], p["topk"], p["nwin"], float(scale),
           _CFG["dkv"], _CFG["rows"], _CFG["win"], _CFG["bk"], _CFG["bq"],
           _CFG["bwthreads"], rd, bool(_CFG["rope_in"]))
    if key in _BUILT:
        return
    _fwd_kernel(p["h"], p["hp"], p["d"], p["topk"], float(scale), rd)
    _bwd_kernel(p["h"], p["hp"], p["d"], p["topk"], p["nwin"], float(scale),
                _CFG["dkv"] == "all", _CFG["rows"], rd)
    if _CFG["dkv"] == "split" and (_CFG["win"] != "triton" or not HAS_TRITON):
        _dkv_win_kernel(p["h"], p["d"], p["win"], float(scale),
                        _CFG["bk"], _CFG["bq"], _CFG["bwthreads"], _CFG["win"])
    _BUILT[key] = True


def sparse_attn_v3(q, kv, attn_sink, topk_idxs, softmax_scale, freqs=None, rd=0):
    """与 `patched_env.sparse_attn` 同签名（+可选的 rope-in 参数）；不满足时回退 v1/fast_attn。

    可选 `freqs [m, rd]` fp32 是 q 位置轴的 RoPE cos/sin（re,im 交错，start_pos==0 时的
    位置 0..m-1）。只有 `NANOV41_ATTN_V3_ROPE_IN=1` 且形状满足时才会把 q 的 RoPE 折进
    kernel；否则 `freqs` 被忽略（调用方仍走外部 rope_tail）。
    """
    _STATS["calls"] += 1
    d = q.size(-1)
    rope_use = bool(_CFG["rope_in"] and freqs is not None and rd and rd % 2 == 0
                    and rd < d and freqs.dtype == torch.float32
                    and freqs.shape[0] == q.size(1) and freqs.size(1) == rd)
    fr = freqs if rope_use else None
    rdv = int(rd) if rope_use else 0
    if not (HAS_TILELANG and torch.cuda.is_available() and q.is_cuda):
        _STATS["fallback"] += 1
        return _fallback(q, kv, attn_sink, topk_idxs, softmax_scale, fr, rdv)
    if topk_idxs.numel() == 0:
        _STATS["fallback"] += 1
        return _fallback(q, kv, attn_sink, topk_idxs, softmax_scale, fr, rdv)
    p = plan(q, kv, topk_idxs)
    if p is None or _BROKEN["on"]:
        _STATS["fallback"] += 1
        return _fallback(q, kv, attn_sink, topk_idxs, softmax_scale, fr, rdv)
    try:
        _ensure_compiled(p, softmax_scale, rdv)
    except Exception as exc:  # noqa: BLE001
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise
        _BROKEN["on"] = True
        print(f"[tl_sparse_attn_v3] kernel 编译失败，永久回退 v1/fast_attn "
              f"({type(exc).__name__}: {exc})", flush=True)
        _STATS["fallback"] += 1
        return _fallback(q, kv, attn_sink, topk_idxs, softmax_scale, fr, rdv)
    _STATS["v3_calls"] += 1
    q = q.contiguous()
    kv = kv.contiguous()
    idxs = topk_idxs.contiguous()
    sink = attn_sink.float().contiguous()
    if fr is not None:
        fr = fr.contiguous()
    return _V3.apply(q, kv, sink, idxs, float(softmax_scale), p["win"], p["nc"], p,
                     fr, rdv)


def available():
    return bool(HAS_TILELANG and torch.cuda.is_available())


_PREV = {}


def install(M):
    """幂等安装：`M.sparse_attn = sparse_attn_v3`，返回被替换的原实现。"""
    cur = getattr(M, "sparse_attn", None)
    if cur is sparse_attn_v3:
        return _PREV.get(id(M))
    _PREV[id(M)] = cur
    M.sparse_attn = sparse_attn_v3
    return cur


def uninstall(M):
    if id(M) not in _PREV:
        return
    M.sparse_attn = _PREV.pop(id(M))

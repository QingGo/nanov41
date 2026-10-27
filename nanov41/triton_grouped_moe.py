"""Graph-safe grouped MoE: Triton grouped GEMM with **device-side** offsets.

## 为什么需要它

CUDA graph 是目前测到的最大单笔收益（4090 b=8：323 → 175.6 ms，1.84×；parity 逐位一致）。
但图里不能有 host 同步：

- `torch._grouped_mm` 内部把 offs 拷回 host 算 tile 数 → 捕获失败；
- ref/loop 版用 `bincount().tolist()` 给 Python 循环切片 → 捕获失败；
- "static-routing"（捕获前把每组 token 数烤死）只在**权重不变**时逐位正确；真实训练每步权重都变，
  路由会漂移。

正解是 DeepGEMM 那类 **device-side offsets** 的 grouped GEMM：把 (token, slot) 按 expert 排序并补齐到
BM 的整数倍，使每个 M-tile 只属于一个 expert；kernel 用 device tensor 查 expert/行数，Python 侧只给
静态 grid。于是排序、散射、GEMM 全部可进 CUDA graph，且路由是动态的、每步都能变。

## 布局（全部 GPU kernel op，无 host 同步）

    flat_expert [T*k]                 每个 (token, slot) 的被选 expert
    order       = argsort(flat, stable=True)
    counts[e]   = histogram(flat)                          (index_add_)
    pad_off[e]  = Σ_{j<e} ceil(counts[j]/BM)*BM            ([E+1] int32)
    local       = arange(Tk) - pad_off[:-1][expert_sorted]
    pad_pos     = pad_off[:-1][expert_sorted] + local
    x_pad[pad_pos] = x_sorted (= x[tok], tok = order // k)
    tile_e[t]   = searchsorted(pad_off, t*BM) - 1          （越界 → -1，kernel 直接 return）

## 计算（数学等价于 ref 的逐 expert Expert.forward）

    gate = x_pad @ W1[e,dim→inter];  up = x_pad @ W3[e]
    act  = silu(clamp(gate,max=lim)) * clamp(up,±lim) * ws_pad
    y_pad= act @ W2[e,inter→dim]
    out  = scatter_out(y_pad).view(T,k,dim).sum(1) + shared_expert(x)

反向：dW 用"组内行归约"GEMM（grid 固定 (E,K/BK,N/BN)），dX 用两个 grouped GEMM（Wᵀ）。
权重以 stacked 形式传入（`stack(...)`），autograd 会把梯度拆回每个 expert 的 Parameter。

## 正确性口径

与 `train_utils.moe_grouped_forward` 对比：bf16 下 y/梯度 rel 应该在 1e-2 量级（累加顺序不同），
fp32 不适用（Triton dot 走 bf16 tensor core）。测试：`tools/test_triton_moe.py`（GPU）。
"""
from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:  # noqa: BLE001 - Mac 上没有 triton
    HAS_TRITON = False
    triton = None
    tl = None

# 布局实现：sort = argsort(stable)（确定性，逐位可复现）；atomic = 原子计数排序（更快，顺序不定但数学等价）
# 实测：fused scatter-pad 更慢（135.1 vs 131.5 ms）且缓存缓冲会被上一层残留污染 ⇒ 默认关闭
_FUSED_PAD = os.environ.get("NANOV41_MOE_FUSED_PAD", "0").lower() not in ("0", "false", "no", "off")
_LAYOUT = os.environ.get("NANOV41_MOE_LAYOUT", "sort").lower()
assert _LAYOUT in ("sort", "atomic", "fused"), _LAYOUT
# R44：dW split-K（更多 CTA → 更好 SM 利用率；fp32 partial + torch sum，确定性）与 dx 融合
DW_SPLIT = int(os.environ.get("NANOV41_MOE_DW_SPLIT", "0") or 0)
FUSED_DX = os.environ.get("NANOV41_MOE_FUSED_DX", "0").lower() not in ("0", "false", "no", "off")
# R46：MoE 前向尾部（unpad gather + 按 order 散射 + 按 token 对 k 个 slot 求和）合成 1 个 kernel；
# 反传（dy -> dy_pad 散射）也是 1 个 kernel。旧路径是 3 个 aten kernel × ~140 MB/层。
FUSED_TAIL = os.environ.get("NANOV41_MOE_FUSED_TAIL", "0").lower() not in ("0", "false", "no", "off")
# R46：bwd 里 dws_pad = Σ_inter dact*act0 的 8 个 fp32 aten kernel 合成 1 个 Triton kernel
FUSED_DWS = os.environ.get("NANOV41_MOE_FUSED_DWS", "0").lower() not in ("0", "false", "no", "off")
# R48：bwd 不再对 expert 权重做 3 次 [E,dim,inter] transpose（census: 36 次 copy = 0.83 ms eager），
# 改成 grouped GEMM 的 TRANS_B 变体（权重按 [E,N,K] 原布局、kernel 内 trans）
TB = os.environ.get("NANOV41_MOE_TB", "0").lower() not in ("0", "false", "no", "off")
# R50：MoE 前向不再物化 x_pad（gather+zeros+scatter 三层，~100 MB/层）：
# swiglu_fwd 直接从原始 x 按 flat_of_row 间接取行；bwd 的 dW13 同理。act/gate/up 仍写 padding=0。
NOPAD = os.environ.get("NANOV41_MOE_NOPAD", "0").lower() not in ("0", "false", "no", "off")
# R50：MoE 的 fwd/dx/dact 三段走 fp8（e4m3）：权重每步 per-tensor 量化；激活在 kernel 内按
# "每个 k-tile 的行 max" 量化（per-row scale），fp32 累加后按 s_row*wscale 还原。
# dW 两段本轮仍走 bf16（R51 再做）。开关 NANOV41_MOE_FP8=1（依赖 NOPAD）。
FP8 = os.environ.get("NANOV41_MOE_FP8", "0").lower() not in ("0", "false", "no", "off")
FP8_MAX = 448.0

# 默认值 = 4090 上扫描出的最优（BM64/BN64/BK128/warps4/stages2，比默认快 ~8%）
# R43：可用 NANOV41_MOE_TILE="BM,BN,BK,WARPS,STAGES" 在 5090 上重扫（必须能整除对应维度）
BM = 64
BN = 64
BK = 128
WARPS = 4
STAGES = 2
_TILE_ENV = os.environ.get("NANOV41_MOE_TILE", "").strip()
if _TILE_ENV:
    _bm, _bn, _bk, _w, _st = (int(x) for x in _TILE_ENV.split(","))
    assert 384 % _bk == 0 and 384 % _bn == 0 and _bm >= 16
    BM, BN, BK, WARPS, STAGES = _bm, _bn, _bk, _w, _st
# R44：dW kernel 独立 tile（5090 上扫出的最优 128/32/64/4/3，比旧的跟随全局值 64/128/64/4/2 快 ~10%）
DW_BM, DW_BK, DW_BN, DW_WARPS, DW_STAGES = 128, 32, 64, 4, 3
_DWT = os.environ.get("NANOV41_MOE_DW_TILE", "").strip()
if _DWT:
    DW_BM, DW_BK, DW_BN, DW_WARPS, DW_STAGES = (int(x) for x in _DWT.split(","))
    assert 384 % DW_BK == 0 and 384 % DW_BN == 0 and DW_BM >= 16


# --------------------------------------------------------------------------- kernels
if HAS_TRITON:

    @triton.jit
    def _scatter_pad_kernel(X, WS, TOK, ORDER, PAD_POS, XP, WSP, TK,
                            D: tl.constexpr, BD: tl.constexpr, BM: tl.constexpr):
        """把 x[tok]、ws[order] 一次散射进 padding 后的缓冲（缓冲由调用方保证 padding 行恒为 0）。

        替代原来 6 个算子：x[tok] gather + zeros + scatter + ws[order] gather + zeros + scatter。
        """
        pid = tl.program_id(0) * BM + tl.arange(0, BM)
        mask = pid < TK
        tok = tl.load(TOK + pid, mask=mask, other=0).to(tl.int64)
        ordr = tl.load(ORDER + pid, mask=mask, other=0).to(tl.int64)
        dst = tl.load(PAD_POS + pid, mask=mask, other=0).to(tl.int64)
        rd = tl.arange(0, BD)
        m2 = mask[:, None] & (rd[None, :] < D)
        vals = tl.load(X + tok[:, None] * D + rd[None, :], mask=m2, other=0.0)
        tl.store(XP + dst[:, None] * D + rd[None, :], vals, mask=m2)
        w = tl.load(WS + ordr, mask=mask, other=0.0)
        tl.store(WSP + dst, w, mask=mask)

    @triton.jit
    def _layout_meta_kernel(FLAT, PAD_OFF, CUM_EXCL, TILE_E, TK, E: tl.constexpr,
                            BM: tl.constexpr, NT: tl.constexpr):
        """单 block：直方图 → pad_off / cum_excl → tile_e（全部确定性，无 host 同步）。"""
        re_ = tl.arange(0, E)
        cnt = tl.zeros((E,), dtype=tl.int32)
        for p in range(0, TK):
            e = tl.load(FLAT + p).to(tl.int32)
            cnt += tl.where(re_ == e, 1, 0)          # 用掩码代替原子（E=16 很便宜且确定）
        padded = ((cnt + BM - 1) // BM) * BM
        # 前缀和（E 很小，直接扫）
        off = tl.zeros((E,), dtype=tl.int32)
        acc = 0
        cum = tl.zeros((E,), dtype=tl.int32)
        for i in tl.static_range(E):
            ci = tl.sum(tl.where(re_ == i, cnt, 0))
            pi = tl.sum(tl.where(re_ == i, padded, 0))
            off = tl.where(re_ == i, acc, off)
            cum = tl.where(re_ == i, acc, cum)
            acc += pi
        tl.store(CUM_EXCL + re_, cum)
        tl.store(PAD_OFF + re_, off)
        tl.store(PAD_OFF + E, acc)
        # tile_e：第 t 个 tile 的起点落在哪个 expert 的 [off, off+padded) 区间
        for t in tl.static_range(NT):
            start = t * BM
            e_of = -1
            for i in tl.static_range(E):
                oi = tl.sum(tl.where(re_ == i, off, 0))
                pi = tl.sum(tl.where(re_ == i, padded, 0))
                hit = (start >= oi) & (start < oi + pi)
                e_of = tl.where(hit, i, e_of)
            hit_any = start < acc
            tl.store(TILE_E + t, tl.where(hit_any, e_of, -1).to(tl.int32))

    @triton.jit
    def _layout_map_kernel(FLAT, ORDER, PAD_OFF, CUM_EXCL, PAD_POS, TOK, TK,
                           K: tl.constexpr, BLOCK: tl.constexpr):
        """按排序后的位置 s 生成 tok / pad_pos（一次扫过 TK 个元素）。"""
        pid = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = pid < TK
        o = tl.load(ORDER + pid, mask=mask, other=0).to(tl.int32)
        e = tl.load(FLAT + o, mask=mask, other=0).to(tl.int32)
        off = tl.load(PAD_OFF + e, mask=mask, other=0)
        cum = tl.load(CUM_EXCL + e, mask=mask, other=0)
        tl.store(PAD_POS + pid, (off + pid - cum).to(PAD_POS.dtype.element_ty), mask=mask)
        tl.store(TOK + pid, (o // K).to(TOK.dtype.element_ty), mask=mask)

    @triton.jit
    def _count_sort_kernel(FLAT, CNT, PAD_OFF, CUM_EXCL, ORDER, PAD_POS, TK,
                           BLOCK: tl.constexpr):
        """原子计数排序：一次扫过 (token,slot) 直接写出 order / pad_pos。

        与 `argsort(flat, stable=True)` 的区别：expert 内部的行序由原子加决定（不确定），
        但**数学等价**（Dw 只是求和顺序不同；前向/反向逐行独立）。省掉 cub radix sort。
        """
        pid = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = pid < TK
        e = tl.load(FLAT + pid, mask=mask, other=0).to(tl.int64)
        slot = tl.atomic_add(CNT + e, 1, mask=mask)
        base_sorted = tl.load(CUM_EXCL + e, mask=mask)
        base_pad = tl.load(PAD_OFF + e, mask=mask)
        s_idx = base_sorted + slot
        pos = base_pad + slot
        tl.store(ORDER + s_idx, pid.to(ORDER.dtype.element_ty), mask=mask)
        tl.store(PAD_POS + s_idx, pos.to(PAD_POS.dtype.element_ty), mask=mask)

    @triton.jit
    def _aggemm_kernel(C, A, W, TILE_E, K, N,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """C[P,N] = A[P,K] @ W[e,K,N]；每个 M-tile 的 expert 来自 TILE_E。"""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        e = tl.load(TILE_E + pid_m)
        if e < 0:
            return
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        a_ptrs = A + rm[:, None] * K + rk[None, :]
        w_ptrs = W + e.to(tl.int64) * K * N + rk[:, None] * N + rn[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(0, K, BK):
            a = tl.load(a_ptrs)
            w = tl.load(w_ptrs)
            acc += tl.dot(a, w)
            a_ptrs += BK
            w_ptrs += BK * N
        c_ptrs = C + rm[:, None] * N + rn[None, :]
        tl.store(c_ptrs, acc.to(C.dtype.element_ty))

    @triton.jit
    def _aggemm_tb_kernel(C, A, W, TILE_E, K, N,
                          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """C[P,N] = A[P,K] @ W[e,K,N]，但权重按 **[N,K]** 存（transposed-B，省 host 侧 transpose）。"""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        e = tl.load(TILE_E + pid_m)
        if e < 0:
            return
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        a_ptrs = A + rm[:, None] * K + rk[None, :]
        w_ptrs = W + e.to(tl.int64) * N * K + rn[:, None] * K + rk[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(0, K, BK):
            a = tl.load(a_ptrs)
            w = tl.load(w_ptrs)                     # [BN, BK]
            acc += tl.dot(a, tl.trans(w))
            a_ptrs += BK
            w_ptrs += BK
        c_ptrs = C + rm[:, None] * N + rn[None, :]
        tl.store(c_ptrs, acc.to(C.dtype.element_ty))

    @triton.jit
    def _swiglu_bwd_tb_kernel(DGATE, DUP, DACT, DY, W2, GATE_C, UP_C, WS, TILE_E, K, INTER, LIM,
                              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """同 `_swiglu_bwd_kernel`，但 W2 按 **[E,INTER,dim]=[N,K]** 原布局（transposed-B）。"""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        e = tl.load(TILE_E + pid_m)
        if e < 0:
            return
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        dy_ptrs = DY + rm[:, None] * K + rk[None, :]
        w_ptrs = W2 + e.to(tl.int64) * INTER * K + rn[:, None] * K + rk[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(0, K, BK):
            acc += tl.dot(tl.load(dy_ptrs), tl.trans(tl.load(w_ptrs)))
            dy_ptrs += BK
            w_ptrs += BK
        off = rm[:, None] * INTER + rn[None, :]
        gate = tl.load(GATE_C + off).to(tl.float32)
        up = tl.load(UP_C + off).to(tl.float32)
        ws = tl.load(WS + rm).to(tl.float32)
        dact = acc * ws[:, None]
        tl.store(DACT + off, acc.to(DACT.dtype.element_ty))
        gate_c = tl.minimum(gate, LIM)
        up_c = tl.maximum(tl.minimum(up, LIM), -LIM)
        sig = tl.sigmoid(gate_c)
        d_silu = sig * (1.0 + gate_c * (1.0 - sig))
        dgate = dact * up_c * d_silu
        dup = dact * (gate_c * sig)
        dgate = tl.where(gate_c < LIM, dgate, 0.0)
        dup = tl.where(tl.abs(up_c) < LIM, dup, 0.0)
        tl.store(DGATE + off, dgate.to(DGATE.dtype.element_ty))
        tl.store(DUP + off, dup.to(DUP.dtype.element_ty))

    @triton.jit
    def _dw_kernel(DW, A, DC, SEG_START, SEG_END, K, N,
                   BM: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr):
        """DW[e,K,N] = A[rows,K]^T @ DC[rows,N]，rows ∈ expert e。grid=(E, K//BK, N//BN)。"""
        pid_e = tl.program_id(0)
        pid_k = tl.program_id(1)
        pid_n = tl.program_id(2)
        start = tl.load(SEG_START + pid_e)
        end = tl.load(SEG_END + pid_e)
        rk = pid_k * BK + tl.arange(0, BK)
        rn = pid_n * BN + tl.arange(0, BN)
        acc = tl.zeros((BK, BN), dtype=tl.float32)
        for r0 in range(0, end - start, BM):
            rm = start + r0 + tl.arange(0, BM)
            mask = rm < end
            a = tl.load(A + rm[:, None] * K + rk[None, :], mask=mask[:, None], other=0.0)
            dc = tl.load(DC + rm[:, None] * N + rn[None, :], mask=mask[:, None], other=0.0)
            acc += tl.dot(tl.trans(a), dc)
        dw_ptrs = DW + pid_e.to(tl.int64) * K * N + rk[:, None] * N + rn[None, :]
        tl.store(dw_ptrs, acc.to(DW.dtype.element_ty))

    @triton.jit
    def _dw_split_kernel(DWP, A, DC, SEG_START, SEG_END, K, N,
                         S: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr):
        """DW partials[s] = A[rows,K]ᵀ @ DC[rows,N]，rows 按 s 均分成 S 段（fp32 partial，后处理 sum）。"""
        pid_e = tl.program_id(0)
        pid_k = tl.program_id(1)
        pid_n = tl.program_id(2)
        pid_s = tl.program_id(3)
        start = tl.load(SEG_START + pid_e)
        end = tl.load(SEG_END + pid_e)
        n_rows = end - start
        chunk = (n_rows + S - 1) // S
        r_lo = pid_s * chunk
        r_hi = tl.minimum(r_lo + chunk, n_rows)
        rk = pid_k * BK + tl.arange(0, BK)
        rn = pid_n * BN + tl.arange(0, BN)
        acc = tl.zeros((BK, BN), dtype=tl.float32)
        for r0 in range(r_lo, r_hi, BM):
            rm = start + r0 + tl.arange(0, BM)
            mask = rm < (start + r_hi)
            a = tl.load(A + rm[:, None] * K + rk[None, :], mask=mask[:, None], other=0.0)
            dc = tl.load(DC + rm[:, None] * N + rn[None, :], mask=mask[:, None], other=0.0)
            acc += tl.dot(tl.trans(a), dc)
        n_e = tl.num_programs(0)
        off = ((pid_s * n_e + pid_e).to(tl.int64) * K + rk[:, None]) * N + rn[None, :]
        tl.store(DWP + off, acc)

    @triton.jit
    def _dw13_split_kernel(DW1P, DW3P, A, DGATE, DUP, SEG_START, SEG_END, K, INTER,
                           S: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr):
        """dW1/dW3 的 split-K 版本（共享 A tile 读取）。"""
        pid_e = tl.program_id(0)
        pid_k = tl.program_id(1)
        pid_n = tl.program_id(2)
        pid_s = tl.program_id(3)
        start = tl.load(SEG_START + pid_e)
        end = tl.load(SEG_END + pid_e)
        n_rows = end - start
        chunk = (n_rows + S - 1) // S
        r_lo = pid_s * chunk
        r_hi = tl.minimum(r_lo + chunk, n_rows)
        rk = pid_k * BK + tl.arange(0, BK)
        rn = pid_n * BN + tl.arange(0, BN)
        acc1 = tl.zeros((BK, BN), dtype=tl.float32)
        acc3 = tl.zeros((BK, BN), dtype=tl.float32)
        for r0 in range(r_lo, r_hi, BM):
            rm = start + r0 + tl.arange(0, BM)
            mask = rm < (start + r_hi)
            a = tl.load(A + rm[:, None] * K + rk[None, :], mask=mask[:, None], other=0.0)
            dg = tl.load(DGATE + rm[:, None] * INTER + rn[None, :], mask=mask[:, None], other=0.0)
            dp = tl.load(DUP + rm[:, None] * INTER + rn[None, :], mask=mask[:, None], other=0.0)
            acc1 += tl.dot(tl.trans(a), dg)
            acc3 += tl.dot(tl.trans(a), dp)
        n_e = tl.num_programs(0)
        off = ((pid_s * n_e + pid_e).to(tl.int64) * K + rk[:, None]) * INTER + rn[None, :]
        tl.store(DW1P + off, acc1)
        tl.store(DW3P + off, acc3)

    @triton.jit
    def _aggemm2_kernel(C, A1, W1, A2, W2, TILE_E, K, N,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """C[P,N] = A1[P,K] @ W1[e,K,N] + A2[P,K] @ W2[e,K,N]（一次 kernel 出两个 dx 之和）。"""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        e = tl.load(TILE_E + pid_m)
        if e < 0:
            return
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        a1_ptrs = A1 + rm[:, None] * K + rk[None, :]
        a2_ptrs = A2 + rm[:, None] * K + rk[None, :]
        w1_ptrs = W1 + e.to(tl.int64) * K * N + rk[:, None] * N + rn[None, :]
        w2_ptrs = W2 + e.to(tl.int64) * K * N + rk[:, None] * N + rn[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(0, K, BK):
            acc += tl.dot(tl.load(a1_ptrs), tl.load(w1_ptrs))
            acc += tl.dot(tl.load(a2_ptrs), tl.load(w2_ptrs))
            a1_ptrs += BK
            a2_ptrs += BK
            w1_ptrs += BK * N
            w2_ptrs += BK * N
        tl.store(C + rm[:, None] * N + rn[None, :], acc.to(C.dtype.element_ty))

    @triton.jit
    def _aggemm2_t_kernel(C, A1, W1, A2, W2, TILE_E, K, N,
                          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """同 _aggemm2，但 W1/W2 以**未转置**的 [E,N,K] 存储（kernel 内按转置读，省掉 host 侧 transpose+contiguous）。"""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        e = tl.load(TILE_E + pid_m)
        if e < 0:
            return
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        a1_ptrs = A1 + rm[:, None] * K + rk[None, :]
        a2_ptrs = A2 + rm[:, None] * K + rk[None, :]
        w1_ptrs = W1 + e.to(tl.int64) * K * N + rn[None, :] * K + rk[:, None]
        w2_ptrs = W2 + e.to(tl.int64) * K * N + rn[None, :] * K + rk[:, None]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(0, K, BK):
            acc += tl.dot(tl.load(a1_ptrs), tl.load(w1_ptrs))
            acc += tl.dot(tl.load(a2_ptrs), tl.load(w2_ptrs))
            a1_ptrs += BK
            a2_ptrs += BK
            w1_ptrs += BK
            w2_ptrs += BK
        tl.store(C + rm[:, None] * N + rn[None, :], acc.to(C.dtype.element_ty))

    def agg_forward(a_pad, w_stack, tile_e, n_tiles_m, n_out):
        """a_pad [P,K] bf16, w_stack [E,K,N] → [P,N]（新的空 buffer）。"""
        C = torch.empty((a_pad.shape[0], n_out), dtype=a_pad.dtype, device=a_pad.device)
        _aggemm_kernel[(n_tiles_m, n_out // BN)](
            C, a_pad, w_stack, tile_e, a_pad.shape[1], n_out,
            BM=BM, BN=BN, BK=BK, num_warps=WARPS, num_stages=STAGES)
        return C

    def agg_forward2_t(a1_pad, w1_stored, a2_pad, w2_stored, tile_e, n_tiles_m, n_out):
        """dx = A1 @ W1ᵀ + A2 @ W2ᵀ，W1/W2 以 [E,N,K]（未转置）存储；省掉两次 host transpose+contiguous。"""
        C = torch.empty((a1_pad.shape[0], n_out), dtype=a1_pad.dtype, device=a1_pad.device)
        _aggemm2_t_kernel[(n_tiles_m, n_out // BN)](
            C, a1_pad, w1_stored, a2_pad, w2_stored, tile_e,
            a1_pad.shape[1], n_out,
            BM=BM, BN=BN, BK=BK, num_warps=WARPS, num_stages=STAGES)
        return C

    def agg_dw(a_pad, dc_pad, seg_start, seg_end, n_experts):
        """→ DW [E,K,N] (fp32 累加后 cast 回 a_pad.dtype)。DW_SPLIT>1 时走 split-K。"""
        K = a_pad.shape[1]
        N = dc_pad.shape[1]
        if DW_SPLIT > 1:
            part = torch.empty((DW_SPLIT, n_experts, K, N), dtype=torch.float32,
                               device=a_pad.device)
            _dw_split_kernel[(n_experts, K // BK, N // BN, DW_SPLIT)](
                part, a_pad, dc_pad, seg_start, seg_end, K, N,
                S=DW_SPLIT, BM=BM, BK=BK, BN=BN, num_warps=WARPS, num_stages=STAGES)
            return part.sum(0).to(a_pad.dtype)
        DW = torch.empty((n_experts, K, N), dtype=a_pad.dtype, device=a_pad.device)
        _dw_kernel[(n_experts, K // DW_BK, N // DW_BN)](
            DW, a_pad, dc_pad, seg_start, seg_end, K, N,
            BM=DW_BM, BK=DW_BK, BN=DW_BN, num_warps=DW_WARPS, num_stages=DW_STAGES)
        return DW
else:  # pragma: no cover
    def agg_forward(*a, **k):
        raise RuntimeError("triton 不可用（Mac）")

    def agg_dw(*a, **k):
        raise RuntimeError("triton 不可用（Mac）")


# --------------------------------------------------------------------------- layout
_PAD_CACHE: dict = {}


class MoELayout:
    """一次 MoE forward 的布局。构造只用 GPU kernel op（可进 CUDA graph）。"""

    def __init__(self, flat_expert: torch.Tensor, n_experts: int, k: int):
        dev = flat_expert.device
        Tk = flat_expert.numel()
        self.Tk, self.k, self.E = Tk, k, n_experts
        self.T = Tk // k
        idx = flat_expert.to(torch.int64)
        counts = torch.zeros(n_experts, dtype=torch.int64, device=dev).index_add_(
            0, idx, torch.ones(Tk, dtype=torch.int64, device=dev))
        self.counts = counts
        padded = ((counts + BM - 1) // BM) * BM
        pad_off = torch.cat([torch.zeros(1, dtype=torch.int64, device=dev), padded.cumsum(0)])
        self.pad_off = pad_off
        # 未补齐的累计起点（= counts 的 exclusive cumsum）：算 expert 内局部行号要用它，
        # 不能用 pad_off（pad_off 是补齐后的起点，前一个 expert 的 padding 会错位）
        self.cum_excl = torch.cat([torch.zeros(1, dtype=torch.int64, device=dev),
                                   counts.cumsum(0)[:-1]])
        self.seg_start = pad_off[:-1].to(torch.int32)
        self.seg_end = pad_off[1:].to(torch.int32)
        self.P_max = int(((Tk + n_experts * BM) // BM) * BM)
        self.n_tiles_m = self.P_max // BM
        if _LAYOUT == "fused":
            # 确定性融合布局：argsort（cub）+ 2 个 kernel 算完 pad_off/cum_excl/tile_e/tok/pad_pos
            order = torch.argsort(idx, stable=True)
            pad_off = torch.empty(n_experts + 1, dtype=torch.int32, device=dev)
            cum_excl = torch.empty(n_experts, dtype=torch.int32, device=dev)
            tile_e = torch.empty(self.n_tiles_m, dtype=torch.int32, device=dev)
            _layout_meta_kernel[(1,)](idx.to(torch.int32), pad_off, cum_excl, tile_e, Tk,
                                      E=n_experts, BM=BM, NT=self.n_tiles_m, num_warps=1)
            tok = torch.empty(Tk, dtype=torch.int64, device=dev)
            pad_pos = torch.empty(Tk, dtype=torch.int64, device=dev)
            _layout_map_kernel[(triton.cdiv(Tk, 1024),)](idx.to(torch.int32), order, pad_off,
                                                         cum_excl, pad_pos, tok, Tk,
                                                         K=k, BLOCK=1024, num_warps=4)
            self.order, self.tok, self.pad_pos, self.tile_e = order, tok, pad_pos, tile_e
            self.seg_start = pad_off[:-1]
            self.seg_end = pad_off[1:]
            self.pad_off_i32 = pad_off
            self.counts = None
            if FUSED_TAIL or NOPAD:
                rof = torch.empty(Tk, dtype=torch.int64, device=dev)
                rof[self.order] = self.pad_pos
                self.row_of_flat = rof
            if NOPAD:
                fof = torch.zeros(self.P_max, dtype=torch.int32, device=dev)
                fof[self.row_of_flat] = torch.arange(Tk, dtype=torch.int32, device=dev)
                self.flat_of_row = fof
                tail = torch.full((1,), Tk, dtype=torch.int64, device=dev)
                ce = torch.cat([cum_excl.to(torch.int64), tail])
                self.count_i32 = torch.diff(ce).to(torch.int32)
            # 下游只用 seg_start/seg_end/tile_e/pad_pos/tok/order 与 P_max
            return
        if _LAYOUT == "atomic":
            # 原子计数排序：一个 kernel 直接出 order / pad_pos（expert 内行序不确定，数学等价）
            cnt = torch.zeros(n_experts, dtype=torch.int64, device=dev)
            order = torch.empty(Tk, dtype=torch.int64, device=dev)
            pad_pos = torch.empty(Tk, dtype=torch.int64, device=dev)
            _count_sort_kernel[(triton.cdiv(Tk, 1024),)](
                idx, cnt, pad_off[:-1], self.cum_excl, order, pad_pos, Tk, BLOCK=1024)
            self.order = order
            self.tok = torch.div(order, k, rounding_mode="floor")
            self.pad_pos = pad_pos
        else:
            order = torch.argsort(idx, stable=True)
            self.order = order
            self.tok = torch.div(order, k, rounding_mode="floor")
            local = torch.arange(Tk, device=dev) - self.cum_excl[idx[order]]
            self.pad_pos = pad_off[:-1][idx[order]] + local
        tile_start_rows = torch.arange(self.n_tiles_m, device=dev) * BM
        e_of_tile = torch.searchsorted(pad_off, tile_start_rows, right=True) - 1
        e_of_tile = torch.where(tile_start_rows < pad_off[-1], e_of_tile,
                                torch.full_like(e_of_tile, -1))
        self.tile_e = e_of_tile.to(torch.int32)
        if FUSED_TAIL or NOPAD:
            # row_of_flat[f] = 第 f 个 (token,slot) 在 padded 缓冲里的行号
            rof = torch.empty(Tk, dtype=torch.int64, device=dev)
            rof[self.order] = self.pad_pos
            self.row_of_flat = rof
        if NOPAD:
            fof = torch.zeros(self.P_max, dtype=torch.int32, device=dev)
            fof[self.row_of_flat] = torch.arange(Tk, dtype=torch.int32, device=dev)
            self.flat_of_row = fof
            tail = torch.full((1,), Tk, dtype=torch.int64, device=dev)
            ce = torch.cat([self.cum_excl.to(torch.int64), tail])
            self.count_i32 = torch.diff(ce).to(torch.int32)

    def scatter_pad(self, x: torch.Tensor, ws_flat: torch.Tensor):
        """一次 kernel 出 (x_pad [P_max, D], ws_pad [P_max])；padding 行由缓存缓冲保证为 0。"""
        t = x.size(0)
        d = x.size(1)
        key = (self.P_max, d, x.device)
        buf = _PAD_CACHE.get(key)
        if buf is None:
            xp = torch.zeros(self.P_max, d, dtype=x.dtype, device=x.device)
            wsp = torch.zeros(self.P_max, dtype=ws_flat.dtype, device=x.device)
            _PAD_CACHE[key] = (xp, wsp)
        else:  # pragma: no cover - 复用缓冲（padding 行恒为 0）
            xp, wsp = buf
        _scatter_pad_kernel[(triton.cdiv(t, 1024),)](x, ws_flat, self.tok, self.order,
                                                     self.pad_pos, xp, wsp, t,
                                                     D=d, BD=512, BM=1024, num_warps=8)
        return xp, wsp

    def pad(self, flat: torch.Tensor, width: int) -> torch.Tensor:
        out = torch.zeros((self.P_max, width), dtype=flat.dtype, device=flat.device)
        out[self.pad_pos] = flat
        return out

    def unpad(self, padded: torch.Tensor) -> torch.Tensor:
        return padded[self.pad_pos]


def stack_expert_weights(experts, attr: str) -> torch.Tensor:
    """[E, out, in] → [E, in, out]（contiguous），供 grouped GEMM 用。"""
    w = torch.stack([getattr(e, attr).weight for e in experts])      # [E, out, in]
    return w.transpose(1, 2).contiguous()


# --------------------------------------------------------------------------- autograd
class _GroupedMoE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_sorted_pad, ws_pad, w1, w3, w2, tile_e, seg_start, seg_end,
                n_tiles_m, swiglu_limit):
        """
        x_sorted_pad: [P_max, dim]（已按 expert 排序并散射，padding 行为 0）
        ws_pad: [P_max]（routing 权重，已按同布局散射）
        w1,w3: [E, dim, inter]；w2: [E, inter, dim]
        """
        P = x_sorted_pad.shape[0]
        dim = x_sorted_pad.shape[1]
        inter = w1.shape[2]
        gate = agg_forward(x_sorted_pad, w1, tile_e, n_tiles_m, inter)
        up = agg_forward(x_sorted_pad, w3, tile_e, n_tiles_m, inter)
        gate_f = gate.float()
        up_f = up.float()
        lim = swiglu_limit
        if lim and lim > 0:
            up_c = torch.clamp(up_f, min=-lim, max=lim)
            gate_c = torch.clamp(gate_f, max=lim)
        else:  # pragma: no cover
            up_c, gate_c = up_f, gate_f
        act = (torch.nn.functional.silu(gate_c) * up_c) * ws_pad.float()[:, None]
        act_bf = act.to(x_sorted_pad.dtype)
        y_pad = agg_forward(act_bf, w2, tile_e, n_tiles_m, dim)
        ctx.save_for_backward(x_sorted_pad, ws_pad, w1, w3, w2, tile_e, seg_start, seg_end,
                              gate_f, up_f, act_bf)
        ctx.n_tiles_m = n_tiles_m
        ctx.lim = lim
        return y_pad

    @staticmethod
    def backward(ctx, dy_pad):
        (x_pad, ws_pad, w1, w3, w2, tile_e, seg_start, seg_end,
         gate_f, up_f, act_bf) = ctx.saved_tensors
        E = w1.shape[0]
        dim = w2.shape[2]
        inter = w1.shape[2]
        lim = ctx.lim
        n_tiles_m = ctx.n_tiles_m
        dtype = x_pad.dtype
        dW2 = agg_dw(act_bf, dy_pad, seg_start, seg_end, E)
        w2t = w2.transpose(1, 2).contiguous()
        dact = agg_forward(dy_pad, w2t, tile_e, n_tiles_m, inter).float()
        ws = ws_pad.float()[:, None]
        if lim and lim > 0:
            sig = torch.sigmoid(gate_f)
            d_silu = sig * (1 + gate_f * (1 - sig))
            dgate = dact * ws * up_f * d_silu
            dgate = torch.where(gate_f <= lim, dgate, torch.zeros_like(dgate))
            dup = dact * ws * torch.nn.functional.silu(gate_f)
            dup = torch.where(up_f.abs() <= lim, dup, torch.zeros_like(dup))
        else:  # pragma: no cover
            sig = torch.sigmoid(gate_f)
            dgate = dact * ws * up_f * (sig * (1 + gate_f * (1 - sig)))
            dup = dact * ws * torch.nn.functional.silu(gate_f)
        # routing 权重 ws_pad 来自 gate(x)（依赖 x），它的梯度不能丢：
        # act = act0 * ws，所以 d(ws) = Σ_inter dact * act0
        if lim and lim > 0:
            up_c = torch.clamp(up_f, min=-lim, max=lim)
            gate_c = torch.clamp(gate_f, max=lim)
        else:  # pragma: no cover
            up_c, gate_c = up_f, gate_f
        act0 = torch.nn.functional.silu(gate_c) * up_c          # [P, inter] fp32
        dws_pad = (dact * act0).sum(-1)                          # [P]
        dgate_bf = dgate.to(dtype)
        dup_bf = dup.to(dtype)
        dW1 = agg_dw(x_pad, dgate_bf, seg_start, seg_end, E)
        dW3 = agg_dw(x_pad, dup_bf, seg_start, seg_end, E)
        w1t = w1.transpose(1, 2).contiguous()
        w3t = w3.transpose(1, 2).contiguous()
        dx_pad = agg_forward(dgate_bf, w1t, tile_e, n_tiles_m, dim)
        dx_pad = dx_pad + agg_forward(dup_bf, w3t, tile_e, n_tiles_m, dim)
        return (dx_pad, dws_pad, dW1, dW3, dW2, None, None, None, None, None)


def grouped_moe_experts(x, w1, w2, w3, layout: MoELayout, ws_flat: torch.Tensor,
                        swiglu_limit: float = 10.0):
    """routed experts 的 fwd。
    x [T,dim]；w1/w3 [E,dim,inter]；w2 [E,inter,dim]；ws_flat [T*k]（按 (token,slot) 展开的 routing 权重）
    返回 y [T, dim]（已按 slot 求和）。
    """
    x_sorted = x[layout.tok]
    x_pad = layout.pad(x_sorted, x.shape[1])
    # ws_flat 是 (token, slot) 展开顺序；pad() 要的是按 expert 排序后的顺序 → 先按 order 重排
    ws_sorted = ws_flat[layout.order]
    ws_pad = layout.pad(ws_sorted[:, None], 1).squeeze(1)
    y_pad = _GroupedMoE.apply(x_pad, ws_pad, w1, w3, w2, layout.tile_e,
                              layout.seg_start, layout.seg_end,
                              layout.n_tiles_m, swiglu_limit)
    y_sorted = layout.unpad(y_pad)                       # [Tk, dim]，顺序 = order（按 expert 排序）
    y_flat = torch.empty_like(y_sorted)
    y_flat[layout.order] = y_sorted                      # 还原成 (token, slot) 展开顺序
    return y_flat.view(layout.T, layout.k, x.shape[1]).sum(1)


# =========================================================================== v2: fused kernels
# v1 的代价：fwd 里 gate/up 两个 GEMM 各读一遍 A（18432×384 bf16 = 14 MB），SwiGLU 又是 3-4 个
# elementwise kernel；bwd 里 dgate./dup 也是独立 elementwise。v2 把 SwiGLU（含 clamp 与 routing
# 权重）融进 GEMM epilogue，并把 dW1/dW3 合成一个 kernel（共享 A）。
if HAS_TRITON:

    @triton.jit
    def _swiglu_fwd_kernel(ACT, GATE_C, UP_C, A, W1, W3, WS, TILE_E, K, INTER, LIM,
                           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """ACT=bf16[M,INTER], GATE_C/UP_C=fp32[M,INTER]；
        act = silu(clamp(gate,max=LIM)) * clamp(up,±LIM) * ws。"""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        e = tl.load(TILE_E + pid_m)
        if e < 0:
            return
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        a_ptrs = A + rm[:, None] * K + rk[None, :]
        w1_ptrs = W1 + e.to(tl.int64) * K * INTER + rk[:, None] * INTER + rn[None, :]
        w3_ptrs = W3 + e.to(tl.int64) * K * INTER + rk[:, None] * INTER + rn[None, :]
        acc_g = tl.zeros((BM, BN), dtype=tl.float32)
        acc_u = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(0, K, BK):
            a = tl.load(a_ptrs)
            acc_g += tl.dot(a, tl.load(w1_ptrs))
            acc_u += tl.dot(a, tl.load(w3_ptrs))
            a_ptrs += BK
            w1_ptrs += BK * INTER
            w3_ptrs += BK * INTER
        ws = tl.load(WS + rm).to(tl.float32)
        # ref 的 Linear 输出是 bf16、再 .float() → 这里也必须先 round 到 bf16 才能逐位对齐
        acc_g = acc_g.to(tl.bfloat16).to(tl.float32)
        acc_u = acc_u.to(tl.bfloat16).to(tl.float32)
        gate = tl.minimum(acc_g, LIM)
        up = tl.maximum(tl.minimum(acc_u, LIM), -LIM)
        act = (gate * tl.sigmoid(gate)) * up * ws[:, None]
        off = rm[:, None] * INTER + rn[None, :]
        tl.store(ACT + off, act.to(ACT.dtype.element_ty))
        tl.store(GATE_C + off, acc_g.to(GATE_C.dtype.element_ty))
        tl.store(UP_C + off, acc_u.to(UP_C.dtype.element_ty))

    @triton.jit
    def _swiglu_fwd_nopad_kernel(ACT, GATE_C, UP_C, X, WS, W1, W3, FLAT, TILE_E,
                                 SEG_START, COUNT,
                                 D: tl.constexpr, KS: tl.constexpr, INTER: tl.constexpr,
                                 LIM: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                                 BK: tl.constexpr):
        """与 `_swiglu_fwd_kernel` 数学一致，但 A 不再来自物化的 x_pad：
        对 padded 行 rm，flat=FLAT[rm] 给出 (token,slot) 平坦下标，token=flat//KS ⇒ 直接从原始 X 取行。
        padding 行在 ACT/GATE_C/UP_C 里显式写 0（下游 dW 会乘 0，但必须是 0 而不是垃圾）。"""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        e = tl.load(TILE_E + pid_m)
        if e < 0:
            return
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        start = tl.load(SEG_START + e)
        cnt = tl.load(COUNT + e)
        valid = (rm - start) < cnt
        flat = tl.load(FLAT + rm, mask=valid, other=0).to(tl.int32)
        src = flat // KS
        a_ptrs = X + src[:, None] * D + rk[None, :]
        w1_ptrs = W1 + e.to(tl.int64) * D * INTER + rk[:, None] * INTER + rn[None, :]
        w3_ptrs = W3 + e.to(tl.int64) * D * INTER + rk[:, None] * INTER + rn[None, :]
        acc_g = tl.zeros((BM, BN), dtype=tl.float32)
        acc_u = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(0, D, BK):
            a = tl.load(a_ptrs, mask=valid[:, None], other=0.0)
            acc_g += tl.dot(a, tl.load(w1_ptrs))
            acc_u += tl.dot(a, tl.load(w3_ptrs))
            a_ptrs += BK
            w1_ptrs += BK * INTER
            w3_ptrs += BK * INTER
        ws = tl.load(WS + flat, mask=valid, other=0.0).to(tl.float32)
        acc_g = acc_g.to(tl.bfloat16).to(tl.float32)
        acc_u = acc_u.to(tl.bfloat16).to(tl.float32)
        gate = tl.minimum(acc_g, LIM)
        up = tl.maximum(tl.minimum(acc_u, LIM), -LIM)
        act = (gate * tl.sigmoid(gate)) * up * ws[:, None]
        off = rm[:, None] * INTER + rn[None, :]
        vm = valid[:, None]
        tl.store(ACT + off, tl.where(vm, act, 0.0).to(ACT.dtype.element_ty))
        tl.store(GATE_C + off, tl.where(vm, acc_g, 0.0).to(GATE_C.dtype.element_ty))
        tl.store(UP_C + off, tl.where(vm, acc_u, 0.0).to(UP_C.dtype.element_ty))

    @triton.jit
    def _swiglu_bwd_kernel(DGATE, DUP, DACT, DY, W2T, GATE_C, UP_C, WS, TILE_E, K, INTER, LIM,
                           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """dact = DY @ W2ᵀ，然后算 dgate/dup（含 clamp 掩码）并写 bf16。"""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        e = tl.load(TILE_E + pid_m)
        if e < 0:
            return
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        dy_ptrs = DY + rm[:, None] * K + rk[None, :]
        w_ptrs = W2T + e.to(tl.int64) * K * INTER + rk[:, None] * INTER + rn[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(0, K, BK):
            acc += tl.dot(tl.load(dy_ptrs), tl.load(w_ptrs))
            dy_ptrs += BK
            w_ptrs += BK * INTER
        off = rm[:, None] * INTER + rn[None, :]
        gate = tl.load(GATE_C + off).to(tl.float32)
        up = tl.load(UP_C + off).to(tl.float32)
        ws = tl.load(WS + rm).to(tl.float32)
        dact = acc * ws[:, None]          # ∂L/∂act0（dgate/dup 用）
        tl.store(DACT + off, acc.to(DACT.dtype.element_ty))   # ∂L/∂act（dws 用，未乘 ws）
        gate_c = tl.minimum(gate, LIM)
        up_c = tl.maximum(tl.minimum(up, LIM), -LIM)
        sig = tl.sigmoid(gate_c)
        d_silu = sig * (1.0 + gate_c * (1.0 - sig))
        dgate = dact * up_c * d_silu
        dup = dact * (gate_c * sig)
        dgate = tl.where(gate_c < LIM, dgate, 0.0)
        dup = tl.where(tl.abs(up_c) < LIM, dup, 0.0)
        tl.store(DGATE + off, dgate.to(DGATE.dtype.element_ty))
        tl.store(DUP + off, dup.to(DUP.dtype.element_ty))

    @triton.jit
    def _dw13_kernel(DW1, DW3, A, DGATE, DUP, SEG_START, SEG_END, K, INTER,
                     BM: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr):
        """DW1/DW3 = A[rows,K]ᵀ @ DGATE/DUP[rows,INTER]，共享 A tile 的读取。"""
        pid_e = tl.program_id(0)
        pid_k = tl.program_id(1)
        pid_n = tl.program_id(2)
        start = tl.load(SEG_START + pid_e)
        end = tl.load(SEG_END + pid_e)
        rk = pid_k * BK + tl.arange(0, BK)
        rn = pid_n * BN + tl.arange(0, BN)
        acc1 = tl.zeros((BK, BN), dtype=tl.float32)
        acc3 = tl.zeros((BK, BN), dtype=tl.float32)
        for r0 in range(0, end - start, BM):
            rm = start + r0 + tl.arange(0, BM)
            mask = rm < end
            a = tl.load(A + rm[:, None] * K + rk[None, :], mask=mask[:, None], other=0.0)
            dg = tl.load(DGATE + rm[:, None] * INTER + rn[None, :], mask=mask[:, None], other=0.0)
            dp = tl.load(DUP + rm[:, None] * INTER + rn[None, :], mask=mask[:, None], other=0.0)
            acc1 += tl.dot(tl.trans(a), dg)
            acc3 += tl.dot(tl.trans(a), dp)
        o1 = DW1 + pid_e.to(tl.int64) * K * INTER + rk[:, None] * INTER + rn[None, :]
        o3 = DW3 + pid_e.to(tl.int64) * K * INTER + rk[:, None] * INTER + rn[None, :]
        tl.store(o1, acc1.to(DW1.dtype.element_ty))
        tl.store(o3, acc3.to(DW3.dtype.element_ty))

    def swiglu_fwd(a_pad, w1, w3, ws_pad, tile_e, n_tiles_m, inter, lim):
        P = a_pad.shape[0]
        dev = a_pad.device
        act = torch.empty((P, inter), dtype=a_pad.dtype, device=dev)
        gate = torch.empty((P, inter), dtype=a_pad.dtype, device=dev)   # bf16：省带宽
        up = torch.empty((P, inter), dtype=a_pad.dtype, device=dev)
        _swiglu_fwd_kernel[(n_tiles_m, inter // BN)](
            act, gate, up, a_pad, w1, w3, ws_pad, tile_e, a_pad.shape[1], inter, float(lim),
            BM=BM, BN=BN, BK=BK, num_warps=WARPS, num_stages=STAGES)
        return act, gate, up

    def swiglu_bwd(dy_pad, w2t, gate_c, up_c, ws_pad, tile_e, n_tiles_m, inter, lim):
        P = dy_pad.shape[0]
        dev = dy_pad.device
        dgate = torch.empty((P, inter), dtype=dy_pad.dtype, device=dev)
        dup = torch.empty((P, inter), dtype=dy_pad.dtype, device=dev)
        dact = torch.empty((P, inter), dtype=dy_pad.dtype, device=dev)
        _swiglu_bwd_kernel[(n_tiles_m, inter // BN)](
            dgate, dup, dact, dy_pad, w2t, gate_c, up_c, ws_pad, tile_e, dy_pad.shape[1], inter,
            float(lim), BM=BM, BN=BN, BK=BK, num_warps=WARPS, num_stages=STAGES)
        return dgate, dup, dact

    def agg_forward_tb(a, w, tile_e, n_tiles_m, n_out):
        """a [P,K] @ w[e,K,N]（w 按 [E,N,K] 存）。"""
        kdim = a.shape[1]
        out = torch.empty((a.shape[0], n_out), dtype=a.dtype, device=a.device)
        _aggemm_tb_kernel[(n_tiles_m, n_out // BN)](
            out, a, w, tile_e, kdim, n_out, BM=BM, BN=BN, BK=BK,
            num_warps=WARPS, num_stages=STAGES)
        return out

    def swiglu_bwd_tb(dy_pad, w2, gate_c, up_c, ws_pad, tile_e, n_tiles_m, inter, lim):
        P = dy_pad.shape[0]
        dev = dy_pad.device
        dgate = torch.empty((P, inter), dtype=dy_pad.dtype, device=dev)
        dup = torch.empty((P, inter), dtype=dy_pad.dtype, device=dev)
        dact = torch.empty((P, inter), dtype=dy_pad.dtype, device=dev)
        _swiglu_bwd_tb_kernel[(n_tiles_m, inter // BN)](
            dgate, dup, dact, dy_pad, w2, gate_c, up_c, ws_pad, tile_e,
            dy_pad.shape[1], inter, float(lim), BM=BM, BN=BN, BK=BK,
            num_warps=WARPS, num_stages=STAGES)
        return dgate, dup, dact

    @triton.jit
    def _dw13_nopad_kernel(DW1, DW3, X, DGATE, DUP, FLAT, SEG_START, COUNT,
                           D: tl.constexpr, KS: tl.constexpr, INTER: tl.constexpr,
                           BM: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr):
        """DW1/DW3 = Aᵀ @ DGATE/DUP，A 行由 FLAT 映射回原始 X（不物化 x_pad）。"""
        pid_e = tl.program_id(0)
        pid_k = tl.program_id(1)
        pid_n = tl.program_id(2)
        start = tl.load(SEG_START + pid_e)
        cnt = tl.load(COUNT + pid_e)
        rk = pid_k * BK + tl.arange(0, BK)
        rn = pid_n * BN + tl.arange(0, BN)
        acc1 = tl.zeros((BK, BN), dtype=tl.float32)
        acc3 = tl.zeros((BK, BN), dtype=tl.float32)
        for r0 in range(0, cnt, BM):
            local = r0 + tl.arange(0, BM)
            m = local < cnt
            padded = start + local
            flat = tl.load(FLAT + padded, mask=m, other=0).to(tl.int32)
            src = flat // KS
            a = tl.load(X + src[:, None] * D + rk[None, :], mask=m[:, None], other=0.0)
            dg = tl.load(DGATE + padded[:, None] * INTER + rn[None, :], mask=m[:, None], other=0.0)
            dp = tl.load(DUP + padded[:, None] * INTER + rn[None, :], mask=m[:, None], other=0.0)
            acc1 += tl.dot(tl.trans(a), dg)
            acc3 += tl.dot(tl.trans(a), dp)
        o1 = DW1 + pid_e.to(tl.int64) * D * INTER + rk[:, None] * INTER + rn[None, :]
        o3 = DW3 + pid_e.to(tl.int64) * D * INTER + rk[:, None] * INTER + rn[None, :]
        tl.store(o1, acc1.to(DW1.dtype.element_ty))
        tl.store(o3, acc3.to(DW3.dtype.element_ty))
        return

    def swiglu_fwd_nopad(x, ws_flat, w1, w3, flat, tile_e, seg_start, count, n_tiles_m, inter, lim, ks):
        P = flat.shape[0]
        dev = x.device
        act = torch.empty((P, inter), dtype=x.dtype, device=dev)
        gate = torch.empty((P, inter), dtype=x.dtype, device=dev)
        up = torch.empty((P, inter), dtype=x.dtype, device=dev)
        _swiglu_fwd_nopad_kernel[(n_tiles_m, inter // BN)](
            act, gate, up, x, ws_flat, w1, w3, flat, tile_e, seg_start, count,
            D=x.shape[1], KS=ks, INTER=inter, LIM=float(lim),
            BM=BM, BN=BN, BK=BK, num_warps=WARPS, num_stages=STAGES)
        return act, gate, up

    def dw13_nopad(x, flat, seg_start, count, dgate, dup, n_experts, inter, ks):
        d = x.shape[1]
        dev = x.device
        dw1 = torch.empty((n_experts, d, inter), dtype=x.dtype, device=dev)
        dw3 = torch.empty_like(dw1)
        _dw13_nopad_kernel[(n_experts, d // DW_BK, inter // DW_BN)](
            dw1, dw3, x, dgate, dup, flat, seg_start, count,
            D=d, KS=ks, INTER=inter, BM=DW_BM, BK=DW_BK, BN=DW_BN,
            num_warps=DW_WARPS, num_stages=DW_STAGES)
        return dw1, dw3

    @triton.jit
    def _amax_kernel(W, PART, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        idx = pid * BLOCK + tl.arange(0, BLOCK)
        mask = idx < N
        w = tl.load(W + idx, mask=mask, other=0.0).to(tl.float32)
        tl.store(PART + pid, tl.max(tl.abs(w)))

    @triton.jit
    def _quant_apply_kernel(W, WQ, SCALE, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        idx = pid * BLOCK + tl.arange(0, BLOCK)
        mask = idx < N
        sc = tl.load(SCALE)
        w = tl.load(W + idx, mask=mask, other=0.0).to(tl.float32) / sc
        w = tl.minimum(tl.maximum(w, -448.0), 448.0)
        tl.store(WQ + idx, w.to(tl.float8e4nv), mask=mask)

    def quant_weight(w):
        """bf16 权重 -> (fp8 e4m3, per-tensor scale)。两段 kernel，graph 内安全。"""
        n = w.numel()
        nb = triton.cdiv(n, 8192)
        part = torch.empty(nb, dtype=torch.float32, device=w.device)
        _amax_kernel[(nb,)](w, part, n, BLOCK=8192, num_warps=4)
        scale = torch.clamp(part.max(), min=1e-12) / 448.0
        wq = torch.empty_like(w, dtype=torch.float8_e4m3fn)
        _quant_apply_kernel[(nb,)](w, wq, scale, n, BLOCK=8192, num_warps=4)
        return wq, scale

    @triton.jit
    def _swiglu_fwd_fp8_kernel(ACT, GATE_C, UP_C, A, W1Q, W3Q, WS, TILE_E, K, INTER, LIM,
                               WS1, WS3,
                               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """padding 版 fp8 SwiGLU 前向：A=x_pad 的 k-tile 按行 max 量化，W 已 fp8+per-tensor scale。"""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        e = tl.load(TILE_E + pid_m)
        if e < 0:
            return
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        a_ptrs = A + rm[:, None] * K + rk[None, :]
        w1_ptrs = W1Q + e.to(tl.int64) * K * INTER + rk[:, None] * INTER + rn[None, :]
        w3_ptrs = W3Q + e.to(tl.int64) * K * INTER + rk[:, None] * INTER + rn[None, :]
        acc_g = tl.zeros((BM, BN), dtype=tl.float32)
        acc_u = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(0, K, BK):
            a = tl.load(a_ptrs).to(tl.float32)
            s = tl.maximum(tl.max(tl.abs(a), axis=1) / 448.0, 1e-12)
            aq = (a / s[:, None]).to(tl.float8e4nv)
            acc_g += tl.dot(aq, tl.load(w1_ptrs)) * s[:, None]
            acc_u += tl.dot(aq, tl.load(w3_ptrs)) * s[:, None]
            a_ptrs += BK
            w1_ptrs += BK * INTER
            w3_ptrs += BK * INTER
        acc_g = acc_g * tl.load(WS1)
        acc_u = acc_u * tl.load(WS3)
        ws = tl.load(WS + rm).to(tl.float32)
        acc_g = acc_g.to(tl.bfloat16).to(tl.float32)
        acc_u = acc_u.to(tl.bfloat16).to(tl.float32)
        gate = tl.minimum(acc_g, LIM)
        up = tl.maximum(tl.minimum(acc_u, LIM), -LIM)
        act = (gate * tl.sigmoid(gate)) * up * ws[:, None]
        off = rm[:, None] * INTER + rn[None, :]
        tl.store(ACT + off, act.to(ACT.dtype.element_ty))
        tl.store(GATE_C + off, acc_g.to(GATE_C.dtype.element_ty))
        tl.store(UP_C + off, acc_u.to(UP_C.dtype.element_ty))

    def swiglu_fwd_fp8(a_pad, ws_pad, w1q, w3q, ws1, ws3, tile_e, n_tiles_m, inter, lim):
        P = a_pad.shape[0]
        dev = a_pad.device
        act = torch.empty((P, inter), dtype=a_pad.dtype, device=dev)
        gate = torch.empty((P, inter), dtype=a_pad.dtype, device=dev)
        up = torch.empty((P, inter), dtype=a_pad.dtype, device=dev)
        _swiglu_fwd_fp8_kernel[(n_tiles_m, inter // BN)](
            act, gate, up, a_pad, w1q, w3q, ws_pad, tile_e, a_pad.shape[1], inter, float(lim),
            ws1, ws3, BM=BM, BN=BN, BK=BK, num_warps=WARPS, num_stages=STAGES)
        return act, gate, up

    @triton.jit
    def _swiglu_fwd_nopad_fp8_kernel(ACT, GATE_C, UP_C, X, WS, W1Q, W3Q, FLAT, TILE_E,
                                     SEG_START, COUNT, WS1, WS3,
                                     D: tl.constexpr, KS: tl.constexpr, INTER: tl.constexpr,
                                     LIM: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                                     BK: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        e = tl.load(TILE_E + pid_m)
        if e < 0:
            return
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        start = tl.load(SEG_START + e)
        cnt = tl.load(COUNT + e)
        valid = (rm - start) < cnt
        flat = tl.load(FLAT + rm, mask=valid, other=0).to(tl.int32)
        src = flat // KS
        a_ptrs = X + src[:, None] * D + rk[None, :]
        w1_ptrs = W1Q + e.to(tl.int64) * D * INTER + rk[:, None] * INTER + rn[None, :]
        w3_ptrs = W3Q + e.to(tl.int64) * D * INTER + rk[:, None] * INTER + rn[None, :]
        acc_g = tl.zeros((BM, BN), dtype=tl.float32)
        acc_u = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(0, D, BK):
            a = tl.load(a_ptrs, mask=valid[:, None], other=0.0).to(tl.float32)
            s = tl.max(tl.abs(a), axis=1) / 448.0
            s = tl.maximum(s, 1e-12)
            aq = (a / s[:, None]).to(tl.float8e4nv)
            acc_g += tl.dot(aq, tl.load(w1_ptrs)) * s[:, None]
            acc_u += tl.dot(aq, tl.load(w3_ptrs)) * s[:, None]
            a_ptrs += BK
            w1_ptrs += BK * INTER
            w3_ptrs += BK * INTER
        acc_g = acc_g * tl.load(WS1)
        acc_u = acc_u * tl.load(WS3)
        ws = tl.load(WS + flat, mask=valid, other=0.0).to(tl.float32)
        acc_g = acc_g.to(tl.bfloat16).to(tl.float32)
        acc_u = acc_u.to(tl.bfloat16).to(tl.float32)
        gate = tl.minimum(acc_g, LIM)
        up = tl.maximum(tl.minimum(acc_u, LIM), -LIM)
        act = (gate * tl.sigmoid(gate)) * up * ws[:, None]
        off = rm[:, None] * INTER + rn[None, :]
        vm = valid[:, None]
        tl.store(ACT + off, tl.where(vm, act, 0.0).to(ACT.dtype.element_ty))
        tl.store(GATE_C + off, tl.where(vm, acc_g, 0.0).to(GATE_C.dtype.element_ty))
        tl.store(UP_C + off, tl.where(vm, acc_u, 0.0).to(UP_C.dtype.element_ty))

    @triton.jit
    def _aggemm_fp8_kernel(C, A, WQ, TILE_E, K, N, WSC,
                           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """C = A[P,K] @ W[e,K,N]，A 每 k-tile 行量化，W 已 fp8 + per-tensor scale。"""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        e = tl.load(TILE_E + pid_m)
        if e < 0:
            return
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        a_ptrs = A + rm[:, None] * K + rk[None, :]
        w_ptrs = WQ + e.to(tl.int64) * K * N + rk[:, None] * N + rn[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(0, K, BK):
            a = tl.load(a_ptrs).to(tl.float32)
            s = tl.maximum(tl.max(tl.abs(a), axis=1) / 448.0, 1e-12)
            aq = (a / s[:, None]).to(tl.float8e4nv)
            acc += tl.dot(aq, tl.load(w_ptrs)) * s[:, None]
            a_ptrs += BK
            w_ptrs += BK * N
        acc = acc * tl.load(WSC)
        tl.store(C + rm[:, None] * N + rn[None, :], acc.to(C.dtype.element_ty))

    @triton.jit
    def _aggemm_tb_fp8_kernel(C, A, WQ, TILE_E, K, N, WSC,
                              BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        """C = A[P,K] @ W[e,K,N]，W 按 [E,N,K] 存（transposed-B），fp8。"""
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        e = tl.load(TILE_E + pid_m)
        if e < 0:
            return
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        a_ptrs = A + rm[:, None] * K + rk[None, :]
        w_ptrs = WQ + e.to(tl.int64) * N * K + rn[:, None] * K + rk[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(0, K, BK):
            a = tl.load(a_ptrs).to(tl.float32)
            s = tl.maximum(tl.max(tl.abs(a), axis=1) / 448.0, 1e-12)
            aq = (a / s[:, None]).to(tl.float8e4nv)
            acc += tl.dot(aq, tl.trans(tl.load(w_ptrs))) * s[:, None]
            a_ptrs += BK
            w_ptrs += BK
        acc = acc * tl.load(WSC)
        tl.store(C + rm[:, None] * N + rn[None, :], acc.to(C.dtype.element_ty))

    @triton.jit
    def _swiglu_bwd_tb_fp8_kernel(DGATE, DUP, DACT, DY, WQ, GATE_C, UP_C, WS, TILE_E,
                                  K, INTER, LIM, WSC,
                                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        e = tl.load(TILE_E + pid_m)
        if e < 0:
            return
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        dy_ptrs = DY + rm[:, None] * K + rk[None, :]
        w_ptrs = WQ + e.to(tl.int64) * INTER * K + rn[:, None] * K + rk[None, :]
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for _ in range(0, K, BK):
            a = tl.load(dy_ptrs).to(tl.float32)
            s = tl.maximum(tl.max(tl.abs(a), axis=1) / 448.0, 1e-12)
            aq = (a / s[:, None]).to(tl.float8e4nv)
            acc += tl.dot(aq, tl.trans(tl.load(w_ptrs))) * s[:, None]
            dy_ptrs += BK
            w_ptrs += BK
        acc = acc * tl.load(WSC)
        off = rm[:, None] * INTER + rn[None, :]
        gate = tl.load(GATE_C + off).to(tl.float32)
        up = tl.load(UP_C + off).to(tl.float32)
        ws = tl.load(WS + rm).to(tl.float32)
        dact = acc * ws[:, None]
        tl.store(DACT + off, acc.to(DACT.dtype.element_ty))
        gate_c = tl.minimum(gate, LIM)
        up_c = tl.maximum(tl.minimum(up, LIM), -LIM)
        sig = tl.sigmoid(gate_c)
        d_silu = sig * (1.0 + gate_c * (1.0 - sig))
        dgate = dact * up_c * d_silu
        dup = dact * (gate_c * sig)
        dgate = tl.where(gate_c < LIM, dgate, 0.0)
        dup = tl.where(tl.abs(up_c) < LIM, dup, 0.0)
        tl.store(DGATE + off, dgate.to(DGATE.dtype.element_ty))
        tl.store(DUP + off, dup.to(DUP.dtype.element_ty))

    def swiglu_fwd_nopad_fp8(x, ws_flat, w1q, w3q, ws1, ws3, flat, tile_e, seg_start,
                             count, n_tiles_m, inter, lim, ks):
        P = flat.shape[0]
        dev = x.device
        act = torch.empty((P, inter), dtype=x.dtype, device=dev)
        gate = torch.empty((P, inter), dtype=x.dtype, device=dev)
        up = torch.empty((P, inter), dtype=x.dtype, device=dev)
        _swiglu_fwd_nopad_fp8_kernel[(n_tiles_m, inter // BN)](
            act, gate, up, x, ws_flat, w1q, w3q, flat, tile_e, seg_start, count,
            ws1, ws3, D=x.shape[1], KS=ks, INTER=inter, LIM=float(lim),
            BM=BM, BN=BN, BK=BK, num_warps=WARPS, num_stages=STAGES)
        return act, gate, up

    def agg_forward_fp8(a, wq, wsc, tile_e, n_tiles_m, n_out):
        kdim = a.shape[1]
        out = torch.empty((a.shape[0], n_out), dtype=a.dtype, device=a.device)
        _aggemm_fp8_kernel[(n_tiles_m, n_out // BN)](
            out, a, wq, tile_e, kdim, n_out, wsc, BM=BM, BN=BN, BK=BK,
            num_warps=WARPS, num_stages=STAGES)
        return out

    def agg_forward_tb_fp8(a, wq, wsc, tile_e, n_tiles_m, n_out):
        kdim = a.shape[1]
        out = torch.empty((a.shape[0], n_out), dtype=a.dtype, device=a.device)
        _aggemm_tb_fp8_kernel[(n_tiles_m, n_out // BN)](
            out, a, wq, tile_e, kdim, n_out, wsc, BM=BM, BN=BN, BK=BK,
            num_warps=WARPS, num_stages=STAGES)
        return out

    def swiglu_bwd_tb_fp8(dy_pad, w2q, ws2, gate_c, up_c, ws_pad, tile_e, n_tiles_m, inter, lim):
        P = dy_pad.shape[0]
        dev = dy_pad.device
        dgate = torch.empty((P, inter), dtype=dy_pad.dtype, device=dev)
        dup = torch.empty((P, inter), dtype=dy_pad.dtype, device=dev)
        dact = torch.empty((P, inter), dtype=dy_pad.dtype, device=dev)
        _swiglu_bwd_tb_fp8_kernel[(n_tiles_m, inter // BN)](
            dgate, dup, dact, dy_pad, w2q, gate_c, up_c, ws_pad, tile_e,
            dy_pad.shape[1], inter, float(lim), ws2,
            BM=BM, BN=BN, BK=BK, num_warps=WARPS, num_stages=STAGES)
        return dgate, dup, dact

    def dw13_fused(a_pad, dgate, dup, seg_start, seg_end, n_experts, inter):
        K = a_pad.shape[1]
        dev = a_pad.device
        if DW_SPLIT > 1:
            p1 = torch.empty((DW_SPLIT, n_experts, K, inter), dtype=torch.float32, device=dev)
            p3 = torch.empty_like(p1)
            _dw13_split_kernel[(n_experts, K // BK, inter // BN, DW_SPLIT)](
                p1, p3, a_pad, dgate, dup, seg_start, seg_end, K, inter,
                S=DW_SPLIT, BM=BM, BK=BK, BN=BN, num_warps=WARPS, num_stages=STAGES)
            return p1.sum(0).to(a_pad.dtype), p3.sum(0).to(a_pad.dtype)
        dw1 = torch.empty((n_experts, K, inter), dtype=a_pad.dtype, device=dev)
        dw3 = torch.empty((n_experts, K, inter), dtype=a_pad.dtype, device=dev)
        _dw13_kernel[(n_experts, K // DW_BK, inter // DW_BN)](
            dw1, dw3, a_pad, dgate, dup, seg_start, seg_end, K, inter,
            BM=DW_BM, BK=DW_BK, BN=DW_BN, num_warps=DW_WARPS, num_stages=DW_STAGES)
        return dw1, dw3
    @triton.jit
    def _moe_tail_fwd_kernel(YP, ROF, OUT, T, D: tl.constexpr, K: tl.constexpr,
                             BD: tl.constexpr, BM: tl.constexpr):
        """out[t] = Σ_j y_pad[row_of_flat[t*K+j]]（尾部 3 个 aten kernel 融合）。"""
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        rmask = rows < T
        cols = tl.arange(0, BD)
        cmask = cols < D
        acc = tl.zeros((BM, BD), dtype=tl.float32)
        for j in tl.static_range(0, K):
            r = tl.load(ROF + rows * K + j, mask=rmask, other=0)
            acc += tl.load(YP + r[:, None] * D + cols[None, :],
                           mask=rmask[:, None] & cmask[None, :], other=0.0).to(tl.float32)
        tl.store(OUT + rows[:, None] * D + cols[None, :], acc.to(OUT.dtype.element_ty),
                 mask=rmask[:, None] & cmask[None, :])

    @triton.jit
    def _moe_tail_bwd_kernel(DOUT, ROF, DYP, TK, D: tl.constexpr, K: tl.constexpr,
                             BD: tl.constexpr, BM: tl.constexpr):
        """dy_pad[row_of_flat[f]] = dout[f // K]（每个 padded row 只被写一次，无原子）。"""
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        rmask = rows < TK
        cols = tl.arange(0, BD)
        cmask = cols < D
        r = tl.load(ROF + rows, mask=rmask, other=0)
        t = rows // K
        v = tl.load(DOUT + t[:, None] * D + cols[None, :],
                    mask=rmask[:, None] & cmask[None, :], other=0.0)
        tl.store(DYP + r[:, None] * D + cols[None, :], v, mask=rmask[:, None] & cmask[None, :])

    def moe_tail_fwd(y_pad, row_of_flat, T, k):
        D = y_pad.shape[1]
        out = torch.empty(T, D, dtype=y_pad.dtype, device=y_pad.device)
        _moe_tail_fwd_kernel[(triton.cdiv(T, 8),)](
            y_pad, row_of_flat, out, T, D=D, K=k, BD=triton.next_power_of_2(D),
            BM=8, num_warps=4)
        return out

    def moe_tail_bwd(dout, row_of_flat, P, k):
        D = dout.shape[1]
        # 必须 zeros：padded 行不参与 loss，但在 agg_dw/agg_forward 的反传里会被读到；
        # 旧路径靠 index 反传自动给 0，这里用 memset 显式清（~25 MB，15 µs）。
        dyp = torch.zeros(P, D, dtype=dout.dtype, device=dout.device)
        _moe_tail_bwd_kernel[(triton.cdiv(dout.shape[0] * k, 8),)](
            dout, row_of_flat, dyp, dout.shape[0] * k, D=D, K=k, BD=triton.next_power_of_2(D),
            BM=8, num_warps=4)
        return dyp

    @triton.jit
    def _moe_dws_kernel(DACT, GATE, UP, DWSP, P, INTER: tl.constexpr, LIM: tl.constexpr,
                        BD: tl.constexpr, BM: tl.constexpr):
        """dws[p] = Σ_inter dact * (silu(clamp(gate)) * clamp(up))（fp32 归约，1 kernel）。"""
        pid = tl.program_id(0)
        rows = pid * BM + tl.arange(0, BM)
        rmask = rows < P
        cols = tl.arange(0, BD)
        cmask = cols < INTER
        m2 = rmask[:, None] & cmask[None, :]
        g = tl.load(GATE + rows[:, None] * INTER + cols[None, :], mask=m2, other=0.0).to(tl.float32)
        u = tl.load(UP + rows[:, None] * INTER + cols[None, :], mask=m2, other=0.0).to(tl.float32)
        da = tl.load(DACT + rows[:, None] * INTER + cols[None, :], mask=m2, other=0.0).to(tl.float32)
        g = tl.minimum(g, LIM)
        u = tl.minimum(tl.maximum(u, -LIM), LIM)
        act0 = (g * tl.sigmoid(g)) * u
        tl.store(DWSP + rows, tl.sum(da * act0, axis=1), mask=rmask)

    def moe_dws(dact, gate, up, lim):
        P = gate.shape[0]
        inter = gate.shape[1]
        out = torch.empty(P, dtype=torch.float32, device=gate.device)
        _moe_dws_kernel[(triton.cdiv(P, 8),)](
            dact, gate, up, out, P, INTER=inter, LIM=float(lim),
            BD=triton.next_power_of_2(inter), BM=8, num_warps=4)
        return out

else:  # pragma: no cover
    def swiglu_fwd(*a, **k):
        raise RuntimeError("triton 不可用（Mac）")

    def swiglu_bwd(*a, **k):
        raise RuntimeError("triton 不可用（Mac）")

    def dw13_fused(*a, **k):
        raise RuntimeError("triton 不可用（Mac）")

    def moe_tail_fwd(*a, **k):
        raise RuntimeError("triton 不可用（Mac）")

    def moe_tail_bwd(*a, **k):
        raise RuntimeError("triton 不可用（Mac）")

    def moe_dws(*a, **k):
        raise RuntimeError("triton 不可用（Mac）")

    def agg_forward_tb(*a, **k):
        raise RuntimeError("triton 不可用（Mac）")

    def swiglu_fwd_nopad(*a, **k):
        raise RuntimeError("triton 不可用（Mac）")

    def swiglu_fwd_fp8(*a, **k):
        raise RuntimeError("triton 不可用（Mac）")

    def agg_forward_fp8(*a, **k):
        raise RuntimeError("triton 不可用（Mac）")

    def agg_forward_tb_fp8(*a, **k):
        raise RuntimeError("triton 不可用（Mac）")

    def swiglu_bwd_tb_fp8(*a, **k):
        raise RuntimeError("triton 不可用（Mac）")

    def quant_weight(*a, **k):
        raise RuntimeError("triton 不可用（Mac）")

    def dw13_nopad(*a, **k):
        raise RuntimeError("triton 不可用（Mac）")

    def swiglu_bwd_tb(*a, **k):
        raise RuntimeError("triton 不可用（Mac）")


class _GroupedMoEv2(torch.autograd.Function):
    """v2：fwd 一个 kernel 融合 gate/up GEMM + SwiGLU + routing 权重；bwd 融合 dact+SwiGLU 反向，
    并让 dW1/dW3 共享 A 的读取。"""

    @staticmethod
    def forward(ctx, x_pad, ws_pad, w1, w3, w2, tile_e, seg_start, seg_end, n_tiles_m, lim):
        P = x_pad.shape[0]
        dim = x_pad.shape[1]
        inter = w1.shape[2]
        act, gate_f, up_f = swiglu_fwd(x_pad, w1, w3, ws_pad, tile_e, n_tiles_m, inter, lim)
        y_pad = agg_forward(act, w2, tile_e, n_tiles_m, dim)
        ctx.save_for_backward(x_pad, ws_pad, w1, w3, w2, tile_e, seg_start, seg_end,
                              gate_f, up_f, act)
        ctx.n_tiles_m = n_tiles_m
        ctx.lim = float(lim)
        return y_pad

    @staticmethod
    def backward(ctx, dy_pad):
        (x_pad, ws_pad, w1, w3, w2, tile_e, seg_start, seg_end,
         gate_f, up_f, act) = ctx.saved_tensors
        E = w1.shape[0]
        dim = w2.shape[2]
        inter = w1.shape[2]
        lim = ctx.lim
        n_tiles_m = ctx.n_tiles_m
        dW2 = agg_dw(act, dy_pad, seg_start, seg_end, E)                 # [E, inter, dim]
        if TB:
            dgate, dup, dact = swiglu_bwd_tb(dy_pad, w2, gate_f, up_f, ws_pad, tile_e,
                                             n_tiles_m, inter, lim)
        else:
            w2t = w2.transpose(1, 2).contiguous()                        # [E, dim, inter]
            dgate, dup, dact = swiglu_bwd(dy_pad, w2t, gate_f, up_f, ws_pad, tile_e,
                                          n_tiles_m, inter, lim)
        # routing 权重 ws 来自 gate(x)，必须回传梯度：dws = Σ_inter dact * act0，act0 = silu(gate_c)*up_c
        if FUSED_DWS:
            dws_pad = moe_dws(dact, gate_f, up_f, lim)
        else:
            gate_c = torch.clamp(gate_f.float(), max=lim)
            up_c = torch.clamp(up_f.float(), min=-lim, max=lim)
            act0 = torch.nn.functional.silu(gate_c) * up_c
            dws_pad = (dact.float() * act0).sum(-1)
        dW1, dW3 = dw13_fused(x_pad, dgate, dup, seg_start, seg_end, E, inter)
        if FUSED_DX:
            dx_pad = agg_forward2_t(dgate, w1, dup, w3, tile_e, n_tiles_m, dim)
        elif TB:
            dx_pad = agg_forward_tb(dgate, w1, tile_e, n_tiles_m, dim)
            dx_pad = dx_pad + agg_forward_tb(dup, w3, tile_e, n_tiles_m, dim)
        else:
            w1t = w1.transpose(1, 2).contiguous()
            w3t = w3.transpose(1, 2).contiguous()
            dx_pad = agg_forward(dgate, w1t, tile_e, n_tiles_m, dim)
            dx_pad = dx_pad + agg_forward(dup, w3t, tile_e, n_tiles_m, dim)
        return (dx_pad, dws_pad, dW1, dW3, dW2, None, None, None, None, None)


class _GroupedMoEv2NoPad(torch.autograd.Function):
    """R50：前向不物化 x_pad 的 grouped MoE（A 行用 FLAT 映射回原始 x）。

    与 `_GroupedMoEv2` 数学一致：act = silu(clamp(gate))*clamp(up)*ws，y = act @ w2；
    反传里 dW1/dW3 也从原始 x 按映射取行（`_dw13_nopad_kernel`），dx 用 tail kernel 汇总回 token。
    """

    @staticmethod
    def forward(ctx, x, ws_flat, ws_pad, flat, rof, seg_start, seg_end, count, tile_e,
                n_tiles_m, w1, w3, w2, ks, lim):
        dim = x.shape[1]
        inter = w1.shape[2]
        act, gate_f, up_f = swiglu_fwd_nopad(x, ws_flat, w1, w3, flat, tile_e, seg_start,
                                             count, n_tiles_m, inter, lim, ks)
        y_pad = agg_forward(act, w2, tile_e, n_tiles_m, dim)
        ctx.save_for_backward(x, ws_pad, flat, rof, seg_start, seg_end, count, tile_e,
                              act, gate_f, up_f, w1, w3, w2)
        ctx.n_tiles_m = n_tiles_m
        ctx.ks = ks
        ctx.lim = float(lim)
        ctx.T = rof.shape[0] // ks
        return y_pad

    @staticmethod
    def backward(ctx, dy_pad):
        (x, ws_pad, flat, rof, seg_start, seg_end, count, tile_e,
         act, gate_f, up_f, w1, w3, w2) = ctx.saved_tensors
        E = w1.shape[0]
        dim = w2.shape[2]
        inter = w1.shape[2]
        lim = ctx.lim
        n_tiles = ctx.n_tiles_m
        ks = ctx.ks
        dW2 = agg_dw(act, dy_pad, seg_start, seg_end, E)
        if TB:
            dgate, dup, dact = swiglu_bwd_tb(dy_pad, w2, gate_f, up_f, ws_pad, tile_e,
                                             n_tiles, inter, lim)
        else:  # pragma: no cover - 默认 TB 开
            w2t = w2.transpose(1, 2).contiguous()
            dgate, dup, dact = swiglu_bwd(dy_pad, w2t, gate_f, up_f, ws_pad, tile_e,
                                          n_tiles, inter, lim)
        if FUSED_DWS:
            dws_pad = moe_dws(dact, gate_f, up_f, lim)
        else:  # pragma: no cover
            gate_c = torch.clamp(gate_f.float(), max=lim)
            up_c = torch.clamp(up_f.float(), min=-lim, max=lim)
            act0 = torch.nn.functional.silu(gate_c) * up_c
            dws_pad = (dact.float() * act0).sum(-1)
        dW1, dW3 = dw13_nopad(x, flat, seg_start, count, dgate, dup, E, inter, ks)
        if FUSED_DX:  # pragma: no cover
            dx_pad = agg_forward2_t(dgate, w1, dup, w3, tile_e, n_tiles, dim)
        elif TB:
            dx_pad = agg_forward_tb(dgate, w1, tile_e, n_tiles, dim)
            dx_pad = dx_pad + agg_forward_tb(dup, w3, tile_e, n_tiles, dim)
        else:  # pragma: no cover
            w1t = w1.transpose(1, 2).contiguous()
            w3t = w3.transpose(1, 2).contiguous()
            dx_pad = agg_forward(dgate, w1t, tile_e, n_tiles, dim)
            dx_pad = dx_pad + agg_forward(dup, w3t, tile_e, n_tiles, dim)
        # dx: padded 行 → token（Σ over k slots），复用 tail kernel
        dx = moe_tail_fwd(dx_pad, rof, ctx.T, ks)
        # dws 映射回 ws_flat：dws_flat[f] = dws_pad[row_of_flat[f]]
        dws_flat = dws_pad[rof]
        return (dx, dws_flat, None, None, None, None, None, None, None, None,
                dW1, dW3, dW2, None, None)


class _GroupedMoEv2FP8(torch.autograd.Function):
    """MoE fwd/dx/dact 三段走 fp8（e4m3）；dW 两段仍 bf16（数值更敏感，R51 再换）。

    权重每步 per-tensor 量化（`quant_weight`，6 个小 kernel/层）；激活在 GEMM kernel 内
    按"每个 k-tile 的行 max"量化（per-row scale），fp32 累加后乘 s_row 与 wscale 还原。
    """

    @staticmethod
    def forward(ctx, x_pad, ws_pad, w1, w3, w2, tile_e, seg_start, seg_end, n_tiles_m, lim):
        inter = w1.shape[2]
        w1q, ws1 = quant_weight(w1)
        w3q, ws3 = quant_weight(w3)
        w2q, ws2 = quant_weight(w2)
        act, gate_f, up_f = swiglu_fwd_fp8(x_pad, ws_pad, w1q, w3q, ws1, ws3, tile_e,
                                           n_tiles_m, inter, lim)
        y_pad = agg_forward_fp8(act, w2q, ws2, tile_e, n_tiles_m, w2.shape[2])
        ctx.save_for_backward(x_pad, ws_pad, w1, w3, w2, w1q, w3q, w2q,
                              tile_e, seg_start, seg_end, act, gate_f, up_f, ws1, ws3, ws2)
        ctx.n_tiles_m = n_tiles_m
        ctx.lim = float(lim)
        return y_pad

    @staticmethod
    def backward(ctx, dy_pad):
        (x_pad, ws_pad, w1, w3, w2, w1q, w3q, w2q,
         tile_e, seg_start, seg_end, act, gate_f, up_f, ws1, ws3, ws2) = ctx.saved_tensors
        E = w1.shape[0]
        dim = w2.shape[2]
        inter = w1.shape[2]
        lim = ctx.lim
        n_tiles = ctx.n_tiles_m
        dW2 = agg_dw(act, dy_pad, seg_start, seg_end, E)
        dgate, dup, dact = swiglu_bwd_tb_fp8(dy_pad, w2q, ws2, gate_f, up_f, ws_pad, tile_e,
                                            n_tiles, inter, lim)
        if FUSED_DWS:
            dws_pad = moe_dws(dact, gate_f, up_f, lim)
        else:  # pragma: no cover
            gate_c = torch.clamp(gate_f.float(), max=lim)
            up_c = torch.clamp(up_f.float(), min=-lim, max=lim)
            act0 = torch.nn.functional.silu(gate_c) * up_c
            dws_pad = (dact.float() * act0).sum(-1)
        dW1, dW3 = dw13_fused(x_pad, dgate, dup, seg_start, seg_end, E, inter)
        dx_pad = agg_forward_tb_fp8(dgate, w1q, ws1, tile_e, n_tiles, dim)
        dx_pad = dx_pad + agg_forward_tb_fp8(dup, w3q, ws3, tile_e, n_tiles, dim)
        return (dx_pad, dws_pad, dW1, dW3, dW2, None, None, None, None, None)


class _MoETailFn(torch.autograd.Function):
    """MoE 前向尾部：unpad gather + 按 order 散射 + 按 token 对 k 个 slot 求和（1 kernel / 方向）。"""

    @staticmethod
    def forward(ctx, y_pad, row_of_flat, T, k):
        out = moe_tail_fwd(y_pad, row_of_flat, T, k)
        ctx.save_for_backward(row_of_flat)
        ctx.P, ctx.k = y_pad.shape[0], k
        return out

    @staticmethod
    def backward(ctx, dout):
        (row_of_flat,) = ctx.saved_tensors
        dyp = moe_tail_bwd(dout, row_of_flat, ctx.P, ctx.k)
        return dyp, None, None, None


def grouped_moe_experts_v2(x, w1, w2, w3, layout: MoELayout, ws_flat: torch.Tensor,
                           swiglu_limit: float = 10.0):
    if FP8:
        if _FUSED_PAD:
            x_pad, ws_pad = layout.scatter_pad(x, ws_flat)
        else:
            x_sorted = x[layout.tok]
            x_pad = layout.pad(x_sorted, x.shape[1])
            ws_sorted = ws_flat[layout.order]
            ws_pad = layout.pad(ws_sorted[:, None], 1).squeeze(1)
        y_pad = _GroupedMoEv2FP8.apply(x_pad, ws_pad, w1, w3, w2, layout.tile_e,
                                       layout.seg_start, layout.seg_end,
                                       layout.n_tiles_m, float(swiglu_limit))
    elif NOPAD:
        # 只物化很小的 ws_pad（bwd 的 swiglu_bwd 用）；x 不再 gather/pad
        ws_sorted = ws_flat[layout.order]
        ws_pad = layout.pad(ws_sorted[:, None], 1).squeeze(1)
        y_pad = _GroupedMoEv2NoPad.apply(
            x, ws_flat.contiguous(), ws_pad, layout.flat_of_row, layout.row_of_flat,
            layout.seg_start, layout.seg_end, layout.count_i32, layout.tile_e,
            layout.n_tiles_m, w1, w3, w2, layout.k, float(swiglu_limit))
    else:
        if _FUSED_PAD:
            x_pad, ws_pad = layout.scatter_pad(x, ws_flat)
        else:
            x_sorted = x[layout.tok]
            x_pad = layout.pad(x_sorted, x.shape[1])
            ws_sorted = ws_flat[layout.order]
            ws_pad = layout.pad(ws_sorted[:, None], 1).squeeze(1)
        y_pad = _GroupedMoEv2.apply(x_pad, ws_pad, w1, w3, w2, layout.tile_e,
                                    layout.seg_start, layout.seg_end,
                                    layout.n_tiles_m, float(swiglu_limit))
    if FUSED_TAIL:
        return _MoETailFn.apply(y_pad, layout.row_of_flat, layout.T, layout.k)
    y_sorted = layout.unpad(y_pad)
    y_flat = torch.empty_like(y_sorted)
    y_flat[layout.order] = y_sorted
    return y_flat.view(layout.T, layout.k, x.shape[1]).sum(1)

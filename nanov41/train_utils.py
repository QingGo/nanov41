"""Training-speed utilities for nano-v41, derived from the DeepSeek-V4.1 report ideas:

- `moe_grouped_forward`: replace the reference per-expert `torch.where` loop with one
  stable sort + contiguous per-expert GEMMs (report: CSA2/Mega-MoE style token grouping;
  keeps exact math, removes host syncs and tiny scatter kernels).
- `head_bf16_forward`: the reference head keeps an fp32 weight and emits fp32 logits; for
  training we run the head GEMM in bf16 (fp32 master weight kept in the optimizer).
- `chunked_cross_entropy`: custom autograd that stores bf16 logits and does the
  softmax/normalization chunk-wise, so we never allocate an fp32 [T, V] buffer.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

try:  # Triton 融合 CE（GPU）；Mac 上不可用则自动退回 torch 路径
    from . import triton_ce as _triton_ce
    _CE_TRITON = os.environ.get("NANOV41_CE_TRITON", "1").lower() not in ("0", "false", "no", "off")
except Exception:  # pragma: no cover
    _triton_ce = None
    _CE_TRITON = False


def moe_grouped_forward(self, x, image_mask=None):
    """Exact-math replacement for reference MoE.forward (sort once, no where/sync)."""
    shape = x.size()
    x = x.view(-1, self.dim)
    w, idx = self.gate(x, None if image_mask is None else image_mask.flatten())
    T, k = idx.shape
    flat = idx.reshape(-1)
    order = torch.argsort(flat, stable=True)
    tok = torch.div(order, k, rounding_mode="floor")
    counts = torch.bincount(flat, minlength=self.n_routed_experts).tolist()
    xs = x[tok]
    ws = w.reshape(-1)[order]
    outs, pos = [], 0
    for e in range(self.experts_start_idx, self.experts_end_idx):
        c = counts[e]
        if c == 0:
            continue
        outs.append(self.experts[e](xs[pos:pos + c], ws[pos:pos + c, None]))
        pos += c
    y_sorted = torch.cat(outs, 0)
    flat_out = torch.empty(T * k, self.dim, dtype=y_sorted.dtype, device=y_sorted.device)
    flat_out[order] = y_sorted
    y = flat_out.view(T, k, self.dim).sum(1).to(x.dtype)
    return (y + self.shared_experts(x)).view(shape)


_GROUPED_MM = getattr(torch, "_grouped_mm", None)


def moe_grouped_forward_gmm(self, x, image_mask=None):
    """MoE forward using torch._grouped_mm（CUTLASS/cuBLASLt grouped GEMM）。

    2026-10-21 在 4090 上实测：与 `moe_grouped_forward`（逐 expert 循环）**逐位一致**
    （fwd 与 dx/dW1/dW3/dW2 的 max_abs_diff = 0.0），单层 fwd+bwd 11.63 → 5.10 ms（2.28×）。
    torch 2.12 的 _grouped_mm 在 sm89 也能跑，autograd 完整支持（对 stacking 的权重梯度正确）。

    前置条件：world_size == 1（nano 配置恒为真）。offs 语义：**每组结束位置**的 int32 数组。
    """
    assert _GROUPED_MM is not None, "torch._grouped_mm 不可用"
    assert torch.distributed.is_initialized() is False or torch.distributed.get_world_size() == 1
    shape = x.size()
    x = x.view(-1, self.dim)
    w, idx = self.gate(x, None if image_mask is None else image_mask.flatten())
    T, k = idx.shape
    flat = idx.reshape(-1)
    order = torch.argsort(flat, stable=True)
    tok = torch.div(order, k, rounding_mode="floor")
    counts = torch.bincount(flat, minlength=self.n_routed_experts)
    offs = torch.cumsum(counts, 0).to(torch.int32)          # 每组结束位置，长度 E
    xs = x[tok]
    ws = w.reshape(-1)[order]
    experts = [self.experts[i] for i in range(self.experts_start_idx, self.experts_end_idx)]
    W1 = torch.stack([e.w1.weight for e in experts]).transpose(1, 2)   # [E, dim, inter]
    W3 = torch.stack([e.w3.weight for e in experts]).transpose(1, 2)
    W2 = torch.stack([e.w2.weight for e in experts]).transpose(1, 2)   # [E, inter, dim]
    gate = _GROUPED_MM(xs, W1, offs=offs).float()
    up = _GROUPED_MM(xs, W3, offs=offs).float()
    lim = experts[0].swiglu_limit
    if lim and lim > 0:
        up = torch.clamp(up, min=-lim, max=lim)
        gate = torch.clamp(gate, max=lim)
    h = ws[:, None] * (F.silu(gate) * up)                  # 与 Expert.forward 的运算顺序一致
    outs = _GROUPED_MM(h.to(x.dtype), W2, offs=offs)
    flat_out = torch.empty(T * k, self.dim, dtype=outs.dtype, device=outs.device)
    flat_out[order] = outs
    y = flat_out.view(T, k, self.dim).sum(1).to(x.dtype)
    return (y + self.shared_experts(x)).view(shape)


def head_bf16_forward(self, x, full_logits=False):
    if not full_logits:
        x = x[:, -1]
    return F.linear(x.to(torch.bfloat16), self.weight.to(torch.bfloat16))


class _ChunkedCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, chunk=2048, ignore_index=-100):
        V = logits.size(-1)
        targets = targets.reshape(-1)
        T = targets.numel()
        total = torch.zeros((), dtype=torch.float32, device=logits.device)
        count = 0
        flat = logits.reshape(-1, V)
        for i in range(0, T, chunk):
            lc = flat[i:i + chunk].float()
            tc = targets[i:i + chunk]
            total = total + F.cross_entropy(lc, tc, reduction="sum", ignore_index=ignore_index)
            count += (tc != ignore_index).sum()
        ctx.save_for_backward(logits, targets)
        ctx.chunk = chunk
        ctx.ignore_index = ignore_index
        return total / count

    @staticmethod
    def backward(ctx, grad_out):
        logits, targets = ctx.saved_tensors
        V = logits.size(-1)
        flat = logits.reshape(-1, V)
        targets = targets.reshape(-1)
        T = targets.numel()
        g = torch.empty_like(flat)
        scale = grad_out.float() / (targets != ctx.ignore_index).sum()
        for i in range(0, T, ctx.chunk):
            lc = flat[i:i + ctx.chunk].float()
            tc = targets[i:i + ctx.chunk]
            d = torch.softmax(lc, dim=-1)
            d.sub_(F.one_hot(tc.clamp_min(0), V).to(d.dtype))
            d.mul_(scale)
            if (tc == ctx.ignore_index).any():
                d[tc == ctx.ignore_index] = 0        # ignore_index 行 CE 梯度必须为 0
            g[i:i + ctx.chunk] = d.to(g.dtype)
        return g.view_as(logits), None, None, None


def chunked_cross_entropy(logits, targets, chunk=2048, ignore_index=-100):
    return _ChunkedCE.apply(logits, targets, chunk, ignore_index)


class _ChunkedCEShift(torch.autograd.Function):
    """chunked CE with the shift done in-place instead of materializing logits[:, :-1].

    旧实现的两个问题（b=16 OOM 的根因，见 runs/p0_b8.json 的 direct_copy kernel 26ms/step）：
      1. `logits[:, :-1].reshape(-1, V)`：非连续 view 会**整块复制**（b=8 已是 2.1 GB，
         b=16 4.2 GB；forward + backward 各一次）；
      2. backward 里 `F.one_hot(t, V)` 是 int64，chunk=2048 时每块 2.1 GB。
    这里改成：logits 保持连续，用 scatter 构造 shift 后的 target（只有 b*s 个元素），
    flat = logits.view(b*s, V) 是零拷贝 view；backward 用 scatter_ 直接减 1，
    不再分配 one_hot。接口：logits [b,s,V] 连续、ids [b,s]。
    """

    @staticmethod
    def forward(ctx, logits, ids, chunk=1024, ignore_index=-100):
        b, s, V = logits.shape
        n = b * s
        ctx_aligned = chunk % s == 0   # 对齐时可以用无同步的 strided 清零（CUDA graph 需要）
        flat = logits.reshape(n, V)                       # contiguous -> view（零拷贝）
        target = torch.full((n,), ignore_index, dtype=torch.long, device=logits.device)
        target.view(b, s)[:, :-1] = ids[:, 1:]            # row (i,t) 预测 ids[i,t+1]
        total = torch.zeros((), dtype=torch.float32, device=logits.device)
        count = 0
        for i in range(0, n, chunk):
            lc = flat[i:i + chunk].float()
            tc = target[i:i + chunk]
            total = total + F.cross_entropy(lc, tc, reduction="sum", ignore_index=ignore_index)
            count += (tc != ignore_index).sum()
        ctx.save_for_backward(logits, target)
        ctx.chunk = chunk
        ctx.ignore_index = ignore_index
        ctx.s = s
        ctx.aligned = ctx_aligned
        ctx.count = count
        return total / count

    @staticmethod
    def backward(ctx, grad_out):
        logits, target = ctx.saved_tensors
        b, s, V = logits.shape
        n = b * s
        flat = logits.reshape(n, V)
        g = torch.empty_like(flat)                        # dlogits，bf16
        scale = grad_out.float() / ctx.count
        for i in range(0, n, ctx.chunk):
            tc = target[i:i + ctx.chunk]
            d = torch.softmax(flat[i:i + ctx.chunk].float(), dim=-1)
            d.mul_(scale)
            idx = tc.clamp_min(0).unsqueeze(1)
            d.scatter_(1, idx, d.gather(1, idx) - scale)  # p*scale - 1*scale
            # ignore_index 行 = shift 后每条序列最后一个位置（位置由 shape 决定、与数据无关），
            # 梯度必须为 0。不能用 `(tc == ig).any()`：那是 device→host 同步，会让 CUDA graph
            # 捕获直接失败（实测 4090/5090）。这里按 stride 清零，只写 chunk/s 行。
            # shift 造出的 target：每条序列最后一个位置恒为 ignore_index（默认 -100），
            # 其 CE 梯度必须为 0。注意不能按 `ignore_index >= 0` 判断是否启用（-100 也是启用）。
            if ctx.aligned:
                d[ctx.s - 1::ctx.s] = 0
            else:  # 非对齐（只有测试/边角场景）：动态 mask 路径，不做 graph 承诺
                mask = tc == ctx.ignore_index
                if mask.any():
                    d[mask] = 0
            g[i:i + ctx.chunk] = d.to(g.dtype)
        return g.view_as(logits), None, None, None


def chunked_cross_entropy_shift(logits, ids, chunk=1024, ignore_index=-100):
    """fwd+shift+CE，零整块拷贝。logits [b,s,V] 连续，ids [b,s]。"""
    return _ChunkedCEShift.apply(logits, ids, chunk, ignore_index)


class _FusedHeadCE(torch.autograd.Function):
    """Fused head projection + shift + CE：分块算 logits，**不落全量 [T, V]、也不落 fp32 logits**。

    为什么（2026-10-21 在 4090 上 op 级 profile 的结论）：
      * `cunn_SoftMaxForward<float>`（CE 的 softmax）单步 31.7 ms/16 次调用，是最大的单类 kernel；
        根因是 fp32 logits 的写+读（b=8 每 chunk 530MB ×2 遍）。
      * head 权重 [129280, 384] bf16 = 99MB，每 chunk 重读一次；chunk 越小重读越多。

    v2 做法（数学等价，误差在 bf16 输出舍入量级）：
      1. GEMM 直接出 bf16 logits，**不 `.float()`**：bf16 的 log_softmax/softmax kernel 内部
         就是 fp32 累加，fp32 副本纯属额外搬运；
      2. loss 用 `cross_entropy(reduction="none").float().sum()`，跨 chunk 在 fp32 里累加
         （8192 行随机舍入误差 ≈ 2e-5 相对，远优于 2e-3 验收阈值）；
      3. backward 的 scale 从 d 挪到 matmul **之后**（d 保持 bf16，省掉一次 fp32 广播乘）；
      4. chunk 默认 2048（= 2×seq1024），head 权重重读次数从 8 次降到 4 次。
    显存：chunk=2048 时 bf16 logits 530MB + softmax 530MB ≈ 1.1GB 峰值（与 v1 相当）。
    """

    @staticmethod
    def forward(ctx, hidden, weight, ids, chunk=2048, ignore_index=-100, z_loss=0.0):
        b, s, D = hidden.shape
        V = weight.shape[0]
        n = b * s
        ctx_aligned = chunk % s == 0   # 对齐时用无同步的 strided 清零（CUDA graph 需要）
        hf = hidden.reshape(n, D)                       # contiguous view
        target = torch.full((n,), ignore_index, dtype=torch.long, device=hidden.device)
        target.view(b, s)[:, :-1] = ids[:, 1:]
        wb = weight.to(torch.bfloat16)
        total = torch.zeros((), dtype=torch.float32, device=hidden.device)
        z2 = torch.zeros((), dtype=torch.float32, device=hidden.device)
        count = torch.zeros((), dtype=torch.float32, device=hidden.device)
        # 只有 CUDA 张量才走 Triton（CPU 对拍/门禁在 GPU 机器上跑时不能误走 GPU kernel）
        use_triton_ce = (_CE_TRITON and hidden.is_cuda and _triton_ce is not None
                         and _triton_ce.available() and z_loss == 0)
        lse_all = (torch.empty(n, dtype=torch.float32, device=hidden.device)
                   if use_triton_ce else None)
        for i in range(0, n, chunk):
            lc = F.linear(hf[i:i + chunk].to(torch.bfloat16), wb)   # bf16 logits（不 cast fp32）
            tc = target[i:i + chunk]
            if use_triton_ce:
                s_rows, lse = _triton_ce.ce_fwd(lc, tc, ignore_index)   # 在线 lse，不物化 log_softmax
                total = total + s_rows
                lse_all[i:i + chunk] = lse
            else:
                row = F.cross_entropy(lc, tc, reduction="none", ignore_index=ignore_index)
                total = total + row.float().sum()                        # 跨 chunk fp32 累加
            count = count + (tc != ignore_index).sum()
            if z_loss:  # z-loss 路径很少用，保持 fp32 语义
                z2 = z2 + torch.logsumexp(lc.float(), -1).pow(2).sum()
        loss = total / count
        if z_loss:
            loss = loss + z_loss * z2 / n
        ctx.save_for_backward(hidden, weight, target)
        ctx.chunk = chunk
        ctx.count = count
        ctx.z_loss = z_loss
        ctx.n = n
        ctx.s = s
        ctx.aligned = ctx_aligned
        ctx.ignore_index = ignore_index
        ctx.use_triton_ce = use_triton_ce
        ctx.lse_all = lse_all
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        hidden, weight, target = ctx.saved_tensors
        b, s, D = hidden.shape
        n = b * s
        hf = hidden.reshape(n, D)
        wb = weight.to(torch.bfloat16)
        gh = torch.empty_like(hf)
        gw = torch.zeros_like(weight)                   # fp32 master 累加
        scale = grad_out.float() / ctx.count
        for i in range(0, n, ctx.chunk):
            tc = target[i:i + ctx.chunk]
            xc = hf[i:i + ctx.chunk].to(torch.bfloat16)
            lc = F.linear(xc, wb)                       # bf16 logits（重算，省显存）
            d = torch.softmax(lc, dim=-1)               # bf16 输出，内部 fp32 归约
            idx = tc.clamp_min(0).unsqueeze(1)
            d.scatter_(1, idx, d.gather(1, idx) - 1.0)  # p - onehot（bf16，误差在输出舍入量级）
            # ignore_index 行 = shift 后每条序列最后一个位置（由 shape 决定），梯度须为 0。
            # shift 造出的 target 每条序列最后一个位置恒为 ignore_index，CE 梯度必须为 0
            # （不能用 `ignore_index >= 0` 判断是否启用：默认值 -100 也是启用状态）
            if ctx.aligned:
                d[ctx.s - 1::ctx.s] = 0
            else:  # 非对齐（仅测试/边角）：动态 mask 路径
                mask = tc == ctx.ignore_index
                if mask.any():
                    d[mask] = 0
            if ctx.z_loss:
                # d += (2*z_loss/n) * z * softmax(logits)，z = logsumexp
                lcf = lc.float()
                z = torch.logsumexp(lcf, -1, keepdim=True)
                d = d.float() + (2.0 * ctx.z_loss / ctx.n) * z * torch.softmax(lcf, dim=-1)
                d = d.to(wb.dtype)
            # 非 z-loss 路径：d 直接保持 bf16。原来多一步 `d.float()`，但紧接着就 `.to(bf16)` 喂 matmul，
            # 等于白做一次 [chunk, V] 的 fp32 往返（op profile 里 copy_ [2048,129280] 20.45 ms/step 的主因）。
            gh[i:i + ctx.chunk] = (torch.matmul(d, wb) * scale).to(gh.dtype)
            gw += torch.matmul(d.t(), xc).float() * scale
        return gh.view_as(hidden), gw, None, None, None, None


def fused_head_ce(hidden, weight, ids, chunk=None, ignore_index=-100, z_loss=0.0):
    """hidden [b,s,D] → loss；head 权重 [V,D]；ids [b,s]。等价于
    head_bf16_forward + chunked_cross_entropy_shift（+ 可选 z-loss），但显存 O(chunk·V)。"""
    if chunk is None:
        chunk = int(os.environ.get("NANOV41_CE_CHUNK", "2048"))
    _fp8 = os.environ.get("NANOV41_HEAD_FP8", "").lower()
    if _fp8 in ("2", "v2") and z_loss == 0:
        try:
            from . import fp8_head_v2 as _f8v2
            if _f8v2.available():
                return _f8v2.fused_head_ce_fp8(hidden, weight, ids, chunk, ignore_index, z_loss)
        except Exception:  # noqa: BLE001
            pass
    return _FusedHeadCE.apply(hidden, weight, ids, chunk, ignore_index, z_loss)

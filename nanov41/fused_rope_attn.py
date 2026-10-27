"""fused_rope_attn — 把 attention 里的 q/kv/o RoPE 换成"单 kernel 原地旋转"（R49）。

官方 `apply_rotary_emb`（`_ref_patched_model.py:402`）每次调用是 3–4 个 aten kernel 且全程 fp32 中间量：

    x = torch.view_as_complex(x.float()...)   # bf16 -> fp32 copy
    x = x * freqs_cis                         # complex64 复数乘（fp32）
    y.copy_(x)                                # fp32 -> bf16，原地写回切片

R45 试过"compute kernel + 可追踪 copy_"（2 kernel）→ 无收益（`fused_rope.py` 已证伪）。
R49 换思路：**patch `Attention.forward` / `Attention._window_kv` 的调用点，让 rope 返回张量并被赋值**，
于是可以做成"原地单 kernel + custom Function"，不需要 copy_、也不经过 fp32 中间量：

    q = rope_tail(q, freqs_cis, False)         # 1 kernel：读/写 [b,s,h,16] 的 tail（bf16 直进直出）
    o = rope_tail(o, freqs_cis, True)
    kv = rope_tail(kv, freqs_cis, False)       # 在 _window_kv 里

流量从 ~24 MB/次降到 ~8 MB/次，kernel 数 3–4 → 1。
反传：`d(x) = rotate^{-1}(dy)`（同一个 kernel，逆旋转）。

安装：`install(M)` 复制 `Attention.forward` / `Attention._window_kv` 并在相应位置换成 `rope_tail`
（复制而不是包 layer，因为需要把"忽略返回值"改成"赋值"）。
"""
from __future__ import annotations

import os

import torch

try:  # pragma: no cover - triton is GPU-only
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # noqa: BLE001
    HAS_TRITON = False
    triton = None
    tl = None

__all__ = ["rope_tail", "apply_rotary_emb_ref", "install", "uninstall", "available"]

_ENABLED = os.environ.get("NANOV41_ROPE_ATTN", "0").lower() not in ("0", "false", "no", "off")
# A/B 诊断：patch 后的 forward 里仍用**官方 rope**（证明"patched forward 与官方语义一致"），
# 用来隔离"loss 差异到底来自 rope kernel 舍入还是 forward 复制走样"
_REF_ROPE = os.environ.get("NANOV41_ROPE_ATTN_REF", "0").lower() not in ("0", "false", "no", "off")
_INSTALLED = None
_MOD = None


def apply_rotary_emb_ref(x, freqs_cis, inverse=False):
    """官方实现逐行复刻（CPU fallback / 测试用）。"""
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x.ndim == 3:
        freqs_cis = freqs_cis.view(1, x.size(1), x.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
    y.copy_(x)
    return y


def _freqs_2d(freqs_cis: torch.Tensor):
    """复数 freqs [n, rd/2] -> (fp32 [n, rd] 视图或拷贝, 行跨度(元素))。"""
    assert freqs_cis.is_complex()
    k = freqs_cis.shape[-1]
    rd = 2 * k
    base = freqs_cis.reshape(-1, k)
    row_stride = base.stride(0) * 2
    flat = torch.view_as_real(base)
    if flat.stride(-1) != 1 or flat.stride(-2) != 2:
        flat = flat.contiguous()
        row_stride = rd
    return flat.reshape(-1, rd), row_stride


def _layout(x: torch.Tensor):
    """(B, S, H, 父行跨度 SP, batch stride SB, token stride ST)。"""
    if x.dim() == 3:
        b, s, _ = x.shape
        return b, s, 1, x.stride(1), x.stride(0), x.stride(1)
    if x.dim() == 4:
        b, s, h, _ = x.shape
        return b, s, h, x.stride(2), x.stride(0), x.stride(1)
    raise ValueError(f"rope_tail: unsupported ndim {x.dim()}")


if HAS_TRITON:

    @triton.jit
    def _rope_tail_kernel(
        X, F, T,
        SB, ST, SP,
        RD: tl.constexpr, H: tl.constexpr, BR: tl.constexpr, FS,
        INVERSE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        t = pid % T
        b = pid // T
        j = tl.arange(0, BR)
        pmask = j < RD // 2
        f_off = t * FS + 2 * j
        fre = tl.load(F + f_off, mask=pmask, other=0.0)
        fim = tl.load(F + f_off + 1, mask=pmask, other=0.0)
        if INVERSE:
            fim = -fim
        for h in tl.static_range(H):
            base = b * SB + t * ST + h * SP
            off = base + 2 * j
            re = tl.load(X + off, mask=pmask, other=0.0).to(tl.float32)
            im = tl.load(X + off + 1, mask=pmask, other=0.0).to(tl.float32)
            tl.store(X + off, (re * fre - im * fim).to(X.dtype.element_ty), mask=pmask)
            tl.store(X + off + 1, (re * fim + im * fre).to(X.dtype.element_ty), mask=pmask)

    def _launch(x, freqs_cis, inverse):
        b, s, h, sp, sb, st = _layout(x)
        fr, fs = _freqs_2d(freqs_cis)
        rd = x.shape[-1]
        br = triton.next_power_of_2(rd // 2)
        _rope_tail_kernel[(b * s,)](x, fr, s, sb, st, sp, RD=rd, H=h, BR=br, FS=fs,
                                    INVERSE=bool(inverse), num_warps=4)


class _RopeTailFn(torch.autograd.Function):
    """fwd: 原地旋转 x 的最后 rd 维（返回 x 本体，调用点赋值 ⇒ Function 在计算图里）；
    bwd: d(x) = 逆旋转(dy 的 tail)。"""

    @staticmethod
    def forward(ctx, x, freqs_cis, inverse, rd):
        _launch(x[..., -rd:], freqs_cis, inverse)   # tail 视图：只被本 kernel 读写
        ctx.save_for_backward(freqs_cis)
        ctx.inverse = bool(inverse)
        ctx.rd = int(rd)
        return x

    @staticmethod
    def backward(ctx, dy):
        (freqs_cis,) = ctx.saved_tensors
        rd = ctx.rd
        dx = dy if dy.is_contiguous() else dy.contiguous()
        dx = dx.clone() if dx is dy else dx
        # dx 是完整张量（连续）：tail 视图的 stride(-2) = head_dim，kernel 会自动用父行跨度
        _launch(dx[..., -rd:], freqs_cis, not ctx.inverse)
        return dx, None, None, None


def rope_tail(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False):
    """原地旋转 x 的最后 rd 维（rd = 2*freqs.shape[-1]），返回 x 本体。"""
    rd = 2 * freqs_cis.shape[-1]
    if _REF_ROPE:
        apply_rotary_emb_ref(x[..., -rd:], freqs_cis, inverse)
        return x
    if not HAS_TRITON or not x.is_cuda:
        apply_rotary_emb_ref(x[..., -rd:], freqs_cis, inverse)
        return x
    return _RopeTailFn.apply(x, freqs_cis, bool(inverse), rd)


# --------------------------------------------------------------------------- model patch
def _attn_forward(self, x: torch.Tensor, start_pos: int):
    M = _MOD
    bsz, seqlen, _ = x.size()
    freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
    rd = self.rope_head_dim

    qr = self.q_norm(self.wq_a(x))
    q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
    q = rope_tail(q, freqs_cis)

    kv, topk_idxs = self._window_kv(x, freqs_cis, start_pos)
    if self.compress_ratio:
        compress_kv, compress_idxs = self._compress_kv(x, qr, start_pos, kv.size(1))
        kv = torch.cat([kv, compress_kv], dim=1)
        topk_idxs = torch.cat([topk_idxs, compress_idxs], dim=-1)

    o = M.sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
    o = rope_tail(o, freqs_cis, True)

    o = o.view(bsz, seqlen, self.n_local_groups, -1)
    M.act_quant(o, M.fp8_block_size, M.scale_fmt, M.scale_dtype, True)
    wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
    o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
    x = self.wo_b(o.flatten(2))
    return x


def _window_kv(self, x, freqs_cis, start_pos):
    M = _MOD
    bsz, seqlen, _ = x.size()
    win = self.window_size
    kv = self.kv_norm(self.wkv(x))
    kv = rope_tail(kv, freqs_cis)
    M.act_quant(kv, M.fp8_block_size, M.scale_fmt, M.scale_dtype, True)
    if start_pos == 0:  # prefill: attend over this chunk, seeding the ring buffer for decode
        if seqlen <= win:
            self.window_kv_cache[:bsz, :seqlen] = kv
        else:
            cutoff = seqlen % win
            self.window_kv_cache[:bsz, cutoff:win], self.window_kv_cache[:bsz, :cutoff] = kv[:, -win:].split(
                [win - cutoff, cutoff], dim=1
            )
        window_kv = kv
    else:  # decode: one token into the ring buffer, attend over the whole window
        self.window_kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
        window_kv = self.window_kv_cache[:bsz]
    return window_kv, M.get_window_topk_idxs(win, bsz, seqlen, start_pos)


def install(M):
    """复制并替换 Attention.forward / Attention._window_kv（幂等）。"""
    global _INSTALLED, _MOD
    _MOD = M
    if _INSTALLED is not None and _INSTALLED[0] is M.Attention:
        return _INSTALLED[1]
    prev = (M.Attention.__dict__.get("forward"), M.Attention.__dict__.get("_window_kv"))
    M.Attention.forward = _attn_forward
    M.Attention._window_kv = _window_kv
    _INSTALLED = (M.Attention, prev)
    return prev


def uninstall(M=None):
    global _INSTALLED
    if _INSTALLED is None:
        return
    cls, prev = _INSTALLED
    if prev[0] is not None:
        cls.forward = prev[0]
    if prev[1] is not None:
        cls._window_kv = prev[1]
    _INSTALLED = None


def available() -> bool:
    return HAS_TRITON and torch.cuda.is_available()

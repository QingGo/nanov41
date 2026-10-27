"""fused_indexer — HSI indexer 的 score 段融合（R47）。

官方 `Indexer.forward`（`_ref_patched_model.py:537`）在 2 个 index-source 层里做：

    index_score = torch.einsum("bshd,btd->bsht", q, index_k)   # [b,s,h,ncomp]，ncomp 最大 1024
    index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)

index_score 是 **[16,1024,4,ncomp] 的 bf16 大张量**（ncomp=1024 时 134 MB），relu/mul/sum 三趟 aten
再加 einsum 的输出写回，合计约 500 MB/次；只有 2 次/step，但仍是 ~0.6 ms 的长尾。
本模块用一个 Triton kernel 完成 einsum + relu + head 加权求和（保持 bf16 舍入语义、输出 [b,s,ncomp] bf16），
下游的 mask/topk/sort 完全不动。

用法：
    import nanov41.fused_indexer as fi
    fi.install(M)          # M = patched_env.import_patched()
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

__all__ = ["index_scores_fused", "index_scores_ref", "install", "uninstall", "available"]

_ENABLED = os.environ.get("NANOV41_IDX_FUSED", "0").lower() not in ("0", "false", "no", "off")
_INSTALLED = None
BT = int(os.environ.get("NANOV41_IDX_BT", "64"))
BC = int(os.environ.get("NANOV41_IDX_BC", "64"))


def index_scores_ref(q, index_k, weights):
    """官方两行：einsum -> relu*mul -> sum(head)。CPU fallback / 测试用。"""
    s = torch.einsum("bshd,btd->bsht", q, index_k)
    return (s.relu_() * weights.unsqueeze(-1)).sum(dim=2)


if HAS_TRITON:

    @triton.jit
    def _idx_scores_kernel(
        Q, K, W, OUT, S, N,
        QB, QS, KB, KN,
        H: tl.constexpr, D: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_b = tl.program_id(1)
        pid_c = tl.program_id(2)
        ts = pid_t * BT + tl.arange(0, BT)
        cs = pid_c * BC + tl.arange(0, BC)
        tm = ts < S
        cm = cs < N
        rd = tl.arange(0, D)
        acc = tl.zeros((BT, BC), dtype=tl.float32)
        for h in tl.static_range(H):
            q = tl.load(Q + pid_b * QB + ts[:, None] * QS + h * D + rd[None, :],
                        mask=tm[:, None], other=0.0)
            k = tl.load(K + pid_b * KB + cs[:, None] * KN + rd[None, :],
                        mask=cm[:, None], other=0.0)
            s = tl.dot(q, tl.trans(k))  # bf16 x bf16 -> fp32
            # 复刻官方的 bf16 舍入次序：dot->bf16, relu, *w->bf16, sum(head) 在 fp32 累加
            sb = tl.maximum(s, 0.0).to(tl.bfloat16)
            wv = tl.load(W + pid_b * (S * H) + ts * H + h, mask=tm, other=0.0).to(tl.float32)
            acc += (sb.to(tl.float32) * wv[:, None]).to(tl.bfloat16).to(tl.float32)
        tl.store(OUT + pid_b * (S * N) + ts[:, None] * N + cs[None, :],
                 acc.to(OUT.dtype.element_ty), mask=tm[:, None] & cm[None, :])

    def index_scores_fused(q, index_k, weights):
        """q [b,s,h,d] bf16, index_k [b,n,d] bf16, weights [b,s,h] bf16 ->
        [b,s,n] bf16（= relu(q·k) 按 head 用 weights 加权求和）。"""
        b, s, h, d = q.shape
        n = index_k.shape[1]
        out = torch.empty(b, s, n, dtype=torch.bfloat16, device=q.device)
        _idx_scores_kernel[(triton.cdiv(s, BT), b, triton.cdiv(n, BC))](
            q, index_k, weights, out, s, n,
            q.stride(0), q.stride(1), index_k.stride(0), index_k.stride(1),
            H=h, D=d, BT=BT, BC=BC, num_warps=4, num_stages=2,
        )
        return out


def _indexer_forward(self, x: torch.Tensor, qr: torch.Tensor, latent: torch.Tensor,
                     start_pos: int, offset: int):
    """官方 `Indexer.forward` 的复刻，仅把 score 段替换为融合 kernel。"""
    M = _MOD
    assert self.freqs_cis is not None
    bsz, seqlen, _ = x.size()
    ratio, rd, end_pos = self.compress_ratio, self.rope_head_dim, start_pos + seqlen

    if self.owns_k and latent is not None:
        freqs = (
            self.freqs_cis[: seqlen - seqlen % ratio : ratio]
            if start_pos == 0
            else self.freqs_cis[start_pos + 1 - ratio].unsqueeze(0)
        )
        k = self.k_norm(self.wk(latent))
        M.apply_rotary_emb(k[..., -rd:], freqs)
        M.fp4_act_quant(k, M.fp4_block_size, True)
        self.k_cache[:bsz, start_pos // ratio : start_pos // ratio + k.size(1)] = k
    if self.owns_k:
        M.shared_attn.index_k = k if (start_pos == 0 and latent is not None) else self.k_cache

    q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.index_head_dim))
    M.apply_rotary_emb(q[..., -rd:], self.freqs_cis[start_pos:end_pos])
    M.fp4_act_quant(q, M.fp4_block_size, True)

    index_k = M.shared_attn.index_k[:bsz, : end_pos // ratio]
    weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
    if _ENABLED and q.is_cuda:
        index_score = index_scores_fused(q, index_k, weights)
    else:
        index_score = torch.einsum("bshd,btd->bsht", q, index_k)
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
    if M.world_size > 1:
        M.dist.all_reduce(index_score)

    if start_pos == 0:
        compress_lens = (torch.arange(1, seqlen + 1, device=x.device) // ratio).unsqueeze(-1)
        index_score.masked_fill_(torch.arange(seqlen // ratio, device=x.device) >= compress_lens,
                                 -torch.inf)
    else:
        compress_lens = end_pos // ratio

    if self.is_candidate_source:
        M.shared_attn.candidates = M.select_candidate_blocks(
            index_score, compress_lens, self.candidate_topk_blocks, self.candidate_block_size)
    elif self.uses_candidates:
        index_score = index_score.masked_fill(~M.shared_attn.candidates, -torch.inf)

    topk = min(self.index_topk, end_pos // ratio)
    idxs = index_score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
    return torch.where(idxs < compress_lens, idxs + offset, -1).int()


_MOD = None


def install(M, enabled: bool | None = None):
    """把 `M.Indexer.forward` 换成融合版（幂等）。enabled=False 时只装上但走官方两行。"""
    global _INSTALLED, _MOD, _ENABLED
    _MOD = M
    if enabled is not None:
        _ENABLED = bool(enabled)
    if _INSTALLED is not None and _INSTALLED[0] is M.Indexer:
        return _INSTALLED[1]
    prev = M.Indexer.__dict__.get("forward")
    M.Indexer.forward = _indexer_forward
    _INSTALLED = (M.Indexer, prev)
    return prev


def uninstall(M=None):
    global _INSTALLED
    if _INSTALLED is None:
        return
    cls, prev = _INSTALLED
    if prev is not None:
        cls.forward = prev
    _INSTALLED = None


def available() -> bool:
    return HAS_TRITON and torch.cuda.is_available()

# SPDX-License-Identifier: MIT
"""Decode-side attention fast path for the native inference engine.

Why this exists: on a real P2b checkpoint one decode step is ~1730 CUDA launches but only
~3.9 ms of GPU busy time (``runs/p2b/real_decode_profile.json``) -- the reference attention is a
long chain of tiny torch ops (window ring gather, sparse softmax with an attention sink, RMSNorm,
RoPE, output RoPE) whose per-op dispatch dominates.  This module keeps the *same* math but
collapses the elementwise/reduction tail into three Triton kernels:

- ``rmsnorm_kernel``: RMSNorm over the last dim in float32, cast back to the input dtype;
- ``rope_kernel``: the complex rotation used by ``apply_rotary_emb`` (forward or inverse),
  operating on a strided last-dim slice of the parent tensor so it stays in place;
- ``sparse_attn_decode_kernel``: window ring buffer + selected compressed positions -> one
  softmax with the attention sink in the denominator -> weighted sum.

Everything else (the ``Linear`` projections, the compressor and the indexer) is reused unchanged,
so the projection GEMMs are exactly the reference cublas path.  ``enable_attn_decode`` patches
``Attention.forward`` of an inference copy; prefill / ``start_pos == 0`` / multi-token calls fall
back to the reference forward.

Torch fallbacks mirror the kernels so ``tests/test_attn_decode.py`` can run without CUDA.  The
kernel path differs from the reference only in fp32 reduction order inside the softmax / RMSNorm /
output sum, so results agree to bf16 round-off (RoPE itself is bit-exact: same fp32 complex
multiply).
"""
from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception:  # noqa: BLE001
    HAS_TRITON = False
    triton = None
    tl = None


def _use_triton() -> bool:
    flag = os.environ.get("NANOV41_ATTN_DECODE", "triton").strip().lower()
    if flag in ("0", "torch", "off", "false", "no"):
        return False
    return HAS_TRITON and torch.cuda.is_available()


# --------------------------------------------------------------------------------------- torch
def rmsnorm_torch(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Exactly ``RMSNorm.forward``: float32 statistics, dtype cast at the end."""
    dtype = x.dtype
    xf = x.float()
    var = xf.square().mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    return (weight * xf).to(dtype)


def rope_torch(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """In-place ``apply_rotary_emb`` on a [..., rope_dim] slice (same ops as the reference)."""
    view = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    freqs = freqs_cis.conj() if inverse else freqs_cis
    if view.ndim == 3:  # [b, s, f]
        freqs = freqs.view(1, view.size(1), view.size(-1))
    else:  # [b, s, h, f]
        freqs = freqs.view(1, view.size(1), 1, view.size(-1))
    x.copy_(torch.view_as_real(view * freqs).flatten(-2))
    return x


def sparse_attn_decode_torch(q: torch.Tensor, window_kv: torch.Tensor,
                             compress_kv: torch.Tensor | None, compress_idxs: torch.Tensor | None,
                             attn_sink: torch.Tensor, softmax_scale: float, start_pos: int,
                             window_size: int) -> torch.Tensor:
    """Torch mirror of ``sparse_attn_decode_kernel``.

    q: [B, 1, H, D]; window_kv: [B, win, D]; compress_kv: [B, nc, D].  ``compress_idxs`` carries
    the reference ``+window_size`` offset (values in [win, win+nc) or -1); window ring slots are
    derived arithmetically from ``start_pos`` exactly like ``get_window_topk_idxs``.
    """
    b, m, h, d = q.shape
    assert m == 1
    device = q.device
    win = window_size
    oldest = start_pos % win + 1
    ring = torch.cat([torch.arange(oldest, win, device=device),
                      torch.arange(oldest, device=device)])            # [win] slot order
    win_valid = ring <= start_pos                                      # [win]
    kvg = window_kv.reshape(b, win, d).gather(
        1, ring.view(1, win, 1).expand(b, win, d))                     # [b, win, d]
    valid = win_valid.view(1, win).expand(b, win)
    if compress_kv is not None and compress_idxs is not None and compress_idxs.numel():
        nc = compress_idxs.size(-1)
        cvalid = (compress_idxs >= 0).reshape(b, nc)                   # [b, n]
        clocal = (compress_idxs - win).clamp_min(0).reshape(b, nc)     # [b, n]
        ckv = compress_kv.reshape(b, -1, d).gather(
            1, clocal.unsqueeze(-1).expand(b, nc, d))                  # [b, n, d]
        ckv = ckv * cvalid.unsqueeze(-1).to(ckv.dtype)                 # zero invalid rows
        kvg = torch.cat([kvg, ckv], dim=1)
        valid = torch.cat([valid, cvalid], dim=1)
    scores = torch.einsum("bhd,btd->bht", q[:, 0].float(), kvg.float()) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
    mx = scores.amax(dim=-1)
    finite = torch.isfinite(mx)
    mx_safe = torch.where(finite, mx, torch.zeros_like(mx))
    p = torch.where(valid.unsqueeze(1), (scores - mx_safe.unsqueeze(-1)).exp(),
                    torch.zeros_like(scores))
    denom = p.sum(dim=-1) + torch.exp(attn_sink.view(1, h) - mx_safe)
    out = torch.einsum("bht,btd->bhd", p, kvg.float()) / denom.unsqueeze(-1)
    out = torch.where(torch.isfinite(out), out, torch.zeros_like(out)).to(q.dtype)
    return out.unsqueeze(1)


# ---------------------------------------------------------------------------------- triton cm
if HAS_TRITON:
    @triton.jit
    def rmsnorm_kernel(X, W, OUT, N: tl.constexpr, eps, BLOCK: tl.constexpr):
        """One program: one row of N elements -> RMSNorm (float32 math, dtype cast on store).

        M3: the row offset comes from ``program_id(0)`` so the same kernel serves every sequence
        row of a batched decode step (M1/M2 only ever launched grid=(1,)).
        """
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(OUT + row * N + cols, (x * rstd * w).to(OUT.dtype.element_ty), mask=mask)

    @triton.jit
    def rope_kernel(X, COS, SIN, OFFSET, ROW_STRIDE, N: tl.constexpr, INVERSE: tl.constexpr):
        """Rotate adjacent float pairs of a [rows, N] slice at OFFSET inside each parent row."""
        row = tl.program_id(0)
        HALF: tl.constexpr = N // 2
        j = tl.arange(0, HALF)
        cos = tl.load(COS + j)
        sin = tl.load(SIN + j)
        base = X + OFFSET + row * ROW_STRIDE
        xr = tl.load(base + 2 * j).to(tl.float32)
        xi = tl.load(base + 2 * j + 1).to(tl.float32)
        if INVERSE:
            sin = -sin
        tl.store(base + 2 * j, (xr * cos - xi * sin).to(X.dtype.element_ty))
        tl.store(base + 2 * j + 1, (xr * sin + xi * cos).to(X.dtype.element_ty))

    @triton.jit
    def _sparse_attn_decode_kernel_v1(
        Q, WIN_KV, CMP_KV, CMP_IDX, SINK, OUT,
        P, D: tl.constexpr, WIN, NC, SCALE: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        """``P`` is a pointer to a scalar position tensor so a captured CUDA graph reads the
        updated position on every replay (a Python-int argument would be frozen at capture)."""
        """One program per head: combined window + compressed softmax with sink, weighted sum.

        q: [H, D]; window_kv: [WIN, D]; compress_kv: [NC, D]; cmp_idx: [NC] int32 with the
        reference ``+WIN`` offset or -1.  Window slot order is ``(p % WIN + 1 + t) % WIN`` and a
        slot is valid iff ``slot <= p`` (the reference ``get_window_topk_idxs`` rule).  Invalid
        slots are folded in with masked loads and ``-inf``/0 weights, so no data-dependent `if`
        is needed (the Triton loop stays uniform).
        """
        h = tl.program_id(0)
        p = tl.load(P).to(tl.int32)
        cols = tl.arange(0, BLOCK_D)
        mask_d = cols < D
        qv = tl.load(Q + h * D + cols, mask=mask_d, other=0.0).to(tl.float32)
        oldest = p % WIN + 1

        # pass 1: max score over the valid slots (window slots first, then compressed)
        mx = -float("inf")
        for t in range(0, WIN):
            slot = (oldest + t) % WIN
            valid = slot <= p
            kv = tl.load(WIN_KV + slot * D + cols, mask=mask_d, other=0.0).to(tl.float32)
            score = tl.sum(qv * kv) * SCALE
            mx = tl.maximum(mx, tl.where(valid, score, -float("inf")))
        for t in range(0, NC):
            idx = tl.load(CMP_IDX + t)
            valid = idx >= 0
            local = tl.maximum(idx - WIN, 0)
            kv = tl.load(CMP_KV + local * D + cols, mask=mask_d, other=0.0).to(tl.float32)
            score = tl.sum(qv * kv) * SCALE
            mx = tl.maximum(mx, tl.where(valid, score, -float("inf")))
        mx_safe = tl.where(mx == -float("inf"), 0.0, mx)

        # pass 2: denominator (attention sink included) and weighted value sum
        den = tl.exp(tl.load(SINK + h) - mx_safe)
        acc = tl.zeros((BLOCK_D,), tl.float32)
        for t in range(0, WIN):
            slot = (oldest + t) % WIN
            valid = slot <= p
            kv = tl.load(WIN_KV + slot * D + cols, mask=mask_d, other=0.0).to(tl.float32)
            e = tl.where(valid, tl.exp(tl.sum(qv * kv) * SCALE - mx_safe), 0.0)
            den += e
            acc += e * kv
        for t in range(0, NC):
            idx = tl.load(CMP_IDX + t)
            valid = idx >= 0
            local = tl.maximum(idx - WIN, 0)
            kv = tl.load(CMP_KV + local * D + cols, mask=mask_d, other=0.0).to(tl.float32)
            e = tl.where(valid, tl.exp(tl.sum(qv * kv) * SCALE - mx_safe), 0.0)
            den += e
            acc += e * kv
        tl.store(OUT + h * D + cols, (acc / den).to(OUT.dtype.element_ty), mask=mask_d)



    @triton.jit
    def sparse_attn_decode_kernel(
        Q, WIN_KV, CMP_KV, CMP_IDX, SINK, OUT,
        P, D: tl.constexpr, WIN: tl.constexpr, NC: tl.constexpr,
        SCALE: tl.constexpr, BLOCK_D: tl.constexpr, BT: tl.constexpr,
    ):
        """Vectorised decode sparse attention (M2): one program per head, chunked 2-D tiles.

        The v1 kernel walked slots one at a time with a scalar dot, so at T=1 only 8 tiny programs
        ran and each did ~96 serial iterations.  v2 loads ``BT`` KV rows at once and reduces over D
        with ``tl.sum``, keeping only the online-softmax state (max / denominator / weighted sum)
        alive across the (static) window and compressed chunk loops.  The head's attention sink only
        enters the denominator, exactly as the reference `sparse_attn`.
        """
        h = tl.program_id(0)
        p = tl.load(P).to(tl.int32)
        cols = tl.arange(0, BLOCK_D)
        mask_d = cols < D
        qv = tl.load(Q + h * D + cols, mask=mask_d, other=0.0).to(tl.float32)
        oldest = p % WIN + 1
        lane = tl.arange(0, BT)

        m = -float("inf")
        den = 0.0
        acc = tl.zeros((BLOCK_D,), tl.float32)

        # window ring: slots are (oldest + t) % WIN, valid iff the slot index <= p
        for start in tl.static_range(0, WIN, BT):
            t = start + lane
            slot = (oldest + t) % WIN
            valid = slot <= p
            kv = tl.load(WIN_KV + slot[:, None] * D + cols[None, :],
                         mask=valid[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
            s = tl.sum(qv[None, :] * kv, axis=1) * SCALE
            s = tl.where(valid, s, -float("inf"))
            cmax = tl.max(s, axis=0)
            new_m = tl.maximum(m, cmax)
            # -inf - -inf would be NaN if a whole chunk is invalid and m is still -inf
            alpha = tl.where(m == -float("inf"), 0.0, tl.exp(m - new_m))
            e = tl.where(valid, tl.exp(s - new_m), 0.0)
            den = den * alpha + tl.sum(e, axis=0)
            acc = acc * alpha + tl.sum(e[:, None] * kv, axis=0)
            m = new_m

        # compressed slots: index buffer carries the reference +WIN offset, -1 = invalid
        for start in tl.static_range(0, NC, BT):
            t = start + lane
            idx = tl.load(CMP_IDX + t)
            valid = idx >= 0
            local = tl.maximum(idx - WIN, 0)
            kv = tl.load(CMP_KV + local[:, None] * D + cols[None, :],
                         mask=valid[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
            s = tl.sum(qv[None, :] * kv, axis=1) * SCALE
            s = tl.where(valid, s, -float("inf"))
            cmax = tl.max(s, axis=0)
            new_m = tl.maximum(m, cmax)
            # -inf - -inf would be NaN if a whole chunk is invalid and m is still -inf
            alpha = tl.where(m == -float("inf"), 0.0, tl.exp(m - new_m))
            e = tl.where(valid, tl.exp(s - new_m), 0.0)
            den = den * alpha + tl.sum(e, axis=0)
            acc = acc * alpha + tl.sum(e[:, None] * kv, axis=0)
            m = new_m

        den += tl.exp(tl.load(SINK + h) - m)
        tl.store(OUT + h * D + cols, (acc / den).to(OUT.dtype.element_ty), mask=mask_d)

    # ------------------------------------------------------------------ batch decode (M3)
    @triton.jit
    def rope_batch_kernel(X, FREQ, OFFSET, ROW_STRIDE,
                          ROWS_PER_SEQ: tl.constexpr, N: tl.constexpr,
                          INVERSE: tl.constexpr):
        """RoPE with a per-sequence frequency row (decode batch).

        ``FREQ`` is the ``view_as_real`` (float32) form of the complex freqs rows: [B, N//2, 2].
        ``row = b * ROWS_PER_SEQ + r`` selects the same memory layout as ``rope_kernel``, so the
        same call shape works for q [B,1,H,D] (ROWS_PER_SEQ=H) and for single-row tensors [B,1,D].
        """
        row = tl.program_id(0)
        HALF: tl.constexpr = N // 2
        b = row // ROWS_PER_SEQ
        j = tl.arange(0, HALF)
        cos = tl.load(FREQ + b * 2 * HALF + 2 * j)
        sin = tl.load(FREQ + b * 2 * HALF + 2 * j + 1)
        base = X + OFFSET + row * ROW_STRIDE
        xr = tl.load(base + 2 * j).to(tl.float32)
        xi = tl.load(base + 2 * j + 1).to(tl.float32)
        if INVERSE:
            sin = -sin
        tl.store(base + 2 * j, (xr * cos - xi * sin).to(X.dtype.element_ty))
        tl.store(base + 2 * j + 1, (xr * sin + xi * cos).to(X.dtype.element_ty))

    @triton.jit
    def sparse_attn_decode_batch_kernel(
        Q, WIN_KV, CMP_KV, CMP_IDX, SINK, OUT, P,
        D: tl.constexpr, H: tl.constexpr, WIN: tl.constexpr, NC: tl.constexpr,
        SCALE: tl.constexpr, BLOCK_D: tl.constexpr, BT: tl.constexpr,
        WIN_BST, CMP_BST,
    ):
        """Batched ``sparse_attn_decode_kernel``: one program per (sequence, head)."""
        b = tl.program_id(0)
        h = tl.program_id(1)
        p = tl.load(P + b).to(tl.int32)
        cols = tl.arange(0, BLOCK_D)
        mask_d = cols < D
        qv = tl.load(Q + (b * H + h) * D + cols, mask=mask_d, other=0.0).to(tl.float32)
        win_base = WIN_KV + b * WIN_BST
        cmp_base = CMP_KV + b * CMP_BST
        idx_base = CMP_IDX + b * NC
        oldest = p % WIN + 1
        lane = tl.arange(0, BT)

        m = -float("inf")
        den = 0.0
        acc = tl.zeros((BLOCK_D,), tl.float32)

        for start in tl.static_range(0, WIN, BT):
            t = start + lane
            slot = (oldest + t) % WIN
            valid = slot <= p
            kv = tl.load(win_base + slot[:, None] * D + cols[None, :],
                         mask=valid[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
            s = tl.sum(qv[None, :] * kv, axis=1) * SCALE
            s = tl.where(valid, s, -float("inf"))
            new_m = tl.maximum(m, tl.max(s, axis=0))
            alpha = tl.where(m == -float("inf"), 0.0, tl.exp(m - new_m))
            e = tl.where(valid, tl.exp(s - new_m), 0.0)
            den = den * alpha + tl.sum(e, axis=0)
            acc = acc * alpha + tl.sum(e[:, None] * kv, axis=0)
            m = new_m

        if NC > 0:
            for start in tl.static_range(0, NC, BT):
                t = start + lane
                idx = tl.load(idx_base + t)
                valid = idx >= 0
                local = tl.maximum(idx - WIN, 0)
                kv = tl.load(cmp_base + local[:, None] * D + cols[None, :],
                             mask=valid[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
                s = tl.sum(qv[None, :] * kv, axis=1) * SCALE
                s = tl.where(valid, s, -float("inf"))
                new_m = tl.maximum(m, tl.max(s, axis=0))
                alpha = tl.where(m == -float("inf"), 0.0, tl.exp(m - new_m))
                e = tl.where(valid, tl.exp(s - new_m), 0.0)
                den = den * alpha + tl.sum(e, axis=0)
                acc = acc * alpha + tl.sum(e[:, None] * kv, axis=0)
                m = new_m

        den += tl.exp(tl.load(SINK + h) - m)
        tl.store(OUT + (b * H + h) * D + cols, (acc / den).to(OUT.dtype.element_ty),
                 mask=mask_d)


# --------------------------------------------------------------------------------- fast path
def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if not _use_triton() or x.numel() == 0:
        return rmsnorm_torch(x, weight, eps)
    n = x.shape[-1]
    out = torch.empty_like(x)
    rows = x.numel() // n
    rmsnorm_kernel[(rows,)](x.reshape(-1, n), weight, out.reshape(-1, n), n, float(eps),
                             triton.next_power_of_2(n))
    return out


def _rope(parent: torch.Tensor, offset: int, rows: int, n: int, freqs_cis: torch.Tensor,
          inverse: bool = False) -> None:
    """In-place RoPE on ``parent[..., offset:offset+n]`` (parent's last dim must be contiguous)."""
    if not _use_triton():
        rope_torch(parent[..., offset:offset + n], freqs_cis, inverse)
        return
    # the kernel applies the inverse by negating sin, so do NOT pre-conjugate here
    flat = freqs_cis.reshape(-1, n // 2)[:1]  # decode: one position
    cos = flat.real.reshape(-1).contiguous()
    sin = flat.imag.reshape(-1).contiguous()
    rope_kernel[(rows,)](parent, cos, sin, offset, parent.stride(-2), n, inverse)


def rope_row(parent: torch.Tensor, offset: int, rows: int, n: int, freqs_row: torch.Tensor,
             inverse: bool = False) -> None:
    """In-place RoPE on ``parent[..., offset:offset+n]`` using a precomputed freqs row.

    ``freqs_row`` is a complex [1, n//2] tensor (e.g. ``freqs_cis[pos]``); in a captured graph the
    row is produced by indexing the position buffer, so the same kernel replays for every position.
    """
    if not _use_triton():
        rope_torch(parent[..., offset:offset + n], freqs_row, inverse)
        return
    flat = freqs_row.reshape(-1, n // 2)[:1]
    cos = flat.real.reshape(-1).contiguous()
    sin = flat.imag.reshape(-1).contiguous()
    rope_kernel[(rows,)](parent, cos, sin, offset, parent.stride(-2), n, inverse)


def sparse_attn_decode(q: torch.Tensor, window_kv: torch.Tensor,
                       compress_kv: torch.Tensor | None, compress_idxs: torch.Tensor | None,
                       attn_sink: torch.Tensor, softmax_scale: float, start_pos: int,
                       window_size: int) -> torch.Tensor:
    """Fused sparse attention for one decode token; see ``sparse_attn_decode_torch``."""
    if not _use_triton():
        return sparse_attn_decode_torch(q, window_kv, compress_kv, compress_idxs, attn_sink,
                                        softmax_scale, start_pos, window_size)
    b, m, h, d = q.shape
    assert b == 1 and m == 1, "decode fast path is single-token, single-batch"
    nc = 0 if compress_idxs is None else compress_idxs.numel()
    cmp_idx = (compress_idxs.reshape(-1).to(torch.int32).contiguous() if nc
               else torch.zeros(1, dtype=torch.int32, device=q.device))
    cmp_kv = (compress_kv.reshape(-1, d).contiguous() if nc
              else torch.zeros(1, d, dtype=q.dtype, device=q.device))
    if torch.is_tensor(start_pos):
        pos = start_pos.to(torch.int32).reshape(1)
    else:
        # torch.full launches a fill kernel instead of a host->device copy, so this path is also
        # usable while a CUDA graph is being captured (the scalar is then frozen, as expected)
        pos = torch.full((1,), int(start_pos), dtype=torch.int32, device=q.device)
    out = torch.empty((1, 1, h, d), dtype=q.dtype, device=q.device)
    bt = int(os.environ.get("NANOV41_ATTN_BT", "16"))
    sparse_attn_decode_kernel[(h,)](
        q.reshape(h, d), window_kv.reshape(-1, d), cmp_kv, cmp_idx, attn_sink,
        out.reshape(h, d), pos, d, window_size, nc, float(softmax_scale),
        triton.next_power_of_2(d), bt,
    )
    return out


# ------------------------------------------------------------------------------ batch decode
def rope_rows_batch(parent: torch.Tensor, offset: int, rows_per_seq: int, n: int,
                    freqs_rows: torch.Tensor, inverse: bool = False) -> None:
    """In-place RoPE on ``parent[..., offset:offset+n]`` with one freqs row per sequence.

    ``parent`` is [B, 1, ROWS_PER_SEQ, D] (q/o) or [B, 1, D] (kv/latent); ``freqs_rows`` is a
    complex [B, n//2] tensor (e.g. ``freqs_cis[pos]`` indexed by a pos *tensor*).
    """
    if not _use_triton():
        view = torch.view_as_complex(parent[..., offset:offset + n].float().unflatten(-1, (-1, 2)))
        freqs = freqs_rows.conj() if inverse else freqs_rows
        if view.dim() == 4:  # [B, 1, rows_per_seq, half]
            freqs = freqs.view(freqs.shape[0], 1, 1, -1)
        else:
            freqs = freqs.view(freqs.shape[0], 1, -1)
        parent[..., offset:offset + n].copy_(torch.view_as_real(view * freqs).flatten(-2))
        return
    b = int(parent.shape[0])
    rows = b * int(rows_per_seq)
    freq_real = torch.view_as_real(freqs_rows.contiguous())      # [B, n//2, 2] float32
    rope_batch_kernel[(rows,)](parent, freq_real, offset, parent.stride(-2),
                               ROWS_PER_SEQ=int(rows_per_seq), N=int(n), INVERSE=inverse)


def sparse_attn_decode_batch(q: torch.Tensor, window_kv: torch.Tensor,
                             compress_kv: torch.Tensor | None,
                             compress_idxs: torch.Tensor | None,
                             attn_sink: torch.Tensor, softmax_scale: float,
                             pos: torch.Tensor, window_size: int) -> torch.Tensor:
    """Batched fused sparse attention for one decode token per sequence.

    q: [B, 1, H, D]; window_kv: [B, win, D]; compress_kv: [B, nc_cache, D] (the group cache);
    compress_idxs: [B, N] int32 with the reference +window offset (or None); pos: int64 [B].
    """
    b, _, h, d = q.shape
    assert b >= 1 and q.shape[1] == 1, "batch decode path is one token per sequence"
    bt = int(os.environ.get("NANOV41_ATTN_BT", "16"))
    out = torch.empty((b, 1, h, d), dtype=q.dtype, device=q.device)
    if compress_idxs is not None and compress_idxs.numel():
        nc = int(compress_idxs.shape[-1])
        cmp_idx = compress_idxs.reshape(b, nc).to(torch.int32).contiguous()
        cmp_kv = compress_kv
        cmp_bst = int(compress_kv.stride(0))
    else:
        nc = 0
        cmp_idx = torch.zeros(1, dtype=torch.int32, device=q.device)
        cmp_kv = torch.zeros(1, 1, d, dtype=q.dtype, device=q.device)
        cmp_bst = 0
    sparse_attn_decode_batch_kernel[(b, h)](
        q, window_kv, cmp_kv, cmp_idx, attn_sink, out, pos,
        d, h, window_size, nc, float(softmax_scale),
        triton.next_power_of_2(d), bt,
        int(window_kv.stride(0)), cmp_bst,
    )
    return out


def blockdiag_wo_a(o: torch.Tensor, weight: torch.Tensor, groups: int, rank: int) -> torch.Tensor:
    """Group-block-diagonal ``wo_a`` projection in fp32 (batch-size invariant, one bf16 rounding).

    The reference uses ``einsum("bsgd,grd->bsgr", o, wo_a)``; torch's einsum dispatches a different
    bf16 GEMM for batch 1 vs batch >1, which makes batched decode drift from the single-sequence
    runtime, so the decode paths share this explicit fp32 contraction instead.
    """
    b = int(o.shape[0])
    ov = o.view(b, 1, groups, -1).float()
    wa = weight.view(groups, rank, -1).float()
    return (ov.unsqueeze(-2) * wa.unsqueeze(0).unsqueeze(0)).sum(-1).to(o.dtype)


# ----------------------------------------------------------------------------------- patching
def attn_decode_forward(self, x: torch.Tensor, start_pos: int) -> torch.Tensor:
    """``Attention.forward`` for a single decode token; falls back above one token."""
    bsz, seqlen, _ = x.size()
    if seqlen != 1 or start_pos == 0:
        return self._attn_decode_orig_forward(x, start_pos)
    rd = self.rope_head_dim
    win = self.window_size
    freqs_cis = self.freqs_cis[start_pos:start_pos + 1]

    # q: low-rank projection, q_norm, then RoPE on the tail of each head
    qr = _rmsnorm(self.wq_a(x), self.q_norm.weight, self.q_norm.eps)
    q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
    _rope(q, self.head_dim - rd, self.n_local_heads, rd, freqs_cis)

    # kv: projection, kv_norm, RoPE, then one slot into the window ring buffer
    kv = _rmsnorm(self.wkv(x), self.kv_norm.weight, self.kv_norm.eps)
    _rope(kv, self.head_dim - rd, bsz, rd, freqs_cis)
    self.window_kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
    window_kv = self.window_kv_cache[:bsz]

    compress_kv = compress_idxs = None
    if self.compress_ratio:
        compress_kv, compress_idxs = self._compress_kv(x, qr, start_pos, window_kv.size(1))

    o = sparse_attn_decode(q, window_kv, compress_kv, compress_idxs, self.attn_sink,
                           self.softmax_scale, start_pos, win)
    _rope(o, self.head_dim - rd, self.n_local_heads, rd, freqs_cis, inverse=True)

    o = o.view(bsz, 1, self.n_local_groups, -1)
    o = blockdiag_wo_a(o, self.wo_a.weight, self.n_local_groups, self.o_lora_rank)
    return self.wo_b(o.flatten(2))


def enable_attn_decode(model) -> int:
    """Patch every ``Attention.forward`` with the single-token decode fast path."""
    inner = getattr(model, "inner", model)
    n = 0
    for layer in getattr(inner, "layers", []):
        attn = getattr(layer, "attn", None)
        if attn is None or hasattr(attn, "_attn_decode_orig_forward"):
            continue
        attn._attn_decode_orig_forward = attn.forward
        attn.forward = attn_decode_forward.__get__(attn, type(attn))
        n += 1
    return n


def disable_attn_decode(model) -> int:
    inner = getattr(model, "inner", model)
    n = 0
    for layer in getattr(inner, "layers", []):
        attn = getattr(layer, "attn", None)
        if attn is None or not hasattr(attn, "_attn_decode_orig_forward"):
            continue
        attn.forward = attn._attn_decode_orig_forward
        del attn._attn_decode_orig_forward
        n += 1
    return n


__all__ = ["enable_attn_decode", "disable_attn_decode", "attn_decode_forward",
           "sparse_attn_decode", "sparse_attn_decode_torch", "rmsnorm_torch", "rope_torch",
           "rope_row", "HAS_TRITON"]

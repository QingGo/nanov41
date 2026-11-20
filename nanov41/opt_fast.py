# SPDX-License-Identifier: MIT
"""Fast *out-of-graph* optimizer / Engram-harvest paths for the nano-v41 training loop (R77).

The R72 loop profile (b2/seq8192/accum32, 2^19 tokens/step) attributes ~254 ms/step of the
3337 ms step to work that happens **outside** the captured graph:

    engram harvest 42.7 ms | Muon 113 ms | SinkhornEngram 33.2 ms | Sinkhorn 16.8 ms
    AdamW 6.3 ms | clip 6.2 ms | h2d 2.9 ms | host gap 56 ms

This module replaces the four biggest of those (harvest, SinkhornEngram, Sinkhorn, clip) — plus an
optional fused AdamW — with implementations that are **mathematically identical** to the current
ones but touch far less memory:

1. **harvest** (`engram_accumulate`): the current path sorts every per-forward token-id record
   (``segment_sum_rows``: stable argsort -> unique_consecutive -> segment_reduce), merges the
   per-forward results and then ``index_add_``s them into the dense fp32 grad buffer.  At d12 the
   two Engram tables receive 24 hash ids per token per table, i.e. ~393 k ids per micro-batch per
   table -> ~25 MB of leaf grad and a 393 k-element sort every forward.  The fast path scatters the
   leaf grads straight into the fp32 grad buffer (a Triton atomic row-add, or ``index_add_`` as a
   torch fallback): identical accumulation over duplicate ids, no sort / unique / segment pass.
   NOTE: the fused atomic path sums duplicate ids in hardware order, so the harvested fp32 buffer
   differs from the deterministic segment-sum by fp32 re-association only (~1e-7 rel).  Set
   ``NANOV41_OPT_FAST_HARVEST=det`` to keep the bitwise-deterministic sorted reduction.
2. **SinkhornEngram** (`engram_step`): the current path materialises the full [rows, 32] fp32
   matrix and runs 11 alternating row/column L2 normalisations on it (each one a full read plus a
   full read-modify-write of the matrix), then writes the table back in chunks.  All those
   operations are **diagonal scalings**, so the same Algorithm-1 update can be computed from
   row/column scaling vectors ``a``/``b`` (``U = diag(a) G~ diag(b)``): each Sinkhorn iteration is
   then a single read-only pass over ``G~``, and the final (k=K) row pass is fused into the
   write-back -> 7 read passes instead of 11 x (read+write).  Numerics: fp32 everywhere, with the
   same ``eps`` / ``tau`` masking semantics and the same recurrence.
   **R78 fix (the R77 failure was numerical, not algebraic):** every product now carries *both*
   scale vectors -- ``(a_i * G~_ij) * b_j`` -- so it lives at the magnitude of the materialised
   matrix; and the pair is renormalised by an *exact power of two* after every row pass
   (``_canonicalise_pair_``).  R77 formed sums from ``(G~_ij * b_j)^2`` with ``b_j ~ 1e-13`` at
   k=11, i.e. summands ~1e-41 (fp32 subnormal), so ``a`` blew up to ``1/eps`` and the direction
   diverged from the materialised Algorithm 1 (rms_rel 0.148 on real sparse Engram state).
3. **Sinkhorn** (`sinkhorn_step`): same treatment for the dense Sinkhorn group (token embedding +
   prediction head, [129280, 512] bf16).  The update writeback also reproduces the current
   ``p.add_(delta.to(p.dtype), alpha=-coef)`` bf16 rounding.
4. **clip** (`clip_grad_norm_`): keeps torch's total-norm computation (bitwise-identical norm ->
   identical clip coefficient) but avoids the 752 M-parameter in-place grad scaling pass for the
   groups whose optimizers can fold the coefficient into a read they already do: Muon's grad
   conversion (``to_fp32_scaled``, included here) and the dense Sinkhorn kernel.  That removes
   ~2.4 GB of grad read+write per step.  The AdamW (small) grads are still scaled in place with the
   same ``_foreach_mul_`` op torch uses, so the *effective* post-clip grads are bitwise identical.
5. **AdamW** (``adamw_step``): the v41 AdamW holds a few hundred tiny 1-D tensors; the non-fused
   foreach path launches ~6 kernels per tensor (~1.2 k launches -> 6.3 ms).  ``torch._fused_adam_``
   is one kernel per tensor; ``build_optimizers`` can construct the fused optimizer behind the same
   env flag (state dict format is unchanged -> resume works across the flag).

Everything here is **default-off** and only reachable through the env gates in
``nanov41/muon.py``::

    NANOV41_OPT_FAST=1            # master switch for the wired-in fast paths
    NANOV41_OPT_FAST_HARVEST=...  # 1 (default) | 0 | det (deterministic sorted harvest)
    NANOV41_OPT_FAST_SINKHORN=...      # dense SinkhornMomentum
    NANOV41_OPT_FAST_SINKHORN_ENG=...  # SinkhornEngram step
    NANOV41_OPT_FAST_GRAD_SCALE=...    # Muon grad-scale folding (used by clip_grad_norm_)
    NANOV41_OPT_FAST_CLIP=...          # clip_grad_norm_ fast path
    NANOV41_OPT_FAST_ADAMW=...         # fused AdamW construction

Sub-flags unset inherit the master flag; the module is importable standalone (``import
nanov41.opt_fast``) and does nothing unless a fast entry point is called.

R77 measured status (RTX 5090, b2/seq8192/accum32, see ``runs/r77/opt_fast_bench.json``):

======================  =============  ==========================  =====================
path                    device ms/step  pipelined wall (MoE+attn3)  parity
======================  =============  ==========================  =====================
engram harvest          -30.2 (42.7->12.5 measured 37.5->7.3 as  gaps 55.7->41.8 ms,   bitwise
                        bench per step)  +52.2 ms wall saved,       (both tables)
                                         170,982 -> 173,942 tok/s
clip (+grad fold)       -2.9            **+115 ms host gap -> OFF** total norm/coef/grads
                                                                   bitwise (device only)
Sinkhorn (dense)        -8.8            untested in the final stack   max_rel 2.1e-3 -> OFF
SinkhornEngram          -24.0           -                          **FAILS real data -> OFF**
======================  =============  ==========================  =====================

**Safe production config (validated on the Lead's MoE+attn3 stack):**
``NANOV41_OPT_FAST=1 NANOV41_OPT_FAST_CLIP=0 NANOV41_OPT_FAST_SINKHORN=0
NANOV41_OPT_FAST_SINKHORN_ENG=0 NANOV41_OPT_FAST_ADAMW=0`` -> harvest only (+52 ms/step wall on
that stack, gaps still 1.4%).  Everything else is available behind its sub-flag for future
same-stack re-validation.

The SinkhornEngram failure was a *numerical* one, not a wiring bug: the vector form kept
``U = diag(a) G~ diag(b)`` factorised and formed the row/column norms from **one** scale vector
only (``(G~ * b)^2``), which on the real Engram state (mean row norm ~2e-7, ``b`` down to ~1e-13
after 5 column passes) underflows fp32: summands ~1e-41 -> ``a`` grew to ``1/eps`` and the k=11
row rewrite diverged.  **R78 fixes it** by (a) forming every product with *both* scale vectors,
``(a_i * G~_ij) * b_j``, so it sits where the materialised matrix sits, and (b) renormalising
``(a, b)`` by exact powers of two after every row pass so ``a`` cannot run away.  Same recurrence,
same eps/tau semantics; ``tools/bench_muon_fast.py`` re-checks parity against
:func:`nanov41.muon._sinkhorn_direction_` on real-state-class sparse gradients.
"""
from __future__ import annotations

import math
import os

import torch

try:  # pragma: no cover - exercised on the GPU host only
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # noqa: BLE001 - optional dependency (Mac/CPU dev boxes)
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False


# =========================================================================== env flags
def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return bool(default)
    return raw not in ("0", "false", "no", "off")


MASTER = _flag("NANOV41_OPT_FAST", False)


def _sub(name: str) -> str:
    """Resolve a sub-flag: unset -> master; else the raw lowercase string."""
    raw = os.environ.get(name, "").strip().lower()
    return ("1" if MASTER else "0") if not raw else raw


_HARVEST = _sub("NANOV41_OPT_FAST_HARVEST")
#: Dense Sinkhorn (embed/head).  **Default OFF even under the master flag** (R77: 16.8->8.1 ms
#: device, parity max_rel 2.1e-3 bf16-grade; not part of the validated final config).
SINKHORN = _flag("NANOV41_OPT_FAST_SINKHORN", False)
#: Engram (dense-table) Sinkhorn step. **Default OFF even under the master flag**: R77 measured
#: that the first scaling-vector factorisation diverged from the materialised Algorithm-1 on the
#: real sparse/tiny-magnitude Engram gradient state (rms_rel 0.148; row scales reached 1/eps).
#: **R78 fixed the numerics** (every product carries both scale vectors + exact power-of-two
#: renormalisation of the pair, see `_canonicalise_pair_`); still default OFF until the same-process
#: real-state parity gate has passed -- `tools/bench_muon_fast.py` -> `runs/r78/muon_fast_bench.json`.
SINKHORN_ENG = _flag("NANOV41_OPT_FAST_SINKHORN_ENG", False)

#: Fast clip (+ Muon grad-scale folding).  **Default OFF even under the master flag**: it removes
#: 2.9 ms of device work but R77 measured +115 ms/step of *pipelined host gap* on the MoE+attn3
#: stack (the pure-torch rewrite of the fold is in place, but it is not yet re-validated on GPU; a
#: Triton launch costs ~0.3-0.4 ms CPU here).  Enable only after a fresh same-stack profile.
CLIP = _flag("NANOV41_OPT_FAST_CLIP", False)
#: Fused-AdamW construction.  Default OFF (the probe is rejected on this host; not validated).
ADAMW = _flag("NANOV41_OPT_FAST_ADAMW", False)
#: Muon grad-scale folding - only meaningful together with the fast ``CLIP`` path.
GRAD_SCALE = _flag("NANOV41_OPT_FAST_GRAD_SCALE", CLIP)
HARVEST = _HARVEST not in ("0", "false", "no", "off")
HARVEST_DET = _HARVEST == "det"

#: fp32 element tile per Triton program for the Sinkhorn passes (rows-per-program = TILE // D).
#: 4096 fp32 = 16 KiB of registers per program, which keeps the 2-D row/col reductions at a
#: comfortable occupancy on sm_120 (8192 spilled in the first draft).
_D_TILE = 4096
_NUM_WARPS = 4


def triton_ok() -> bool:
    return bool(_HAS_TRITON) and torch.cuda.is_available()


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _br_for(dim: int) -> int:
    """Rows per Triton program for a [rows, dim] fp32 tile of at most _D_TILE elements."""
    return max(1, min(256, _D_TILE // dim))


def _reduce_block(dim: int) -> tuple[int, int]:
    """(BLOCK_D, BLOCK_P) for the deterministic column-partial reduction kernel."""
    block_d = 32 if dim <= 32 else (64 if dim <= 64 else (128 if dim <= 128 else 128))
    block_p = max(1, min(256, 4096 // block_d))
    return block_d, block_p


# =========================================================================== Triton kernels
if _HAS_TRITON:

    @triton.jit
    def _nest_kernel(g_ptr, m_ptr, gt_ptr, rho_ptr, scale_ptr, rows,
                     D: tl.constexpr, BR: tl.constexpr, mom, om,
                     NESTEROV: tl.constexpr, HAS_SCALE: tl.constexpr,
                     ROUND_GRAD: tl.constexpr, EMIT_RHO: tl.constexpr):
        """One pass: M <- mom*M + (1-mom)*G ; G~ <- (1-mom)*G + mom*M ; rho <- ||G~_row||.

        ``g_ptr``/``gt_ptr`` may alias (the Engram dense grad buffer is reused in place); each row
        is owned by exactly one program, so the read-before-write order is safe.
        """
        pid = tl.program_id(0)
        ri = pid * BR + tl.arange(0, BR)
        rm = ri < rows
        od = tl.arange(0, D)
        idx = ri[:, None] * D + od[None, :]
        g = tl.load(g_ptr + idx, mask=rm[:, None], other=0.0).to(tl.float32)
        if HAS_SCALE:
            sc = tl.load(scale_ptr)
            if ROUND_GRAD:
                sc = sc.to(g_ptr.dtype.element_ty).to(tl.float32)
            g = g * sc
            if ROUND_GRAD:
                g = g.to(g_ptr.dtype.element_ty).to(tl.float32)
        m = tl.load(m_ptr + idx, mask=rm[:, None], other=0.0)
        mn = mom * m + om * g
        if NESTEROV:
            gt = om * g + mom * mn
        else:
            gt = mn
        tl.store(m_ptr + idx, mn, mask=rm[:, None])
        tl.store(gt_ptr + idx, gt, mask=rm[:, None])
        if EMIT_RHO:
            r = tl.sqrt(tl.sum(gt * gt, axis=1))
            tl.store(rho_ptr + ri, r, mask=rm)

    @triton.jit
    def _pass_kernel(gt_ptr, a_ptr, b_ptr, part_ptr, rows,
                     D: tl.constexpr, BR: tl.constexpr, eps,
                     DO_ROW: tl.constexpr, DO_COL: tl.constexpr):
        """One read-only pass over ``G~`` with the *scaling-vector* Algorithm-1 iterate.

        The current matrix is ``U = diag(a) G~ diag(b)``; this pass never materialises it, but it
        forms each element as ``(a_i * G~_ij) * b_j`` **including both scale vectors** so the
        floating-point product stays where the materialised matrix lives (O(1) .. O(1/sqrt(rows))).

        * ``DO_ROW``: ``rn_i = ||U_i||``; ``a_i <- a_i / (rn_i + eps)`` (row L2 normalisation, k odd)
          and, when ``DO_COL``, the column partial sums of the *new* ``U`` for the k+1 column pass.
        * otherwise: column partial sums of the current ``U`` (k even).

        R77 bug fixed here: the old kernel formed ``(G~_ij * b_j)^2`` with **no** ``a_i``, i.e.
        ``b_j ~ 1e-13`` at k=11 gave summands ~1e-41 -> fp32 subnormal/underflow -> ``a`` blew up to
        ``1/eps`` and the update diverged on real Engram state (rms_rel 0.148).
        """
        pid = tl.program_id(0)
        ri = pid * BR + tl.arange(0, BR)
        rm = ri < rows
        od = tl.arange(0, D)
        idx = ri[:, None] * D + od[None, :]
        g = tl.load(gt_ptr + idx, mask=rm[:, None], other=0.0)
        b = tl.load(b_ptr + od)
        a_old = tl.load(a_ptr + ri, mask=rm, other=0.0)
        t = (a_old[:, None] * g) * b[None, :]
        if DO_ROW:
            rn = tl.sqrt(tl.sum(t * t, axis=1))
            a_new = tl.where(a_old > 0.0, a_old / (rn + eps), 0.0)
            tl.store(a_ptr + ri, a_new, mask=rm)
            if DO_COL:
                u = t / (rn[:, None] + eps)
                acc = tl.sum(u * u, axis=0)
                tl.store(part_ptr + pid * D + od, acc)
        else:
            acc = tl.sum(t * t, axis=0)
            tl.store(part_ptr + pid * D + od, acc)

    @triton.jit
    def _reduce_kernel(part_ptr, b_ptr, nprog, eps,
                       D: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_P: tl.constexpr):
        """``b_j <- b_j / (||U_col_j|| + eps)`` (deterministic, fixed pid order).

        ``||U_col_j|| = sqrt(sum_i (a_i G~_ij b_j)^2)`` is already the *norm of the current
        column* (it includes ``b_j``), hence the division of the old ``b_j``.  All-zero columns
        keep ``b_j = 0``: the reference leaves that column zero forever (``0 / (0+eps) = 0``),
        whereas ``b_old / eps`` would grow unboundedly over 11 passes.
        """
        pd = tl.program_id(0)
        od = pd * BLOCK_D + tl.arange(0, BLOCK_D)
        dm = od < D
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for p0 in range(0, nprog, BLOCK_P):
            pp = p0 + tl.arange(0, BLOCK_P)
            pm = pp < nprog
            v = tl.load(part_ptr + pp[:, None] * D + od[None, :],
                        mask=pm[:, None] & dm[None, :], other=0.0)
            acc += tl.sum(v, axis=0)
        b_old = tl.load(b_ptr + od, mask=dm, other=1.0)
        b_new = tl.where(acc > 0.0, b_old / (tl.sqrt(acc) + eps), 0.0)
        tl.store(b_ptr + od, b_new, mask=dm)

    @triton.jit
    def _final_kernel(gt_ptr, w_ptr, a_ptr, b_ptr, coef_ptr, rows,
                      D: tl.constexpr, BR: tl.constexpr, eps,
                      ROUND_DIR: tl.constexpr, ZERO_GT: tl.constexpr):
        """Final (k=K, odd -> row) pass fused with the write-back:

        ``U <- U / (||U_i|| + eps)``; ``w <- w - coef * U`` (and optionally re-zero ``G~``).
        Saves one full read pass over ``G~`` vs doing the k=K row pass and the update separately.
        """
        pid = tl.program_id(0)
        ri = pid * BR + tl.arange(0, BR)
        rm = ri < rows
        od = tl.arange(0, D)
        idx = ri[:, None] * D + od[None, :]
        g = tl.load(gt_ptr + idx, mask=rm[:, None], other=0.0)
        a = tl.load(a_ptr + ri, mask=rm, other=0.0)
        b = tl.load(b_ptr + od)
        t = (a[:, None] * g) * b[None, :]
        rn = tl.sqrt(tl.sum(t * t, axis=1))
        u = t / (rn[:, None] + eps)
        coef = tl.load(coef_ptr)
        if ROUND_DIR:
            # current SinkhornMomentum rounds delta to the parameter dtype *before* the add
            u = u.to(w_ptr.dtype.element_ty).to(tl.float32)
        w = tl.load(w_ptr + idx, mask=rm[:, None], other=0.0).to(tl.float32)
        wn = w - coef * u
        tl.store(w_ptr + idx, wn.to(w_ptr.dtype.element_ty), mask=rm[:, None])
        if ZERO_GT:
            tl.store(gt_ptr + idx, g * 0.0, mask=rm[:, None])

    @triton.jit
    def _scatter_add_kernel(ids_ptr, g_ptr, dest_ptr, n,
                            D: tl.constexpr, BR: tl.constexpr):
        """dest[ids[i]] += g[i] (fp32 atomics; duplicate ids are summed, order not fixed)."""
        pid = tl.program_id(0)
        i = pid * BR + tl.arange(0, BR)
        im = i < n
        ids = tl.load(ids_ptr + i, mask=im, other=0).to(tl.int64)
        od = tl.arange(0, D)
        g = tl.load(g_ptr + i[:, None] * D + od[None, :],
                    mask=im[:, None], other=0.0).to(tl.float32)
        tl.atomic_add(dest_ptr + ids[:, None] * D + od[None, :], g, mask=im[:, None])

    @triton.jit
    def _scale_inplace_kernel(dst_ptr, n, scale_ptr, BLOCK: tl.constexpr, EMULATE: tl.constexpr):
        """dst *= dtype_round(scale) with the same rounding as an in-place ``mul_`` (async)."""
        pid = tl.program_id(0)
        i = pid * BLOCK + tl.arange(0, BLOCK)
        im = i < n
        sc = tl.load(scale_ptr)
        if EMULATE:
            sc = sc.to(dst_ptr.dtype.element_ty).to(tl.float32)
        v = tl.load(dst_ptr + i, mask=im, other=0.0).to(tl.float32) * sc
        tl.store(dst_ptr + i, v.to(dst_ptr.dtype.element_ty), mask=im)

    @triton.jit
    def _to_fp32_scaled_kernel(src_ptr, dst_ptr, n, scale_ptr,
                               BLOCK: tl.constexpr, EMULATE: tl.constexpr):
        """fp32 out = float(dtype_round(src * dtype_round(scale))) when EMULATE.

        This reproduces `torch._foreach_mul_(bf16_grads, cuda_fp32_0dim_coef)` bitwise: on CUDA
        that call takes TensorIterator's *tensor* path (a CUDA 0-dim tensor is not a wrapped
        number), so the fp32 coefficient is cast to the grad dtype first, the product is computed
        in fp32 opmath and rounded back to the grad dtype.
        """
        pid = tl.program_id(0)
        i = pid * BLOCK + tl.arange(0, BLOCK)
        im = i < n
        sc = tl.load(scale_ptr)
        if EMULATE:
            sc = sc.to(src_ptr.dtype.element_ty).to(tl.float32)
        v = tl.load(src_ptr + i, mask=im, other=0.0).to(tl.float32) * sc
        if EMULATE:
            v = v.to(src_ptr.dtype.element_ty).to(tl.float32)
        tl.store(dst_ptr + i, v, mask=im)


# =========================================================================== small helpers
def _scalar(value: float, device: torch.device) -> torch.Tensor:
    """0-dim fp32 device tensor filled *asynchronously*.

    ``torch.tensor(float, device="cuda")`` stages through pageable host memory and synchronises the
    current stream (``cudaStreamSynchronize``); in the pipelined training loop that drains the
    CUDA queue and cost ~400 ms/step of host stall (R77).  ``empty() + fill_()`` only launches a
    fill kernel, so the host never waits for the device.
    """
    buf = torch.empty((), dtype=torch.float32, device=device)
    buf.fill_(float(value))
    return buf


def _empty(rows: int, cols: int, device: torch.device) -> torch.Tensor:
    return torch.empty((rows, cols), dtype=torch.float32, device=device)


def _check_matrix(w: torch.Tensor) -> bool:
    """True when a table/parameter matrix can go through the Triton Sinkhorn pipeline."""
    if not w.is_cuda or w.ndim != 2 or not w.is_contiguous():
        return False
    dim = int(w.shape[1])
    return _pow2(dim) and 8 <= dim <= 4096 and w.dtype in (torch.bfloat16, torch.float16,
                                                           torch.float32)


# =========================================================================== Sinkhorn pipeline
#: fp32 smallest normal: ``frexp`` anchor for the scale-vector canonicalisation.
_F32_TINY = 1.1754943508222875e-38


def _canonicalise_pair_(a: torch.Tensor, b: torch.Tensor) -> None:
    """Rescale ``(a, b)`` by reciprocal powers of two so ``max|a|`` lands in ``[0.5, 1)``.

    Exact in floating point (a power-of-two multiply rounds nothing) and the Algorithm-1 iterate
    ``U = diag(a) G~ diag(b)`` is invariant under ``a -> a/s, b -> b*s``, so this changes no math.
    It is what keeps the *factorised* form inside fp32 range: without it ``a`` grows by
    ~sqrt(rows/dim) on every row pass (5+ passes -> ~1e19 for the 3.4M-row Engram tables) while
    ``b`` shrinks to ~1e-13, and products formed from only one of them underflow (the R77 failure).
    ``eps`` stays negligible because the *matrix* (hence every norm in the recurrence) is unchanged
    and remains O(1) .. O(1/sqrt(rows)), exactly as in the materialised reference.
    """
    amax = a.abs().max().clamp_min(_F32_TINY)
    amax = torch.nan_to_num(amax, nan=1.0, posinf=1.0, neginf=1.0)
    _, exp = torch.frexp(amax)                       # amax = mant * 2**exp, |mant| in [0.5, 1)
    one = torch.ones((), dtype=torch.float32, device=a.device)
    a.mul_(torch.ldexp(one, -exp))                   # a *= 2**-exp  (exact)
    b.mul_(torch.ldexp(one, exp))                    # b *= 2**exp   (exact)


def _sinkhorn_pipeline(w, gt_src, gt, M, rho, a, b, part, rows, dim, br, nprog,
                       mom, nesterov, iters, tau, eps, coef, scale,
                       round_dir, round_grad, zero_gt):
    """Run the vector-form Algorithm-1 Sinkhorn and apply ``w -= coef * U`` in place.

    ``gt`` is the fp32 working buffer that receives G~ (may alias ``gt_src``).  ``w`` is the
    matrix (bf16/fp16/fp32) that is updated.  ``scale`` (0-dim cuda tensor or None) folds a grad
    scale (the clip coefficient) into the *first* pass, exactly like an in-place ``_foreach_mul_``
    with the same dtype rounding.

    Faithful translation of ``_sinkhorn_direction_`` (the materialised reference)::

        rho = ||G~_i||; mask rho_i <= tau*rho_bar; U = masked G~
        for k in 1..K: U /= (||U_i||+eps)  [k odd, rows] or (||U_j||+eps)  [k even, cols]
        w -= coef * U

    The iterate is kept factorised as ``U = diag(a) G~ diag(b)`` so each pass is a *read-only*
    pass over ``G~`` (the official kernel trick: maintain row/column scaling vectors, never rewrite
    the normalised matrix).  ``k=1`` and ``k=K`` need no extra pass: ``k=1`` comes from ``rho`` and
    the final row pass is fused with the write-back.  -> 7 reads of ``G~`` instead of 11 x
    (read+write) for the materialised version.
    """
    has_scale = scale is not None
    floor = scale if has_scale else rho   # never dereferenced when HAS_SCALE is False
    _nest_kernel[(nprog,)](gt_src, M, gt, rho, floor, rows, dim, br, mom, 1.0 - mom,
                           nesterov, has_scale, round_grad, True, num_warps=_NUM_WARPS)
    # k=1 (row): a <- mask / (||G~_row|| + eps)   [the tau near-zero-row mask lives in `a`]
    if tau and tau > 0:
        rho_bar = rho.mean()
        a.copy_(torch.where(rho > tau * rho_bar, 1.0, 0.0).div_(rho + eps))
    else:
        a.copy_(torch.reciprocal(rho + eps))
    b.fill_(1.0)
    _canonicalise_pair_(a, b)
    reduce_block_d, reduce_block_p = _reduce_block(dim)
    grid_red = (cdiv(dim, reduce_block_d),)
    # k=2 (col): b <- b / (||U_col|| + eps)
    if iters >= 2:
        _pass_kernel[(nprog,)](gt, a, b, part, rows, dim, br, eps, False, True,
                               num_warps=_NUM_WARPS)
        _reduce_kernel[grid_red](part, b, nprog, eps, dim, reduce_block_d, reduce_block_p,
                                 num_warps=4)
    # k=3,5,...,K-2 (row), each pass also producing the k+1 column partials
    for k in range(3, iters, 2):
        do_col = (k + 1) <= iters
        _pass_kernel[(nprog,)](gt, a, b, part, rows, dim, br, eps, True, do_col,
                               num_warps=_NUM_WARPS)
        if do_col:
            _reduce_kernel[grid_red](part, b, nprog, eps, dim, reduce_block_d, reduce_block_p,
                                     num_warps=4)
        _canonicalise_pair_(a, b)
    # k=K (odd -> row) fused with `w -= coef * U`
    _final_kernel[(nprog,)](gt, w, a, b, coef, rows, dim, br, eps, round_dir, zero_gt,
                            num_warps=_NUM_WARPS)
    return a, b


# --------------------------------------------------------------------------- engram step
def engram_step(eng) -> bool:
    """Fast ``SinkhornEngram.step`` (dense fp32 Sinkhorn on the sparse-accumulated table grads)."""
    if not SINKHORN_ENG or not triton_ok():
        return False
    tables = list(eng.tables)
    if not tables:
        return True
    if int(eng.K) % 2 == 0:  # Algorithm 1 requires the last (odd) pass to be a row one
        return False
    for table in tables:  # validate everything before mutating any state
        if not _check_matrix(table.weight) or int(table.weight.shape[1]) != int(table.dim):
            return False
    for table in tables:
        w = table.weight
        rows, dim = int(w.shape[0]), int(w.shape[1])
        st = eng._ensure(table)
        M, G = st["momentum"], st["grad"]
        if M.shape != w.shape or G.shape != w.shape or not M.is_contiguous() \
                or not G.is_contiguous():
            return False
        br = _br_for(dim)
        nprog = cdiv(rows, br)
        rho = torch.empty(rows, dtype=torch.float32, device=w.device)
        a = torch.empty(rows, dtype=torch.float32, device=w.device)
        b = torch.empty(dim, dtype=torch.float32, device=w.device)
        part = torch.empty(nprog * dim, dtype=torch.float32, device=w.device)
        coef = _scalar(eng.lr * eng.gamma * math.sqrt(dim), w.device)
        _sinkhorn_pipeline(w, G, G, M, rho, a, b, part, rows, dim, br, nprog,
                           float(eng.momentum), bool(eng.nesterov), int(eng.K),
                           float(eng.tau), float(eng.eps), coef,
                           scale=None, round_dir=False, round_grad=False, zero_gt=True)
    return True


# --------------------------------------------------------------------------- dense Sinkhorn
def _sinkhorn_specs(opt) -> list[tuple[str, torch.Tensor, int, int]] | None:
    """(name, param, rows, dim) for every dense Sinkhorn param, or None when unsupported."""
    if not (SINKHORN and triton_ok()):
        return None
    if int(opt.K) % 2 == 0:  # Algorithm 1 requires the last (odd) pass to be a row one
        return None
    specs = []
    for name, p in opt._named:
        if p.grad is None:
            continue
        if not _check_matrix(p) or not p.grad.is_contiguous() or p.grad.dtype != p.dtype:
            return None
        specs.append((name, p, int(p.shape[0]), int(p.shape[1])))
    return specs


def sinkhorn_step(opt) -> bool:
    """Fast ``SinkhornMomentum.step`` (token embedding + prediction head).

    ``opt`` is the :class:`nanov41.muon.SinkhornMomentum` instance; when the clip fast path
    stashed a coefficient in ``opt._opt_fast_grad_scale`` the grad scaling is folded into the
    momentum pass instead of a separate in-place pass over the grads.
    """
    scale = getattr(opt, "_opt_fast_grad_scale", None)
    specs = _sinkhorn_specs(opt)
    if specs is None:
        if scale is not None:  # clip deferred the scaling to us -> apply it the slow way
            grads = [p.grad for _, p in opt._named if p.grad is not None]
            if grads:
                _foreach_mul_grouped(grads, scale)
            opt._opt_fast_grad_scale = None
        return False
    group = opt.param_groups[0]
    lr, mom = float(group["lr"]), float(group["momentum"])
    nesterov = bool(group["nesterov"])
    for name, p, rows, dim in specs:
        st = opt.state[p]
        if "momentum" not in st:
            st["momentum"] = torch.zeros(p.shape, dtype=torch.float32, device=p.device)
        M = st["momentum"]
        if M.shape != p.shape or not M.is_contiguous() or M.dtype != torch.float32:
            return False
        gt = _empty(rows, dim, p.device)
        br = _br_for(dim)
        nprog = cdiv(rows, br)
        rho = torch.empty(rows, dtype=torch.float32, device=p.device)
        a = torch.empty(rows, dtype=torch.float32, device=p.device)
        b = torch.empty(dim, dtype=torch.float32, device=p.device)
        part = torch.empty(nprog * dim, dtype=torch.float32, device=p.device)
        coef = _scalar(lr * opt.lr_mults.get(name, 1.0) * opt.gamma * math.sqrt(dim), p.device)
        _sinkhorn_pipeline(p, p.grad, gt, M, rho, a, b, part, rows, dim, br, nprog,
                           mom, nesterov, int(opt.K), float(opt.tau), float(opt.eps), coef,
                           scale=scale, round_dir=(p.dtype != torch.float32),
                           round_grad=(p.grad.dtype != torch.float32), zero_gt=False)
        del gt
    if scale is not None:
        opt._opt_fast_grad_scale = None
    return True


# =========================================================================== Engram harvest
def _scatter_add_(dest: torch.Tensor, ids: torch.Tensor, grads: torch.Tensor) -> None:
    """dest[ids] += grads (fp32 dest [rows, D]); duplicate ids are summed."""
    dim = int(grads.shape[1])
    n = int(grads.shape[0])
    if n == 0:
        return
    if triton_ok() and _pow2(dim) and dim <= 128 and grads.is_contiguous():
        br = 128 if dim <= 32 else 64
        _scatter_add_kernel[(cdiv(n, br),)](ids, grads, dest, n, dim, br, num_warps=4)
    else:
        dest.index_add_(0, ids.to(torch.int64), grads.to(torch.float32))


def _consume_records(table) -> None:
    """Replicate ``NanoEngramTable.collect_row_grads`` post-processing exactly."""
    if table._static_records:
        if table._static_zero_grads:
            for _, values in table._records:
                if values.grad is not None:
                    values.grad.zero_()
    else:
        for _, values in table._records:
            values.grad = None
        table._records.clear()


def engram_accumulate(eng) -> bool:
    """Fast ``SinkhornEngram.accumulate``: scatter leaf grads straight into the dense fp32 buffer."""
    if not HARVEST:
        return False
    for table in eng.tables:
        st = eng._ensure(table)
        dest = st["grad"]
        dim = int(table.dim)
        for ids, values in list(table._records):
            grad = values.grad
            if grad is None:
                continue
            ids_flat = ids.reshape(-1)
            g2 = grad.reshape(-1, dim)
            if ids_flat.numel() == 0:
                continue
            if HARVEST_DET:
                from .nano_engram import segment_sum_rows

                rows, sums = segment_sum_rows(ids_flat, g2)
                if rows.numel():
                    dest.index_add_(0, rows.to(torch.int64), sums.to(torch.float32))
            else:
                _scatter_add_(dest, ids_flat, g2)
        _consume_records(table)
        table._pending.clear()  # merged_row_grads/zero_row_grad bookkeeping (unused here)
    return True


# =========================================================================== clip / grad scale
def to_fp32_scaled(grad: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """fp32 copy of ``grad`` with the clip coefficient folded in, bitwise-identical to
    ``_to_fp32(grad * coef)`` after torch's in-place ``_foreach_mul_``.

    Implemented with two aten kernels (``mul`` + ``float``) rather than a Triton kernel: on this
    host a Triton launch costs ~0.3-0.4 ms of CPU, and Muon converts ~183 tensors/step, so a
    per-tensor Triton kernel here costs ~70 ms/step of *host* time (which showed up as a pipelined
    host gap even though device time improved).  The two-kernel torch form is bitwise-equal:
    torch's CUDA in-place mul by a device scalar computes ``dtype(float(g) * float(dtype(coef)))``.
    """
    if grad.dtype in (torch.bfloat16, torch.float16):
        return torch.mul(grad, scale.to(grad.dtype)).float()
    return torch.mul(grad, scale)


def _total_norm(tensors, norm_type: float = 2.0) -> torch.Tensor:
    """Total grad norm, computed exactly like ``torch.nn.utils.clip_grad_norm_``."""
    get_total_norm = getattr(torch.nn.utils, "get_total_norm", None)
    if get_total_norm is not None:
        return get_total_norm(tensors, norm_type)
    norms = torch._foreach_norm(tensors, norm_type)
    if norm_type == math.inf:
        return torch.stack([n.max() for n in norms]).max()
    return torch.linalg.vector_norm(torch.stack(list(norms)), norm_type)


def _scale_inplace_(tensors, coef: torch.Tensor) -> None:
    """tensors *= coef in place, bitwise like torch's own clip (``_foreach_mul_`` per dtype/device).

    Uses torch's foreach (what ``torch.nn.utils.clip_grad_norm_`` uses too) instead of one Triton
    kernel per tensor: only ~105 tensors here and a Triton launch costs ~0.3-0.4 ms of host CPU on
    this box, which would turn the 2.9 ms device saving into a ~40 ms host gap.  `clip_coef` is the
    same 0-dim device tensor torch's clip passes, so the arithmetic is unchanged.
    """
    by_key: dict = {}
    for t in tensors:
        by_key.setdefault((t.device, t.dtype), []).append(t)
    for group in by_key.values():
        torch._foreach_mul_(group, coef)


def _foreach_mul_grouped(tensors, coef: torch.Tensor) -> None:
    """Back-compat alias for :func:`_scale_inplace_`."""
    _scale_inplace_(tensors, coef)


def clip_grad_norm_(bundle, params, max_norm: float, norm_type: float = 2.0) -> torch.Tensor:
    """``torch.nn.utils.clip_grad_norm_`` equivalent that skips the big in-place grad scaling.

    The returned total norm is the same tensor torch returns (bitwise-identical computation), so
    the clip coefficient is exactly the same and the *effective* post-clip grads are identical:
    the Muon group's coefficient is folded into its ``to_fp32_scaled`` grad conversion, the dense
    Sinkhorn group's into its momentum pass, and only the (small) remaining grads are scaled in
    place with the same ``_foreach_mul_`` op torch uses.
    """
    if not CLIP:  # behave exactly like the torch helper when the fast path is off
        return torch.nn.utils.clip_grad_norm_(params, max_norm, norm_type)
    params = [p for p in params if p.grad is not None]
    if not params:
        return _scalar(0.0, torch.device("cpu"))
    grads = [p.grad for p in params]
    total = _total_norm(grads, norm_type)
    coef = torch.clamp(max_norm / (total + 1e-6), max=1.0)
    coef_dev = coef.to(grads[0].device)
    muon = getattr(bundle, "muon", None) if bundle is not None else None
    sink = getattr(bundle, "sinkhorn", None) if bundle is not None else None
    fold_muon = bool(GRAD_SCALE and muon is not None)
    fold_sink = sink is not None and _sinkhorn_specs(sink) is not None
    fast_ids: set[int] = set()
    if fold_muon:
        fast_ids.update(id(p) for _, p in muon._named)
    if fold_sink:
        fast_ids.update(id(p) for _, p in sink._named)
    rest = [p.grad for p in params if id(p) not in fast_ids]
    if rest:
        _scale_inplace_(rest, coef_dev)
    if fold_muon:
        muon._opt_fast_grad_scale = coef_dev
    if fold_sink:
        sink._opt_fast_grad_scale = coef_dev
    return total


# =========================================================================== AdamW
def adamw_step(opt) -> bool:
    """One fused-AdamW step for a ``torch.optim.AdamW`` built with ``fused=True``."""
    if not ADAMW:
        return False
    try:
        opt.step()
        return True
    except Exception:  # noqa: BLE001 - never kill the run; muon.py falls back to the dense path
        return False


#: Reason the last ``adamw_fused_supported()`` probe failed (empty when it succeeded).
_ADAMW_PROBE_ERR = ""


def adamw_fused_supported() -> bool:
    """Whether ``torch.optim.AdamW(fused=True)`` is usable on this host (per-dtype groups)."""
    global _ADAMW_PROBE_ERR
    if not (ADAMW and torch.cuda.is_available()):
        _ADAMW_PROBE_ERR = f"ADAMW flag={ADAMW} cuda={torch.cuda.is_available()}"
        return False
    try:
        groups = []
        for dtype in (torch.float32, torch.bfloat16):
            p = torch.zeros(4, dtype=dtype, device="cuda", requires_grad=True)
            p.grad = torch.zeros_like(p)
            groups.append({"params": [p]})
        torch.optim.AdamW(groups, lr=1e-3, fused=True).step()
        _ADAMW_PROBE_ERR = ""
        return True
    except Exception as exc:  # noqa: BLE001
        _ADAMW_PROBE_ERR = f"{type(exc).__name__}: {exc}"
        return False


# =========================================================================== R79 pre-warm
def prewarm(dims=(32, 512), rows: int = 256) -> dict:
    """Compile/launch the scaling-vector Sinkhorn kernels for the given column dims (no-op off GPU).

    R79 item 3: Triton JIT-compiles the four kernels on their first launch, which showed up as a
    fake "+83ms host bubble" in the first profile / A-B step (R78).  ``D`` is a ``tl.constexpr``
    here, so every production column dim needs its own compile; ``ROUND_DIR``/``ZERO_GT`` (engram
    vs dense Sinkhorn write-back) and ``HAS_SCALE`` (clip folding) are constexprs too, so we run
    the full constexpr matrix once on a tiny dummy matrix.  Returns the number of pipeline runs.
    """
    if not triton_ok():
        return {"runs": 0}
    dev = torch.device("cuda")
    runs = 0
    for dim in sorted({int(d) for d in dims if int(d) > 0}):
        br = _br_for(dim)
        nprog = cdiv(rows, br)
        rho = torch.empty(rows, dtype=torch.float32, device=dev)
        a = torch.empty(rows, dtype=torch.float32, device=dev)
        b = torch.empty(dim, dtype=torch.float32, device=dev)
        part = torch.empty(nprog * dim, dtype=torch.float32, device=dev)
        coef = _scalar(1e-3, dev)
        scale = _scalar(1.0, dev)
        for round_dir, zero_gt, round_grad, has_scale in (
                (False, True, False, False),   # engram, no clip fold
                (True, False, False, False),   # dense Sinkhorn, no clip fold
                (True, False, True, True),     # dense Sinkhorn + clip fold
                (False, True, True, True)):    # engram + (unused) clip-fold variant
            G = torch.full((rows, dim), 1e-3, dtype=torch.float32, device=dev)
            M = torch.zeros((rows, dim), dtype=torch.float32, device=dev)
            w = torch.zeros((rows, dim), dtype=torch.float32, device=dev)
            _sinkhorn_pipeline(w, G, G, M, rho, a, b, part, rows, dim, br, nprog,
                               0.95, True, 11, 1e-3, 1e-20, coef,
                               scale=(scale if has_scale else None),
                               round_dir=round_dir, round_grad=round_grad, zero_gt=zero_gt)
            runs += 1
    torch.cuda.synchronize()
    return {"runs": runs, "dims": sorted({int(d) for d in dims if int(d) > 0})}


# =========================================================================== diagnostics
def info() -> dict:
    """Flags + availability (handy for the bench JSON / run logs)."""
    return {
        "triton": bool(_HAS_TRITON),
        "cuda": bool(torch.cuda.is_available()),
        "master": MASTER,
        "harvest": _HARVEST,
        "sinkhorn": SINKHORN,
        "sinkhorn_eng": SINKHORN_ENG,
        "grad_scale": GRAD_SCALE,
        "clip": CLIP,
        "adamw": ADAMW, "adamw_probe_error": _ADAMW_PROBE_ERR,
        "d_tile": _D_TILE,
        "num_warps": _NUM_WARPS,
    }


def _nest_pass_(g_src, M, gt, rho, rows, dim, mom, nesterov, scale=None, round_grad=False):
    """Run just the momentum/nesterov/rho kernel (bench debug helper)."""
    br = _br_for(dim)
    nprog = cdiv(rows, br)
    has_scale = scale is not None
    floor = scale if has_scale else rho
    _nest_kernel[(nprog,)](g_src, M, gt, rho, floor, rows, dim, br, mom, 1.0 - mom,
                           nesterov, has_scale, round_grad, True, num_warps=_NUM_WARPS)


__all__ = [
    "MASTER", "HARVEST", "HARVEST_DET", "SINKHORN", "SINKHORN_ENG", "GRAD_SCALE", "CLIP",
    "ADAMW",
    "triton_ok", "info", "prewarm",
    "engram_accumulate", "engram_step", "sinkhorn_step",
    "to_fp32_scaled", "clip_grad_norm_", "adamw_step", "adamw_fused_supported",
    "_nest_pass_", "_sinkhorn_pipeline", "_br_for", "cdiv", "_nest_kernel",
    "_scatter_add_",
]

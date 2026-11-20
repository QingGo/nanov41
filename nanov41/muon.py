"""DeepSeek-V4.1-Flash optimizer suite, pure PyTorch (single GPU).

Implements V4.1 report §2.5 / §4.2.2 and Algorithm 1:

1. **Muon** for the weight matrices of the backbone linear transformations, the Engram
   projection layers (and, when present, the VL projector):
   Nesterov momentum, decoupled weight decay 0.1, ``head-wise`` Newton-Schulz for the Query
   weights, and the **RMS of each update matrix rescaled to 0.18** so that the AdamW learning
   rate can be reused.  Newton-Schulz runs on shape-grouped stacked matrices in bf16 (one
   batched matmul per group per iteration) so ~4.6 k expert matrices at d12/aligned cost a few
   hundred MB of scratch instead of one NS launch per matrix.
2. **Sinkhorn-balanced momentum** for the Engram embedding tables, the token embedding and the
   prediction head (Algorithm 1 verbatim): ``K=11`` alternating row/column L2 normalisations,
   ``tau=1e-3`` near-zero row masking, ``eps=1e-20``, ``gamma=0.18`` learning-rate correction,
   Nesterov momentum, **no weight decay**.  Memory is a single momentum buffer (no Adam
   second moment), which is the report's stated motivation for this optimizer.
3. **AdamW** for norm-layer weights (weight decay 0.1) and all other non-matrix parameters --
   biases, scaling factors, attention sinks (weight decay 0) -- betas (0.9, 0.95), eps 1e-20.

Momentum convention (report Algorithm 1, eq. 1-2)::

    M_t = beta * M_{t-1} + (1 - beta) * G_t
    G~_t = beta * M_t + (1 - beta) * G_t

i.e. the classic "lerp" form used by Moonlight/Kimi-K2 Muon, *not* ``buf = beta*buf + g``.

Engram tables are trained through sparse row gradients (``nanov41.nano_engram``); the table
analogue of the Sinkhorn optimizer is :class:`SinkhornEngram`, which scatters the per-step row
gradients into a dense fp32 buffer, applies the *same* Algorithm-1 update to the full table and
writes the result back to the (bf16) table.  That is the faithful reading of the report (their
kernels "maintain row/column scaling vectors across iterations to avoid repeated writes of the
full normalized matrix").

Deviations / decisions (documented in ``docs/v41-aligned-spec.md`` §2):
  * Head-wise applies to Q weights (``*.attn.wq_b.weight`` and ``*.attn.indexer.wq_b.weight``).
    The released reference model has a *single shared* KV latent (``attn.wkv.weight``, no head
    structure), so there is no per-head K matrix to split; the report's "Query and Key" wording
    cannot be expressed on this architecture beyond Q.
  * "Engram learning rate x5" is applied to all Engram-module parameters: the tables
    (Sinkhorn), the ``wkv`` projections (Muon) and the 1-D gate weights (AdamW).
  * ``gate.bias`` / ``gate.bias_vl`` are *never* produced by an optimizer here: they follow the
    auxiliary-loss-free update rule (see ``nanov41/v41_balance.py``).

Configurable knobs (env-gated; **unset == the report defaults**, so existing runs are unchanged):
  * ``NANOV41_MUON_RMS``        Muon update-RMS target (default 0.18).  The update is
    ``lr * (NS_direction rescaled to this RMS)``, hence linear: 0.36 == Muon-group lr x2,
    0.72 == x4.
  * ``NANOV41_SINKHORN_GAMMA``  Sinkhorn ``gamma`` (default 0.18); same linear semantics
    (update = ``lr * gamma * sqrt(n) * U``), so 0.36 == Sinkhorn-group lr x2.
  * ``NANOV41_ENGRAM_LR_MULT``  Engram lr multiplier (default 5.0 = report ``x5``).
  The values are read once at import; ``build_optimizers(update_rms=...,
  sinkhorn_gamma=..., engram_lr_mult=...)`` still accepts explicit overrides for tests/experiments.

R78 O5a implementation-speed knobs (**default OFF; unset == the report-literal code path**)::

    NANOV41_MUON_FAST=1            # master switch for the math-preserving Muon speed-ups
    NANOV41_MUON_FAST_FOREACH=...  # multi-tensor momentum/Nesterov (`torch._foreach_*`), bitwise
    NANOV41_MUON_FAST_NS=...       # baddbmm-fused Newton-Schulz epilogues (<=1 bf16 ulp)
    NANOV41_MUON_FAST_WRITE=...    # chunk-level RMS/cast + grouped wd/add write-out (bitwise)
    NANOV41_MUON_FAST_FUSED=...    # Triton fused momentum+Nesterov+stack-cast (opt-in, see below)
    NANOV41_MUON_FAST_FMA=...      # explicit-FMA form of the fused kernel (both bitwise)
    NANOV41_MUON_MOMENTUM_BF16=1   # R79: Muon momentum state bf16 (frees ~1.25 GiB; math-changing)
    NANOV41_MUON_CLIP_FOLD=...     # R79: fold the clip coefficient into the fused momentum kernel

Sub-flags unset inherit the master.  The fast path never changes the *math*: the momentum /
Nesterov chain, the update-RMS rescale and the decoupled-wd update are arithmetically identical
(same operation order, same scalars); the Newton-Schulz iterate differs only in where bf16
rounding happens (fused GEMM epilogues) and is gated on the same-process <=1 bf16-ulp parity
check.  See ``tools/bench_muon_fast.py`` / ``runs/r78/muon_fast_bench.json`` for the validation.

Why: the literal path walks 183 Muon tensors with ~6 kernels each (>1.1 k launches/step) and
~32 GB of elementwise traffic per step (fp32 momentum read/write, per-tensor fp32 casts and
stacked bf16 writes); R72/R77 profile Muon at 113 ms/step (4.0% of the step).
"""

from __future__ import annotations

import math
import os
import re

import torch

try:  # optional: the R78 fused momentum kernel needs Triton (GPU hosts only)
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # noqa: BLE001 - CPU/Mac dev boxes have no Triton
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _HAS_TRITON = False


def _env_float(name: str, default: float) -> float:
    """Read a float knob from the environment, falling back to ``default`` when unset.

    Empty/whitespace means "unset".  A non-float value raises ``ValueError`` instead of silently
    keeping the default: a typo in an A/B run must fail loudly (otherwise the run's headline
    number would be produced by a different optimizer than the one requested).
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError as exc:  # pragma: no cover - exercised by tests
        raise ValueError(f"{name} must be a float, got {raw!r}") from exc


def _env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean knob from the environment (empty/whitespace means unset)."""
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return bool(default)
    return raw not in ("0", "false", "no", "off")


# =========================================================================== R78 O5a flags
#: Master switch for the math-preserving Muon implementation speed-ups (R78 O5a).  Default OFF:
#: with the env unset `Muon.step` executes the report-literal code path unchanged.
MUON_FAST = _env_flag("NANOV41_MUON_FAST", False)


def _muon_fast_sub(name: str) -> bool:
    """Sub-flag of the O5a Muon fast path: unset inherits ``NANOV41_MUON_FAST``."""
    return _env_flag(name, MUON_FAST)


def _muon_fast_flags() -> dict[str, bool]:
    """Resolved per-stage flags (also the bench's knob: mutate ``opt.fast`` in place).

    ``NANOV41_MUON_FAST=1`` enables **only the validated bitwise stage** (the fused Triton
    momentum/Nesterov/stack-cast kernel).  The R78 bench measured the other stages as either
    slower (``foreach`` +2.0ms, ``write`` +2.5ms whole-step) or math-changing (``ns`` baddbmm
    fusion: rms_rel 2.7e-2, angle 1.4-4.7deg), so they are explicit opt-ins -- never inherited.
    """
    return {
        # validated bitwise (R78 O5a, tools/bench_muon_fast.py -> runs/r78/muon_fast_bench.json)
        "fused": _env_flag("NANOV41_MUON_FAST_FUSED", MUON_FAST),
        # R78 negative results: multi-tensor chains lose to the per-tensor loops on this host
        "foreach": _env_flag("NANOV41_MUON_FAST_FOREACH", False),
        "write": _env_flag("NANOV41_MUON_FAST_WRITE", False),
        # math-changing (O5c-class; needs a 20-step loss A/B before it may ship)
        "ns": _env_flag("NANOV41_MUON_FAST_NS", False),
        # Explicit-FMA form of the Triton momentum kernel (both variants were bitwise; default 0).
        "fma": _env_flag("NANOV41_MUON_FAST_FMA", False),
    }


def _opt_fast_module():
    """R77 fast out-of-graph optimizer paths (``nanov41/opt_fast.py``); None when unavailable.

    Everything in that module is **env-gated and default-off** (``NANOV41_OPT_FAST=1`` + the
    per-path sub-flags), so importing it here can never change the report-literal behaviour of
    this file.  The helpers below dispatch only when the corresponding flag is on; each fast
    implementation returns False when it cannot handle the shapes/host, in which case the literal
    code path in this file runs unchanged.
    """
    try:
        from . import opt_fast
        return opt_fast
    except Exception:  # noqa: BLE001 - optional fast path, never kill the run
        return None


# --------------------------------------------------------------------------- report constants
NS_STEPS = 5
NS_A, NS_B, NS_C = 3.4445, -4.7750, 2.0315

MUON_MOMENTUM = 0.95
MUON_WEIGHT_DECAY = 0.1
#: Muon update-RMS target (report: 0.18).  **Env-overridable via ``NANOV41_MUON_RMS``.**
#: The NS direction is normalised and rescaled to this RMS *before* being multiplied by the
#: learning rate, so the per-element update magnitude is LINEAR in the target:
#: ``NANOV41_MUON_RMS=0.36`` is exactly "Muon-group lr x2", ``0.72`` is "x4" — the clean knob
#: for the M1 question "is the shared muP LR (8.22e-4) right for the V4.1 suite?" without
#: touching the AdamW/Sinkhorn groups.  Defaults (unset env) reproduce the report exactly.
MUON_UPDATE_RMS = _env_float("NANOV41_MUON_RMS", 0.18)

ADAMW_BETAS = (0.9, 0.95)
ADAMW_EPS = 1e-20
ADAMW_WEIGHT_DECAY = 0.1

SINKHORN_MOMENTUM = 0.95
#: Sinkhorn learning-rate correction ``gamma`` (report: 0.18).  **Env-overridable via
#: ``NANOV41_SINKHORN_GAMMA``** with the same linear semantics as MUON_UPDATE_RMS (the update is
#: ``lr * gamma * sqrt(n) * U``), i.e. 0.36 == Sinkhorn-group lr x2.
SINKHORN_GAMMA = _env_float("NANOV41_SINKHORN_GAMMA", 0.18)
SINKHORN_ITERS = 11
SINKHORN_TAU = 1e-3
SINKHORN_EPS = 1e-20
SINKHORN_WEIGHT_DECAY = 0.0

#: R79 item 1: Muon momentum buffer in **bf16** (default OFF, fp32 = report-literal).
#: Frees ``numel * (4-2) = ~1.25 GiB`` of optimizer state for the aligned-d12 Muon group (623.8M
#: elems) — the accum64/b3 memory unlock — and halves the momentum read/write traffic.  Math delta:
#: the *state* is rounded to bf16 once per step (``M_t = bf16(beta*M + (1-beta)*G)``, and the
#: Nesterov direction uses the stored bf16 state), so this is bf16-class noise on the update
#: direction, NOT 1-ulp-fp32; it needs a loss A/B and is therefore env-gated and default OFF.
MUON_MOMENTUM_BF16 = _env_flag("NANOV41_MUON_MOMENTUM_BF16", False)

#: "Engram learning rate x5" (report §2.5) applied to all Engram-module parameters.
#: **Env-overridable via ``NANOV41_ENGRAM_LR_MULT``** (default 5.0 == report).
ENGRAM_LR_MULT = _env_float("NANOV41_ENGRAM_LR_MULT", 5.0)

#: Muon batched-NS scratch budget in elements.  Matrices are chunked so that one chunk's stacked
#: gradient + NS intermediates fit this budget; the peak transient of the optimizer step is
#: roughly (1 x stacked + 2 x [N,m,m] + 1 x [N,m,n]) bf16 of this size.  Lowering it trades a few
#: percent of NS speed for gigabytes of memory (the chunk8192 / 16k-micro-batch enabler).
MUON_MAX_BATCH_ELEMS = int(_env_float("NANOV41_MUON_MAX_BATCH_ELEMS", float(1 << 25)))

#: SinkhornEngram dense-update chunk (table rows per in-place update); env-overridable.
ENGRAM_UPDATE_CHUNK_ROWS = int(_env_float("NANOV41_ENGRAM_UPDATE_CHUNK_ROWS", float(1 << 20)))

#: parameters the aux-free load-balancing rule owns (never optimizer-owned)
GATE_BIAS_SUFFIXES = (".gate.bias", ".gate.bias_vl")


# =========================================================================== Newton-Schulz
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = NS_STEPS,
                                eps: float = 1e-7) -> torch.Tensor:
    """Reference single-matrix Newton-Schulz orthogonalisation in fp32 (CPU-parity baseline).

    ``G`` is 2-D.  Matrices with ``rows > cols`` are transposed first, so the returned matrix is
    semi-orthogonal in the same orientation as ``G``.
    """
    X = G.to(torch.float32)
    if X.ndim != 2:
        raise ValueError(f"zeropower_via_newtonschulz5 expects a 2-D matrix, got {tuple(X.shape)}")
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = NS_B * A + NS_C * (A @ A)
        X = NS_A * X + B @ X
    return X.T if transposed else X


def _ns_dtype(device: torch.device) -> torch.dtype:
    """bf16 NS math on GPU (report's kernel precision); fp32 on CPU so unit tests stay portable."""
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def _batched_newton_schulz(G: torch.Tensor, steps: int = NS_STEPS,
                           eps: float = 1e-7) -> torch.Tensor:
    """Batched NS on ``[N, m, n]`` (same coefficients/order as the reference; dtype see _ns_dtype)."""
    X = G.to(_ns_dtype(G.device))
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.transpose(-1, -2)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.transpose(-1, -2)
        B = NS_B * A + NS_C * (A @ A)
        X = NS_A * X + B @ X
    if transposed:
        X = X.transpose(-1, -2)
    return X


def _batched_newton_schulz_fused(G: torch.Tensor, steps: int = NS_STEPS,
                                 eps: float = 1e-7) -> torch.Tensor:
    """R78 O5a: ``_batched_newton_schulz`` with the two elementwise epilogues folded into the
    batched GEMMs via ``torch.baddbmm(..., beta=..., alpha=...)``.

    The reference computes, per iteration, ``B = NS_B*A + NS_C*(A@A)`` and ``X = NS_A*X + B@X``
    with *separate* bf16-rounded elementwise ops (4 + 3 passes over the bf16 intermediates); the
    fused form lets the GEMM epilogue apply ``beta*input + alpha*(matmul)`` in one pass.  Same
    iterate in exact arithmetic; the only difference is where the bf16 rounding happens (one
    rounding per epilogue instead of two) -> validated on the same-process parity gate at
    <=1 bf16 ulp (``tools/bench_muon_fast.py``).
    """
    X = G.to(_ns_dtype(G.device))
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.transpose(-1, -2)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.transpose(-1, -2)
        B = torch.baddbmm(A, A, A, beta=NS_B, alpha=NS_C)     # NS_B*A + NS_C*(A@A)
        X = torch.baddbmm(X, B, X, beta=NS_A, alpha=1.0)      # NS_A*X + B@X
    if transposed:
        X = X.transpose(-1, -2)
    return X


def muon_direction(G: torch.Tensor, steps: int = NS_STEPS,
                   update_rms: float = MUON_UPDATE_RMS, eps: float = 1e-7) -> torch.Tensor:
    """One Muon direction for a single 2-D gradient: NS-orthogonalise, then rescale to ``update_rms``.

    Returns an fp32 tensor shaped like ``G``.  Used by tests/parity harnesses; the optimizer uses
    the batched path.
    """
    u = _batched_newton_schulz(G.to(torch.float32).unsqueeze(0), steps=steps, eps=eps)
    u = u.squeeze(0).float()
    rms = u.square().mean().sqrt().clamp_min(1e-12)
    return u * (update_rms / rms)


# =========================================================================== helpers
def _to_fp32(t: torch.Tensor) -> torch.Tensor:
    """Detached fp32 copy of a gradient (never aliases ``t``)."""
    t = t.detach()
    return t.float() if t.dtype != torch.float32 else t.clone()


# =========================================================================== R78 O5a fused momentum
#: Elements per Triton program for the fused momentum/Nesterov kernel.
_MOM_BLOCK = 1024

if _HAS_TRITON:

    @triton.jit
    def _muon_momentum_kernel(g_ptr, m_ptr, out_ptr, scale_ptr, out_off, n, mom, om,
                              GRAD_F32: tl.constexpr, M_BF16: tl.constexpr,
                              HAS_SCALE: tl.constexpr, NESTEROV: tl.constexpr,
                              FMA: tl.constexpr, BLOCK: tl.constexpr):
        """M <- mom*M + (1-mom)*G ; out <- Nesterov(G, M_t) ; one pass, one launch, one kernel.

        Replaces the reference chain (5 kernels + a fp32 cast + a cast-copy per tensor)::

            g   = fp32(p.grad)                        # cast kernel
            buf.mul_(mom)                             # t  = rn(mom*M)
            buf.add_(g, alpha=1-mom)                  # M' = rn((1-mom)*G + t)   [fused alpha]
            g.mul_(1-mom)                             # o  = rn((1-mom)*G)
            g.add_(buf, alpha=mom)                    # G~ = rn(mom*M' + o)      [fused alpha]
            stacked[...].copy_(g)                     # bf16 cast-copy

        torch's CUDA ``add_`` computes ``a + alpha*b`` contracted to a single-rounding FMA, so the
        bitwise-equivalent form is ``FMA=1`` (explicit PTX ``fma.rn.f32``); ``FMA=0`` let the
        backend contract freely.  ``tools/bench_muon_fast.py`` reports which one is bitwise.
        """
        pid = tl.program_id(0)
        i = pid * BLOCK + tl.arange(0, BLOCK)
        mask = i < n
        g = tl.load(g_ptr + i, mask=mask, other=0.0)
        m = tl.load(m_ptr + i, mask=mask, other=0.0)
        if not GRAD_F32:
            g = g.to(tl.float32)
        if HAS_SCALE:
            # R79 clip folding: replace the separate in-place `grad *= coef` pass with exactly
            # what torch's clip does, emulated in-register:
            #   bf16/fp16 grads: float(bf16(grad * bf16(coef)))   (torch mul opmath -> back to dtype)
            #   fp32 grads:      grad * float(coef)
            sc = tl.load(scale_ptr)
            if GRAD_F32:
                g = g * sc
            else:
                sc = sc.to(g_ptr.dtype.element_ty).to(tl.float32)
                g = (g * sc).to(g_ptr.dtype.element_ty).to(tl.float32)
        t = mom * m
        og = om * g
        if FMA:
            # inline PTX needs tensor operands -> broadcast the two scalars to the block shape
            omv = tl.zeros([BLOCK], tl.float32) + om
            momv = tl.zeros([BLOCK], tl.float32) + mom
            mn = tl.inline_asm_elementwise("fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f",
                                           [omv, g, t], dtype=tl.float32, is_pure=True, pack=1)
        else:
            mn = t + og
        if M_BF16:
            # bf16 state: the stored (rounded) momentum is authoritative -> G~ uses it
            mnr = mn.to(tl.bfloat16)
            tl.store(m_ptr + i, mnr, mask=mask)
            mn = mnr.to(tl.float32)
        else:
            tl.store(m_ptr + i, mn, mask=mask)
        if NESTEROV:
            if FMA:
                gt = tl.inline_asm_elementwise("fma.rn.f32 $0, $1, $2, $3;", "=f,f,f,f",
                                               [momv, mn, og], dtype=tl.float32, is_pure=True,
                                               pack=1)
            else:
                gt = mom * mn + og
        else:
            gt = mn
        tl.store(out_ptr + out_off + i, gt.to(out_ptr.dtype.element_ty), mask=mask)


def _launch_muon_momentum_fused(grad: torch.Tensor, buf: torch.Tensor, out_flat: torch.Tensor,
                                out_off: int, mom: float, nesterov: bool,
                                fma: bool = False, scale: torch.Tensor | None = None) -> bool:
    """Launch :func:`_muon_momentum_kernel` for one parameter; False when unsupported.

    ``scale`` (0-dim fp32 CUDA tensor or None) folds the gradient-clip coefficient into the read
    that this kernel already does (R79 item 2): the effective grad is exactly
    ``torch.nn.utils.clip_grad_norm_``'s in-place ``grad *= coef`` (see ``opt_fast.to_fp32_scaled``)
    so the post-clip grads are bitwise identical.  ``buf`` may be fp32 (default) or bf16
    (``NANOV41_MUON_MOMENTUM_BF16=1``).
    """
    if not _HAS_TRITON or not grad.is_cuda:
        return False
    n = int(grad.numel())
    if n == 0 or int(buf.numel()) != n:
        return False
    grid = ((n + _MOM_BLOCK - 1) // _MOM_BLOCK,)
    _muon_momentum_kernel[grid](grad.detach(), buf, out_flat, scale if scale is not None else grad,
                                int(out_off), n, float(mom), 1.0 - float(mom),
                                GRAD_F32=(grad.dtype == torch.float32),
                                M_BF16=(buf.dtype == torch.bfloat16),
                                HAS_SCALE=(scale is not None),
                                NESTEROV=bool(nesterov), FMA=bool(fma),
                                BLOCK=_MOM_BLOCK, num_warps=4)
    return True


#: Probe cache for ``torch._foreach_copy_`` across dtypes (bf16 stacked <- fp32 grads).
_FOREACH_COPY_CAST: bool | None = None


def _foreach_copy_cast(dst: list, src: list) -> bool:
    """``torch._foreach_copy_`` with dtype conversion, probed once; False -> caller loops."""
    global _FOREACH_COPY_CAST
    if _FOREACH_COPY_CAST is None:
        try:
            a = torch.zeros(8, dtype=torch.bfloat16, device=src[0].device)
            b = torch.zeros(8, dtype=torch.float32, device=src[0].device)
            torch._foreach_copy_([a], [b])
            _FOREACH_COPY_CAST = True
        except Exception:  # noqa: BLE001 - older torch requires identical dtypes
            _FOREACH_COPY_CAST = False
    if not _FOREACH_COPY_CAST:
        return False
    try:
        torch._foreach_copy_(dst, src)
        return True
    except Exception:  # noqa: BLE001
        return False


def _sinkhorn_direction_(U: torch.Tensor, iters: int = SINKHORN_ITERS,
                         tau: float = SINKHORN_TAU, eps: float = SINKHORN_EPS) -> torch.Tensor:
    """Algorithm 1 lines 3-16: mask near-zero rows, then K alternating L2 normalisations.

    ``U`` is modified in place (and also returned).  ``K`` must be odd (report uses 11; the
    last normalisation is then a row one, so rows end up unit-norm).  The caller applies the
    ``sqrt(n)`` factor and ``gamma*eta``.
    """
    if iters % 2 == 0:
        raise ValueError(f"Sinkhorn iteration count must be odd (report uses 11), got {iters}")
    if U.ndim != 2:
        raise ValueError(f"Sinkhorn expects a 2-D matrix, got {tuple(U.shape)}")
    if tau and tau > 0:
        rho = U.norm(dim=1)
        rho_bar = rho.mean()
        U.masked_fill_(rho.unsqueeze(1) <= tau * rho_bar, 0.0)
    for k in range(1, iters + 1):
        if k % 2 == 1:  # odd: normalise each row
            U.div_(U.norm(dim=1, keepdim=True) + eps)
        else:           # even: normalise each column
            U.div_(U.norm(dim=0, keepdim=True) + eps)
    return U


# =========================================================================== Muon
class Muon(torch.optim.Optimizer):
    """Muon (MomentUm Orthogonalised by Newton-Schulz) with shape-grouped batched NS.

    Parameters are supplied as ``(name, param)`` pairs so that head-wise Q splitting and the
    Engram 5x learning rate can be attached by *name*.  Supported layouts:
      * 2-D matrix ``[m, n]``                       -> one NS matrix
      * 3-D ``[E, m, n]`` (stacked experts, etc.)   -> E independent NS matrices
      * head-wise 2-D ``[H*m, n]`` with ``headwise[name] = H`` -> H matrices ``[m, n]``

    Every per-matrix update is rescaled to RMS ``update_rms`` (report: 0.18) before the
    decoupled-weight-decay + ``p -= lr * delta`` step.
    """

    def __init__(self, named_params, lr: float = 2.6e-4, momentum: float = MUON_MOMENTUM,
                 nesterov: bool = True, ns_steps: int = NS_STEPS,
                 weight_decay: float = MUON_WEIGHT_DECAY,
                 update_rms: float = MUON_UPDATE_RMS,
                 headwise: dict[str, int] | None = None,
                 lr_mults: dict[str, float] | None = None,
                 max_batch_elems: int = MUON_MAX_BATCH_ELEMS):
        self._named = [(str(n), p) for n, p in named_params]
        self.headwise = dict(headwise or {})
        self.lr_mults = dict(lr_mults or {})
        self.ns_steps = int(ns_steps)
        self.max_batch_elems = int(max_batch_elems)
        #: R78 O5a per-stage fast-path flags (resolved from the env at construction; the bench
        #: mutates this dict in place to A/B individual stages).  All False == report-literal.
        self.fast = _muon_fast_flags()
        #: R79 item 1: dtype of the momentum buffer (fp32 = report-literal; bf16 frees ~1.25 GiB).
        #: A per-instance override (``opt.momentum_dtype = torch.bfloat16``) is the bench's knob.
        self.momentum_dtype = torch.bfloat16 if MUON_MOMENTUM_BF16 else torch.float32
        for name, n_heads in self.headwise.items():
            p = dict(self._named)[name]
            if p.ndim != 2 or p.shape[0] % n_heads:
                raise ValueError(f"head-wise param {name} shape {tuple(p.shape)} not divisible "
                                 f"by {n_heads} heads")
        super().__init__([dict(params=[p for _, p in self._named], lr=float(lr),
                               momentum=float(momentum), nesterov=bool(nesterov),
                               weight_decay=float(weight_decay),
                               update_rms=float(update_rms))],
                         defaults=dict(lr=float(lr), momentum=float(momentum),
                                       nesterov=bool(nesterov),
                                       weight_decay=float(weight_decay),
                                       update_rms=float(update_rms)))

    # ---------------------------------------------------------------- layout
    def _layout(self, name: str, p: torch.Tensor) -> tuple[int, int, int]:
        """(num_matrices, m, n) for a parameter."""
        if name in self.headwise:
            h = self.headwise[name]
            return h, p.shape[0] // h, p.shape[1]
        if p.ndim == 2:
            return 1, p.shape[0], p.shape[1]
        if p.ndim == 3:
            return p.shape[0], p.shape[-2], p.shape[-1]
        raise ValueError(f"Muon got non-matrix parameter {name} with shape {tuple(p.shape)}")

    # ---------------------------------------------------------------- step
    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        group = self.param_groups[0]
        lr, mom = group["lr"], group["momentum"]
        nesterov, wd = group["nesterov"], group["weight_decay"]
        update_rms = group["update_rms"]
        # R77 clip folding: when `opt_fast.clip_grad_norm_` deferred the clip coefficient into the
        # grad conversion below, `_opt_fast_grad_scale` holds it (0-dim cuda tensor).  Default-off.
        _of = _opt_fast_module()
        gscale = None
        if _of is not None and _of.GRAD_SCALE:
            gscale = getattr(self, "_opt_fast_grad_scale", None)
        # R78 O5a: implementation-level fast path (default OFF).  Since R79 the deferred clip
        # coefficient (if any) is folded into the fast momentum path as well (fused kernel: free;
        # foreach/literal momentum: via `opt_fast.to_fp32_scaled`), so no gscale fallback is needed.
        if self._fast_enabled():
            return self._step_fast(loss, gscale)

        # 1) bucket the parameter *names* by (m, n); no gradient-sized fp32 buffers are kept
        #    alive here.  The previous version built the whole Nesterov update for every param
        #    before chunking (~2.4 GiB of fp32 for the 4608 expert matrices), which dominated the
        #    optimizer's transient memory on 32 GB cards (see docs/m1-optimizer.md §7/§9).
        buckets: dict[tuple[int, int], list[tuple[str, torch.Tensor, int]]] = {}
        for name, p in self._named:
            if p.grad is None:
                continue
            cnt, m, n = self._layout(name, p)
            st = self.state[p]
            if "momentum" not in st:
                st["momentum"] = torch.zeros(p.shape, dtype=self.momentum_dtype,
                                             device=p.device)
            buckets.setdefault((m, n), []).append((name, p, cnt))

        # 2) per shape bucket, chunk by element count: momentum/Nesterov -> stacked bf16 ->
        #    batched NS -> RMS rescale -> update.  Peak scratch is one chunk, not one bucket.
        #    NOTE the chunk budget counts *matrices*, not tensors: a 3-D stacked-expert tensor
        #    holds cnt (=E) matrices, so a tensor-count chunk of e.g. 32 would put thousands of
        #    matrices in one stacked buffer + NS intermediates (~1 GiB each) and blow the
        #    optimizer's transient memory.  Greedy-pack items until the matrix count would
        #    exceed `max_batch_elems // m/n` so the peak scratch is bounded by the budget.
        for (m, n), items in buckets.items():
            per_matrix = m * n
            max_mats = max(1, self.max_batch_elems // max(per_matrix, 1))
            # A single tensor is the minimum granularity (its `cnt` matrices are one bmm
            # input); the budget is therefore a soft cap that can be exceeded by one tensor.
            max_mats = max(max_mats, max(c for _, _, c in items))
            chunks: list[list[tuple[str, torch.Tensor, int]]] = []
            cur: list[tuple[str, torch.Tensor, int]] = []
            cur_mats = 0
            for it in items:
                c = it[2]
                if cur and cur_mats + c > max_mats:
                    chunks.append(cur)
                    cur, cur_mats = [], 0
                cur.append(it)
                cur_mats += c
            if cur:
                chunks.append(cur)
            for part in chunks:
                total = sum(c for _, _, c in part)
                device = part[0][1].device
                stacked = torch.empty((total, m, n), dtype=_ns_dtype(device), device=device)
                off = 0
                for name, p, cnt in part:
                    buf = self.state[p]["momentum"]
                    if gscale is not None:
                        # fold the clip coefficient into the grad -> fp32 conversion (one read of
                        # p.grad instead of a separate in-place `_foreach_mul_` over all grads)
                        g = _of.to_fp32_scaled(p.grad, gscale)
                    else:
                        g = _to_fp32(p.grad)
                    if buf.dtype == torch.float32:
                        buf.mul_(mom).add_(g, alpha=1.0 - mom)        # M_t = beta*M+(1-beta)*G
                        mt = buf
                    else:  # R79: bf16 momentum state -> one bf16 rounding of M_t per step
                        mt = buf.float()
                        mt.mul_(mom).add_(g, alpha=1.0 - mom)
                        buf.copy_(mt)                                 # bf16 state is authoritative
                        mt = buf.float()
                    if nesterov:
                        g.mul_(1.0 - mom).add_(mt, alpha=mom)         # G~ = beta*M_t+(1-beta)*G
                    else:
                        g.copy_(mt)
                    del mt
                    stacked[off:off + cnt].copy_(g.reshape(cnt, m, n))
                    off += cnt
                    del g
                u = _batched_newton_schulz(stacked, steps=self.ns_steps).float()
                rms = u.square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-12)
                u.mul_(update_rms / rms)
                off = 0
                for name, p, cnt in part:
                    delta = u[off:off + cnt].reshape(p.shape)
                    off += cnt
                    plr = lr * self.lr_mults.get(name, 1.0)
                    if wd:
                        p.mul_(1.0 - plr * wd)
                    p.add_(delta.to(p.dtype), alpha=-plr)
                del stacked, u
        if gscale is not None:  # clip coefficient consumed once
            self._opt_fast_grad_scale = None
        return loss

    # ---------------------------------------------------------------- batch helpers
    def set_lr(self, lr: float) -> None:
        for group in self.param_groups:
            group["lr"] = float(lr)

    # ---------------------------------------------------------------- R79 memory helpers
    @torch.no_grad()
    def ensure_states_(self) -> int:
        """Pre-allocate every momentum buffer before graph capture; returns the bytes allocated.

        R79 item: the optimizer state is currently created on the *first* step, i.e. after the
        capture pool is reserved, which is what OOMs the larger-step configs (accum64/b3).  Calling
        this before ``torch.cuda.graph(...)`` moves that allocation into the (smaller) pre-capture
        live set and lets the allocator reuse the warm-up scratch.  Idempotent, no math.
        """
        total = 0
        for name, p in self._named:
            st = self.state[p]
            if "momentum" not in st:
                st["momentum"] = torch.zeros(p.shape, dtype=self.momentum_dtype,
                                             device=p.device)
            total += st["momentum"].numel() * st["momentum"].element_size()
        return total

    def state_bytes(self) -> int:
        """Bytes held by this optimizer's state (for the memory matrix; 0 before first step)."""
        total = 0
        for p, st in self.state.items():
            for v in st.values():
                if isinstance(v, torch.Tensor):
                    total += v.numel() * v.element_size()
        return total

    def prewarm(self) -> int:
        """Compile/launch the R78 fused momentum kernel once so the first timed step is warm.

        R78 observed a fake "+83ms host bubble" in the first profile/A-B step: Triton JIT-compiles
        ``_muon_momentum_kernel`` on its first launch.  This launches it on a tiny dummy for every
        constexpr variant the current config can select (grad dtype x momentum dtype x clip-fold).
        The kernel is shape-generic, so one dummy per variant is enough.  Returns #launches.
        """
        if not (_HAS_TRITON and torch.cuda.is_available()):
            return 0
        dev = torch.device("cuda")
        n = 4096
        out = torch.zeros(n, dtype=torch.bfloat16, device=dev)
        n_launch = 0
        for grad_dtype in (torch.bfloat16, torch.float32):
            g = torch.zeros(n, dtype=grad_dtype, device=dev)
            for mdtype in (self.momentum_dtype,):
                m = torch.zeros(n, dtype=mdtype, device=dev)
                for scale in (None, torch.ones((), dtype=torch.float32, device=dev)):
                    if _launch_muon_momentum_fused(g, m, out, 0, 0.95, True,
                                                   scale=scale):
                        n_launch += 1
        torch.cuda.synchronize()
        return n_launch

    def load_state_dict(self, state_dict) -> None:
        """Load, but keep the momentum buffer in ``self.momentum_dtype`` (R79).

        torch's ``Optimizer.load_state_dict`` casts floating-point state tensors to the *parameter*
        dtype, which would silently turn the fp32 momentum into bf16 on resume for bf16 params
        (a latent R72-era inconsistency) and would fight ``NANOV41_MUON_MOMENTUM_BF16``.  We keep
        the saved values verbatim and convert them straight to the configured dtype (single cast).
        """
        saved = {}
        for k, v in (state_dict.get("state") or {}).items():
            try:
                saved[int(k)] = dict(v)
            except (TypeError, ValueError):
                continue
        super().load_state_dict(state_dict)
        for idx, (name, p) in enumerate(self._named):
            st = self.state.get(p)
            src = saved.get(idx, {}).get("momentum")
            if st is None or src is None:
                continue
            st["momentum"] = src.to(device=p.device, dtype=self.momentum_dtype)

    # ---------------------------------------------------------------- R78 O5a fast path
    def _fast_enabled(self) -> bool:
        """True when the fast path is configured AND every parameter can be handled by it."""
        flags = getattr(self, "fast", None)
        if not flags or not any(flags.values()):
            return False
        if flags.get("fused") and not (_HAS_TRITON and torch.cuda.is_available()):
            return False
        for name, p in self._named:
            if p.grad is None:
                continue
            g = p.grad
            if not (p.is_cuda and g.is_cuda and p.is_contiguous() and g.is_contiguous()):
                return False
            if p.dtype not in (torch.bfloat16, torch.float32):
                return False
            if g.dtype not in (torch.bfloat16, torch.float32):
                return False
            st = self.state.get(p)
            if st is not None and "momentum" in st:
                mm = st["momentum"]
                if (mm.dtype != self.momentum_dtype or mm.shape != p.shape
                        or not mm.is_contiguous() or mm.device != p.device):
                    return False
        return True

    @torch.no_grad()
    def _step_fast(self, loss=None, gscale=None):
        """O5a fast Muon step: bitwise/<=1-ulp equivalent to :meth:`step` with fewer launches.

        Same bucketing/chunking as :meth:`step` (the chunk budget still bounds the NS scratch), but
        the momentum/Nesterov elementwise chain runs as multi-tensor ops (or one fused Triton
        kernel), the NS epilogues are folded into the batched GEMMs, and the RMS/cast/write-out
        stages are done per chunk (one bf16 cast per dtype) with grouped ``_foreach_*`` updates.

        ``gscale`` (R79 clip folding): 0-dim CUDA tensor from ``opt_fast.clip_grad_norm_`` that has
        to be applied to every Muon grad.  The fused kernel folds it into the read it already does
        (bitwise-equal to torch's in-place ``grad *= coef``); the fallback chain uses
        ``opt_fast.to_fp32_scaled`` exactly like the literal path.
        """
        _of = _opt_fast_module()
        flags = self.fast
        group = self.param_groups[0]
        lr, mom = group["lr"], group["momentum"]
        nesterov, wd = group["nesterov"], group["weight_decay"]
        update_rms = group["update_rms"]
        use_fused = bool(flags["fused"])
        fma = bool(flags.get("fma"))
        buckets: dict[tuple[int, int], list[tuple[str, torch.Tensor, int]]] = {}
        for name, p in self._named:
            if p.grad is None:
                continue
            cnt, m, n = self._layout(name, p)
            st = self.state[p]
            if "momentum" not in st:
                st["momentum"] = torch.zeros(p.shape, dtype=self.momentum_dtype,
                                             device=p.device)
            buckets.setdefault((m, n), []).append((name, p, cnt))
        for (m, n), items in buckets.items():
            per_matrix = m * n
            max_mats = max(1, self.max_batch_elems // max(per_matrix, 1))
            max_mats = max(max_mats, max(c for _, _, c in items))
            chunks: list[list[tuple[str, torch.Tensor, int]]] = []
            cur: list[tuple[str, torch.Tensor, int]] = []
            cur_mats = 0
            for it in items:
                c = it[2]
                if cur and cur_mats + c > max_mats:
                    chunks.append(cur)
                    cur, cur_mats = [], 0
                cur.append(it)
                cur_mats += c
            if cur:
                chunks.append(cur)
            for part in chunks:
                total = sum(c for _, _, c in part)
                device = part[0][1].device
                stacked = torch.empty((total, m, n), dtype=_ns_dtype(device), device=device)
                # ---- (1) momentum + Nesterov -> fp32 G~ -> stacked bf16
                if use_fused:
                    flat = stacked.view(-1)
                    off_mat = 0
                    for name, p, cnt in part:
                        buf = self.state[p]["momentum"]
                        if not _launch_muon_momentum_fused(p.grad, buf, flat,
                                                           off_mat * per_matrix, mom, nesterov,
                                                           fma=fma, scale=gscale):
                            raise RuntimeError(
                                f"R78 fused Muon momentum kernel rejected {name} "
                                f"(shape {tuple(p.shape)}, grad {p.grad.dtype})")
                        off_mat += cnt
                else:
                    g_list: list[torch.Tensor] = []
                    bufs: list[torch.Tensor] = []
                    for name, p, cnt in part:
                        if gscale is not None and _of is not None:
                            g_list.append(_of.to_fp32_scaled(p.grad, gscale))
                        else:
                            g_list.append(_to_fp32(p.grad))
                        bufs.append(self.state[p]["momentum"])
                    bf_state = any(b.dtype != torch.float32 for b in bufs)
                    if bf_state:  # R79: bf16 momentum state -> rounded state is authoritative
                        for g, buf in zip(g_list, bufs):
                            mt = buf.float()
                            mt.mul_(mom).add_(g, alpha=1.0 - mom)
                            buf.copy_(mt)
                            mt = buf.float()
                            if nesterov:
                                g.mul_(1.0 - mom).add_(mt, alpha=mom)
                            else:
                                g.copy_(mt)
                            del mt
                    elif flags["foreach"]:
                        torch._foreach_mul_(bufs, mom)
                        torch._foreach_add_(bufs, g_list, alpha=1.0 - mom)
                        if nesterov:
                            torch._foreach_mul_(g_list, 1.0 - mom)
                            torch._foreach_add_(g_list, bufs, alpha=mom)
                        else:
                            torch._foreach_copy_(g_list, bufs)
                    else:
                        for g, buf in zip(g_list, bufs):
                            buf.mul_(mom).add_(g, alpha=1.0 - mom)
                            if nesterov:
                                g.mul_(1.0 - mom).add_(buf, alpha=mom)
                            else:
                                g.copy_(buf)
                    offs: list[int] = []
                    off = 0
                    for _, _, cnt in part:
                        offs.append(off)
                        off += cnt
                    dst = [stacked[o:o + c] for o, (_, _, c) in zip(offs, part)]
                    if not _foreach_copy_cast(dst, g_list):
                        for o, g, (_, p, c) in zip(offs, g_list, part):
                            stacked[o:o + c].copy_(g.reshape(c, m, n))
                    del g_list, bufs, dst
                # ---- (2) batched Newton-Schulz (optionally baddbmm-fused) + RMS rescale
                if flags["ns"]:
                    u = _batched_newton_schulz_fused(stacked, steps=self.ns_steps).float()
                else:
                    u = _batched_newton_schulz(stacked, steps=self.ns_steps).float()
                rms = u.square().mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-12)
                u.mul_(update_rms / rms)
                # ---- (3) decoupled wd + update (chunk-level cast; grouped _foreach_*)
                if flags["write"]:
                    casts: dict = {}
                    for _, p, _ in part:
                        dt = p.dtype
                        if dt not in casts:
                            casts[dt] = u if dt == torch.float32 else u.to(dt)
                    groups: dict = {}
                    off = 0
                    for name, p, cnt in part:
                        plr = lr * self.lr_mults.get(name, 1.0)
                        params, deltas = groups.setdefault((p.dtype, plr), ([], []))
                        params.append(p)
                        deltas.append(casts[p.dtype][off:off + cnt].reshape(p.shape))
                        off += cnt
                    for (_dt, plr), (params, deltas) in groups.items():
                        if wd:
                            torch._foreach_mul_(params, 1.0 - plr * wd)
                        torch._foreach_add_(params, deltas, alpha=-plr)
                    del casts
                else:
                    off = 0
                    for name, p, cnt in part:
                        delta = u[off:off + cnt].reshape(p.shape)
                        off += cnt
                        plr = lr * self.lr_mults.get(name, 1.0)
                        if wd:
                            p.mul_(1.0 - plr * wd)
                        p.add_(delta.to(p.dtype), alpha=-plr)
                del stacked, u
        if gscale is not None:  # clip coefficient consumed once
            self._opt_fast_grad_scale = None
        return loss


# =========================================================================== Sinkhorn
class SinkhornMomentum(torch.optim.Optimizer):
    """Momentum + Sinkhorn balancing (report Algorithm 1) for 2-D embedding/head matrices."""

    def __init__(self, named_params, lr: float = 2.6e-4,
                 momentum: float = SINKHORN_MOMENTUM, nesterov: bool = True,
                 gamma: float = SINKHORN_GAMMA, sinkhorn_iters: int = SINKHORN_ITERS,
                 tau: float = SINKHORN_TAU, eps: float = SINKHORN_EPS,
                 lr_mults: dict[str, float] | None = None):
        self._named = [(str(n), p) for n, p in named_params]
        for name, p in self._named:
            if p.ndim != 2:
                raise ValueError(f"SinkhornMomentum expects 2-D parameters, {name} is "
                                 f"{tuple(p.shape)}")
        self.gamma = float(gamma)
        self.K = int(sinkhorn_iters)
        self.tau = float(tau)
        self.eps = float(eps)
        self.lr_mults = dict(lr_mults or {})
        super().__init__([dict(params=[p for _, p in self._named], lr=float(lr),
                               momentum=float(momentum), nesterov=bool(nesterov))],
                         defaults=dict(lr=float(lr), momentum=float(momentum),
                                       nesterov=bool(nesterov)))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        _of = _opt_fast_module()
        if _of is not None and _of.SINKHORN and _of.sinkhorn_step(self):
            return loss
        group = self.param_groups[0]
        lr, mom = group["lr"], group["momentum"]
        nesterov = group["nesterov"]
        for name, p in self._named:
            if p.grad is None:
                continue
            st = self.state[p]
            if "momentum" not in st:
                st["momentum"] = torch.zeros(p.shape, dtype=torch.float32, device=p.device)
            buf = st["momentum"]
            g = _to_fp32(p.grad)
            buf.mul_(mom).add_(g, alpha=1.0 - mom)                  # M_t = beta*M+(1-beta)*G
            if nesterov:
                g.mul_(1.0 - mom).add_(buf, alpha=mom)              # G~ = beta*M_t+(1-beta)*G
            else:
                g = buf.clone()
            delta = _sinkhorn_direction_(g, iters=self.K, tau=self.tau, eps=self.eps)
            # eta_eff = gamma * eta ; Delta = sqrt(n) * U (Algorithm 1 lines 17-18)
            coef = lr * self.lr_mults.get(name, 1.0) * self.gamma * math.sqrt(p.size(1))
            p.add_(delta.to(p.dtype), alpha=-coef)
        return loss

    def set_lr(self, lr: float) -> None:
        for group in self.param_groups:
            group["lr"] = float(lr)

    # ---------------------------------------------------------------- R79 memory helpers
    @torch.no_grad()
    def ensure_states_(self) -> int:
        """Pre-allocate the fp32 momentum buffers before graph capture; returns the bytes."""
        total = 0
        for name, p in self._named:
            st = self.state[p]
            if "momentum" not in st:
                st["momentum"] = torch.zeros(p.shape, dtype=torch.float32, device=p.device)
            total += st["momentum"].numel() * st["momentum"].element_size()
        return total

    def state_bytes(self) -> int:
        total = 0
        for p, st in self.state.items():
            for v in st.values():
                if isinstance(v, torch.Tensor):
                    total += v.numel() * v.element_size()
        return total

    def load_state_dict(self, state_dict) -> None:
        """Load, keeping the momentum buffer fp32 (torch's loader would cast it to the param dtype)."""
        saved = {}
        for k, v in (state_dict.get("state") or {}).items():
            try:
                saved[int(k)] = dict(v)
            except (TypeError, ValueError):
                continue
        super().load_state_dict(state_dict)
        for idx, (name, p) in enumerate(self._named):
            st = self.state.get(p)
            src = saved.get(idx, {}).get("momentum")
            if st is None or src is None:
                continue
            st["momentum"] = src.to(device=p.device, dtype=torch.float32)


class SinkhornEngram:
    """Algorithm-1 Sinkhorn update for the sparse-gradient nano-Engram tables.

    The table is a ``[rows, head_dim]`` buffer (not an ``nn.Parameter``); its gradients arrive as
    sparse (row, fp32 grad) segments from :class:`nanov41.nano_engram.NanoEngramTable`.  We
    scatter them into a dense fp32 buffer and run the *same* update as
    :class:`SinkhornMomentum`, then write the (bf16/fp32) table rows back.  Optimizer state is a
    single fp32 momentum buffer per table (``rows*head_dim*4`` bytes), matching the report's
    memory motivation.
    """

    def __init__(self, tables, lr: float = 2.6e-4, momentum: float = SINKHORN_MOMENTUM,
                 nesterov: bool = True, gamma: float = SINKHORN_GAMMA,
                 sinkhorn_iters: int = SINKHORN_ITERS, tau: float = SINKHORN_TAU,
                 eps: float = SINKHORN_EPS,
                 update_chunk_rows: int = ENGRAM_UPDATE_CHUNK_ROWS):
        self.tables = list(tables)
        self.lr = float(lr)
        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)
        self.gamma = float(gamma)
        self.K = int(sinkhorn_iters)
        self.tau = float(tau)
        self.eps = float(eps)
        self.update_chunk_rows = int(update_chunk_rows)
        self._state: dict[int, dict[str, torch.Tensor]] = {}

    # ---------------------------------------------------------------- state
    def _ensure(self, table) -> dict[str, torch.Tensor]:
        w = table.weight
        st = self._state.get(id(table))
        if (st is None or st["momentum"].shape != w.shape
                or st["momentum"].device != w.device):
            st = {
                "momentum": torch.zeros(w.shape, dtype=torch.float32, device=w.device),
                "grad": torch.zeros(w.shape, dtype=torch.float32, device=w.device),
            }
            self._state[id(table)] = st
        return st

    # ---------------------------------------------------------------- gradient accumulation
    @torch.no_grad()
    def accumulate(self) -> None:
        """Harvest the sparse row gradients of the pending forward(s) into the dense buffer."""
        _of = _opt_fast_module()
        if _of is not None and _of.HARVEST and _of.engram_accumulate(self):
            return
        for table in self.tables:
            st = self._ensure(table)
            table.collect_row_grads()
            rows, grads = table.merged_row_grads()
            if rows.numel():
                st["grad"].index_add_(0, rows, grads.to(torch.float32))
            table.zero_row_grad()

    @torch.no_grad()
    def zero_grad(self) -> None:
        for table in self.tables:
            table.zero_row_grad()

    # ---------------------------------------------------------------- R79 memory helpers
    @torch.no_grad()
    def ensure_states_(self) -> int:
        """Pre-allocate every table's (momentum, grad) buffers before graph capture."""
        total = 0
        for table in self.tables:
            st = self._ensure(table)
            for v in st.values():
                total += v.numel() * v.element_size()
        return total

    def state_bytes(self) -> int:
        total = 0
        for st in self._state.values():
            for v in st.values():
                if isinstance(v, torch.Tensor):
                    total += v.numel() * v.element_size()
        return total

    # ---------------------------------------------------------------- optimizer step
    @torch.no_grad()
    def step(self) -> None:
        _of = _opt_fast_module()
        if _of is not None and _of.SINKHORN_ENG and _of.engram_step(self):
            return None
        mom = self.momentum
        for table in self.tables:
            st = self._ensure(table)
            M, G = st["momentum"], st["grad"]          # both fp32, dense
            M.mul_(mom).add_(G, alpha=1.0 - mom)       # M_t
            if self.nesterov:
                G.mul_(1.0 - mom).add_(M, alpha=mom)   # G~ (in place in the grad buffer)
            else:
                G.copy_(M)
            _sinkhorn_direction_(G, iters=self.K, tau=self.tau, eps=self.eps)
            coef = self.lr * self.gamma * math.sqrt(table.weight.size(1))
            w = table.weight
            rows = w.size(0)
            for start in range(0, rows, self.update_chunk_rows):
                end = min(start + self.update_chunk_rows, rows)
                new = w[start:end].float().add_(G[start:end], alpha=-coef)
                w[start:end].copy_(new.to(w.dtype))
            G.zero_()
        return None

    def set_lr(self, lr: float) -> None:
        self.lr = float(lr)

    # ---------------------------------------------------------------- checkpointing
    def state_dict(self, cpu: bool = True) -> dict:
        tables = []
        for table in self.tables:
            st = self._state.get(id(table))
            if st is None:
                tables.append(None)
                continue
            tables.append({
                "momentum": st["momentum"].cpu() if cpu else st["momentum"],
                "grad_shape": tuple(st["grad"].shape),
            })
        return {"lr": self.lr, "momentum": self.momentum, "gamma": self.gamma,
                "K": self.K, "tau": self.tau, "eps": self.eps, "tables": tables}

    def load_state_dict(self, state: dict) -> None:
        self.lr = float(state.get("lr", self.lr))
        self.momentum = float(state.get("momentum", self.momentum))
        self.gamma = float(state.get("gamma", self.gamma))
        self.K = int(state.get("K", self.K))
        self.tau = float(state.get("tau", self.tau))
        self.eps = float(state.get("eps", self.eps))
        for table, saved in zip(self.tables, state.get("tables", [])):
            st = self._ensure(table)
            if saved is None:
                st["momentum"].zero_()
            else:
                st["momentum"].copy_(saved["momentum"].to(st["momentum"].device))
            st["grad"].zero_()


# =========================================================================== parameter split
def _is_norm_weight(name: str, p: torch.Tensor) -> bool:
    return p.ndim == 1 and name.endswith("norm.weight")


def _is_engram_param(name: str) -> bool:
    return ".engram." in f".{name}." or name.startswith("engram.")


def _is_headwise_q(name: str, p: torch.Tensor) -> bool:
    return name.endswith("wq_b.weight") and p.ndim == 2


def is_gate_bias(name: str) -> bool:
    return name.endswith(GATE_BIAS_SUFFIXES)


def classify_parameters(model, cfg: dict | None = None, *,
                        headwise: bool = True) -> dict[str, list[tuple[str, torch.Tensor]]]:
    """Split model parameters into the report's three optimizer classes.

    Returns ``{"muon": [...], "sinkhorn": [...], "adamw_wd": [...], "adamw_nowd": [...],
    "headwise": {name: n_heads}, "skipped": [...]}`` where every entry is ``(name, param)``
    (``headwise``/``skipped`` are name lists/dicts).  ``gate.bias``-style params are *skipped*
    (they belong to the aux-free load-balancing update), and parameters with
    ``requires_grad=False`` are ignored.
    """
    cfg = dict(cfg or {})
    n_heads = int(cfg.get("n_heads") or 0)
    index_n_heads = int(cfg.get("index_n_heads") or 0)
    if not n_heads or not index_n_heads:
        # infer from the built model for callers that pass a bare cfg
        for name, mod in model.named_modules():
            if name.endswith(".attn"):
                n_heads = n_heads or int(getattr(mod, "n_heads", 0))
            if name.endswith(".attn.indexer") or ".indexer" in name:
                index_n_heads = index_n_heads or int(getattr(mod, "n_heads", 0))
    out: dict = {"muon": [], "sinkhorn": [], "adamw_wd": [], "adamw_nowd": [],
                 "headwise": {}, "skipped": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        short = re.sub(r"^(module\.|_orig_mod\.)", "", name)
        if is_gate_bias(short):
            out["skipped"].append(short)
            continue
        if short.endswith(("embed.weight", "head.weight")):
            out["sinkhorn"].append((short, p))
            continue
        if p.ndim >= 2:
            out["muon"].append((short, p))
            if headwise and _is_headwise_q(short, p):
                h = index_n_heads if "indexer" in short else n_heads
                if h > 1 and p.shape[0] % h == 0:
                    out["headwise"][short] = h
        elif _is_norm_weight(short, p):
            out["adamw_wd"].append((short, p))
        else:
            out["adamw_nowd"].append((short, p))
    return out


def _lr_mult_map(entries, mult: float) -> dict[str, float]:
    return {n: mult for n, _ in entries if _is_engram_param(n)}


# =========================================================================== optimizers
def build_optimizers(model, cfg: dict | None = None, *, base_lr: float = 2.6e-4,
                     engram_tables=None, headwise: bool = True,
                     muon_lr_mult: float = 1.0, engram_lr_mult: float = ENGRAM_LR_MULT,
                     muon_wd: float = MUON_WEIGHT_DECAY, muon_momentum: float = MUON_MOMENTUM,
                     ns_steps: int = NS_STEPS, update_rms: float = MUON_UPDATE_RMS,
                     adamw_wd: float = ADAMW_WEIGHT_DECAY, adamw_betas=ADAMW_BETAS,
                     adamw_eps: float = ADAMW_EPS,
                     sinkhorn_iters: int = SINKHORN_ITERS,
                     sinkhorn_gamma: float = SINKHORN_GAMMA,
                     sinkhorn_tau: float = SINKHORN_TAU,
                     sinkhorn_eps: float = SINKHORN_EPS,
                     sinkhorn_momentum: float = SINKHORN_MOMENTUM,
                     prewarm: bool = True, verbose: bool = False) -> "OptimizerBundle":
    """Build the report's (Muon, Sinkhorn, AdamW, SinkhornEngram) optimizer bundle.

    ``base_lr`` is the μP learning rate shared by Muon / AdamW / Sinkhorn (the Sinkhorn
    ``gamma=0.18`` correction lives inside the optimizer, like Muon's 0.18 update-RMS rescale).
    Engram-module parameters get ``base_lr * engram_lr_mult`` (report: x5).
    """
    classes = classify_parameters(model, cfg, headwise=headwise)
    muon_named = classes["muon"]
    sink_named = classes["sinkhorn"]
    muon_lr_mults = _lr_mult_map(muon_named, engram_lr_mult)
    adamw_nowd = classes["adamw_nowd"]
    adamw_nowd_engram = [(n, p) for n, p in adamw_nowd if _is_engram_param(n)]
    adamw_nowd_main = [(n, p) for n, p in adamw_nowd if not _is_engram_param(n)]

    muon = Muon(muon_named, lr=base_lr * muon_lr_mult, momentum=muon_momentum,
                ns_steps=ns_steps, weight_decay=muon_wd, update_rms=update_rms,
                headwise=classes["headwise"], lr_mults=muon_lr_mults) if muon_named else None

    sinkhorn = SinkhornMomentum(sink_named, lr=base_lr, momentum=sinkhorn_momentum,
                                gamma=sinkhorn_gamma, sinkhorn_iters=sinkhorn_iters,
                                tau=sinkhorn_tau, eps=sinkhorn_eps,
                                lr_mults=_lr_mult_map(sink_named, engram_lr_mult)
                                ) if sink_named else None

    adamw_groups = []
    if classes["adamw_wd"]:
        adamw_groups.append({"params": [p for _, p in classes["adamw_wd"]],
                             "lr": base_lr, "weight_decay": adamw_wd})
    if adamw_nowd_main:
        adamw_groups.append({"params": [p for _, p in adamw_nowd_main],
                             "lr": base_lr, "weight_decay": 0.0})
    if adamw_nowd_engram:
        adamw_groups.append({"params": [p for _, p in adamw_nowd_engram],
                             "lr": base_lr * engram_lr_mult, "weight_decay": 0.0,
                             "lr_mult_engram": True})
    # R77 (default OFF): the v41 AdamW owns a few hundred tiny 1-D tensors, and the non-fused
    # foreach path launches ~6 kernels per tensor (~6.3 ms/step measured in R72).  `fused=True` is
    # one kernel per tensor; the *state-dict format is identical*, so checkpoint/resume works
    # across the flag.  torch's fused kernel wants one dtype per call -> split each role group by
    # dtype (group hyper-parameters/`lr_mult_engram` markers are copied verbatim).
    _of = _opt_fast_module()
    fused_adamw = bool(_of is not None and _of.adamw_fused_supported())
    if fused_adamw:
        split_groups: list[dict] = []
        for grp in adamw_groups:
            by_dtype: dict = {}
            for p in grp["params"]:
                by_dtype.setdefault(p.dtype, []).append(p)
            for dtype in by_dtype:
                sub = dict(grp)
                sub["params"] = by_dtype[dtype]
                split_groups.append(sub)
        adamw_groups = split_groups
    adamw = (torch.optim.AdamW(adamw_groups, lr=base_lr, betas=tuple(adamw_betas),
                               eps=float(adamw_eps), weight_decay=0.0, fused=fused_adamw)
             if adamw_groups else None)

    eng = (SinkhornEngram(engram_tables, lr=base_lr * engram_lr_mult,
                          momentum=sinkhorn_momentum, gamma=sinkhorn_gamma,
                          sinkhorn_iters=sinkhorn_iters, tau=sinkhorn_tau,
                          eps=sinkhorn_eps) if engram_tables else None)

    bundle = OptimizerBundle(muon=muon, adamw=adamw, sinkhorn=sinkhorn, engram=eng,
                             base_lr=base_lr, muon_lr_mult=muon_lr_mult,
                             engram_lr_mult=engram_lr_mult)
    if verbose:
        counts = {k: len(v) for k, v in classes.items()
                  if k not in ("headwise", "skipped")}
        nmuon = sum(p.numel() for _, p in muon_named)
        nsink = sum(p.numel() for _, p in sink_named)
        nadam = sum(p.numel() for _, p in classes["adamw_wd"]) + \
            sum(p.numel() for _, p in adamw_nowd)
        neng = sum(t.weight.numel() for t in (engram_tables or []))
        print(f"[v41opt] muon={counts['muon']} tensors/{nmuon/1e6:.1f}M  "
              f"sinkhorn={counts['sinkhorn']} tensors/{nsink/1e6:.1f}M  "
              f"adamw={counts['adamw_wd'] + len(adamw_nowd)} tensors/{nadam/1e6:.2f}M  "
              f"engram={neng/1e6:.1f}M  skipped={classes['skipped']} "
              f"headwise={len(classes['headwise'])}", flush=True)
        _mf = getattr(muon, "fast", None) if muon is not None else None
        if _mf and any(_mf.values()):
            print(f"[v41opt] NANOV41_MUON_FAST stages: {_mf} "
                  f"momentum_dtype={getattr(muon, 'momentum_dtype', None)}", flush=True)
    # R79 item 3: pre-compile the env-enabled Triton fast kernels BEFORE graph capture / any timed
    # region (R78's "+83ms host bubble" was Triton JIT cold start).  Cheap, CPU-side, best-effort:
    # a pre-warm failure must never kill a run (the kernels would just compile on first use).
    if prewarm and torch.cuda.is_available():
        try:
            stats = bundle.prewarm()
            if stats and verbose:
                print(f"[v41opt] Triton kernel pre-warm: {stats}", flush=True)
        except Exception as _exc:  # noqa: BLE001
            if verbose:
                print(f"[v41opt] pre-warm skipped: {type(_exc).__name__}: {_exc}", flush=True)
    return bundle


# =========================================================================== bundle
class OptimizerBundle:
    """Thin orchestration over the report's optimizers (one ``step()`` per training step)."""

    def __init__(self, muon=None, adamw=None, sinkhorn=None, engram=None,
                 base_lr: float = 2.6e-4, muon_lr_mult: float = 1.0,
                 engram_lr_mult: float = ENGRAM_LR_MULT):
        self.muon = muon
        self.adamw = adamw
        self.sinkhorn = sinkhorn
        self.engram = engram
        self.base_lr = float(base_lr)
        self.muon_lr_mult = float(muon_lr_mult)
        self.engram_lr_mult = float(engram_lr_mult)

    # ---------------------------------------------------------------- core
    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in (self.adamw, self.muon, self.sinkhorn):
            if opt is not None:
                opt.zero_grad(set_to_none=set_to_none)
        if self.engram is not None:
            self.engram.zero_grad()

    def step(self) -> None:
        for opt in (self.adamw, self.muon, self.sinkhorn):
            if opt is not None:
                opt.step()
        if self.engram is not None:
            self.engram.step()

    def accumulate(self) -> None:
        """Engram sparse-gradient harvest; call once per micro-batch after the backward."""
        if self.engram is not None:
            self.engram.accumulate()

    # ---------------------------------------------------------------- lr
    def set_lr(self, base_lr: float) -> None:
        self.base_lr = float(base_lr)
        if self.muon is not None:
            self.muon.set_lr(base_lr * self.muon_lr_mult)
        if self.adamw is not None:
            for group in self.adamw.param_groups:
                group["lr"] = base_lr * (self.engram_lr_mult
                                         if group.get("lr_mult_engram", False) else 1.0)
        if self.sinkhorn is not None:
            self.sinkhorn.set_lr(base_lr)
        if self.engram is not None:
            self.engram.set_lr(base_lr * self.engram_lr_mult)

    @property
    def param_groups(self):
        out = []
        for opt in (self.adamw, self.muon, self.sinkhorn):
            if opt is not None:
                out.extend(opt.param_groups)
        return out

    def params(self):
        out = []
        for group in self.param_groups:
            out.extend(group["params"])
        return out

    # ---------------------------------------------------------------- R79 memory / pre-warm
    @torch.no_grad()
    def ensure_states_(self) -> dict[str, int]:
        """Pre-allocate all optimizer state before graph capture (bytes per group).

        ``{'muon': B, 'sinkhorn': B, 'engram': B, 'adamw': 0}``.  No math, idempotent; intended to
        be called by the trainer/profile probe *before* ``torch.cuda.graph(...)`` so the post-capture
        optimizer-state allocation can be served from the pre-capture headroom (R79 accum64/b3).
        """
        out: dict[str, int] = {}
        for key, opt in (("muon", self.muon), ("sinkhorn", self.sinkhorn),
                         ("adamw", self.adamw)):
            if opt is not None and hasattr(opt, "ensure_states_"):
                out[key] = int(opt.ensure_states_())
            else:
                out[key] = 0
        out["engram"] = int(self.engram.ensure_states_()) if self.engram is not None else 0
        return out

    def state_bytes(self) -> dict[str, int]:
        """Current optimizer-state bytes per group (for the memory matrix)."""
        out = {}
        for key, opt in (("muon", self.muon), ("sinkhorn", self.sinkhorn),
                         ("adamw", self.adamw)):
            if opt is not None and hasattr(opt, "state_bytes"):
                out[key] = int(opt.state_bytes())
            else:
                out[key] = 0
        out["engram"] = int(self.engram.state_bytes()) if self.engram is not None else 0
        out["total"] = sum(out.values())
        return out

    def prewarm(self) -> dict[str, int]:
        """Compile/launch the env-enabled Triton fast kernels once (kills first-call bubbles).

        R79 item 3: the R78 "+83ms host bubble" was Triton JIT cold start on the first optimizer
        step, which polluted the first profile / A-B step.  Called automatically from
        :func:`build_optimizers` (before graph capture) unless ``prewarm=False``.
        """
        stats: dict[str, int] = {}
        if self.muon is not None and any(self.muon.fast.values()):
            stats["muon_launches"] = int(self.muon.prewarm())
        _of = _opt_fast_module()
        if _of is not None and torch.cuda.is_available():
            dims = []
            if _of.SINKHORN and self.sinkhorn is not None:
                dims.extend(sorted({int(p.shape[1]) for _, p in self.sinkhorn._named
                                    if p.ndim == 2}))
            if _of.SINKHORN_ENG and self.engram is not None:
                dims.extend(sorted({int(getattr(t, "dim", 0)) for t in self.engram.tables}))
            if dims:
                stats.update(_of.prewarm(dims=sorted(set(dims))))
        return stats

    # ---------------------------------------------------------------- checkpointing
    def state_dict(self, cpu: bool = True, include_engram: bool = True) -> dict:
        state = {"base_lr": self.base_lr, "muon_lr_mult": self.muon_lr_mult,
                 "engram_lr_mult": self.engram_lr_mult}
        for key, opt in (("muon", self.muon), ("adamw", self.adamw),
                         ("sinkhorn", self.sinkhorn)):
            if opt is None:
                state[key] = None
                continue
            sd = opt.state_dict()
            if cpu:
                sd = _state_to_cpu(sd)
            state[key] = sd
        if include_engram and self.engram is not None:
            state["engram"] = self.engram.state_dict(cpu=cpu)
        else:
            state["engram"] = None
        return state

    def load_state_dict(self, state: dict) -> None:
        for key, opt in (("muon", self.muon), ("adamw", self.adamw),
                         ("sinkhorn", self.sinkhorn)):
            saved = state.get(key)
            if opt is None or saved is None:
                continue
            opt.load_state_dict(saved)
            _move_optimizer_state(opt)
        if self.engram is not None and state.get("engram") is not None:
            self.engram.load_state_dict(state["engram"])
        self.base_lr = float(state.get("base_lr", self.base_lr))
        self.muon_lr_mult = float(state.get("muon_lr_mult", self.muon_lr_mult))
        self.engram_lr_mult = float(state.get("engram_lr_mult", self.engram_lr_mult))
        self.set_lr(self.base_lr)


def _move_optimizer_state(opt) -> None:
    """Torch only device-casts optimizer state for fused/capturable Adam; do it for everyone."""
    for p, st in opt.state.items():
        for k, v in list(st.items()):
            if isinstance(v, torch.Tensor) and v.device != p.device:
                st[k] = v.to(p.device)


def _state_to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _state_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_state_to_cpu(v) for v in obj)
    return obj


# =========================================================================== gate bias rule
def gate_bias_params(model) -> list[tuple[str, torch.Tensor]]:
    """The auxiliary-loss-free router biases (owned by the load-balancing rule, not by an optimizer)."""
    return [(n, p) for n, p in model.named_parameters() if is_gate_bias(n)]


@torch.no_grad()
def update_gate_bias(model, load_counts: dict[str, torch.Tensor] | None = None,
                     speed: float = 1e-3) -> dict[str, float]:
    """Loss-free balancing: ``b_i <- b_i - u * sign(c_i - mean(c))`` (overloaded expert -> lower bias).

    ``load_counts`` maps gate-module name -> per-expert count tensor collected during the step
    (see ``nanov41/v41_balance.py``); when None this is a no-op (nothing to balance).
    Returns the load imbalance of each gate (max/mean) for logging (host sync).
    """
    stats: dict[str, float] = {}
    if not load_counts:
        return stats
    for name, bias in gate_bias_params(model):
        gate_name = name.rsplit(".", 1)[0]
        counts = load_counts.get(gate_name)
        if counts is None or counts.numel() != bias.numel():
            continue
        c = counts.to(torch.float32)
        cbar = c.mean()
        bias.sub_(torch.sign(c - cbar), alpha=speed)
        mean = float(cbar)
        stats[gate_name] = float(c.max()) / mean if mean > 0 else 0.0
    return stats


__all__ = [
    # constants
    "NS_STEPS", "MUON_MOMENTUM", "MUON_WEIGHT_DECAY", "MUON_UPDATE_RMS",
    "ADAMW_BETAS", "ADAMW_EPS", "ADAMW_WEIGHT_DECAY",
    "SINKHORN_MOMENTUM", "SINKHORN_GAMMA", "SINKHORN_ITERS", "SINKHORN_TAU",
    "SINKHORN_EPS", "SINKHORN_WEIGHT_DECAY", "ENGRAM_LR_MULT", "GATE_BIAS_SUFFIXES",
    # R78 O5a fast-path knobs / helpers
    "MUON_FAST", "_muon_fast_flags", "_batched_newton_schulz_fused",
    "_launch_muon_momentum_fused", "_HAS_TRITON",
    # primitives
    "zeropower_via_newtonschulz5", "muon_direction",
    # optimizers
    "Muon", "SinkhornMomentum", "SinkhornEngram", "OptimizerBundle", "build_optimizers",
    # splitting helpers
    "classify_parameters", "is_gate_bias", "gate_bias_params", "update_gate_bias",
]

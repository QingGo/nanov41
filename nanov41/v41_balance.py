"""Sequence-level balance loss + auxiliary-loss-free gate-bias update (V4.1 report §4.2.2 / V3).

Why a *sequence*-level loss
---------------------------
A token-level router can pile every token of one sequence onto a few experts while the batch
average still looks balanced. The V3/V4.1 recipe therefore computes, per sequence ``b``::

    f_i = (1 / (k * S)) * #{positions in sequence b with expert i in their top-k}
    P_i = (1 / S) * sum_s scores[b, s, i]
    loss = weight * mean_b sum_i f_i * P_i                      (weight = 1e-4)

``f`` is a count and carries no gradient, so the gradient flows only through ``P`` and the
backward is a constant::

    d loss / d scores[b, s, i] = weight * f_i / (B * S)

There is consequently no reason to keep the ``[B, S, E]`` fp32 score matrix for backward.
``SeqBalanceFn`` exploits that: forward saves only ``f`` (``[B, E]``, tiny) and backward
returns the closed-form gradient, so the router's score matrix is never retained.

Auxiliary-loss-free bias update
-------------------------------
The correction bias ``gate.bias`` never enters the loss and never receives gradient
(``requires_grad_(False)``). Instead every gate counts how often each expert was selected
(``gate._v41_load``, a float32 buffer accumulated with ``index_add_``) and, per update step::

    bias -= speed * sign(count - count.mean())

overloaded experts get a lower routing bias, underloaded ones a higher one. Loss and bias rule
are independent switches: ``weight=0`` trains pure aux-free (no aux scalar is appended and the
aux path is not even built).

Graph-capture safety
--------------------
The wrapper only uses capture-safe tensor ops (``topk``, ``index_add_``, ``mean``, ``sum``):
no ``.item()``/``.tolist()``, no host synchronisation, no value-dependent Python branch. The
only host values inspected are the loss weight and the presence of the load buffer, both fixed
at capture time. ``model._v41_aux_losses`` is appended at capture time, so the captured scalars
are *not* replayed; use them for logging on eager steps only.

Call order (trainer)
--------------------
::

    moe_stacked.patch_model(model, M)            # optional, before install
    moe_triton.install(M)                        # class-level forward patches first
    v41_balance.install(model, M, weight=1e-4)   # instance-level wrap, LAST
    ...
    loss = ce_loss
    loss = loss + sum(v41_balance.take_aux_losses(model))
    loss.backward(); opt.step()
    # every N steps, outside graph capture:
    v41_balance.update_gate_bias(model, speed=1e-3)
    stats = v41_balance.collect_load_stats(model)
    v41_balance.zero_load_buffers(model)

The aux loss must be *added before backward* (it is a separate graph branch into the gate
scores); ``take_aux_losses`` clears the list so the next step starts empty.

Note: the sequence loss recomputes the gate scores from ``x`` (one extra ``[T,d] x [d,E]``
GEMM per MoE layer, the same shape the gate itself computes). That is the price of not
touching the reference ``Gate.forward``. Image-span routing (``gate.bias_vl``) is not modelled;
the nano port builds with ``vision_enabled=False``.

Public API
----------
``SeqBalanceFn`` / ``sequence_balance_loss(scores, indices, weight=1e-4)``
    fused balance loss over ``scores [b,s,E]`` and selected ``indices [b,s,k]``.
``gate_scores(gate, x)``
    ``[T, d] -> [T, E]`` fp32 gate scores, identical to ``Gate.forward``'s pre-bias scores.
``install(model, M, weight=1e-4) -> int``
    instance-level wrap of every ``M.MoE`` in ``model``; returns the number wrapped.
``take_aux_losses(model) -> list[Tensor]`` / ``clear_aux_losses(model) -> None``
``prepare_load_buffers(model) -> int`` / ``zero_load_buffers(model) -> int``
``update_gate_bias(model, speed=1e-3) -> int``
``collect_load_stats(model) -> {"max_over_mean": float, "min_over_mean": float}``
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Official V4.1 balance-loss weight (report §4.2.2).
AUX_LOSS_WEIGHT = 1e-4
# Attribute holding the per-model balance weight; fix it before graph capture.
_WEIGHT_ATTR = "_v41_aux_weight"
_LOSSES_ATTR = "_v41_aux_losses"
_LOAD_ATTR = "_v41_load"


# --------------------------------------------------------------------------- the loss itself
class SeqBalanceFn(torch.autograd.Function):
    """Sequence-level balance loss as one fused op.

    forward:  ``f_i = count_i / (k*S)`` (detached count), ``P_i = mean_s scores[b,s,i]``,
              ``loss = weight * mean_b sum_i f_i P_i``.
    backward: ``d loss / d scores = weight * f_i / (B*S)`` broadcast over ``s`` — the score
              matrix is never saved, only ``f`` (``[B, E]``).
    """

    @staticmethod
    def forward(ctx, scores: torch.Tensor, indices: torch.Tensor, weight: float) -> torch.Tensor:
        if scores.dim() != 3:
            raise ValueError(f"scores must be [b, s, E], got {tuple(scores.shape)}")
        b, s, e = scores.shape
        k = indices.shape[-1]
        # Per-sequence histogram without host sync: one flat index_add over [B*E].
        rows = torch.arange(b, device=scores.device, dtype=torch.long).unsqueeze(1) * e
        gidx = (indices.reshape(b, -1).long() + rows).reshape(-1)
        counts = torch.zeros(b * e, dtype=scores.dtype, device=scores.device)
        counts = counts.index_add(0, gidx, torch.ones(gidx.numel(), dtype=scores.dtype, device=scores.device))
        f = counts.view(b, e) / (k * s)
        p = scores.mean(dim=1)  # [b, e]; value only, deliberately not saved
        loss = (f * p).sum() / b * float(weight)
        ctx.save_for_backward(f)
        ctx.dims = (b, s)
        ctx.weight = float(weight)
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        (f,) = ctx.saved_tensors
        b, s = ctx.dims
        scale = grad_out * (ctx.weight / (b * s))
        d_scores = (f.unsqueeze(1) * scale).expand(b, s, f.shape[-1])
        return d_scores, None, None


def sequence_balance_loss(scores: torch.Tensor, indices: torch.Tensor, weight: float = AUX_LOSS_WEIGHT):
    """``scores [b,s,E]`` (differentiable) x ``indices [b,s,k]`` -> scalar balance loss."""
    return SeqBalanceFn.apply(scores, indices, float(weight))


def gate_scores(gate, x: torch.Tensor) -> torch.Tensor:
    """fp32 gate scores without the routing bias: ``[T, d]`` in, ``[T, E]`` out.

    Mirrors ``Gate.forward`` exactly (``linear(x.float(), weight.float()) / gate_temp`` then the
    configured nonlinearity), so ``topk(scores + gate.bias, gate.topk)`` reproduces the gate's
    expert selection.
    """
    scores = F.linear(x.float(), gate.weight.float()) / gate.gate_temp
    if gate.score_func == "softmax":
        return scores.softmax(dim=-1)
    if gate.score_func == "sigmoid":
        return scores.sigmoid()
    return F.softplus(scores).sqrt()


def _select(gate, scores: torch.Tensor) -> torch.Tensor:
    """Same selection rule as ``Gate.forward``: the bias steers top-k, the weights do not."""
    return torch.topk(scores + gate.bias, gate.topk, dim=-1).indices


# --------------------------------------------------------------------------- load counters
def _iter_gates(model):
    """Yield every gate the installer touched (fallback: ref ``Gate``-shaped modules)."""
    for mod in model.modules():
        if getattr(mod, "_v41_is_gate", False):
            yield mod
        elif type(mod).__name__ == "Gate" and hasattr(mod, "topk") and hasattr(mod, "score_func"):
            yield mod


def _ensure_load_buffer(gate) -> torch.Tensor:
    e = gate.weight.shape[0]
    buf = gate._buffers.get(_LOAD_ATTR)
    if buf is None:
        gate.register_buffer(
            _LOAD_ATTR, torch.zeros(e, dtype=torch.float32, device=gate.weight.device), persistent=False
        )
    elif buf.dtype != torch.float32 or buf.shape[0] != e or buf.device != gate.weight.device:
        # device/dtype moved (e.g. model.to(...)): re-allocate on the weight's device, fp32.
        gate._buffers[_LOAD_ATTR] = torch.zeros(e, dtype=torch.float32, device=gate.weight.device)
    return gate._buffers[_LOAD_ATTR]


def prepare_load_buffers(model) -> int:
    """Allocate ``gate._v41_load`` (float32, ``[n_routed_experts]``) on every gate. Idempotent."""
    n = 0
    for gate in _iter_gates(model):
        _ensure_load_buffer(gate)
        n += 1
    return n


def zero_load_buffers(model) -> int:
    """Reset every gate's selection counters (e.g. after logging)."""
    n = 0
    for gate in _iter_gates(model):
        buf = gate._buffers.get(_LOAD_ATTR)
        if buf is not None:
            buf.zero_()
        n += 1
    return n


def update_gate_bias(model, speed: float = 1e-3) -> int:
    """Aux-free rule: ``bias -= speed * sign(count - count.mean())`` per gate. Returns #gates.

    Host-side: call it on log/update steps, never inside a CUDA graph capture.
    """
    n = 0
    with torch.no_grad():
        for gate in _iter_gates(model):
            buf = gate._buffers.get(_LOAD_ATTR)
            if buf is None or buf.numel() == 0:
                continue
            load = buf.float()
            mean = load.mean()
            if float(mean) == 0.0:  # no forwards counted since zero_load_buffers
                continue
            gate.bias.sub_(speed * torch.sign(load - mean))
            n += 1
    return n


def collect_load_stats(model) -> dict:
    """``{"max_over_mean", "min_over_mean"}`` of the per-expert loads, aggregated over gates.

    ``max/mean == min/mean == 1`` is perfectly balanced; they are per-gate ratios (each layer
    balances itself), max over gates for the first, min over gates for the second. Gates with
    no counted selections are skipped; if none counted, both are 1.0. Host sync allowed here.
    """
    per_gate = []
    for gate in _iter_gates(model):
        buf = gate._buffers.get(_LOAD_ATTR)
        if buf is None or buf.numel() == 0:
            continue
        load = buf.float()
        mean = load.mean()
        if float(mean) <= 0.0:
            continue
        ratios = load / mean
        per_gate.append((float(ratios.max()), float(ratios.min())))
    if not per_gate:
        return {"max_over_mean": 1.0, "min_over_mean": 1.0}
    return {
        "max_over_mean": max(hi for hi, _ in per_gate),
        "min_over_mean": min(lo for _, lo in per_gate),
    }


# --------------------------------------------------------------------------- install
def _accumulate_load(gate, indices: torch.Tensor) -> None:
    """Capture-safe in-place counter: one ``index_add_`` on the gate's buffer."""
    buf = gate._buffers.get(_LOAD_ATTR)
    if buf is None:
        return
    buf.index_add_(
        0, indices.reshape(-1), torch.ones(indices.numel(), dtype=buf.dtype, device=buf.device)
    )


def _make_forward(orig, model):
    """Build the instance-level wrapper around the already-bound forward ``orig``."""

    def forward(self, x, image_mask=None):
        out = orig(x) if image_mask is None else orig(x, image_mask)
        gate = self.gate
        dim = gate.weight.shape[-1]
        lead = x.shape[:-1]
        seq = int(lead[-1]) if len(lead) else 1
        x2 = x.reshape(-1, dim)
        b = x2.shape[0] // seq
        scores = gate_scores(gate, x2)  # [T, E] fp32, differentiable
        with torch.no_grad():
            sel = _select(gate, scores)
            _accumulate_load(gate, sel)
        weight = getattr(model, _WEIGHT_ATTR, AUX_LOSS_WEIGHT)
        if weight != 0.0:  # host constant, fixed at capture time
            loss = SeqBalanceFn.apply(scores.reshape(b, seq, -1), sel.reshape(b, seq, -1), weight)
            model._v41_aux_losses.append(loss)
        return out

    return forward


def install(model, M, weight: float = AUX_LOSS_WEIGHT) -> int:
    """Wrap every ``M.MoE`` of ``model`` at the instance level and disable ``gate.bias`` grads.

    Must run *after* any class/instance-level forward patch (``moe_triton.install(M)``,
    ``moe_stacked.patch_model(model, M)``): the wrapper captures the currently bound forward.
    Idempotent; re-calling updates the loss weight. Returns the number of MoE modules wrapped.
    """
    if getattr(model, _LOSSES_ATTR, None) is None:
        model._v41_aux_losses = []
    setattr(model, _WEIGHT_ATTR, float(weight))

    # The correction bias is driven by sign(count - mean), never by the optimizer.
    for mod in model.modules():
        if isinstance(mod, M.Gate):
            mod.bias.requires_grad_(False)
            mod._v41_is_gate = True
    prepare_load_buffers(model)

    n = 0
    for mod in model.modules():
        if not isinstance(mod, M.MoE):
            continue
        if getattr(mod, "_v41_balance_installed", False):
            continue
        orig = mod.forward  # already bound: the patch in effect at install time
        mod.forward = _make_forward(orig, model).__get__(mod, type(mod))
        mod._v41_balance_installed = True
        n += 1
    return n


def take_aux_losses(model) -> list:
    """Return the aux scalars accumulated since the last call and clear the list."""
    losses = list(getattr(model, _LOSSES_ATTR, None) or ())
    clear_aux_losses(model)
    return losses


def clear_aux_losses(model) -> None:
    """Drop any pending aux scalars (they are graph branches: always take them before backward)."""
    losses = getattr(model, _LOSSES_ATTR, None)
    if losses is not None:
        losses.clear()

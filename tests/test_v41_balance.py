"""CPU tests for the sequence-level balance loss + aux-free gate-bias update (milestone M1).

Covers:
- the custom autograd Function against a naive loop reference (value + gradient + memory),
- gradcheck of the Function in float64,
- install() on a TINY CPU model: aux values, gate.weight grad contribution, bias grad off,
  load counters, bias update, load stats,
- wrapper composition with a class-level forward patch applied *before* install
  (the `moe_triton.forward_triton` situation) and with `moe_stacked`'s instance-level patch,
- router consistency: recomputed indices == `gate(x)` indices.

Run: python3 -m pytest tests/test_v41_balance.py -q
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from nanov41 import configs, v41_balance
from nanov41.model import init_random_weights, inner_forward
from nanov41.patched_env import import_patched

M = import_patched()

# TINY with *square* expert FFNs: moe_stacked's stacked-parameter layout only supports
# dim == moe_inter_dim, and one test below exercises that composition.
TINY_SQUARE = dict(configs.TINY, moe_inter_dim=configs.TINY["dim"])
TOP_K = configs.TINY["n_activated_experts"]  # 2
B, S = 2, 16
AUX_W = 1e-4


# --------------------------------------------------------------------------- helpers
def _build_tiny(cfg: dict | None = None):
    """TINY CPU model in fp32 (the reference builds bf16; fp32 keeps grad deltas exact)."""
    model = M.Transformer(M.ModelArgs(**dict(cfg or configs.TINY)))
    with torch.no_grad():
        for p in model.parameters():
            if p.is_floating_point():
                p.data = p.data.to(torch.float32)
        for buf in model.buffers():
            if buf.is_floating_point():
                buf.data = buf.data.to(torch.float32)
    init_random_weights(model, std=0.02, seed=0)
    model.eval()
    return model


def _ids(b: int = B, s: int = S, seed: int = 7) -> torch.Tensor:
    return torch.randint(0, configs.TINY["vocab_size"], (b, s), generator=torch.Generator().manual_seed(seed))


def _gates(model) -> list:
    return [m.gate for m in model.modules() if isinstance(m, M.MoE)]


def _ce(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(), ids[:, 1:].reshape(-1))


def _step(model, ids, aux_weight: float | None = None):
    """One forward; adds the aux scalars to the CE loss and returns (loss, aux). Clears the list."""
    if aux_weight is not None:
        model._v41_aux_weight = aux_weight
    logits = inner_forward(model, ids, full_logits=True)
    aux = v41_balance.take_aux_losses(model)
    loss = _ce(logits, ids)
    if aux:
        loss = loss + sum(aux)
    return loss, aux


def _naive_sequence_loss(scores: torch.Tensor, indices: torch.Tensor, weight: float) -> torch.Tensor:
    """Loop reference: f_i counted per sequence, P_i = mean_s scores[b,s,i]."""
    b, s, e = scores.shape
    k = indices.shape[-1]
    total = scores.new_zeros(())
    for bi in range(b):
        counts = torch.zeros(e, dtype=scores.dtype, device=scores.device)
        for si in range(s):
            for j in range(k):
                counts[indices[bi, si, j]] += 1
        f = counts / (k * s)
        p = scores[bi].mean(0)
        total = total + (f * p).sum()
    return weight * total / b


# --------------------------------------------------------------------------- the Function
def test_sequence_loss_matches_naive_reference():
    torch.manual_seed(0)
    b, s, e, k = 3, 5, 4, 2
    idx = torch.randint(0, e, (b, s, k))

    s_fused = torch.randn(b, s, e, requires_grad=True)
    s_naive = s_fused.detach().clone().requires_grad_(True)
    loss_fused = v41_balance.sequence_balance_loss(s_fused, idx, AUX_W)
    loss_naive = _naive_sequence_loss(s_naive, idx, AUX_W)
    loss_fused.backward()
    loss_naive.backward()

    assert loss_fused.ndim == 0
    assert torch.allclose(loss_fused.detach(), loss_naive.detach(), rtol=1e-6, atol=0)
    assert torch.allclose(s_fused.grad, s_naive.grad, rtol=1e-6, atol=0)

    # closed form: d loss / d scores[b, s, i] = weight * f_i / (B*S)
    flat = idx.reshape(b, -1)
    counts = torch.stack([torch.bincount(flat[i], minlength=e) for i in range(b)]).to(torch.float32)
    f = counts / (k * s)
    expected = (AUX_W * f / (b * s)).unsqueeze(1).expand(b, s, e)
    assert torch.allclose(s_fused.grad, expected, rtol=1e-6, atol=0)


def test_sequence_loss_saves_no_score_matrix():
    """The [b,s,E] fp32 scores must not be kept for backward; only f ([b,E]) is."""
    torch.manual_seed(0)
    b, s, e, k = 3, 5, 4, 2
    scores = torch.randn(b, s, e, requires_grad=True)
    idx = torch.randint(0, e, (b, s, k))
    saved: list[tuple] = []

    def pack(t):
        saved.append(tuple(t.shape))
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss = v41_balance.SeqBalanceFn.apply(scores, idx, AUX_W)
    loss.backward()

    assert (b, e) in saved, f"expected the detached f to be saved, got {saved}"
    assert (b, s, e) not in saved, f"score matrix must not be saved, got {saved}"


def test_sequence_loss_gradcheck_float64():
    torch.manual_seed(0)
    b, s, e, k = 2, 3, 4, 2
    idx = torch.randint(0, e, (b, s, k))
    scores = torch.randn(b, s, e, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda sc: v41_balance.sequence_balance_loss(sc, idx, AUX_W), (scores,), eps=1e-6, atol=1e-8
    )


# --------------------------------------------------------------------------- router
@pytest.mark.parametrize("score_func", ["sqrtsoftplus", "softmax", "sigmoid"])
def test_gate_scores_reproduce_gate_router(score_func):
    model = _build_tiny()
    gate = model.layers[0].ffn.gate
    gate.score_func = score_func
    dim = gate.weight.shape[-1]
    x = torch.randn(3, 5, dim)
    x2 = x.reshape(-1, dim)

    with torch.no_grad():
        _weights, idx_gate = gate(x2)
        scores = v41_balance.gate_scores(gate, x2)
        idx_module = torch.topk(scores + gate.bias, gate.topk, dim=-1).indices
        # independent re-derivation straight from the report's formula
        pre = x2.float() @ gate.weight.float().T / gate.gate_temp
        if score_func == "softmax":
            pre = pre.softmax(dim=-1)
        elif score_func == "sigmoid":
            pre = pre.sigmoid()
        else:
            pre = F.softplus(pre).sqrt()
        idx_naive = torch.topk(pre + gate.bias, gate.topk, dim=-1).indices

    assert torch.allclose(scores, pre, rtol=1e-5, atol=1e-6)
    assert torch.equal(idx_module, idx_gate)
    assert torch.equal(idx_naive, idx_gate)


# --------------------------------------------------------------------------- install / counters
def test_install_disables_bias_grad_and_counts_loads():
    model = _build_tiny()
    ids = _ids()
    n_layers = configs.TINY["n_layers"]
    gates = _gates(model)

    n = v41_balance.install(model, M, weight=AUX_W)
    assert n == n_layers == len(gates) == 4
    assert all(not g.bias.requires_grad for g in gates)
    assert all(g.weight.requires_grad for g in gates)
    for g in gates:
        buf = g._buffers.get("_v41_load")
        assert buf is not None and buf.dtype == torch.float32
        assert buf.shape == (g.weight.shape[0],) and not buf.requires_grad
    # a non-persistent buffer: checkpoints keep the official state_dict keys
    assert not any("_v41_load" in k for k in model.state_dict())

    model.eval()
    with torch.no_grad():
        logits = inner_forward(model, ids, full_logits=True)
        aux = v41_balance.take_aux_losses(model)

    assert torch.isfinite(logits).all()
    assert len(aux) == n_layers
    assert all(a.ndim == 0 and bool(torch.isfinite(a)) for a in aux)
    assert all(0.0 < float(a) < 1e-2 for a in aux), [float(a) for a in aux]
    # 2 sequences x 16 positions x top-2 selections per gate
    assert [int(g._v41_load.sum().item()) for g in gates] == [B * S * TOP_K] * n_layers

    # take clears, clear is safe on an empty list
    assert v41_balance.take_aux_losses(model) == []
    v41_balance.clear_aux_losses(model)

    stats = v41_balance.collect_load_stats(model)
    assert set(stats) == {"max_over_mean", "min_over_mean"}
    # aggregated over gates: max of the per-gate max/mean, min of the per-gate min/mean
    per_gate_max = [float(g._v41_load.max() / g._v41_load.mean()) for g in gates]
    per_gate_min = [float(g._v41_load.min() / g._v41_load.mean()) for g in gates]
    assert stats["max_over_mean"] == pytest.approx(max(per_gate_max))
    assert stats["min_over_mean"] == pytest.approx(min(per_gate_min))
    assert stats["max_over_mean"] >= 1.0 >= stats["min_over_mean"] >= 0.0

    assert v41_balance.zero_load_buffers(model) == n_layers
    assert all(int(g._v41_load.sum().item()) == 0 for g in gates)
    assert v41_balance.collect_load_stats(model) == {"max_over_mean": 1.0, "min_over_mean": 1.0}
    # nothing counted -> the bias rule is a no-op
    assert v41_balance.update_gate_bias(model) == 0


def test_aux_loss_backward_updates_gate_weight():
    model = _build_tiny()
    ids = _ids()
    gates = _gates(model)
    n_layers = configs.TINY["n_layers"]
    assert v41_balance.install(model, M, weight=AUX_W) == n_layers
    model.train()

    model.zero_grad(set_to_none=True)
    for _ in range(2):
        loss, aux = _step(model, ids)
        assert len(aux) == n_layers
        loss.backward()

    with_aux = []
    for g in gates:
        assert g.weight.grad is not None and bool(torch.isfinite(g.weight.grad).all())
        assert float(g.weight.grad.abs().sum()) > 0.0
        with_aux.append(g.weight.grad.detach().clone())
    # two forwards accumulate two rounds of selected counts
    assert [int(g._v41_load.sum().item()) for g in gates] == [2 * B * S * TOP_K] * n_layers

    # weight == 0 -> no aux branch at all, so gate.weight grads must differ
    model._v41_aux_weight = 0.0
    model.zero_grad(set_to_none=True)
    for _ in range(2):
        loss, aux = _step(model, ids)
        assert aux == [], "weight=0 must not build the aux branch"
        loss.backward()

    for g, g_aux in zip(gates, with_aux):
        grad = g.weight.grad
        assert grad is not None and bool(torch.isfinite(grad).all())
        assert float((grad - g_aux).abs().max()) > 1e-9, "aux path did not reach gate.weight.grad"


def test_update_gate_bias_moves_overloaded_expert_negative():
    model = _build_tiny()
    assert v41_balance.install(model, M, weight=AUX_W) == configs.TINY["n_layers"]
    gates = _gates(model)
    e = configs.TINY["n_routed_experts"]
    biased = torch.tensor([10.0] + [0.0] * (e - 1))  # one overloaded expert, mean = 10/e
    speed = 1e-2

    with torch.no_grad():
        for g in gates:
            g._v41_load.copy_(biased)
            g.bias.zero_()

    stats = v41_balance.collect_load_stats(model)
    assert stats["max_over_mean"] == pytest.approx(e)
    assert stats["min_over_mean"] == pytest.approx(0.0)

    assert v41_balance.update_gate_bias(model, speed=speed) == len(gates)
    for g in gates:
        assert not g.bias.requires_grad
        assert float(g.bias[0]) == pytest.approx(-speed, rel=1e-5, abs=1e-8)
        assert all(float(b) == pytest.approx(speed, rel=1e-5, abs=1e-8) for b in g.bias[1:])
        assert float(g.bias[0]) < 0.0


# --------------------------------------------------------------------------- composition
def test_wrapper_composes_with_class_level_patch():
    """`moe_triton.install(M)` replaces M.MoE.forward at class level; install() runs after it."""
    model = _build_tiny()
    ids = _ids()
    seen: list[int] = []

    def patched_forward(self, x, image_mask=None):
        seen.append(self.layer_id)
        return self.shared_experts(x.reshape(-1, self.dim)).reshape(x.shape)

    prev = M.MoE.forward
    M.MoE.forward = patched_forward
    try:
        assert v41_balance.install(model, M, weight=AUX_W) == configs.TINY["n_layers"]
        model.eval()
        with torch.no_grad():
            inner_forward(model, ids, full_logits=True)
            aux = v41_balance.take_aux_losses(model)
    finally:
        M.MoE.forward = prev

    # the wrapper captured the *patched* bound forward and still computes the aux scalars
    assert sorted(seen) == [0, 1, 2, 3]
    assert len(aux) == configs.TINY["n_layers"]
    assert all(bool(torch.isfinite(a)) for a in aux)

    # capture-by-value: restoring the class forward does not change the instance's wrapper
    model.eval()
    with torch.no_grad():
        inner_forward(model, ids, full_logits=True)
        assert len(v41_balance.take_aux_losses(model)) == configs.TINY["n_layers"]
    assert len(seen) == 8


def test_wrapper_composes_with_moe_stacked():
    from nanov41 import moe_stacked

    model = _build_tiny(TINY_SQUARE)
    ids = _ids()
    n = moe_stacked.patch_model(model, M)  # instance-level forward, before install
    assert n == configs.TINY["n_layers"]
    assert v41_balance.install(model, M, weight=AUX_W) == n

    model.eval()
    with torch.no_grad():
        inner_forward(model, ids, full_logits=True)
        aux = v41_balance.take_aux_losses(model)
    assert len(aux) == n and all(bool(torch.isfinite(a)) for a in aux)

    model.zero_grad(set_to_none=True)
    loss, aux = _step(model, ids)
    assert len(aux) == n
    loss.backward()
    gates = _gates(model)
    assert all(g.weight.grad is not None and bool(torch.isfinite(g.weight.grad).all()) for g in gates)
    assert all(int(g._v41_load.sum().item()) == 2 * B * S * TOP_K for g in gates)


def test_nanov41_bf16_training_path():
    """Official bf16 build: fp32 gate bias / load counters, aux grad reaches the bf16 gate weight."""
    from nanov41.model import NanoV41

    model = NanoV41(configs.TINY, param_dtype="bf16")
    init_random_weights(model, seed=0)
    n_layers = configs.TINY["n_layers"]
    assert v41_balance.install(model, M, weight=AUX_W) == n_layers

    ids = _ids(b=2, s=8)
    logits = model(ids)
    aux = v41_balance.take_aux_losses(model)
    assert len(aux) == n_layers and all(bool(torch.isfinite(a)) for a in aux)
    (logits.float().sum() + sum(aux)).backward()

    gate = model.inner.layers[0].ffn.gate
    assert gate.weight.dtype == torch.bfloat16
    assert gate.bias.dtype == torch.float32 and not gate.bias.requires_grad
    assert gate.weight.grad is not None and bool(torch.isfinite(gate.weight.grad).all())
    assert float(gate.weight.grad.abs().sum()) > 0.0
    buf = gate._buffers["_v41_load"]
    assert buf.dtype == torch.float32 and int(buf.sum().item()) == 2 * 8 * TOP_K
    # bias update works on the official dtype mix
    with torch.no_grad():
        gate._v41_load.copy_(torch.tensor([8.0, 0.0, 0.0, 0.0]))
        gate.bias.zero_()
    assert v41_balance.update_gate_bias(model, speed=1e-3) == n_layers
    assert float(gate.bias[0]) < 0.0

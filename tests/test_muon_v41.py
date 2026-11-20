"""Numerical verification of the DeepSeek-V4.1 optimizer suite (``nanov41/muon.py``).

Spec: V4.1 report §2.5 / Algorithm 1.  The momentum convention used by both Muon and
Sinkhorn is the "lerp" form

    M_t  = beta * M_{t-1} + (1 - beta) * G_t
    G~_t = beta * M_t + (1 - beta) * G_t           (Nesterov)

Reference implementations below are written independently from ``nanov41/muon.py`` (plain
loops over explicitly cloned tensors, spec constants copied from the report) so that a
shared coding bug is unlikely; the assertions compare final *parameter values*.

Report constants (hard-coded here on purpose, not imported):
  Muon    : NS steps 5, (a, b, c) = (3.4445, -4.7750, 2.0315), ||X|| + 1e-7, update RMS 0.18,
            beta 0.95, decoupled weight decay 0.1, head-wise for Q weights.
  Sinkhorn: K = 11 (odd), tau = 1e-3, eps = 1e-20, gamma = 0.18, Delta = sqrt(n) * U,
            no weight decay.
  AdamW   : betas (0.9, 0.95), eps 1e-20, wd 0.1 for norm weights / 0 for the rest.
"""

from __future__ import annotations

import copy
import math

import pytest
import torch

import nanov41.configs as configs
import nanov41.muon as mu

TINY_CFG = dict(configs.TINY)

NS_STEPS = 5
NS_A, NS_B, NS_C = 3.4445, -4.7750, 2.0315
MUON_RMS = 0.18
SINKHORN_GAMMA = 0.18
SINKHORN_ITERS = 11
SINKHORN_TAU = 1e-3
SINKHORN_EPS = 1e-20
MOMENTUM = 0.95


# =========================================================================== independent references
def ref_newton_schulz_direction(G: torch.Tensor, *, steps: int = NS_STEPS,
                                rms: float = MUON_RMS, eps: float = 1e-7) -> torch.Tensor:
    """Report Muon direction for one 2-D matrix: NS-orthogonalise, then scale to RMS ``rms``."""
    X = G.detach().clone().to(torch.float32)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = NS_B * A + NS_C * (A @ A)
        X = NS_A * X + B @ X
    if transposed:
        X = X.T
    return X * (rms / X.square().mean().sqrt())


def ref_sinkhorn_step(grad: torch.Tensor, momentum: torch.Tensor, *, beta: float, lr: float,
                      gamma: float = SINKHORN_GAMMA, iters: int = SINKHORN_ITERS,
                      tau: float = SINKHORN_TAU, eps: float = SINKHORN_EPS
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """Algorithm 1 verbatim for one 2-D matrix.

    Returns ``(new_momentum, update)`` with ``update = lr * gamma * sqrt(n) * U`` and no
    weight decay.
    """
    G = grad.detach().clone().to(torch.float32)
    M = beta * momentum + (1.0 - beta) * G
    Ge = beta * M + (1.0 - beta) * G
    U = Ge.clone()
    rho = U.norm(dim=1)
    rho_bar = rho.mean()
    U[rho <= tau * rho_bar] = 0.0
    for k in range(1, iters + 1):
        if k % 2 == 1:
            U = U / (U.norm(dim=1, keepdim=True) + eps)
        else:
            U = U / (U.norm(dim=0, keepdim=True) + eps)
    return M, lr * gamma * math.sqrt(G.size(1)) * U


# =========================================================================== test doubles
class FakeTable:
    """Minimal stand-in for ``nanov41.nano_engram.NanoEngramTable``'s sparse-grad API."""

    def __init__(self, rows: int, cols: int, dtype: torch.dtype = torch.float32):
        self.weight = torch.randn(rows, cols).to(dtype)
        self._pending: tuple[torch.Tensor, torch.Tensor] | None = None

    def set_grad(self, grad: torch.Tensor) -> None:
        """Pretend every row was touched by the forward pass (dense gradient)."""
        self._pending = (torch.arange(grad.size(0), dtype=torch.long),
                         grad.detach().clone().to(torch.float32))

    def collect_row_grads(self) -> None:
        pass

    def merged_row_grads(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._pending is None:
            return (torch.empty(0, dtype=torch.long),
                    torch.empty(0, self.weight.size(1), dtype=torch.float32))
        return self._pending

    def zero_row_grad(self) -> None:
        self._pending = None


class FakeModel:
    """Duck-typed model: ``classify_parameters`` only needs ``named_parameters()``."""

    def __init__(self, params: list[tuple[str, torch.nn.Parameter]]):
        self._params = list(params)

    def named_parameters(self):
        return list(self._params)


def _tensor_params(seed: int, specs: dict[str, tuple[int, ...]]) -> dict[str, torch.nn.Parameter]:
    gen = torch.Generator().manual_seed(seed)
    return {name: torch.nn.Parameter(torch.randn(shape, generator=gen))
            for name, shape in specs.items()}


def _set_grads(params: list[tuple[str, torch.nn.Parameter]], seed: int) -> None:
    gen = torch.Generator().manual_seed(seed)
    for _, p in params:
        p.grad = torch.randn(p.shape, generator=gen)


@pytest.fixture(scope="module")
def tiny():
    from nanov41.patched_env import import_patched

    M = import_patched()
    args = M.ModelArgs(**TINY_CFG)
    torch.manual_seed(0)
    model = M.Transformer(args)
    return M, model, args


# =========================================================================== constants
def test_report_constants():
    assert mu.NS_STEPS == 5
    assert (mu.NS_A, mu.NS_B, mu.NS_C) == (3.4445, -4.7750, 2.0315)
    assert mu.MUON_MOMENTUM == pytest.approx(0.95)
    assert mu.MUON_WEIGHT_DECAY == pytest.approx(0.1)
    assert mu.MUON_UPDATE_RMS == pytest.approx(0.18)
    assert mu.SINKHORN_MOMENTUM == pytest.approx(0.95)
    assert mu.SINKHORN_GAMMA == pytest.approx(0.18)
    assert mu.SINKHORN_ITERS == 11
    assert mu.SINKHORN_TAU == pytest.approx(1e-3)
    assert mu.SINKHORN_EPS == pytest.approx(1e-20)
    assert mu.SINKHORN_WEIGHT_DECAY == 0.0
    assert mu.ADAMW_BETAS == (0.9, 0.95)
    assert mu.ADAMW_EPS == pytest.approx(1e-20)
    assert mu.ADAMW_WEIGHT_DECAY == pytest.approx(0.1)


# =========================================================================== Sinkhorn (Algorithm 1)
def test_sinkhorn_momentum_matches_algorithm1():
    """Three steps of Algorithm 1 (Nesterov + tau mask + gamma*sqrt(n), no weight decay)."""
    torch.manual_seed(0)
    rows, cols = 16, 12
    beta, lr = MOMENTUM, 0.1
    init = torch.randn(rows, cols)
    p = torch.nn.Parameter(init.clone())
    opt = mu.SinkhornMomentum([("embed.weight", p)], lr=lr)
    assert opt.gamma == pytest.approx(SINKHORN_GAMMA)
    assert opt.K == SINKHORN_ITERS

    momentum = torch.zeros(rows, cols)
    want = init.clone()
    for step in range(3):
        grad = torch.randn(rows, cols)
        grad[3] *= 1e-8  # rho_3 <= tau * rho_bar at every step: must be tau-masked
        p.grad = grad.clone()
        opt.step()
        momentum, update = ref_sinkhorn_step(grad, momentum, beta=beta, lr=lr)
        want = want - update
        assert torch.allclose(p.detach(), want, atol=1e-6, rtol=0), f"step {step}"
        assert torch.allclose(opt.state[p]["momentum"], momentum, atol=1e-6, rtol=0), step
    # the tau-masked row was exactly frozen for all three steps
    assert torch.equal(p.detach()[3], init[3])


def test_sinkhorn_rejects_even_iteration_count():
    torch.manual_seed(10)
    p = torch.nn.Parameter(torch.randn(6, 5))
    opt = mu.SinkhornMomentum([("embed.weight", p)], lr=0.1, sinkhorn_iters=10)
    p.grad = torch.randn(6, 5)
    with pytest.raises(ValueError, match="odd"):
        opt.step()


def test_sinkhorn_near_zero_row_gets_exactly_zero_update():
    torch.manual_seed(11)
    rows, cols = 8, 5
    init = torch.randn(rows, cols)
    p = torch.nn.Parameter(init.clone())
    opt = mu.SinkhornMomentum([("embed.weight", p)], lr=0.1, momentum=0.0)
    grad = torch.randn(rows, cols)
    grad[2] *= 1e-9
    # the mask condition the implementation under test should apply
    rho = grad.norm(dim=1)
    assert rho[2] <= SINKHORN_TAU * rho.mean()
    p.grad = grad
    opt.step()
    assert torch.equal(p.detach()[2], init[2])
    assert (p.detach()[2] - init[2]).abs().max().item() == 0.0
    assert not torch.equal(p.detach()[0], init[0])


# =========================================================================== Muon
def test_muon_beta_zero_matches_reference_direction_and_rms():
    """With beta=0 and no wd the update is lr * NS-direction rescaled to RMS 0.18."""
    torch.manual_seed(20)
    shapes = {"rows_gt_cols": (9, 4), "cols_gt_rows": (6, 8)}
    params = _tensor_params(21, shapes)
    grads = {name: torch.randn_like(p) for name, p in params.items()}
    init = {name: p.detach().clone() for name, p in params.items()}
    opt = mu.Muon(list(params.items()), lr=0.1, momentum=0.0, weight_decay=0.0)
    for name, p in params.items():
        p.grad = grads[name].clone()
    opt.step()
    for name, p in params.items():
        want = init[name] - 0.1 * ref_newton_schulz_direction(grads[name])
        assert torch.allclose(p.detach(), want, atol=1e-6, rtol=0), name
        direction = (init[name] - p.detach()) / 0.1
        rms = direction.square().mean().sqrt().item()
        assert abs(rms - 0.18) <= 1e-3, (name, rms)


def test_muon_headwise_splits_q_weight():
    torch.manual_seed(22)
    heads, per_head, cols = 4, 6, 5  # per_head > cols exercises the transpose path
    name = "layers.0.attn.wq_b.weight"
    init = torch.randn(heads * per_head, cols)
    grad = torch.randn(heads * per_head, cols)
    p_head = torch.nn.Parameter(init.clone())
    opt_head = mu.Muon([(name, p_head)], lr=0.1, momentum=0.0, weight_decay=0.0,
                       headwise={name: heads})
    p_head.grad = grad.clone()
    opt_head.step()
    for h in range(heads):
        sl = slice(h * per_head, (h + 1) * per_head)
        want = init[sl] - 0.1 * ref_newton_schulz_direction(grad[sl])
        assert torch.allclose(p_head.detach()[sl], want, atol=1e-6, rtol=0), h
        direction = (init[sl] - p_head.detach()[sl]) / 0.1
        assert abs(direction.square().mean().sqrt().item() - 0.18) <= 1e-3, h
    # without head-wise splitting the whole [H*m, n] matrix is orthogonalised as one matrix
    p_plain = torch.nn.Parameter(init.clone())
    opt_plain = mu.Muon([(name, p_plain)], lr=0.1, momentum=0.0, weight_decay=0.0)
    p_plain.grad = grad.clone()
    opt_plain.step()
    assert (p_plain.detach() - p_head.detach()).abs().max().item() > 1e-3
    # head count must divide the rows
    with pytest.raises(ValueError, match="head-wise"):
        mu.Muon([(name, torch.nn.Parameter(torch.randn(heads * per_head, cols)))],
                headwise={name: heads + 1})


def test_muon_nesterov_convention():
    """One step from a non-zero momentum buffer, compared with manual M_t / G~_t.

    The buffer is pre-seeded because Newton-Schulz (and Sinkhorn) are invariant to a positive
    scalar rescaling: from a cold start ``G~ = (1-beta)(1+beta) G`` is just a positive multiple
    of ``G``, so every momentum convention would give the same direction.  With ``M_0 != 0``
    the update distinguishes ``G~ = beta*M_t + (1-beta)*G_t`` from both ``G~ = M_t`` and the
    unscaled ``buf = beta*buf + g`` convention.
    """
    torch.manual_seed(23)
    shape = (6, 8)
    init = torch.randn(shape)
    grad = torch.randn(shape)
    momentum0 = torch.randn(shape)
    p = torch.nn.Parameter(init.clone())
    beta, lr = 0.9, 0.1
    opt = mu.Muon([("w", p)], lr=lr, momentum=beta, weight_decay=0.0)
    opt.state[p]["momentum"] = momentum0.clone()  # resumed optimizer state
    p.grad = grad.clone()
    opt.step()
    momentum = beta * momentum0 + (1.0 - beta) * grad   # M_t
    ge = beta * momentum + (1.0 - beta) * grad          # Nesterov G~
    want = init - lr * ref_newton_schulz_direction(ge)
    assert torch.allclose(p.detach(), want, atol=1e-6, rtol=0)
    assert torch.allclose(opt.state[p]["momentum"], momentum, atol=1e-7, rtol=0)
    # genuinely discriminating: plain momentum (G~ = M_t) gives a different update here
    alt = init - lr * ref_newton_schulz_direction(momentum)
    assert (want - alt).abs().max().item() > 1e-3


def test_muon_decoupled_weight_decay():
    torch.manual_seed(24)
    shape = (5, 7)
    init = torch.randn(shape)
    grad = torch.randn(shape)
    p = torch.nn.Parameter(init.clone())
    lr, wd = 0.1, 0.1
    opt = mu.Muon([("w", p)], lr=lr, momentum=0.0, weight_decay=wd)
    p.grad = grad.clone()
    opt.step()
    want = init * (1.0 - lr * wd) - lr * ref_newton_schulz_direction(grad)
    assert torch.allclose(p.detach(), want, atol=1e-6, rtol=0)
    # weight decay is decoupled: the direction is the NS direction of the raw gradient
    p_plain = torch.nn.Parameter(init.clone())
    p_plain.grad = grad.clone()
    opt_plain = mu.Muon([("w", p_plain)], lr=lr, momentum=0.0, weight_decay=0.0)
    opt_plain.step()
    assert torch.allclose(p_plain.detach(), init - lr * ref_newton_schulz_direction(grad),
                          atol=1e-6, rtol=0)


def test_muon_stacked_3d_is_treated_as_independent_matrices():
    torch.manual_seed(25)
    n_mats, rows, cols = 3, 7, 4
    init = torch.randn(n_mats, rows, cols)
    grad = torch.randn(n_mats, rows, cols)
    p = torch.nn.Parameter(init.clone())
    opt = mu.Muon([("stacked", p)], lr=0.1, momentum=0.0, weight_decay=0.0)
    p.grad = grad.clone()
    opt.step()
    want = init.clone()
    for e in range(n_mats):
        want[e] = init[e] - 0.1 * mu.muon_direction(grad[e])
    assert torch.allclose(p.detach(), want, atol=1e-6, rtol=0)
    for e in range(n_mats):
        direction = (init[e] - p.detach()[e]) / 0.1
        assert abs(direction.square().mean().sqrt().item() - 0.18) <= 1e-3, e


# =========================================================================== parameter split / LR
def test_classify_parameters_tiny_model(tiny):
    M, model, args = tiny
    classes = mu.classify_parameters(model, TINY_CFG)
    named = dict(model.named_parameters())
    requires_grad = {name for name, p in named.items() if p.requires_grad}

    seen = [name for key in ("muon", "sinkhorn", "adamw_wd", "adamw_nowd")
            for name, _ in classes[key]]
    seen += list(classes["skipped"])
    assert len(seen) == len(set(seen)) == len(requires_grad)
    assert set(seen) == requires_grad

    # aux-free gate biases are never optimizer-owned
    gate_names = {name for name in named if mu.is_gate_bias(name)}
    assert gate_names
    assert set(classes["skipped"]) == gate_names
    assert len(classes["skipped"]) == args.n_layers  # vision disabled => no gate.bias_vl

    # Engram tables / token embedding / prediction head use Sinkhorn
    sinkhorn_names = {name for name, _ in classes["sinkhorn"]}
    assert {"embed.weight", "head.weight"} <= sinkhorn_names

    # shape-based routing
    for name, p in classes["muon"]:
        assert p.ndim >= 2, name
    for name, p in classes["sinkhorn"]:
        assert p.ndim == 2, name
    for name, p in classes["adamw_wd"]:
        assert p.ndim == 1 and name.endswith("norm.weight"), name
    for name, p in classes["adamw_nowd"]:
        assert p.ndim == 1 and not name.endswith("norm.weight"), name

    # head-wise Q splitting follows the model's head counts
    headwise = classes["headwise"]
    assert set(headwise) == {name for name in named if name.endswith("wq_b.weight")}
    for name, heads in headwise.items():
        assert name.endswith("wq_b.weight")
        expected = args.index_n_heads if "indexer" in name else args.n_heads
        assert heads == expected, (name, heads, expected)
        assert named[name].shape[0] % heads == 0
    assert sum(1 for name in headwise if "indexer" in name) == \
        sum(1 for name in named if "indexer.wq_b" in name)
    assert mu.gate_bias_params(model)  # non-empty


def test_build_optimizers_tiny_model(tiny):
    M, model, args = tiny
    classes = mu.classify_parameters(model, TINY_CFG)
    bundle = mu.build_optimizers(model, TINY_CFG, base_lr=1e-3)
    assert bundle.muon is not None and bundle.sinkhorn is not None and bundle.adamw is not None
    assert bundle.engram is None
    assert bundle.muon.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert bundle.sinkhorn.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert bundle.muon.headwise == classes["headwise"]
    assert bundle.adamw.defaults["betas"] == (0.9, 0.95)
    assert bundle.adamw.defaults["eps"] == pytest.approx(1e-20)

    # every trainable param is owned exactly once; gate.bias is excluded
    owned = [id(p) for p in bundle.params()]
    assert len(owned) == len(set(owned))
    gate_ids = {id(p) for name, p in model.named_parameters() if mu.is_gate_bias(name)}
    assert gate_ids and not (set(owned) & gate_ids)
    expected = {id(p) for name, p in model.named_parameters()
                if p.requires_grad and not mu.is_gate_bias(name)}
    assert set(owned) == expected

    # AdamW: norm weights get wd 0.1, everything else wd 0
    wd_of = {id(p): group["weight_decay"]
             for group in bundle.adamw.param_groups for p in group["params"]}
    for name, p in classes["adamw_wd"]:
        assert wd_of[id(p)] == pytest.approx(0.1), name
    for name, p in classes["adamw_nowd"]:
        assert wd_of[id(p)] == pytest.approx(0.0), name

    # set_lr scales Muon / Sinkhorn / AdamW groups from the shared base (no Engram in TINY)
    bundle.set_lr(2e-3)
    assert bundle.base_lr == pytest.approx(2e-3)
    assert bundle.muon.param_groups[0]["lr"] == pytest.approx(2e-3)
    assert bundle.sinkhorn.param_groups[0]["lr"] == pytest.approx(2e-3)
    assert all(group["lr"] == pytest.approx(2e-3) for group in bundle.adamw.param_groups)


def test_engram_lr_mult_5x():
    """Engram-module params get 5x base lr: Muon lr_mults, AdamW group and SinkhornEngram."""
    torch.manual_seed(30)
    engram_muon = torch.nn.Parameter(torch.randn(4, 4))
    plain_muon = torch.nn.Parameter(torch.randn(4, 4))
    plain_1d = torch.nn.Parameter(torch.randn(4))
    engram_1d = torch.nn.Parameter(torch.randn(4))
    norm = torch.nn.Parameter(torch.ones(4))
    embed = torch.nn.Parameter(torch.randn(6, 4))
    model = FakeModel([
        ("layers.0.engram.wkv.weight", engram_muon),
        ("layers.0.attn.wq_a.weight", plain_muon),
        ("embed.weight", embed),
        ("layers.0.attn.attn_sink", plain_1d),
        ("layers.0.engram.scale", engram_1d),
        ("layers.0.attn_norm.weight", norm),
        ("layers.0.ffn.gate.bias", torch.nn.Parameter(torch.zeros(4))),
    ])
    table = FakeTable(6, 4)
    bundle = mu.build_optimizers(model, {"n_heads": 4, "index_n_heads": 2},
                                 base_lr=1e-3, engram_tables=[table], muon_wd=0.0)
    assert bundle.muon is not None and bundle.sinkhorn is not None
    assert bundle.adamw is not None and bundle.engram is not None
    assert bundle.muon.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert bundle.muon.lr_mults == {"layers.0.engram.wkv.weight": 5.0}
    assert bundle.sinkhorn.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert bundle.engram.lr == pytest.approx(5e-3)

    # one AdamW group per weight-decay class; only the Engram group carries the 5x lr
    wd_groups = [g for g in bundle.adamw.param_groups
                 if g["weight_decay"] == pytest.approx(0.1)]
    nowd_groups = [g for g in bundle.adamw.param_groups
                   if g["weight_decay"] == 0.0 and not g.get("lr_mult_engram", False)]
    engram_groups = [g for g in bundle.adamw.param_groups if g.get("lr_mult_engram", False)]
    assert len(wd_groups) == len(nowd_groups) == len(engram_groups) == 1
    assert wd_groups[0]["lr"] == pytest.approx(1e-3)
    assert nowd_groups[0]["lr"] == pytest.approx(1e-3)
    assert engram_groups[0]["lr"] == pytest.approx(5e-3)

    # functional check: identical gradients -> Engram update is exactly 5x the plain update
    g_muon = torch.randn(4, 4)
    g_1d = torch.randn(4)
    init_engram_muon = engram_muon.detach().clone()
    init_plain_muon = plain_muon.detach().clone()
    init_engram_1d = engram_1d.detach().clone()
    init_plain_1d = plain_1d.detach().clone()
    engram_muon.grad = g_muon.clone()
    plain_muon.grad = g_muon.clone()
    engram_1d.grad = g_1d.clone()
    plain_1d.grad = g_1d.clone()
    norm.grad = torch.ones_like(norm)
    embed.grad = torch.randn_like(embed)
    table.set_grad(torch.randn(6, 4))
    bundle.accumulate()
    bundle.step()
    muon_engram = (init_engram_muon - engram_muon.detach()).square().mean().sqrt().item()
    muon_plain = (init_plain_muon - plain_muon.detach()).square().mean().sqrt().item()
    assert muon_engram / muon_plain == pytest.approx(5.0, rel=1e-4)
    adamw_engram = (init_engram_1d - engram_1d.detach()).abs().mean().item()
    adamw_plain = (init_plain_1d - plain_1d.detach()).abs().mean().item()
    assert adamw_engram / adamw_plain == pytest.approx(5.0, rel=1e-4)

    # set_lr keeps the multipliers
    bundle.set_lr(2e-3)
    assert bundle.muon.param_groups[0]["lr"] == pytest.approx(2e-3)
    assert bundle.sinkhorn.param_groups[0]["lr"] == pytest.approx(2e-3)
    assert wd_groups[0]["lr"] == pytest.approx(2e-3)
    assert nowd_groups[0]["lr"] == pytest.approx(2e-3)
    assert engram_groups[0]["lr"] == pytest.approx(1e-2)
    assert bundle.engram.lr == pytest.approx(1e-2)
    assert bundle.muon.lr_mults == {"layers.0.engram.wkv.weight": 5.0}


# =========================================================================== checkpointing
def test_optimizer_bundle_state_dict_roundtrip():
    torch.manual_seed(40)
    params = [
        ("layers.0.attn.wq_b.weight", torch.nn.Parameter(torch.randn(6, 4))),
        ("layers.0.attn.wq_a.weight", torch.nn.Parameter(torch.randn(5, 4))),
        ("layers.0.ffn.experts.0.w1.weight", torch.nn.Parameter(torch.randn(6, 4))),
        ("embed.weight", torch.nn.Parameter(torch.randn(7, 4))),
        ("head.weight", torch.nn.Parameter(torch.randn(5, 4))),
        ("layers.0.attn_norm.weight", torch.nn.Parameter(torch.randn(4))),
        ("layers.0.attn.attn_sink", torch.nn.Parameter(torch.randn(4))),
    ]
    model = FakeModel(params)
    bundle = mu.build_optimizers(model, {"n_heads": 2, "index_n_heads": 2}, base_lr=1e-2)
    assert bundle.muon is not None and bundle.sinkhorn is not None and bundle.adamw is not None
    assert bundle.engram is None
    assert bundle.muon.headwise == {"layers.0.attn.wq_b.weight": 2}

    _set_grads(params, 1)
    bundle.step()
    _set_grads(params, 2)
    bundle.step()
    state = copy.deepcopy(bundle.state_dict())  # deepcopy: state_dict holds live tensors on CPU
    assert state["muon"]["state"] and state["adamw"]["state"] and state["sinkhorn"]["state"]
    assert state["engram"] is None
    params_before = {name: p.detach().clone() for name, p in params}

    # one probe step, recording the result we must reproduce after restoring the snapshot
    _set_grads(params, 3)
    bundle.step()
    expected = {name: p.detach().clone() for name, p in params}

    bundle.load_state_dict(copy.deepcopy(state))
    with torch.no_grad():
        for name, p in params:
            p.copy_(params_before[name])
    _set_grads(params, 3)
    bundle.step()
    for name, p in params:
        assert torch.equal(p.detach(), expected[name]), name


def test_sinkhorn_engram_state_dict_roundtrip():
    torch.manual_seed(41)
    table = FakeTable(8, 5)
    eng = mu.SinkhornEngram([table], lr=0.05)
    g1 = torch.randn(8, 5)
    g2 = torch.randn(8, 5)
    table.set_grad(g1)
    eng.accumulate()
    eng.step()
    weight_after_step1 = table.weight.clone()
    saved = copy.deepcopy(eng.state_dict())
    assert saved["lr"] == pytest.approx(0.05)
    assert saved["momentum"] == pytest.approx(0.95)
    assert saved["gamma"] == pytest.approx(0.18)
    assert saved["K"] == 11 and saved["tau"] == pytest.approx(1e-3)
    assert len(saved["tables"]) == 1 and saved["tables"][0] is not None
    assert tuple(saved["tables"][0]["momentum"].shape) == (8, 5)
    momentum_after_step1 = saved["tables"][0]["momentum"].clone()

    table.set_grad(g2)
    eng.accumulate()
    eng.step()
    weight_after_step2 = table.weight.clone()

    # corrupt the live hyper-parameters/state, then restore and replay step 2
    eng.lr, eng.gamma, eng.K = 99.0, 0.0, 3
    eng.load_state_dict(copy.deepcopy(saved))
    assert eng.lr == pytest.approx(0.05)
    assert eng.gamma == pytest.approx(0.18)
    assert eng.K == 11
    restored = eng.state_dict()
    assert torch.equal(restored["tables"][0]["momentum"], momentum_after_step1)
    table.weight.copy_(weight_after_step1)
    table.set_grad(g2)
    eng.accumulate()
    eng.step()
    assert torch.equal(table.weight, weight_after_step2)

    # an untouched table has no state and loads back as a zeroed entry
    fresh_table = FakeTable(3, 5)
    fresh = mu.SinkhornEngram([fresh_table], lr=0.1)
    fresh_state = fresh.state_dict()
    assert fresh_state["tables"] == [None]
    fresh.load_state_dict(fresh_state)
    assert torch.equal(fresh.state_dict()["tables"][0]["momentum"],
                       torch.zeros(3, 5))


def test_sinkhorn_engram_matches_sinkhorn_momentum():
    torch.manual_seed(42)
    rows, cols = 16, 12
    table = FakeTable(rows, cols)
    p = torch.nn.Parameter(table.weight.clone())
    sink = mu.SinkhornMomentum([("engram_table", p)], lr=0.05)
    eng = mu.SinkhornEngram([table], lr=0.05)
    for step in range(2):
        grad = torch.randn(rows, cols)
        grad[4] *= 1e-8  # near-zero row => tau masking path
        p.grad = grad.clone()
        sink.step()
        table.set_grad(grad)
        eng.accumulate()
        eng.step()
        assert torch.allclose(table.weight, p.detach(), atol=1e-6, rtol=0), step


# =========================================================================== aux-free gate bias
def test_update_gate_bias_moves_overloaded_expert_down():
    torch.manual_seed(50)
    bias0 = torch.nn.Parameter(torch.zeros(4))
    bias_vl = torch.nn.Parameter(torch.zeros(4))
    bias1 = torch.nn.Parameter(torch.zeros(4))
    model = FakeModel([
        ("layers.0.ffn.gate.bias", bias0),
        ("layers.0.ffn.gate.bias_vl", bias_vl),
        ("layers.1.ffn.gate.bias", bias1),
    ])
    speed = 0.5  # exactly representable
    counts = torch.tensor([10.0, 2.0, 2.0, 2.0])  # expert 0 overloaded, mean = 4
    stats = mu.update_gate_bias(model, {"layers.0.ffn.gate": counts}, speed=speed)
    assert bias0.detach()[0].item() == -speed
    for i in (1, 2, 3):
        assert bias0.detach()[i].item() == speed
    assert torch.equal(bias_vl.detach(), bias0.detach())   # bias_vl follows the same rule
    assert torch.equal(bias1.detach(), torch.zeros(4))     # no counts -> untouched
    assert stats == {"layers.0.ffn.gate": pytest.approx(10.0 / 4.0)}


def test_update_gate_bias_noop_without_counts():
    bias = torch.nn.Parameter(torch.zeros(4))
    model = FakeModel([("layers.0.ffn.gate.bias", bias)])
    assert mu.update_gate_bias(model, None) == {}
    assert mu.update_gate_bias(model, {}) == {}
    # wrong expert count -> ignored
    assert mu.update_gate_bias(model, {"layers.0.ffn.gate": torch.ones(3)}) == {}
    assert torch.equal(bias.detach(), torch.zeros(4))


def test_is_gate_bias():
    assert mu.GATE_BIAS_SUFFIXES == (".gate.bias", ".gate.bias_vl")
    assert mu.is_gate_bias("layers.0.ffn.gate.bias")
    assert mu.is_gate_bias("layers.0.ffn.gate.bias_vl")
    assert mu.is_gate_bias("module.layers.0.ffn.gate.bias")
    assert not mu.is_gate_bias("layers.0.ffn.gate.weight")
    assert not mu.is_gate_bias("layers.0.ffn.gate")
    assert not mu.is_gate_bias("layers.0.ffn.gate.bias_extra")
    assert not mu.is_gate_bias("bias")


# =========================================================================== env-gated LR-scale knobs (M1 round 2)
def test_muon_rms_target_scales_update_linearly():
    """``NANOV41_MUON_RMS``: update magnitude is linear in the target (0.36 == lr x2, 0.72 == x4)."""
    torch.manual_seed(7)
    G = torch.randn(24, 16)
    u018 = mu.muon_direction(G, update_rms=0.18)
    u036 = mu.muon_direction(G, update_rms=0.36)
    u072 = mu.muon_direction(G, update_rms=0.72)

    def rms(x: torch.Tensor) -> float:
        return x.square().mean().sqrt().item()

    assert rms(u018) == pytest.approx(0.18, rel=2e-3)
    assert rms(u036) == pytest.approx(0.36, rel=2e-3)
    assert rms(u072) == pytest.approx(0.72, rel=2e-3)
    assert torch.allclose(u036, 2.0 * u018, rtol=1e-5, atol=1e-6)
    assert torch.allclose(u072, 4.0 * u018, rtol=1e-5, atol=1e-6)


def _muon_one_step(init: torch.Tensor, grad: torch.Tensor, *, lr: float,
                   update_rms: float) -> torch.Tensor:
    p = torch.nn.Parameter(init.clone())
    opt = mu.Muon([("w", p)], lr=lr, momentum=0.0, nesterov=False, ns_steps=NS_STEPS,
                  weight_decay=0.0, update_rms=update_rms)
    p.grad = grad.clone()
    opt.step()
    return p.detach()


def test_muon_rms_target_equals_lr_scale():
    """Semantics check: ``update_rms=0.36`` at ``lr/2`` == ``update_rms=0.18`` at ``lr`` (fp32 CPU)."""
    torch.manual_seed(11)
    init = torch.randn(32, 16)
    grad = torch.randn(32, 16)
    a = _muon_one_step(init, grad, lr=2e-2, update_rms=0.18)
    b = _muon_one_step(init, grad, lr=1e-2, update_rms=0.36)
    da, db = (a - init), (b - init)
    # update_rms doubles while lr halves -> the two updates are identical (not 2x).
    assert torch.allclose(db, da, rtol=2e-5, atol=1e-8)


def test_sinkhorn_gamma_scales_update_linearly():
    """``NANOV41_SINKHORN_GAMMA``: update is linear in gamma (0.36 == Sinkhorn-group lr x2)."""
    torch.manual_seed(13)
    rows, cols = 16, 12
    init = torch.randn(rows, cols)
    grad = torch.randn(rows, cols)

    def one_step(gamma: float, lr: float) -> torch.Tensor:
        p = torch.nn.Parameter(init.clone())
        opt = mu.SinkhornMomentum([("embed.weight", p)], lr=lr, gamma=gamma)
        p.grad = grad.clone()
        opt.step()
        return p.detach()

    d018 = one_step(0.18, 5e-2) - init
    d036 = one_step(0.36, 5e-2) - init
    d_eq = one_step(0.36, 2.5e-2) - init  # gamma x2 @ lr/2 == baseline
    # linear in gamma up to fp32 rounding (tiny entries: absolute floor needed)
    assert torch.allclose(d036, 2.0 * d018, rtol=1e-4, atol=1e-6)
    assert torch.allclose(d_eq, d018, rtol=1e-4, atol=1e-6)


def test_env_float_helper(monkeypatch):
    monkeypatch.delenv("NANOV41_MUON_RMS", raising=False)
    assert mu._env_float("NANOV41_MUON_RMS", 0.18) == pytest.approx(0.18)
    monkeypatch.setenv("NANOV41_MUON_RMS", "0.72")
    assert mu._env_float("NANOV41_MUON_RMS", 0.18) == pytest.approx(0.72)
    monkeypatch.setenv("NANOV41_MUON_RMS", "   ")
    assert mu._env_float("NANOV41_MUON_RMS", 0.18) == pytest.approx(0.18)
    monkeypatch.setenv("NANOV41_MUON_RMS", "abc")
    with pytest.raises(ValueError, match="NANOV41_MUON_RMS"):
        mu._env_float("NANOV41_MUON_RMS", 0.18)


def test_env_reload_overrides_constants_and_build_defaults(monkeypatch):
    """Env vars are read at import and flow into ``build_optimizers`` defaults; reload restores."""
    import importlib
    import inspect

    monkeypatch.setenv("NANOV41_MUON_RMS", "0.72")
    monkeypatch.setenv("NANOV41_SINKHORN_GAMMA", "0.36")
    monkeypatch.setenv("NANOV41_ENGRAM_LR_MULT", "7.5")
    try:
        reloaded = importlib.reload(mu)
        assert reloaded.MUON_UPDATE_RMS == pytest.approx(0.72)
        assert reloaded.SINKHORN_GAMMA == pytest.approx(0.36)
        assert reloaded.ENGRAM_LR_MULT == pytest.approx(7.5)
        params = inspect.signature(reloaded.build_optimizers).parameters
        assert params["update_rms"].default == pytest.approx(0.72)
        assert params["sinkhorn_gamma"].default == pytest.approx(0.36)
        assert params["engram_lr_mult"].default == pytest.approx(7.5)
    finally:
        monkeypatch.undo()
        importlib.reload(mu)
    assert mu.MUON_UPDATE_RMS == pytest.approx(0.18)
    assert mu.SINKHORN_GAMMA == pytest.approx(0.18)
    assert mu.ENGRAM_LR_MULT == pytest.approx(5.0)
    # the module is still usable after the reload dance
    assert mu.muon_direction(torch.randn(8, 6), update_rms=0.18).shape == (8, 6)


def _muon_scratch_budget_parity(budget_a: int, budget_b: int) -> float:
    """Run Muon.step with two NS chunk budgets on identical copies; return max |Δparam|."""
    torch.manual_seed(7)
    experts, m, n = 6, 16, 8                        # 3-D stacked-expert tensors like the real MoE
    shapes = [("stacked", (experts, m, n)), ("proj", (4 * m, n)), ("square", (m, m))]
    params_a = []
    params_b = []
    for name, shape in shapes:
        p = torch.nn.Parameter(torch.randn(shape) * 0.02)
        p.grad = torch.randn(shape)
        params_a.append((name, p))
        params_b.append((name, copy.deepcopy(p.detach())))
        params_b[-1][1].grad = p.grad.clone()
    opts = []
    for named, budget in ((params_a, budget_a), (params_b, budget_b)):
        o = mu.Muon(named, lr=8.22e-4, max_batch_elems=budget)
        o.step()
        opts.append(o)
    return max(float((pa - pb).abs().max()) for (_, pa), (_, pb) in zip(params_a, params_b))


def test_muon_ns_chunk_budget_is_numerically_transparent():
    """The batched-NS chunk budget only groups matrices; the update must not depend on it.

    Regression for the 2026-11-10 bug: the chunk divisor counted *tensors*, so one 3-D
    stacked-expert tensor (128 matrices) with a budget set for a handful of matrices still
    produced a multi-GiB NS scratch (and chunk8192 / 32k-micro-batch OOM'd in the optimizer).
    """
    delta = _muon_scratch_budget_parity(budget_a=1 << 10, budget_b=1 << 20)
    assert delta < 1e-10, delta


def test_muon_ns_chunk_budget_caps_stacked_matrix_count():
    """3-D stacked tensors must be packed by their matrix count, not one tensor per chunk."""
    torch.manual_seed(11)
    p1 = torch.nn.Parameter(torch.randn(4, 16, 8) * 0.02)
    p1.grad = torch.randn(4, 16, 8)
    p2 = torch.nn.Parameter(torch.randn(4, 16, 8) * 0.02)
    p2.grad = torch.randn(4, 16, 8)
    o = mu.Muon([("e0", p1), ("e1", p2)], lr=1e-3, max_batch_elems=16 * 8 * 5)  # = 5 mats
    seen: list[int] = []
    orig = mu._batched_newton_schulz

    def spy(g, steps=mu.NS_STEPS, eps=1e-7):
        seen.append(int(g.shape[0]))
        return orig(g, steps=steps, eps=eps)

    mu._batched_newton_schulz = spy
    try:
        o.step()
    finally:
        mu._batched_newton_schulz = orig
    assert seen and max(seen) <= 5, seen
    assert sum(seen) == 8                          # two tensors x 4 experts
    assert len(seen) == 2, seen                    # budget 5 < 8 => two chunks

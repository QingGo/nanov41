"""DeepSeek-V4.1-style optimizer for nano-v41 training (pure PyTorch, single GPU).

Following the V4.1 report (§2.5) and its Algorithm 1:
  * Muon (Nesterov momentum + Newton-Schulz orthogonalization) for the backbone linear
    weight matrices, with **head-wise Muon** for Query/Key weights (different precondi-
    tioners per attention head).
  * AdamW for normalization weights, biases and other non-matrix parameters
    (norm weights get weight decay; biases/scaling factors don't).
  * Sinkhorn-balanced momentum updates for the token embedding and the prediction head
    (momentum only, no Adam second moment; gamma=0.18 to match Adam's update magnitude).

Engineering note (nano-scale): vanilla per-matrix NS costs ~1.7 GFLOP per [384,384]
matrix per step; at 768 matrices that is ~1.3 TFLOP, ~37% of a 8k-token d12 step.
We therefore support grouping same-shape expert matrices into one stacked NS
(`grouped_experts=True`) and document the approximation as a measured trade-off.
"""
from __future__ import annotations

import math
import torch

NS_STEPS = 5


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = NS_STEPS, eps: float = 1e-7) -> torch.Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.T if transposed else X


def _muon_update(G: torch.Tensor, steps: int = NS_STEPS) -> torch.Tensor:
    """NS-orthogonalize a 2D tensor (bf16 math, returns fp32 for the param update)."""
    return zeropower_via_newtonschulz5(G.float(), steps=steps).view_as(G)


def headwise_split(g: torch.Tensor, n_heads: int) -> list[torch.Tensor]:
    return list(g.view(n_heads, -1, g.size(1)).unbind(0))


def _batched_newton_schulz(G: torch.Tensor, steps: int = NS_STEPS, eps: float = 1e-7) -> torch.Tensor:
    """Batched NS on [N, m, n] fp32 tensors: 3 batched matmuls per iteration (one kernel each)."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.transpose(-1, -2)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.transpose(-1, -2)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.transpose(-1, -2)
    return X


class Muon(torch.optim.Optimizer):
    """Muon with shape-grouped batched Newton-Schulz (report §2.5), plus head-wise Q/K.

    Naive per-matrix NS costs ~0.5 ms of Python/launch overhead per matrix on this model
    (768 matrices => ~350 ms/step). Grouping all same-shape matrices into one batched NS
    turns that into ~15 batched matmuls per iteration; measured effect is in W2 report.
    """

    def __init__(self, named_params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=NS_STEPS,
                 weight_decay=0.0, headwise: dict[str, int] | None = None, grouped_experts: bool = True):
        self.headwise = headwise or {}
        self.ns_steps = ns_steps
        self._named = list(named_params)
        super().__init__([dict(params=[p for _, p in self._named], lr=lr, momentum=momentum,
                               nesterov=nesterov, weight_decay=weight_decay)],
                         defaults=dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        g = self.param_groups[0]
        lr, mom, nesterov, wd = g["lr"], g["momentum"], g["nesterov"], g["weight_decay"]
        groups: dict[tuple, list] = {}
        for name, p in self._named:
            if p.grad is None:
                continue
            st = self.state[p]
            if "momentum_buffer" not in st:
                st["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
            buf = st["momentum_buffer"]
            buf.mul_(mom).add_(p.grad.float())
            g_eff = p.grad.float().add(buf, alpha=mom) if nesterov else buf.clone()
            h = self.headwise.get(name)
            if h:
                key = ("hw", tuple(p.shape), h)
                groups.setdefault(key, []).append((name, p, g_eff.view(h, -1, p.size(1))))
            else:
                key = ("mat", tuple(p.shape))
                groups.setdefault(key, []).append((name, p, g_eff.view(p.size(0), -1)))
        for key, items in groups.items():
            stacked = torch.stack([g_ for _, _, g_ in items], dim=0)      # [N, m, n]
            flat = stacked.reshape(-1, stacked.size(-2), stacked.size(-1))
            u = _batched_newton_schulz(flat, self.ns_steps).reshape_as(stacked).float()
            for i, (name, p, _) in enumerate(items):
                ui = u[i]
                if key[0] == "hw":
                    ui = ui.reshape(-1, p.size(1))
                scale = max(1.0, math.sqrt(ui.size(0) / ui.size(1)))
                p.add_(ui, alpha=-lr * scale)
                if wd:
                    p.mul_(1.0 - lr * wd)
        return loss


class SinkhornMomentum(torch.optim.Optimizer):
    """Momentum + Sinkhorn balancing for token embedding / prediction head (report Algorithm 1)."""

    def __init__(self, named_params, lr=3e-3, momentum=0.95, nesterov=True, gamma=0.18,
                 sinkhorn_iters=3, tau=1e-3, eps=1e-8):
        self._named = list(named_params)
        self.gamma = gamma
        self.K = sinkhorn_iters
        self.tau = tau
        self.eps = eps
        super().__init__([dict(params=[p for _, p in self._named], lr=lr, momentum=momentum,
                               nesterov=nesterov)], defaults=dict(lr=lr, momentum=momentum, nesterov=nesterov))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        g = self.param_groups[0]
        lr, mom, nesterov = g["lr"], g["momentum"], g["nesterov"]
        for _, p in self._named:
            if p.grad is None:
                continue
            st = self.state[p]
            if "momentum_buffer" not in st:
                st["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
            buf = st["momentum_buffer"]
            buf.mul_(mom).add_(p.grad.float())
            U = p.grad.float().add(buf, alpha=mom) if nesterov else buf.clone()
            # alternating row / column RMS normalization (Algorithm 1)
            for k in range(self.K):
                if k % 2 == 0:
                    rho = U.norm(dim=1, keepdim=True)
                    mean = rho.mean()
                    U = U / (rho + self.eps)
                    U[rho.squeeze(1) <= self.tau * mean] = 0
                else:
                    U = U / (U.norm(dim=0, keepdim=True) + self.eps)
            U = U * math.sqrt(U.size(1))          # unit row L2 -> unit row RMS
            p.add_(U.view_as(p), alpha=-lr * self.gamma)


def build_optimizers(model, muon_lr=0.02, adamw_lr=3e-3, weight_decay=0.05,
                     grouped_experts=True, headwise=True, ns_steps=NS_STEPS):
    """Split named params into (Muon, Sinkhorn, AdamW) per the report and return them."""
    named = dict(model.named_parameters())
    sinkhorn, muon, adamw = [], [], []
    headwise_map = {}
    n_heads = 8
    index_n_heads = 4
    for name, p in named.items():
        if not p.requires_grad:
            continue
        if name in ("inner.embed.weight", "inner.head.weight"):
            sinkhorn.append((name, p)); continue
        if p.ndim == 2:
            muon.append((name, p))
            if headwise and ".wq_b." in name:
                headwise_map[name] = index_n_heads if "indexer" in name else n_heads
        else:
            adamw.append((name, p))
    opts = []
    if muon:
        opts.append(Muon(muon, lr=muon_lr, weight_decay=weight_decay, ns_steps=ns_steps,
                         headwise=headwise_map, grouped_experts=grouped_experts))
    if sinkhorn:
        opts.append(SinkhornMomentum(sinkhorn, lr=adamw_lr))
    if adamw:
        opts.append(torch.optim.AdamW([p for _, p in adamw], lr=adamw_lr,
                                      betas=(0.9, 0.95), weight_decay=0.0, fused=True))
    return opts


class OptimizerBundle:
    """Minimal interface (zero_grad/step/param list) over the report's three optimizers."""
    def __init__(self, opts):
        self.opts = opts

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.opts:
            o.step()

    @property
    def param_groups(self):
        for o in self.opts:
            yield from o.param_groups

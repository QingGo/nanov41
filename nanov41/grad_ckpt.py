# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""Activation recomputation (gradient checkpointing) for the nano-V4.1 training loop.

`install()` wraps `Block.forward` in `torch.utils.checkpoint.checkpoint` so a training step stores
only the block *inputs* and recomputes the block forward during backward. This is the standard
lever for long-sequence training; the M3 memory analysis (see the feasibility report) shows it is
necessary but not sufficient for full-sequence 1M BPTT on a 32 GB card:

  block input is the mHC 4-stream residual `[1, T, 4, 384]` bf16 = 3.10 GiB per layer at T = 1M,
  so 12 checkpointed layers need 37.2 GiB just for the stored inputs.

It is the right lever *inside* chunks (chunk=64K -> 0.20 GiB/layer, ~2.4 GiB for 12 layers), where
it lets chunked BPTT use much larger chunks (less truncation) at a small memory cost.

The non-reentrant form is used: the reference blocks write deterministic KV/index caches as a side
effect, and recomputation writes the same values again (idempotent), which is safe here.

Important subtlety: CSA2 layers hand their per-layer state down via a module-global
`shared_attn` runtime (compressed-KV pointer, index-key pointer, top-k indices, candidate mask).
A naive `checkpoint(Block.forward, ...)` therefore recomputes a consumer layer with whatever
`shared_attn` state happens to be live at backward time (the last forward layer's), which crashes or
silently computes the wrong thing. This module snapshots `shared_attn` *before* each block and
restores it before replaying the block; snapshots hold references (not copies), so layers in the
same ratio group share one tensor.
"""
from __future__ import annotations

import torch

_STATE = {"installed": False, "orig": None, "M": None, "use_reentrant": False,
          "replays": 0, "trace": [], "fp8_save": False}


_SHARED_ATTRS = ("compress_kv", "index_k", "topk_idxs", "candidates")


def _collect_shared_state(M):
    """Snapshot the module-global shared_attn tensors (skipping None) for threading through a checkpoint.

    Order is the canonical `_SHARED_ATTRS` order, so a name tuple plus tensors is enough to restore.
    """
    s = M.shared_attn
    names, tensors = [], []
    for name in _SHARED_ATTRS:
        v = getattr(s, name, None)
        if v is not None:
            names.append(name)
            tensors.append(v)
    return tuple(names), tuple(tensors)


def _apply_shared_state(M, names, tensors) -> None:
    s = M.shared_attn
    for name in _SHARED_ATTRS:
        setattr(s, name, None)
    for name, value in zip(names, tensors):
        setattr(s, name, value)


def _collect_block_buffers(blk):
    """Mutable per-layer caches that the chunked-prefill path reads/writes inside a block.

    They must be saved/restored per checkpointed block: the forward pass updates the sliding-window
    ring (and chunked prefill updates the compressed-KV cache), so a later replay would otherwise see
    the post-forward contents instead of the values the block originally read (dense window-only
    chunked BPTT showed 5e-4 gradient errors from exactly this).
    """
    specs = []
    attn = getattr(blk, "attn", None)
    if attn is not None:
        for name in ("window_kv_cache", "compress_kv_cache"):
            buf = getattr(attn, name, None)
            if buf is not None:
                specs.append((0, name))
        indexer = getattr(attn, "indexer", None)
        if indexer is not None and getattr(indexer, "k_cache", None) is not None:
            specs.append((1, "k_cache"))
    return specs


def _restore_block_buffers(blk, specs, clones) -> None:
    for (kind, name), clone in zip(specs, clones):
        target = blk.attn if kind == 0 else blk.attn.indexer
        buf = getattr(target, name)
        if buf is not None and clone is not None:
            with torch.no_grad():
                buf.copy_(clone)


class _BlockStateFn(torch.autograd.Function):
    """Gradient-correct full-block activation checkpoint that threads `shared_attn` through its I/O.

    A plain whole-block checkpoint is wrong for this model whenever a block consumes tensors that an
    earlier block published into the module-global `shared_attn` state (compressed KV, index keys,
    top-k indices): the consumer's replay sees the forward-time tensor as a graph leaf and drops the
    cross-block gradient path (measured ~46% relative error in `start_pos > 0` chunks).

    Here every state tensor is an explicit input and every state tensor the block *changes* is an
    explicit output. The autograd engine therefore routes a downstream consumer's gradient for e.g.
    `compress_kv` back into the producer block's replay graph, exactly like an ordinary tensor edge.

    forward: run the block under no_grad with the state restored from the inputs; save only the
    (small) inputs. backward: restore the state, replay the block with grad, and compute gradients
    for the differentiable inputs and the block's parameters in one autograd.grad call.
    """

    @staticmethod
    def forward(ctx, blk, x, pre_mix, image_mask, start_pos, state_names, *args):
        n_state = _STATE["n_state"]
        state_tensors = args[:n_state]
        buf_specs = _collect_block_buffers(blk)
        buf_clones = list(args[n_state:])
        M = _STATE["M"]
        orig = _STATE["orig_block"]
        _apply_shared_state(M, state_names, state_tensors)
        _restore_block_buffers(blk, buf_specs, buf_clones)
        with torch.no_grad():
            y, pm = orig(blk, x, start_pos, pre_mix, image_mask)
        # what did this block publish / overwrite relative to its inputs?
        names_out, tensors_out = _collect_shared_state(M)
        before = dict(zip(state_names, state_tensors))
        changed = [n for n, t in zip(names_out, tensors_out) if before.get(n) is not t]
        changed_tensors = [t for n, t in zip(names_out, tensors_out) if n in changed]
        changed_names = tuple(changed)
        ctx.blk = blk
        ctx.start_pos = start_pos
        ctx.image_mask = image_mask
        ctx.state_names = state_names
        ctx.changed_names = changed_names
        ctx.has_pm = pm is not None
        t_cache = []
        ctx.x_idx = ctx.pm_idx = None
        if x is not None:
            ctx.x_idx = len(t_cache); t_cache.append(x)
        if pre_mix is not None:
            ctx.pm_idx = len(t_cache); t_cache.append(pre_mix)
        ctx.buf_specs = buf_specs
        ctx.buf_start = len(t_cache)
        t_cache.extend(buf_clones)
        ctx.state_start = len(t_cache)
        t_cache.extend(state_tensors)
        ctx.save_for_backward(*t_cache)
        # only the compressed-KV latent needs a gradient path across blocks; index keys and top-k
        # indices are integer-valued consumers (the indexer output is int, so no gradient flows there)
        ctx.diff_out_names = tuple(n for n in changed_names if n == "compress_kv")
        non_diff = [t for n, t in zip(changed_names, changed_tensors) if n not in ctx.diff_out_names]
        if non_diff:
            ctx.mark_non_differentiable(*non_diff)
        _STATE["last_changed"] = changed_names
        _STATE["last_has_pm"] = pm is not None
        _STATE["replays"] += 1
        if pm is None:
            return (y, *changed_tensors)
        return (y, pm, *changed_tensors)

    @staticmethod
    def backward(ctx, *grad_outputs):
        M = _STATE["M"]
        orig = _STATE["orig_block"]
        saved = ctx.saved_tensors
        x = saved[ctx.x_idx] if ctx.x_idx is not None else None
        pre_mix = saved[ctx.pm_idx] if ctx.pm_idx is not None else None
        buf_clones = list(saved[ctx.buf_start:ctx.state_start])
        state_tensors = list(saved[ctx.state_start:])
        _restore_block_buffers(ctx.blk, ctx.buf_specs, buf_clones)
        gi = 0
        gy = grad_outputs[gi]; gi += 1
        gpm = None
        if ctx.has_pm:
            gpm = grad_outputs[gi]; gi += 1
        gstates_in = grad_outputs[gi:]                      # grads for the *changed* state outputs
        g_changed = dict(zip(ctx.changed_names, gstates_in))
        changed = {n: g_changed[n] for n in ctx.diff_out_names}

        # Replay with *detached* copies of the inputs: the originals are outputs of other
        # _BlockStateFn calls (or of the forward graph), and reusing them would make this replay
        # graph point back into the outer graph (nested backward / freed saved tensors). Detaching
        # turns them into fresh leaves; the gradients w.r.t. those leaves are what we return, so
        # autograd reconnects the chain block by block.
        replay_inputs = []          # positional args in the forward order
        diff_targets = []           # tensors we ask autograd.grad for
        diff_positions = []         # corresponding apply() argument positions
        x_r = x
        if x is not None and x.requires_grad:
            x_r = x.detach().requires_grad_(True)
            diff_targets.append(x_r); diff_positions.append(1)
        replay_inputs.append(x_r)
        pm_r = pre_mix
        if pre_mix is not None and pre_mix.requires_grad:
            pm_r = pre_mix.detach().requires_grad_(True)
            diff_targets.append(pm_r); diff_positions.append(2)
        replay_inputs.append(pm_r)
        replay_states = []
        for k, (name, t) in enumerate(zip(ctx.state_names, state_tensors)):
            if t is None:
                replay_states.append(None); continue
            t_r = t.detach()
            if name == "compress_kv" and t.requires_grad:
                t_r.requires_grad_(True)
                diff_targets.append(t_r); diff_positions.append(6 + k)
            replay_states.append(t_r)
        params = [p for p in ctx.blk.parameters() if p.requires_grad]
        with torch.enable_grad():
            _apply_shared_state(M, ctx.state_names, tuple(replay_states))
            y, pm = orig(ctx.blk, x_r, ctx.start_pos, pm_r, ctx.image_mask)
            names_out, tensors_out = _collect_shared_state(M)
            out_by_name = dict(zip(names_out, tensors_out))
            outs = [y] + ([pm] if pm is not None else [])
            gouts = [gy] + ([gpm] if pm is not None else [])
            for name, gval in changed.items():
                if name not in out_by_name:
                    raise RuntimeError(f"checkpoint replay lost state {name!r}")
                outs.append(out_by_name[name])
                gouts.append(gval)
            grads = torch.autograd.grad(outs, diff_targets + params, gouts,
                                        allow_unused=True, retain_graph=False)
        diff_inputs = diff_targets
        for p, g in zip(params, grads[len(diff_inputs):]):
            if g is not None:
                if p.grad is None:
                    p.grad = g.detach().clone()
                else:
                    p.grad.add_(g)

        # map grads back onto the apply() argument order:
        #   0 blk | 1 x | 2 pre_mix | 3 image_mask | 4 start_pos | 5 state_names | 6.. state tensors
        n_args = 6 + len(state_tensors) + len(ctx.buf_specs)
        out = [None] * n_args
        for pos, arg_pos in enumerate(diff_positions):
            out[arg_pos] = grads[pos]
        return tuple(out)





def _snapshot_shared(M):
    s = M.shared_attn
    return tuple(getattr(s, a) for a in _SHARED_ATTRS)


def _restore_shared(M, snap) -> None:
    s = M.shared_attn
    for a, v in zip(_SHARED_ATTRS, snap):
        setattr(s, a, v)


def _ffn_tail(self, x, attn_pre, image_mask):
    """Second half of Block.forward: mHC mixes -> hc_pre -> ffn_norm -> ffn -> hc_post.

    Pure function of (x, attn_pre) - it never reads/writes the module-global `shared_attn` state,
    so recomputing it during backward reproduces the same gradients. (A full-block checkpoint
    instead loses the cross-block gradient path through the shared compressed KV / index keys in
    `start_pos > 0` chunks - measured ~46% relative gradient error on those params.)
    """
    residual = x
    ffn_pre, ffn_post, ffn_comb = self.hc_mixes(
        x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base
    )
    x = self.hc_pre(x, attn_pre)
    x = self.ffn_norm(x)
    x = self.ffn(x, image_mask)
    x = self.hc_post(x, residual, ffn_post, ffn_comb)
    return x, ffn_pre


def _replay_block(self, x, start_pos, pre_mix, image_mask, *attn_args):
    """Module-level replay used by checkpoint: restores per-block shared_attn state, then re-runs.

    No closure over `snap` (earlier closure version leaked the whole step's activations across
    steps); the snapshot lives on the block instance and is overwritten every forward.
    """
    M = _STATE["M"]
    if _STATE.get("restore_shared", True):
        snap = getattr(self, "_nanov41_ckpt_snap", None)
        if snap is not None:
            _restore_shared(M, snap)
    _STATE["replays"] += 1
    return _STATE["orig_block"](self, x, start_pos, pre_mix, image_mask, *attn_args)


def _make_fp8_save_hooks(min_numel: int = 1 << 20):
    """saved_tensors_hooks that store large 4-stream bf16 tensors as FP8 e4m3 + per-row scale."""

    def pack(t):
        if (t.dtype is torch.bfloat16 and t.dim() == 4 and t.numel() >= min_numel
                and t.is_cuda):
            q, sc = _quant_rows(t)
            return (1, q, sc, t.shape, t.dtype)
        return (0, t)

    def unpack(obj):
        if obj[0] == 0:
            return obj[1]
        _, q, sc, shape, dtype = obj
        return _dequant(q, sc, dtype).view(shape)

    return torch.autograd.graph.saved_tensors_hooks(pack, unpack)


def install(M=None, use_reentrant: bool = False, fp8_save: bool | None = None,
            mode: str | None = None) -> dict:
    global _STATE
    if M is None:
        from .patched_env import import_patched

        M = import_patched()
    _STATE["restore_shared"] = __import__("os").environ.get(
        "NANOV41_CKPT_RESTORE", "1").lower() in ("1", "true", "yes", "on")
    if fp8_save is None:
        import os as _os
        fp8_save = _os.environ.get("NANOV41_CKPT_FP8", "0").lower() in ("1", "true", "yes", "on")
    if mode is None:
        import os as _os2
        mode = _os2.environ.get("NANOV41_CKPT_MODE", "block").strip().lower() or "block"
    mode = mode.strip().lower()
    if mode not in ("block", "ffn", "state"):
        raise ValueError(f"ckpt mode must be 'block', 'ffn' or 'state', got {mode!r}")
    ckpt_every = max(1, int(__import__("os").environ.get("NANOV41_CKPT_EVERY", "1") or 1))
    if (_STATE["installed"] and _STATE["M"] is M and _STATE.get("fp8_save") == bool(fp8_save)
            and _STATE.get("mode") == mode):
        return dict(_STATE)
    if _STATE["installed"]:
        uninstall()
    orig = M.Block.forward
    _STATE.update({"installed": True, "orig": orig, "orig_block": orig, "M": M,
                   "use_reentrant": use_reentrant, "fp8_save": bool(fp8_save), "mode": mode,
                   "ckpt_every": ckpt_every})

    def forward(self, x, start_pos, pre_mix, image_mask, *attn_args):
        if not (torch.is_grad_enabled() and x.requires_grad):
            return _STATE["orig_block"](self, x, start_pos, pre_mix, image_mask, *attn_args)
        if _STATE.get("mode") == "state":
            # whole-block checkpoint whose shared_attn I/O is threaded through the Function so the
            # cross-block gradient path survives (see _BlockStateFn). NANOV41_CKPT_EVERY=k keeps
            # only every k-th block checkpointed: activation storage grows, but the +1 forward of
            # recompute per saved block shrinks proportionally (Pareto knob for the memory budget).
            every = int(_STATE.get("ckpt_every", 1) or 1)
            if every > 1 and getattr(self, "layer_id", 0) % every != 0:
                return _STATE["orig_block"](self, x, start_pos, pre_mix, image_mask, *attn_args)
            if attn_args:
                raise NotImplementedError("state-threaded checkpoint expects no extra attn args")
            names, tensors = _collect_shared_state(M)
            _STATE["n_state"] = len(tensors)
            buf_specs = _collect_block_buffers(self)
            buf_clones = [getattr(self.attn, name) if kind == 0 else self.attn.indexer.k_cache
                          for kind, name in buf_specs]
            buf_clones = [t.clone() for t in buf_clones]
            outs = _BlockStateFn.apply(self, x, pre_mix, image_mask, start_pos, names,
                                       *tensors, *buf_clones)
            y = outs[0]
            pm = outs[1] if _STATE.get("last_has_pm") else None
            return y, pm
        if _STATE.get("mode") == "ffn":
            # attention half runs normally (it reads/writes the shared_attn state); only the FFN
            # half - a pure function of (x, attn_pre) - is checkpointed. Gradient-exact and it
            # still drops the MoE/swiglu activation memory.
            residual = x
            attn_pre, attn_post, attn_comb = self.hc_mixes(
                x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
            x = self.hc_pre(x, pre_mix)
            x = self.attn_norm(x)
            x = self.attn(x, start_pos, *attn_args)
            x = self.hc_post(x, residual, attn_post, attn_comb)
            return torch.utils.checkpoint.checkpoint(
                _ffn_tail, self, x, attn_pre, image_mask, use_reentrant=use_reentrant)
        if _STATE.get("restore_shared", True):
            self._nanov41_ckpt_snap = _snapshot_shared(M)
        fp8 = _STATE.get("fp8_save")
        if fp8:
            with _make_fp8_save_hooks():
                return torch.utils.checkpoint.checkpoint(
                    _replay_block, self, x, start_pos, pre_mix, image_mask, *attn_args,
                    use_reentrant=use_reentrant,
                )
        return torch.utils.checkpoint.checkpoint(
            _replay_block, self, x, start_pos, pre_mix, image_mask, *attn_args,
            use_reentrant=use_reentrant,
        )

    M.Block.forward = forward
    # safety net: free autograd graphs that checkpoint replay attaches to module buffers (see
    # detach_leaked_buffers docstring / feasibility report §3m). Explicit calls after the
    # optimizer step are better (no one-step lag); this wrapper also protects callers that forget.
    import os as _os3

    _STATE["orig_transformer_forward"] = None
    if _os3.environ.get("NANOV41_CKPT_AUTODETACH", "1").lower() in ("1", "true", "yes", "on"):
        _orig_tf = M.Transformer.forward

        def _tf_forward(self, *args, full_logits=False, **kwargs):
            # keep `full_logits` explicit: inner_forward() inspects this signature to decide whether
            # to request per-position logits (a bare *args/**kwargs wrapper silently breaks it)
            detach_leaked_buffers(self, M)
            return _orig_tf(self, *args, full_logits=full_logits, **kwargs)

        M.Transformer.forward = _tf_forward
        _STATE["orig_transformer_forward"] = _orig_tf
    return {"installed": True, "use_reentrant": use_reentrant}


def uninstall() -> None:
    if not _STATE["installed"]:
        return
    M = _STATE["M"]
    if _STATE.get("orig_transformer_forward") is not None:
        M.Transformer.forward = _STATE["orig_transformer_forward"]
    M.Block.forward = _STATE["orig"]
    _STATE.update({"installed": False, "orig": None, "M": None,
                   "orig_transformer_forward": None})


def get_replays() -> int:
    return _STATE["replays"]


def reset_trace() -> None:
    _STATE["replays"] = 0
    _STATE["trace"] = []


def get_trace() -> list:
    return _STATE["trace"]


_FP8_MAX = 448.0


def _quant_rows(x, rows_per_chunk=1 << 20):
    """bf16 [.., d] -> (fp8 e4m3 [.., d], scale bf16 [.., 1]) with per-row absmax scaling."""
    flat = x.reshape(-1, x.shape[-1])
    n, d = flat.shape
    q = torch.empty(n, d, dtype=torch.float8_e4m3fn, device=x.device)
    s = torch.empty(n, 1, dtype=torch.bfloat16, device=x.device)
    for i in range(0, n, rows_per_chunk):
        j = min(n, i + rows_per_chunk)
        c = flat[i:j].to(torch.bfloat16)
        amax = c.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
        sc = (amax / _FP8_MAX).to(torch.bfloat16)
        q[i:j] = (c / sc).to(torch.float8_e4m3fn)
        s[i:j] = sc
    return q.view(x.shape), s.view(x.shape[:-1] + (1,))


def _dequant(q, s, dtype):
    return (q.to(dtype) * s.to(dtype))

def detach_leaked_buffers(root, M=None) -> int:
    """Free autograd graphs accidentally attached to registered buffers.

    Activation-checkpoint replay runs under grad mode; when a block writes a grad-tracking
    activation into a registered buffer (`self.k_cache[...] = k`, `window_kv_cache[...] = kv`, ...)
    autograd attaches a CopySlices node to that buffer. Buffers live on the module for the whole
    training run, so this keeps the step's autograd graph alive across steps (a PyTorch retention
    behaviour; see the feasibility report §3m for upstream issues). Replacing such buffers with
    detached views (same storage, no grad_fn) after the optimizer step releases the old graph.
    Also clears the CSA2 `shared_attn` runtime pointers, which may reference the same tensors.

    Returns the number of buffers detached.
    """
    n = 0
    for mod in root.modules():
        for name, buf in list(getattr(mod, "_buffers", {}).items()):
            if buf is not None and getattr(buf, "grad_fn", None) is not None:
                mod._buffers[name] = buf.detach()
                n += 1
    if M is None:
        try:
            from .patched_env import import_patched

            M = import_patched()
        except Exception:  # noqa: BLE001
            M = None
    if M is not None:
        s = getattr(M, "shared_attn", None)
        if s is not None:
            for a in _SHARED_ATTRS:
                if getattr(s, a, None) is not None:
                    setattr(s, a, None)
    return n


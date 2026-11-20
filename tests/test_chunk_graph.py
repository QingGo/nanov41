# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""ChunkedGraphTrainer: eager plumbing (CPU) and CUDA-graph parity (GPU).

* CPU: `capture=False` must be bitwise equivalent to the manual chunk loop (losses + gradients).
* CUDA: capture one graph per chunk, replay, and require (a) replay gradients and (b) multi-step
  training (optimizer outside the graphs) to be **bitwise identical** to the eager path.  This is
  the production gate for opt-A: the graphs are only usable if they are exact.
"""
from __future__ import annotations

import pytest
import torch

from nanov41.chunk_graph import ChunkedGraphTrainer


def _zero_caches(model) -> None:
    for layer in model.inner.layers:
        attn = layer.attn
        for holder, name in ((attn, "window_kv_cache"), (attn, "compress_kv_cache"),
                             (getattr(attn, "indexer", None), "k_cache")):
            buf = getattr(holder, name, None) if holder is not None else None
            if buf is not None:
                buf.zero_()
        comp = getattr(attn, "compressor", None)
        if comp is not None:
            for name in ("kv_state", "score_state"):
                buf = getattr(comp, name, None)
                if buf is not None:
                    buf.zero_()


def _build_cpu_tiny(seq: int):
    from nanov41.configs import TINY
    from nanov41.model import NanoV41, init_random_weights

    cfg = dict(TINY)
    cfg.update({"max_seq_len": seq, "max_batch_size": 1})
    torch.manual_seed(0)
    model = NanoV41(cfg, param_dtype="float32")
    init_random_weights(model.inner, seed=0)
    model.eval()  # no dropout in this model; eval keeps the trainable path identical
    return model, cfg


def test_chunk_graph_eager_equals_manual_cpu():
    from nanov41 import chunked_indexer as ci
    from nanov41 import chunked_prefill as cp
    from nanov41.model import inner_forward

    seq, chunk = 64, 32
    model, cfg = _build_cpu_tiny(seq)
    vocab = cfg["vocab_size"]
    params = [p for p in model.parameters() if p.requires_grad]
    ids = torch.randint(0, vocab, (1, seq))

    from nanov41.patched_env import import_patched

    M = import_patched()
    ci.install(M, chunk=16, mode="prefix-pad")
    cp.install(M)
    try:
        def loss_fn(c0, c1, buffers):
            h = inner_forward(model.inner, buffers["ids"][:, c0:c1], c0, full_logits=True)
            logits = h[:, :-1].reshape(-1, vocab).float()
            tgt = ids[:, c0 + 1:c1].reshape(-1)
            return torch.nn.functional.cross_entropy(logits, tgt)

        buffers = {"ids": ids.clone()}
        trainer = ChunkedGraphTrainer(model, None, chunks=[(0, chunk), (chunk, seq)],
                                      loss_fn=loss_fn, buffers=buffers, capture=False,
                                      optimizer_in_step=False, zero_caches=True)
        trainer.step(return_loss=False)
        got_chunks = trainer.chunk_losses().clone()
        got_loss = trainer.loss_value()
        got_grads = [p.grad.detach().clone() if p.grad is not None else None for p in params]

        # manual reference: same pass, no trainer
        for p in params:
            p.grad = None
        _zero_caches(model)
        manual_chunks = []
        total = 0.0
        for c0, c1 in [(0, chunk), (chunk, seq)]:
            h = inner_forward(model.inner, ids[:, c0:c1], c0, full_logits=True)
            logits = h[:, :-1].reshape(-1, vocab).float()
            tgt = ids[:, c0 + 1:c1].reshape(-1)
            loss = torch.nn.functional.cross_entropy(logits, tgt)
            manual_chunks.append(float(loss.detach()))
            total += float(loss.detach()) * ((c1 - c0) / seq)
            (loss * ((c1 - c0) / seq)).backward()
        ref_grads = [p.grad.detach().clone() if p.grad is not None else None for p in params]

        assert torch.equal(got_chunks, torch.tensor(manual_chunks))
        assert got_loss == pytest.approx(total, rel=0, abs=1e-6)
        bad = [i for i, (a, b) in enumerate(zip(got_grads, ref_grads, strict=True))
               if (a is None) != (b is None) or (a is not None and not torch.equal(a, b))]
        assert not bad, f"{len(bad)}/{len(params)} gradients differ from the manual pass"
    finally:
        cp.uninstall()
        ci.uninstall()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph capture test")
def test_chunk_graph_replay_and_multistep_match_eager_cuda():
    from nanov41 import chunked_indexer as ci
    from nanov41 import chunked_prefill as cp
    from nanov41 import fast_attn, grad_ckpt
    from nanov41.configs import D12
    from nanov41.model import NanoV41, init_random_weights, inner_forward
    from nanov41.patched_env import import_patched

    seq, chunk = 16384, 8192
    vocab = 512
    cfg = dict(D12)
    cfg.update({"vocab_size": vocab, "max_seq_len": seq, "max_batch_size": 1})
    torch.manual_seed(0)
    model = NanoV41(cfg, param_dtype="bfloat16")
    init_random_weights(model.inner, seed=0)
    model = model.cuda().train()

    from nanov41 import moe_stacked, moe_triton

    M = import_patched()
    prev_sparse = fast_attn.install(M)
    moe_triton.install(M)            # reference MoE loop does bincount().tolist() -> cannot capture
    moe_stacked.patch_model(model, M)
    params = [p for p in model.parameters() if p.requires_grad]
    ids_buf = torch.zeros(1, seq, dtype=torch.int64, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(7)
    step_ids = [torch.randint(0, vocab, (1, seq), device="cuda", generator=gen) for _ in range(2)]

    ci.install(M, chunk=256, mode="prefix-pad")
    cp.install(M)
    grad_ckpt.install(M, mode="state", fp8_save=False, use_reentrant=False)
    try:
        def loss_fn(c0, c1, buffers):
            h = inner_forward(model.inner, buffers["ids"][:, c0:c1], c0, full_logits=True)
            logits = h[:, :-1].reshape(-1, vocab).float()
            tgt = buffers["ids"][:, c0 + 1:c1].reshape(-1)
            return torch.nn.functional.cross_entropy(logits, tgt)

        def make_opt():
            return torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), fused=True)

        def make_trainer(capture, opt):
            return ChunkedGraphTrainer(model, opt, chunks=[(0, chunk), (chunk, seq)],
                                       loss_fn=loss_fn, buffers={"ids": ids_buf},
                                       capture=capture, zero_caches=True)

        init = {k: v.detach().clone() for k, v in model.state_dict().items()}
        eager = make_trainer(False, make_opt())
        eager_losses = [eager.step(ids=sid, return_loss=True) for sid in step_ids]
        eager_w = {k: v.detach().clone() for k, v in model.state_dict().items()}

        # single-pass gradient parity
        model.load_state_dict(init)
        graph = make_trainer(True, make_opt())
        graph.optimizer_in_step = False
        graph.capture()
        graph.step(ids=step_ids[0], return_loss=False)
        graph_grads = [p.grad.detach().clone() if p.grad is not None else None for p in params]
        model.load_state_dict(init)
        eager.optimizer = None
        eager.step(ids=step_ids[0], return_loss=False)
        eager_grads = [p.grad.detach().clone() if p.grad is not None else None for p in params]
        bad = [i for i, (a, b) in enumerate(zip(eager_grads, graph_grads, strict=True))
               if (a is None) != (b is None) or (a is not None and not torch.equal(a, b))]
        assert not bad, f"{len(bad)}/{len(params)} replay gradients are not bitwise equal to eager"

        # multi-step training parity (optimizer outside the graphs)
        model.load_state_dict(init)
        graph.optimizer = make_opt()
        graph.optimizer_in_step = True
        graph_losses = [graph.step(ids=sid, return_loss=True) for sid in step_ids]
        graph_w = {k: v.detach().clone() for k, v in model.state_dict().items()}
        bad_w = [k for k in eager_w if not torch.equal(eager_w[k], graph_w[k])]
        assert not bad_w, f"{len(bad_w)}/{len(eager_w)} tensors differ after two training steps"
        assert eager_losses == graph_losses, (eager_losses, graph_losses)
    finally:
        cp.uninstall()
        ci.uninstall()
        grad_ckpt.uninstall()
        M.sparse_attn = prev_sparse

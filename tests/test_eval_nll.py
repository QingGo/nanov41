from __future__ import annotations

import numpy as np
import torch

from nanov41.eval.provider import ModelProvider


def _tiny_provider():
    return ModelProvider(config_name="tiny", random_init=True, dtype="float32",
                         device="cpu", seed=0)


def test_chunked_nll_matches_full_logits():
    p = _tiny_provider()
    ids = torch.randint(0, p.vocab_size, (2, 16))
    fast = p.nll(ids, chunk=5)
    logits = p.logits(ids)
    ref = (logits[:, :-1].float().logsumexp(-1)
           - logits[:, :-1].float().gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1))
    # Not bitwise: the chunked path runs the head GEMM on smaller M slices, so BLAS
    # reduction order differs from the single full-logits pass (platform dependent).
    np.testing.assert_allclose(fast, ref.double().numpy(), rtol=1e-5, atol=1e-5)
    assert fast.shape == (2, 15)

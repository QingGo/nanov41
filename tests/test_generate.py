from __future__ import annotations

import torch

from nanov41.configs import CONFIGS
from nanov41.generate import generate_kv, generate_no_cache
from nanov41.model import NanoV41, init_random_weights


def test_kv_cache_generate_matches_full_prefix():
    torch.manual_seed(0)
    model = NanoV41(CONFIGS["tiny"], param_dtype="float32")
    init_random_weights(model, seed=0)
    ids = [1, 2, 3, 4, 5, 6, 7, 8]
    kv = generate_kv(model, ids, max_new_tokens=4, temperature=0.0)
    model2 = NanoV41(CONFIGS["tiny"], param_dtype="float32")
    init_random_weights(model2, seed=0)
    nc = generate_no_cache(model2, ids, max_new_tokens=4, temperature=0.0)
    assert kv[:len(ids)] == ids and kv == nc

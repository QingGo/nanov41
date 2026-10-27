"""Configs for the nano DeepSeek-V4.1 port.

Field names mirror the official `inference/model.py:ModelArgs` exactly (that file's
`ModelArgs(**cfg)` consumes these dicts). Shapes follow the released V4.1-Flash
(CED 20+20 layers, 384 experts, hidden 5120, head_dim 512 ...) scaled down here.

Structural invariants that must hold for the reference code to behave:
- `compress_ratios` has one entry per backbone layer; layers sharing a value share one
  compressed KV + one indexer, produced by the FIRST layer of that group, and that first
  layer must be listed in both `kv_source_layers` and `index_source_layers`.
- `index_source_layers` must be a subset of `kv_source_layers` (only KV owners can build
  index keys).
- `n_heads % o_groups == 0` and `(n_heads * head_dim) % o_groups == 0`.
"""

from __future__ import annotations

# Scale-independent values copied from the released V4.1-Flash config.
_COMMON = dict(
    dtype="bf16",  # bf16 path; the fp8/fp4 weight paths need the vendor kernels
    expert_dtype=None,
    score_func="sqrtsoftplus",
    gate_temp=1.0,
    norm_topk_prob=True,
    route_scale=1.5,
    swiglu_limit=10.0,
    norm_eps=1e-20,
    hc_mult=4,
    hc_eps=1e-6,
    compress_rope_theta=1.6e5,
    rope_theta=1e4,
    rope_factor=16,
    beta_fast=32,
    beta_slow=1,
    original_seq_len=0,  # nano models train short-context; YaRN stays off until >original
    candidate_source_layer=-1,  # two-level indexer disabled in v0
    candidate_topk_blocks=0,
    candidate_block_size=0,
    vision_n_layers=0,
    engram_layer_ids=[],  # Engram tables need the tokenizer; enabled in a later milestone
    engram_num_embeddings=[],
    engram_max_ngram_size=0,
    engram_vocab_size=0,
    engram_n_heads=0,
    engram_head_dim=0,
    engram_compressed_vocab_size=0,
    n_mtp_layers=0,
    dspark_block_size=0,
    dspark_target_layer_ids=[],
)

# Tiny config used by the parity tests / CPU smoke (fast, not a real model).
TINY = dict(
    **_COMMON,
    vocab_size=512,
    dim=64,
    moe_inter_dim=128,
    n_layers=4,
    n_heads=4,
    n_routed_experts=4,
    n_shared_experts=1,
    n_activated_experts=2,
    q_lora_rank=32,
    head_dim=32,
    rope_head_dim=8,
    o_groups=2,
    o_lora_rank=32,
    window_size=16,
    compress_ratios=[0, 2, 2, 1],
    kv_source_layers=[1, 3],
    index_source_layers=[1, 3],
    index_n_heads=2,
    index_head_dim=16,
    index_topk=8,
    hc_sinkhorn_iters=4,
    max_batch_size=4,
    max_seq_len=64,
)

# nano-v41 depth-8: 81.6M params total (15.4M non-embedding), 1x4090 friendly.
D8 = dict(
    **_COMMON,
    vocab_size=129280,
    dim=256,
    moe_inter_dim=256,
    n_layers=8,
    n_heads=4,
    n_routed_experts=8,
    n_shared_experts=1,
    n_activated_experts=2,
    q_lora_rank=64,
    head_dim=64,
    rope_head_dim=16,
    o_groups=2,
    o_lora_rank=64,
    window_size=64,
    compress_ratios=[0, 2, 2, 2, 1, 1, 1, 1],
    kv_source_layers=[1, 4],
    index_source_layers=[1, 4],
    index_n_heads=4,
    index_head_dim=32,
    index_topk=32,
    hc_sinkhorn_iters=20,
    max_batch_size=8,
    max_seq_len=1024,
)

# nano-v41 depth-12: ~190M params.
D12 = dict(
    **_COMMON,
    vocab_size=129280,
    dim=384,
    moe_inter_dim=384,
    n_layers=12,
    n_heads=8,
    n_routed_experts=16,
    n_shared_experts=1,
    n_activated_experts=2,
    q_lora_rank=96,
    head_dim=64,
    rope_head_dim=16,
    o_groups=2,
    o_lora_rank=96,
    window_size=64,
    compress_ratios=[0, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1],
    kv_source_layers=[1, 6],
    index_source_layers=[1, 6],
    index_n_heads=4,
    index_head_dim=32,
    index_topk=32,
    hc_sinkhorn_iters=20,
    max_batch_size=8,
    max_seq_len=1024,
)

# Same tiny model but the indexer keeps *all* reachable compressed positions. The official
# indexer applies ReLU to its scores, so a randomly-initialised model produces many exact
# zeros and `topk` tie-breaking then depends on tensor shape (batch length), making
# full-sequence vs incremental-prefix prefill differ at those ties. With topk >= #positions
# the selection is trivial, so the T1 mask/causality check can run with tight tolerances.
TINY_ALL_TOPK = dict(TINY, index_topk=TINY["max_seq_len"])

# "Dense twin" controls for the MoE-vs-dense A/B (W3 step 0): identical total parameters,
# identical tokenizer, only the number of *activated* experts changes. D12_DENSE = all 16
# experts active => the FFN degenerates to a dense FFN at the same parameter count, which
# makes "FLOPs per token" the only controlled variable (measured: 2.04x for d12, 1.23x for d8).
D8_DENSE = dict(D8, n_activated_experts=D8["n_routed_experts"])
D12_DENSE = dict(D12, n_activated_experts=D12["n_routed_experts"])

CONFIGS = {"tiny": TINY, "tiny_all_topk": TINY_ALL_TOPK, "d8": D8, "d8_dense": D8_DENSE,
           "d12": D12, "d12_dense": D12_DENSE}

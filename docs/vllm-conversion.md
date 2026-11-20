# vLLM checkpoint conversion for nano-V41 (prototype)

Status 2026-11-07: **offline conversion prototype done; GPU load test attempted and blocked by vLLM's production-geometry kernels**.
This is deliberately staged: everything that can be validated on CPU is done before renting a
5090/vLLM environment.

## 1. Can we reuse EngramDB?

`code/EngramDB` is useful as a reference, not as a drop-in converter:

- `refs/v41_engram.py` — official DeepSeek-V4.1 `EngramLayout`, n-gram hash, compressed token map.
  Useful for cross-checking our geometry (primes, rows, layer ids, multipliers); it does **not**
  convert training tables.
- `refs/v41_config.json`, `docs/v41-engram-analysis.md` — official V4.1 Engram constants.
- `python/engramdb/vllm_plugin.py` / `vllm.py` — disk-backed PLE/Engram gather for **Qwen PLE**
  (`DiskPleEmbedding`, `PleDiskGather`); the plugin does not know the DeepSeek V4.1 official
  checkpoint layout, and our 128 M table fits in HBM anyway.
- `docs/engine-integration.md` — vLLM upstream PLE disk-offload constraints (mmap, custom op,
  CUDA-graph splitting op); relevant only if the table does not fit in HBM later.

Missing in EngramDB: **bf16 table -> official FP8/E8M0 table** and **ref-style state_dict ->
official HF checkpoint keys/config**. Those are implemented here.

## 2. What is done

`tools/engram_fp8_export.py`:
- bf16 rows [N, 32] -> `float8_e4m3fn` rows + one `ue8m0` scale byte per 32-wide block;
- vLLM dequant semantics: `value * 2**(byte - 127)`;
- on the P2b `E-default_s1235` tables:
  - layer 1: 2,000,136 rows; layer 4: 2,000,370 rows;
  - mean relative error **2.25 %**, p99 **5.88 %** (per-row E8M0 scale, one block/row);
- output keys: `layers.{i}.engram.embed.weight` + `layers.{i}.engram.embed.scale`.

`tools/export_vllm_ckpt.py`:
- strips the training wrapper's `inner.` prefix and emits official checkpoint spelling;
- replaces the two bf16 Engram tables with fp8 + `.scale`;
- casts `head.weight` to bf16 and keeps the rest bf16/fp32;
- writes a small `text_config` from the nano cfg (hidden 384, 12 layers, 8 heads, 16 experts,
  Engram layers [1,4], `engram_max_ngram_size=3`, `n_heads=2`, `head_dim=32`) plus tokenizer files;
- prototype output: `/tmp/p2b_vllm_ckpt` (522 MB, 854 tensors, no `quantization_config`).

The first vLLM attempt uses **bf16 linears + fp8 Engram table**. If vLLM rejects unquantized
DeepSeek-V4.1 linears, the next extension is block-FP8 (32x32) quantisation for attention/MoE/
shared experts, reusing the same E8M0 quantiser.

## 3. What is still needed (GPU session)

1. Rent a 5090; install vLLM main (sm120 may need source build with CUDA 12.8+/arch 12.0/12.1;
   prebuilt wheels may not cover RTX 5090).
2. `vllm.LLM(model="/tmp/p2b_vllm_ckpt", ...)` load; capture the first error (config, mapper,
   quant, kernel).
3. Compare `prompt_logprob_token_ids` NLL/BPB against our T0 eval on a small web slice.
4. Benchmark eval tok/s vs our 78–87 k tok/s batch16 path.

Expected failure points (in order): AutoConfig/Engram config parsing; engram table param names/scale
dtype; mHC/TileLang kernels on sm120; unquantized linear/mega-MoE path.  Budget: one 2–3 h spike
(~¥6–10 GPU), but a vLLM source build can be slow; do it in a dedicated session and stop after
load + one scoring comparison.

## 4. GPU load test (2026-11-07, RTX 4090 + vLLM 0.33.0)

The converted checkpoint reached model construction in vLLM but failed on hardcoded production
shapes (full error trail: `runs/p2b/vllm_load_attempt.log`):

| # | failure | implication |
|---|---|---|
| 1 | `DeepseekCompressor` asserts `head_dim == 512 and rope_head_dim == 64` (ours 64/16) | compressor kernel is production-only |
| 2 | VL wrapper always builds the vision tower; missing `vision.norm.weight`, `image_start/end/newline` | text-only DSV4.1 still needs vision params (or loader skip) |
| 3 | TileLang mHC kernel `assert (x.shape[1] // n_splits) % n_thr == 0` at hidden 384*hc4=1536 | mHC TileLang tiling is production-only; torch fallback exists but lacks fused norm |
| 4 | `fused_inv_rope_fp8_quant` asserts `head_dim % quant_group_size == 0` (recipe group 128, ours 64) | o-projection/FP8 quant path cannot run nano geometry |

**Conclusion: checkpoint conversion alone is not enough.** vLLM's DeepSeek-V4.1 implementation
bakes production geometry (512/64 head dims, 5120 hidden, 1024 o-LoRA, 128 quant groups, mHC
tiling) into its fused kernels. Making nano run under vLLM requires a model-plugin port that swaps
compressor, mHC, o-projection, attention and MoE paths for generic/fallback kernels -- an
engineering project, not a 1-2 h converter.  All local vLLM patches were reverted after the test;
the instance was left clean.

Decision: for nano eval/OPD, use the native KV-cache path (`nanov41/generate.py`, parity-checked)
and a fused-head NLL eval; reserve vLLM for the full-size teacher or a dedicated small-dims port.

## 5. Fallback-model experiment (2026-11-07)

After the production-kernel path failed, we tested the alternative: **register a custom vLLM model
that reuses the generic NanoV41 reference forward**, bypassing vLLM's DSV4.1 kernels entirely.

Implemented in [`nanov41/vllm_plugin.py`](../nanov41/vllm_plugin.py):

- registers ``NanoV41ForCausalLM`` with vLLM's ``ModelRegistry``;
- builds the generic ``NanoV41`` reference model and loads a training checkpoint in ``__init__``
  (``ckpt_path`` in ``nano_cfg.json``; bf16 ref-key safetensors);
- ``forward`` advances the reference KV/Engram caches with ``start_pos`` and returns
  ``[num_tokens, vocab]`` logits; ``compute_logits`` is the identity;
- a tiny dummy ``Attention`` layer exists only so vLLM's scheduler can create a KV-cache group
  (the plugin's real cache is internal); single request (``max_num_seqs=1``).

**Engine-interface test (random weights, 4090 + vLLM 0.33):**

```
LOAD_OK 18.1
GEN: '也给罪恶将与 canc将与 canc将与 canc'   # random weights -- gibberish expected
Processed prompts: 1/1
```

This proves vLLM's scheduler/sampler/request path can drive our generic model; it is a
**correctness fallback**, not a speed path (our forward runs eager, and vLLM's own KV cache is
only a dummy).  It avoids all of the production-kernel assertions listed in §4.

**Real-checkpoint parity (validated locally, CPU):** building the same generic model, calling
``install()`` for the nano bf16 Engram table and loading
``/tmp/p2b_default_ref.safetensors`` (852 tensors, 745 MB) gives **missing=0, unexpected=0,
logits max|diff| = 0.0 and identical argmax** versus the native ``ModelProvider`` on the same
tokens.  Evidence: ``runs/p2b/vllm_fallback_parity.json``.

**Real-weight vLLM test (done on a dedicated 5090, vLLM 0.33):** `LOAD_OK` in 18.75 s, prefill
next-token argmax matched the native model exactly (`[260, 270, 554, 834, 7203]`, top value
12.119), and the generated continuation was `" a part of the world's economy. The country is a part
of the world's economy..."` — the same text the native path produces. One integration gotcha: with
`load_format="dummy"` vLLM initializes weights *after* `__init__`, leaving the plugin random; moving
the checkpoint load into `load_weights` and returning all parameter names fixed it.  Evidence:
`runs/p2b/vllm_fallback_parity.json`.  This is still a **slow fallback** (eager, internal caches,
dummy vLLM KV), not the fast inference path.

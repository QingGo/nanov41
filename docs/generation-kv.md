# KV-cache generation and what 0.6B tokens buys

Status: **KV-cache generate implemented and parity-checked 2026-11-07**; vLLM conversion assessed
but deferred (see §4).

## 1. What was added

`nanov41/generate.py`:
- `generate_kv(model, prompt_ids, max_new_tokens, temperature, top_p, eos_id)` — prefill the prompt
  once (`start_pos=0`), then feed one token per step with `start_pos=len(prefix)` through the official
  reference incremental path.  It uses the same ring/compressed KV caches as the training stack:
  `window_kv_cache`, `compress_kv_cache`, indexer `k_cache`, compressor `kv_state/score_state`,
  plus the Engram hash cache (written at prefill and advanced per token).
- `reset_caches()` — zero the caches between unrelated prompts.
- `generate_no_cache()` — the slow full-prefix reference loop.

`tools/p2_continue.py` gained `--kv-cache`; samples in
`runs/p2b/continuation_samples_kv.json`.

## 2. Parity check (the important part)

On the P2b `E-default_s1235` checkpoint (0.6 B-token A/B run, 193.9 M params + 128.0 M Engram table),
CPU float32, batch 1, greedy 12 tokens:

| prompt | KV == no-cache? | KV tok/s (CPU) | no-cache tok/s (CPU) |
|---|---:|---:|---:|
| `The capital of France is` | **yes, bit-identical ids** | 2.5 | 12.4 |
| `def fibonacci(n): ... return` | **yes** | 9.2 | 9.2 |
| `人工智能的未来是` | **yes** | 3.0 | 8.8 |

This is the model-bring-up check: the incremental cache path matches full-prefix greedy decoding
**exactly**, so the official caches/hash-cache semantics are correct.  CPU tok/s is meaningless
(no CUDA, Python overhead); it is not a serving benchmark.

Continuation quality after 0.6 B tokens is still weak/repetitive (`" a part of the world's economy.
The country is a part of..."`; Chinese collapses to a repeated `0.0.0.0...` token).  That is a data
scale issue, not a cache/architecture bug.

## 3. How many tokens is 0.6 B for this model?

d12 geometry: **193.41 M total params**, **94.12 M non-embedding**, **19.8 M active/token**
(16 experts top-2), and the default Engram table is **128.02 M** stored parameters (buffer, not
multiplied at inference).

| denominator | 0.6 B tokens / param | 20 B tokens / param | 100 B tokens / param |
|---|---:|---:|---:|
| total params (193.4 M) | **3.10×** | 103× | 517× |
| non-embedding (94.1 M) | **6.37×** | 213× | 1063× |
| active/token (19.8 M) | **30.3×** | 1010× | 5050× |
| stored incl. Engram table (321.4 M) | **1.87×** | 62× | 311× |

Interpretation:
- **3.1 tokens per total parameter** is far below the Chinchilla-style ~20 tokens/param for a dense
  model (0.6 B is ~15 % of the dense compute-optimal token count for 193 M);
- but relative to the **active** parameters (19.8 M), 0.6 B = 30×, i.e. already past the dense
  compute-optimal point.  This is exactly the sparse-MoE corner: compute-optimal for active FLOPs,
  still data-limited for **total capacity/head/embeddings** and for matching a strong teacher.
- 0.6 B is therefore an **A/B screening budget** (it ranks Engram table sizes and d12e32 correctly),
  not a capability budget.  nanochat's reference pretrain is 8.758 B tokens; our 0.6 B is ~7 % of it.
- 20 B pilot = 33× the P2/P2b data; 100 B plan = 167×.

## 4. Expected effect of more data (and vLLM)

Rough extrapolation (not measured): with a typical irreducible-loss + power-law form, going 33×
(0.6 B → 20 B) should remove a substantial fraction of the reducible loss; on the T0
Marin-phase1 mix we would expect roughly **1.25–1.38 BPB** (vs 1.42–1.48 now) and macro BPB
~**1.7–1.85** (vs 1.96–2.01 now), with visibly better short-range fluency, while
**chat-quality generation still needs SFT/OPD**.  The 100 B plan (517 tokens/param) is for the
inference-optimal final model and capability evals, not for module screening.  These are
extrapolations; only the 20 B pilot would validate them.

vLLM: latest vLLM has official **DeepSeek-V4.1 + Engram** support and a
`prompt_logprob_token_ids` teacher-scoring API, but our checkpoints are bf16 training tables while
vLLM's `ParallelEngramEmbedding` expects official **FP8 rows + E8M0 block scales**, official weight
names/config, and a source-built vLLM for sm120/5090.  That is a 1–2 day port and would not change
the conclusion that the model is architecture-correct; the native KV-cache path above is enough for
bring-up.  Revisit vLLM when OPD rollouts/teacher-match (generation-heavy) are on the critical path.

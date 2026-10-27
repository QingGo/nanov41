# Architecture

This document describes what the port actually implements, as expressed by the official
DeepSeek-V4.1-Flash inference code in [`ref/`](../ref/) and the nano configs in
[`nanov41/configs.py`](../nanov41/configs.py).

## 1. Model family

DeepSeek-V4.1-Flash is a **Causal Encoder-Decoder (CED)** transformer:

- every layer attends a **sliding window of raw KV**;
- layer groups additionally share a **compressed KV latent**, produced by the first layer of the
  group (`kv_source_layers` / `compress_ratios`);
- a small **lightning indexer** selects `index_topk` compressed positions per query;
- the released config's layer 20 starts a `ratio-1` group; that is how decoder layers get a
  per-token global KV projection.

Components preserved by the port:

| Component | Notes |
|---|---|
| Hyper-Connections (HC) | `hc_mult` residual copies + Sinkhorn `comb`; numerically sensitive (see limitations) |
| Engram | n-gram hash memory; **disabled** in nano configs (`engram_layer_ids=[]`) because it needs the DeepSeek tokenizer |
| MoE | sqrtsoftplus gate, shared expert, SwiGLU clamps, `norm_topk_prob`, `route_scale=1.5` |
| Attention | MLA-style latent attention with grouped low-rank output projection + compression/ indexer paths |
| MTP / DSpark heads | stubbed out of scope for v0 (`n_mtp_layers=0`, `dspark_block_size=0`) |
| Vision encoder | stubbed out of scope for v0 (`vision_n_layers=0`) |

Scale-independent values are copied from the released config in `nanov41/configs.py` (`_COMMON`).

## 2. Nano configs

| Config | Layers | dim | experts (routed/activated) | window | compress ratios | Params (total / non-embedding) |
|---|---|---|---|---|---|---|
| `tiny` / `tiny_all_topk` | 4 | 64 | 4 / all | 16 | `[0,2,2,1]` | 0.68M / 0.62M |
| `d8` | 8 | 256 | 8 / 2 | 32 | `[0,2,2,2,1,1,1,1]` | 81.62M / 15.43M |
| `d12` | 12 | 384 | 16 / 2 | 32 | `[0,2,2,2,2,2,1,1,1,1,1,1]` | 193.41M / 94.12M |

The vocabulary (129,280 tokens) dominates the `d8` total parameter count; the non-embedding
numbers above match `train.py --config <cfg> --steps 0` output.

## 3. Structural invariants (enforced by the configs)

These must hold for the reference code to behave correctly:

1. `compress_ratios` has one entry per backbone layer.
2. Consecutive layers with the same ratio share one compressed KV + one indexer, produced by the
   **first layer of the group**.
3. That first layer must be listed in both `kv_source_layers` and `index_source_layers`.
4. A `ratio-1` group is the CED decoder: every token gets a global KV projection.
5. `index_source_layers ⊆ kv_source_layers` (only KV owners can build index keys).
6. `n_heads % o_groups == 0` and `(n_heads * head_dim) % o_groups == 0`.

## 4. Training port: five documented patches

`ref/model.py` is inference-first: buffers detach the compression/indexer paths (fine for
serving, fatal for gradients) and `Transformer.forward` is decorated with
`torch.inference_mode()`. `tools/make_patched_ref.py` copies the file and applies exactly five
patches; the generated file is checked in (`nanov41/_ref_patched_model.py`) and verified by
`--check` in CI:

1. **P1** — remove `@torch.inference_mode()` decorators (training needs autograd).
2. **P2** — prefill compression shares the **local, grad-carrying** latent instead of the detached
   KV buffer.
3. **P3** — prefill index keys likewise stay local so the indexer receives gradients.
4. **P4** — `Transformer.forward` gains a `full_logits` parameter.
5. **P5** — `forward` returns per-position logits when `full_logits=True` (training / parity).

Everything else stays byte-identical to upstream, which is what makes the T2 parity check
(patched vs pristine on the same `state_dict`) meaningful.

## 5. Runtime fallbacks

`nanov41/patched_env.py` replaces the official TileLang kernels with pure-torch implementations
of identical math, so the model can run on CPU/Mac:

- `act_quant` / `fp4_act_quant` → no-op (training runs bf16);
- `sparse_attn` → gather + online softmax, with the attention sink only entering the denominator;
- `hc_split_sinkhorn` → torch Sinkhorn (softmax → column normalise → row/column iterations);
- `fp8_gemm` / `fp4_gemm` → explicit `NotImplementedError` (quantised weight paths are not supported);
- `vision` / `image_processor` → stubs (v0 excludes the vision encoder).

`ref/` is therefore a **subset** of the official `inference/` directory and is not a standalone
inference entry point: it is importable only through `patched_env` (which installs the stubs), and
the missing `vision.py` / `image_processor.py` / `encoding/` are intentional.

## 6. Parity harness (`parity.py`)

- **T1** — full-sequence logits vs prefix-by-prefix incremental last-token logits: validates the
  sliding-window mask, compressed-KV causality and indexer top-k causality.
- **T2** — patched training model vs pristine `ref/model.py` on the same weights and prefixes:
  validates that the patches changed gradients, not math.
- **T3** — backward smoke: loss finite, gradients finite and non-zero.
- **T1b** — informational probe of indexer top-k tie-breaking (device-dependent; see README
  limitations).

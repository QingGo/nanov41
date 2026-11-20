# Eval harness: T0 multi-domain BPB (+ reserved CORE-lite)

Status: **built and tested 2026-11-06** (CPU + fake-provider + tiny-model smoke; pinned
tokenizer/template end-to-end tests pass). This implements step 1 of the eval ladder in
[`eval-proxies.md`](eval-proxies.md): *pin the tokenizer, hold out fixed multi-domain slices,
build the evaluator* — then add CORE-lite.

The harness answers one narrow question honestly and cheaply: **given fixed held-out text,
what are the bits-per-byte of a checkpoint per domain and per token type?** It does **not**
generate, sample or judge: T0 is a ranking/screening instrument within one model family, not a
capability claim (see [`eval-proxies.md`](eval-proxies.md) §1–2 for when loss lies).

## 1. What was built

| Path | Role | Needs |
|---|---|---|
| `nanov41/eval/data.py` | On-disk contract: `.npy` shards, JSON manifest, token-type masks, exact byte table | numpy |
| `nanov41/eval/bpb.py` | BPB accumulator, per-type metrics, window-level bootstrap CI | numpy |
| `nanov41/eval/tokenizer.py` | Build-time tokenizer helpers: pinned-artifact loading, **exact per-token UTF-8 byte table**, encoding with offsets | `tokenizers` |
| `nanov41/eval/chat.py` | Jinja2 rendering of the pinned `chat_template.jinja` + exact assistant/tool token tagging | `jinja2` |
| `nanov41/eval/provider.py` | `LogitsProvider` protocol: `ModelProvider` (NanoV41 checkpoint) and analytic `FakeProvider` | torch (model path) |
| `nanov41/eval/mix.py` | Marin-8B reference mixes (phase-1 DCLM mix, jellyfish cooldown) + token-mix weighted BPB | numpy |
| `nanov41/eval/harness.py` | Windowing, NLL/byte accumulation, report JSON, table output | torch/numpy |
| `nanov41/eval/core.py` | **Reserved** CORE-lite interface: registry, JSONL loader, likelihood continuation scorer | torch/numpy |
| `tools/make_eval_slices.py` | Build slices from JSONL (text or agent trajectories) or synthetic smoke data | tokenizers (+jinja2 for trajectories) |
| `tools/fetch_eval_data.py` | Streaming fetch of Marin-referenced held-out raw text (DCLM, ProofPile-2, codeparrot, wikitext, StackExchange, COIG) + provenance sidecar | requests, pyarrow (parquet), zstandard (optional) |
| `tools/eval_t0.py` | CLI: evaluate a checkpoint / fake provider, write report JSON, optional `--tier t1` | torch/numpy |
| `tests/test_eval_harness.py` | 34 CPU tests: data contract, BPB closed-form math, reference mixes, CLI, reserved CORE | — |
| `tests/test_fetch_eval_data.py` | 11 CPU tests: format parsers, EOF-safe HTTP reader, fetch bookkeeping and provenance (no network) | — |
| `tests/test_eval_trajectory.py` | 8 pinned-artifact tests: byte table, template+tagging, slice builder end-to-end | tokenizer/template env |

Evaluation itself needs **no tokenizer and no network**: shards are pre-tokenized, so a
checkpoint on a 5090 can be scored offline. That is deliberate — tokenizer drift is a
correctness risk, so it is pinned once at data-build time and dead afterwards.

## 2. Data contract

```
<dataset>/
  manifest.json                     # format, tokenizer provenance, domains, shard list
  token_bytes.npy                   # uint8 [vocab_size]  UTF-8 bytes per token id
  <domain>/shard-00000.tokens.npy   # uint32 [n]
  <domain>/shard-00000.types.npy    # uint8  [n]  (omitted if every token is TEXT)
```

Token types are attached to the **label** token (the token being predicted):

| type | name | meaning |
|---:|---|---|
| 0 | IGNORE | excluded from every metric (padding / deliberately skipped) |
| 1 | TEXT | plain corpus text, system prompt, user turns |
| 2 | ASSISTANT | one assistant turn: `<｜Assistant｜>` header + reasoning + content + tool-call mark-up + EOS |
| 3 | TOOL | tool output inside `<tool_result>...</tool_result>` (wrapper included) |

Metrics are reported for `all` (union of 1/2/3) and per type (`text`, `assistant`,
`tool_output`). A domain without a types file is a pure-text domain.

`manifest.json` records the pinned artifacts, so any slice is reproducible:

```json
{
  "format": "nanov41-eval-data/1",
  "tokenizer": {"path": "...tokenizer.json", "sha256": "c90dfa01...", "vocab_size": 129280,
                "pad_id": 2, "pinned_v41_flash": true,
                "chat_template_sha256": "d959d804..."},
  "token_bytes": {"path": "token_bytes.npy", "sha256": "...", "dtype": "uint8", "n": 129280},
  "domains": {
    "agent": {"kind": "trajectory", "shards": [{"tokens": "...", "types": "...", "n_tokens": 1048576}],
              "eval": {"window": 2048, "max_tokens": null, "weight": 1.0},
              "stats": {"n_tokens": 1048576, "by_type": {"assistant": {"tokens": 140000, "bytes": 700000}}}}
  },
  "mixes": {"marin-8b-phase1": {"weights": {"web": 0.926, "code": 0.061, "math": 0.013},
                                "source": "https://github.com/marin-community/marin/.../marin-8b-retro.md"}}
}
```

### Byte accounting (why BPB is exact here)

`token_bytes[i]` is the number of UTF-8 bytes token `i` represents, computed by inverting the
GPT-2 byte-level BPE map for ordinary tokens (DeepSeek V4.1-Flash is a BBPE) and using the
literal UTF-8 length for added/special tokens (`<｜User｜>`, `<｜begin▁of▁sentence｜>`, ...).
The builder asserts the invariant on every record **before writing**:

```
sum(token_bytes[t] for t in encode(text)) == len(text.encode("utf-8"))
```

so the denominator of BPB is the byte length of exactly the tokens that were scored — not
"tokens × average bytes/token". A wrong tokenizer therefore fails the build instead of
silently producing a different metric. See [`tokenizer-contract.md`](tokenizer-contract.md).

### Windowing rule

Each shard is cut into non-overlapping windows of `window` tokens; the first token of a window
has no in-window context and is not scored. So a window contributes `window - 1` predictions,
and a trailing partial window is dropped. At `window=2048` this omits
1/2048 ≈ 0.05 % of tokens; the count is reported as `tokens_dropped`. `max_tokens` in the
manifest (or `--max-tokens-per-domain`) caps a domain so a smoke run can be minutes.

## 2b. Data sources and mix (Marin-8B reference)

Our own pretraining mix is not frozen yet, so the harness does not pretend one number is the
objective.  The reference is **Marin-8B**: a fully open model that publishes both its pretraining
mixture and its cooldown mixture
([marin-8b-retro.md](https://github.com/marin-community/marin/blob/main/docs/reports/marin-8b-retro.md)),
which makes its percentages a defensible set of domain weights.  Two mixes are defined in
`nanov41/eval/mix.py` and recorded in the dataset manifest:

| Marin phase | tokens | mixture (from the retro report) |
|---|---:|---|
| phase 1/2 Kestrel+Ocelot | 0 → 3.78 T | DCLM-Baseline **92.6 %**, StarCoder Data **6.1 %**, ProofPile-2 **1.3 %** |
| phase 3 Jellyfish cooldown | 3.78 → 4.78 T | Dolmino DCLM HQ **67.8 %**, Dolma peS2o 10.8 %, FineMath-3+ 6.3 %, Dolma Arxiv 5.2 %, Dolma StackExchange 3.2 %, StarCoder 2.2 %, Dolma Algebraic Stack 2.1 %, Dolma Open Web Math 0.9 %, Dolma Megawika 0.8 %, Dolma Wikipedia 0.7 % |

(For reference: phase 4 Phoenix switches the web source to Nemotron-CC + StarCoder; later
cooldowns reuse the cooldown pattern with new datasets.)  The jellyfish table is aggregated to the
granularity of eval domains in code — academic = peS2o + Arxiv, math = FineMath-3+ + Algebraic
Stack + Open Web Math, wikipedia = Megawika + Wikipedia — while `source_weights` keeps the
original 10-row table verbatim for auditability.

Mix weights are **token proportions**, so a mix BPB is the ratio of two weighted means, not an
average of domain BPBs:

```
mix_bpb = ( Σ w'_d · nll_d/tokens_d ) / ln2 / ( Σ w'_d · bytes_d/tokens_d ),  w'_d = w_d / coverage
```

`coverage` is the summed weight of domains present in the dataset; missing domains are listed and
the remaining weights are renormalised for the reported number.  **Never compare mix BPB across
different coverage**, and treat it as a reference summary — per-domain BPB remains primary.

### Eval-domain ↔ source mapping used by `tools/fetch_eval_data.py`

| eval domain | Marin role | source (HF repo) | holdout basis |
|---|---|---|---|
| `web` | phase-1 92.6 % / jellyfish 67.8 % | `mlfoundations/dclm-baseline-1.0`, **global-shard_10_of_10** | global shard 10 reserved as eval; never train on it |
| `code` | StarCoder 6.1 % / 2.2 % | `codeparrot/codeparrot-clean-valid` | official validation split; StarCoder Data is gated, documented fallback |
| `math` | ProofPile-2 1.3 %; jellyfish 9.3 % | `EleutherAI/proof-pile-2` **test**: algebraic-stack + open-web-math | official test split |
| `academic` | peS2o 10.8 % + Arxiv 5.2 % | `EleutherAI/proof-pile-2` **arxiv/test** | official test split (proxy for the Dolma sources) |
| `wikipedia` | Dolma Wikipedia 0.7 % + Megawika 0.8 % | `Salesforce/wikitext` wikitext-103-raw **test+validation** | official held-out split (proxy) |
| `stackexchange` | Dolma StackExchange 3.2 % | `HuggingFaceH4/stack-exchange-preferences`, three small sites | exclude those sites from training (name-level) |
| `zh` | not in Marin | `m-a-p/COIG-CQIA` wiki/zhihu JSONL | auxiliary domain, weight 0; Chinese held-out text |
| `books`, `agent` | not in Marin | not wired yet | see §7 |

`tools/fetch_eval_data.py` streams JSONL (plain/gzip/zstd) and downloads only small parquet files
(wikitext, StackExchange) under `--max-download-mb`; it stops as soon as the character budget is
reached.  Every fetch writes `<out>.provenance.json` with: dataset repo, **pinned revision sha**,
file names, record/char counts, endpoint and the holdout rationale — keep it with the data, it is
the audit trail for the disjointness claim.

```sh
# raw text (Marin-referenced domains; ~10-12 M chars ~= 3 M tokens each)
for d in web code math academic; do
  python tools/fetch_eval_data.py --domain $d --out runs/eval/raw/$d.jsonl --max-chars 12000000
done
for d in wikipedia stackexchange zh; do
  python tools/fetch_eval_data.py --domain $d --out runs/eval/raw/$d.jsonl --max-chars 4000000
done
# then tokenize into T0 shards (this is where the pinned tokenizer is used)
for d in web code math academic wikipedia stackexchange zh; do
  python tools/make_eval_slices.py --out runs/eval/data_pilot --domain $d --kind text \
      --input runs/eval/raw/$d.jsonl --window 2048 --max-tokens 3000000
done
```

`make_eval_slices.py` auto-records every reference mix whose domains are fully present, so the
pilot dataset above gets both `marin-8b-phase1` (web+code+math) and `marin-8b-jellyfish`
(all six Marin domains); `zh` is reported but carries weight 0.

### Pilot dataset (2026-11-06)

Built with the commands above (the raw JSONL + provenance live under `runs/eval/raw/`, which is
git-ignored; the shards under `runs/eval/data_pilot/`):

| domain | source (holdout basis) | tokens | bytes/token |
|---|---|---:|---:|
| `web` | DCLM-Baseline `global-shard_10_of_10` (reserved shard) | 2,738,942 | 4.401 |
| `code` | codeparrot-clean-valid (official validation split; StarCoder gated fallback) | 2,548,432 | 3.939 |
| `math` | ProofPile-2 test: algebraic-stack + open-web-math | 2,839,946 | 3.542 |
| `academic` | ProofPile-2 `arxiv/test` (proxy for Dolma Arxiv/peS2o) | 2,868,853 | 3.507 |
| `wikipedia` | WikiText-103 raw test+validation (official held-out) | 535,354 | 4.513 |
| `stackexchange` | stack-exchange-preferences, 3 small sites (exclude sites from training) | 909,956 | 4.403 |
| `zh` | COIG-CQIA wiki+zhihu (auxiliary, weight 0) | 2,589,052 | 4.380 |
| **total** | | **15,030,535** | **3.99** |

Closed-form sanity check (no model needed): with `--provider fake --fake-kind uniform` every
token has NLL = ln(129,280), so mix BPB must equal `log2(129280) / (weighted bytes/token)`.
Pilot run (`--window 256 --max-tokens-per-domain 64000`, 7 domains): `marin-8b-phase1`
3.9602, `marin-8b-jellyfish` 4.2094, macro 4.3919, byte-weighted 4.2371 — all consistent with
the per-domain byte ratios above, and both mixes report `coverage=1.0`, `missing=[]`.
These are **pipeline sanity numbers, not capability**: real scores require a checkpoint
(`tools/eval_t0.py --ckpt ...`) or the d12/d16/d20 screening runs.

This is a **pipeline-validation scale** (~1/10 of the recommended per-domain budget), enough to
exercise fetch → tokenize → shard → T0 → mix report; scale to 20–50 M tokens/domain before using
the numbers for model decisions.

## 3. Building slices

The pinned artifacts are **not vendored** (tokenizer 6.37 MB, template 15,348 B); point the
builders at them. From [`tokenizer-contract.md`](tokenizer-contract.md) (mirror works from CN):

```sh
B=https://hf-mirror.com/deepseek-ai/DeepSeek-V4.1-Flash/resolve/main
curl -L $B/tokenizer.json        -o /tmp/dsv41_tokenizer.json         # sha256 c90dfa01...
curl -L $B/chat_template.jinja   -o /tmp/dsv41_chat_template.jinja    # sha256 d959d804...
export NANOV41_DS_TOKENIZER_JSON=/tmp/dsv41_tokenizer.json
export NANOV41_DS_CHAT_TEMPLATE=/tmp/dsv41_chat_template.jinja
```

### Plain held-out text domains

One JSONL per domain, one object per line with a `text` field (any field via `--text-field`):

```sh
for d in web code math zh books; do
  python tools/make_eval_slices.py --out runs/eval/data --domain $d --kind text \
      --input /data/heldout/$d.jsonl --window 2048
done
```

Recommended held-out budget: **20–50 M tokens per domain** (fixed, never trained on; disjoint
by URL/document id from the pretraining mix). 50 M tokens at `uint32` = 200 MB/domain.

### Agent-trajectory domains

JSONL records with `messages` (HuggingFace-style roles; `reasoning_content` for V4.1
reasoning; tool calls in OpenAI `tool_calls` form; `role: tool` results) and optional `tools`:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."},
              {"role": "assistant", "reasoning_content": "...", "content": "",
               "tool_calls": [{"id": "c1", "type": "function",
                               "function": {"name": "read_file", "arguments": {"path": "x"}}}]},
              {"role": "tool", "tool_call_id": "c1", "content": "..."},
              {"role": "assistant", "content": "..."}],
 "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {...}}}]}
```

```sh
python tools/make_eval_slices.py --out runs/eval/data --domain agent --kind trajectory \
    --input /data/heldout/agent.jsonl --window 4096 \
    --thinking-mode thinking --drop-thinking false
```

Rendering uses `chat_template.jinja` exactly as the model repo intends (thinking/chat modes,
tool mark-up); the template hash goes into the manifest. `--drop-reasoning` strips
`reasoning_content`/`thinking` from inputs before rendering if you want to score visible
content only. `--drop-thinking` is the template's own option (default true, matching the V4.1
template default).

Token tagging is template-agnostic but V4.1-specific and **exact**: role headers are single
special-token ids (`<｜Assistant｜>` = 128,804, ...), so an assistant span runs from an
`<｜Assistant｜>` id to the next role header; tool results are literal
`<tool_result>...</tool_result>` spans mapped through tokenizer character offsets. Consecutive
assistant messages merge into one span; the builder reports a `*_count_mismatch` warning rather
than guessing.

**Held-out hygiene for agent data.** (a) Split prompts/tasks so eval trajectories are disjoint
from training/OPD rollouts — contamination is trivial to introduce here; (b) keep the template
pinned (manifest records its hash) — a template change invalidates comparability; (c) at this
scale trajectory BPB is a weak proxy for task success — add a small real agentic success eval
(50–200 tasks with tool execution) once the rollout engine exists; (d) always report the
per-type breakdown, never one mixed number (the system prompt + tool schemas can dominate
`text`: in the synthetic fixture, prompt tokens were 71 % of the trajectory).

### Synthetic smoke data (no downloads)

```sh
python tools/make_eval_slices.py --out runs/eval/data/smoke --domain smoke --kind text \
    --synthetic 64 --window 256
python tools/make_eval_slices.py --out runs/eval/data/smoke --domain smoke_agent \
    --kind trajectory --synthetic 16 --window 256 --drop-thinking false
```

## 4. Running T0

```sh
# data contract + harness check: no model, no GPU, no tokenizer
python tools/eval_t0.py --dataset runs/eval/data/smoke --provider fake --window 256 --json --no-save

# d12 checkpoint on the 5090 (bf16, batch 2, 4096-token windows)
python tools/eval_t0.py --dataset runs/eval/data --ckpt runs/d12.pt \
    --device cuda --dtype bf16 --window 4096 --batch-size 2 --out runs/eval/t0_d12.json

# CPU smoke: random tiny model (tiny config, window <= 64)
python tools/eval_t0.py --dataset runs/eval/data/tiny --config tiny --random-init --window 64

# subset of domains / token budget / report location
python tools/eval_t0.py --dataset runs/eval/data --domains web,code,agent \
    --ckpt runs/d12.pt --max-tokens-per-domain 5000000 --out runs/eval/t0_d12.json
```

`--provider fake --fake-kind uniform` has a closed-form answer: every token NLL = `ln(vocab)`,
so BPB = `log2(vocab) / (bytes per token)` — a free end-to-end sanity check of data, masks,
windowing and arithmetic. `--fake-kind oracle` gives BPB 0.0.

### Cost estimate (T0, forward only)

d12 forward = 141.3 MFLOP/token (measured; 138.9 analytic). At the measured training
throughput of 124 TFLOP/s effective (59.4 % raw MFU, R69), 50 M tokens/domain takes
50e6 × 141.3e6 / 124e12 ≈ **57 s/domain** on one 5090. Six domains × 50 M ≈ 6 min plus
checkpoint load. Evaluation is ~1/3 of a training step per token, so even 6 domains is
negligible next to the runs it screens — exactly why multi-domain BPB should be run on every
candidate.

### Reading the report

```json
{"schema": "nanov41-eval-report/1",
 "dataset": {"domains": {"agent": {"metrics": {"all": {"bpb": 0.98, "tokens": 4088, "bytes": 21600},
                                                "assistant": {"bpb": 0.91, ...},
                                                "tool_output": {"bpb": 1.42, ...}}}},
             "summary": {"macro_bpb": 1.02, "byte_weighted_bpb": 0.99}}}
```

- `macro_bpb` treats domains equally; `byte_weighted_bpb` pools NLL/bytes. Report both — they
  answer different questions.
- `mixes` (section 2b) adds reference-weighted BPBs: `marin-8b-phase1` /
  `marin-8b-jellyfish`, each with `coverage`, `missing` domains and per-domain contributions.
  A mix is only recorded when the dataset fully contains its domains; partial coverage is
  possible for hand-written mixes and is always reported.
- Bootstrap CI (optional `--bootstrap 1000`) resamples windows; it is a **rough stability
  check**, not a rigorous CI (windows are correlated, model uncertainty dominates).
- Comparisons are only meaningful on the same manifest hash, same window/batch size, same
  dtype. `eval_t0.py` records all of them.

### Determinism notes

Forward-only, fixed windows, no dropout in the model, no shuffle. GPU reductions can differ in
the last bits across batch sizes/kernels: keep `--batch-size`/`--window`/`--dtype` fixed when
comparing checkpoints (same discipline as the MFU A/B rules). MoE routing is deterministic in
eval (top-k over logits). Random-init smoke runs are seeded but not informative.

## 5. CORE-lite (T1) — reserved interface

`nanov41/eval/core.py` implements the reusable half of T1 and deliberately stops before the
data port:

- **implemented**: task registry (`arc_easy, arc_challenge, hellaswag, piqa, winogrande, boolq,
  openbookqa`), JSONL loader with validation, continuation NLL via the same `LogitsProvider`
  primitive, normalisation (`none` / `token` / `byte`), accuracy + centred accuracy aggregation;
- **not implemented** (pinned when the data lands, not guessed): official task files, few-shot
  prompt formatting, and the exact normalisation used by the reference CORE evaluator — those
  choices move absolute scores, so they belong to the data port.

Expected layout — `<dataset>/core/<task>.jsonl`, one object per line:

```json
{"context": "<few-shot prompt + question>", "choices": ["answer A", "answer B"], "label": 0,
 "example_id": "optional"}
```

`context` is used verbatim (few-shot assembly is part of the data port); each choice is
tokenised as a continuation and scored after the tokenised context:

```sh
python tools/eval_t0.py --dataset runs/eval/data --tier t1 --ckpt runs/d12.pt \
    --device cuda --dtype bf16
```

Without task data the CLI prints `[t1] CORE-lite data missing (reserved interface ...)` and the
T0 report is unaffected. When data exists, `--tier t1` adds a `t1` block to the report JSON.
For external comparability, CORE is the boundary object vs nanochat (GPT-2 CORE 0.2565); do
not compare loss/token or tokens/s across tokenizers.

## 6. Tests

```sh
python -m pytest tests/test_eval_harness.py tests/test_fetch_eval_data.py -q   # 45 CPU tests
NANOV41_DS_TOKENIZER_JSON=/tmp/dsv41_tokenizer.json \
NANOV41_DS_CHAT_TEMPLATE=/tmp/dsv41_chat_template.jinja \
python -m pytest tests/test_eval_trajectory.py -q                  # 8 pinned-artifact tests
```

The CPU tests cover: manifest/shard round-trip, token-bytes mixing rejection, windowing edge
cases, closed-form BPB for uniform (log2 vocab) and oracle (0) providers, per-type masks,
IGNORE semantics, `max_tokens`, bootstrap CI, vocab-mismatch guard, CLI end-to-end (fake +
oracle + `--list-domains` + error paths), tiny-model forward on CPU, and the reserved CORE-lite
loader/scorer.

## 7. What remains (in order)

1. **Data (CPU work, free):** ✅ pilot built 2026-11-06 (Marin-referenced sources, §2b,
   ~0.5–3 M tokens/domain). Remaining: scale to 20–50 M tokens/domain; wire the two planned
   domains — `books` (manu/project_gutenberg) and `agent` (public tool-call trajectories rendered
   with the pinned template, disjoint from OPD rollouts); build into `runs/eval/data/`.
   Consider a uint32 → 17-bit packed storage pass only if disk becomes the constraint
   ([`tokenizer-contract.md`](tokenizer-contract.md) §3).
2. **CORE-lite data port:** port the official DCLM CORE task files + few-shot formatting +
   reference normalisation into `core/*.jsonl`; validate the scorer on a known checkpoint.
3. **Screening:** run T0 on d12 vs d16 vs d20 at C≈1e18 FLOPs (3 models ≈ ¥18.7) and read the
   multi-domain BPB; then extend the top-2 to C≈2e18 (CORE-lite) only if the slope is positive —
   the ¥20/¥40 decision table is in
   [`capacity-vs-compute-codesign.md`](capacity-vs-compute-codesign.md) §4c.
4. **Agent eval (later):** small real agentic success eval (task success / tool-call validity)
   once the rollout engine exists; BPB stays the cheap regression guard.
5. **T5 long-context:** RULER/needle needs the chunked-prefill path (`nanov41/chunked_prefill.py`)
   rather than T0 windows; not started.

## 8. Non-goals / caveats

- T0 is a **screening** metric, not a capability score; nothing here generates text.
- Absolute BPB is tokenizer-dependent *unless* the held-out bytes are identical — our manifest
  pins one tokenizer, so within-family comparisons are valid; cross-tokenizer claims are not.
- The window rule scores documents independently at boundaries; for held-out text this is
  negligible at 20–50 M tokens, but do not compare against long-context metrics computed with
  document-level windows.
- `INCLUDE system prompt` in trajectory `text`: with tool schemas the system prompt can be most
  of the tokens. Use the `assistant` / `tool_output` subsets for agent-domain decisions; the
  `all` number is still useful as a format/modeling check.

## NLL fast path (native inference kernels, 2026-11-07)

`token_nll_batch` now prefers a provider-level `nll(input_ids)` that computes the head GEMM and
logsumexp tiled over positions instead of materialising `[B,S,129280]` logits:

- `ModelProvider.nll` (chunked, exact parity vs the legacy logits path; test
  `tests/test_eval_nll.py`) — peak memory roughly halves and **batch 32 now fits where the legacy
  path OOMs**; `--nll-mode chunked|auto` (default auto) selects it.
- optional `--nll-mode fused`: `fused_head_nll_v3` (the training FP8 fused head, forward-only).  On
  RTX 5090, b=16×1024: **128.1k tok/s, 3.24 GB peak** vs 90.9k/9.04 GB chunked vs 88.0k/17.5 GB
  legacy; batch32 exceeds v3's int32 `TP*V < 2^31` limit so it is split/falls back.
- **accuracy caveat**: fused v3 is an FP8 head; per-token NLL vs the bf16 chunked head has
  mean |delta| 0.060 nats (p99 0.237, bias +0.0058 nats ≈ 0.008 BPB).  Keep `auto`/`chunked` for
  arm-to-arm reports; use `fused` only when both runs use it or for speed/exploration.
- `--nll-mode logits` + `NANOV41_EVAL_LEGACY_LOGITS=1` keeps the old path for regression checks.

Raw numbers: `runs/p2b/eval_nll_benchmark.json`.

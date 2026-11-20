# Tokenizer contract: reuse DeepSeek-V4.1-Flash verbatim (no tokenizer training)

Status: verified 2026-11-06. The project does **not** train a tokenizer. The model config's
129,280-token vocabulary, the Engram compressed-hash map, and the future OPD teacher API's top-20
token ids all require **exactly the DeepSeek-V4.1-Flash tokenizer**, so we reuse it as an immutable
artifact.

## 1. Pinned artifacts

| artifact | source | size | sha256 |
|---|---|---:|---|
| `tokenizer.json` | `deepseek-ai/DeepSeek-V4.1-Flash` | 6,367,257 B | `c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b` |
| `tokenizer_config.json` | same repo | 801 B | (no chat template; bos 0 / eos 1 / pad 2) |
| `chat_template.jinja` | same repo | 15,348 B | chat/thinking template (needed for SFT/OPD rollouts) |
| `config.json` → `text_config` | same repo | 3,311 B | vocab 129,280; pad_token_id 2; engram fields match `ref/config.json` |

Fetch (mirror works from CN):

```sh
B=https://hf-mirror.com/deepseek-ai/DeepSeek-V4.1-Flash/resolve/main
curl -L $B/tokenizer.json          -o /tmp/dsv41_tokenizer.json
curl -L $B/tokenizer_config.json   -o /tmp/dsv41_tokenizer_config.json
curl -L $B/chat_template.jinja     -o /tmp/dsv41_chat_template.jinja
```

## 2. Verified contract (`tests/test_tokenizer_engram.py`)

| property | value | where it matters |
|---|---:|---|
| vocab size | **129,280** | `nanov41/configs.py` (`vocab_size`), embedding + head shape |
| pad token id | **2** (`<｜▁pad▁｜>`) | `engram_pad_id`, CE ignore/pad handling |
| bos / eos | 0 / 1 | sequence framing / SFT |
| Engram compressed-token map | **99,092** | `engram_compressed_vocab_size`; every hash multiplier derives from it |
| char-normalization | NFKC + lowercase + accent strip + whitespace collapse | n-gram hashing in `ref/engram.py` |

Run it with the pinned artifact:

```sh
NANOV41_DS_TOKENIZER_JSON=/tmp/dsv41_tokenizer.json pytest tests/test_tokenizer_engram.py -q
```

**Version pitfall:** the older **DeepSeek-V3** tokenizer looks similar but does *not* satisfy the
contract — vocab 128,815 and compressed map 98,627. Using it silently re-hashes every Engram table
and breaks OPD id alignment. Pin V4.1-Flash (hash above).

## 3. Consequences

- **No tokenizer training / no tokenizer cost.** The data pipeline is only
  tokenize → dedupe/filter → pack → shard, with this tokenizer; it is days of CPU work, not weeks.
- **Storage encoding:** ids reach 129,279 (> 65,535), so `uint16` is not enough. Either
  `uint32` (4 B/token ⇒ 100B tokens = 400 GB) or 17-bit packed (~2.125 B/token ⇒ ~213 GB).
- **OPD alignment:** the teacher API must be the V4.1-Flash endpoint; its top-20 token ids are in
  this vocabulary. Pin the teacher model/version and cache the tokenizer hash alongside the data.
- **Engram:** the compressed map is verified, so the hash-parity P0 (ref/engram.py vs
  engram-peft vs engramdb-keygen) can run now; `engram_pad_id=2` is confirmed.
- **SFT/OPD chat formatting:** use `chat_template.jinja` from the model repo (thinking/chat modes,
  tool-call markers) instead of hand-rolling prompts.

## 4. What actually remains (updated from earlier plans)

| item | status |
|---|---|
| tokenizer | ✅ reuse V4.1-Flash verbatim (this note) |
| data pipeline (tokenize/dedupe/pack/shard, uint32 or packed) | to build — CPU, days |
| eval harness (held-out BPB + CORE/task proxies + long-context RULER/needle) | to build — CPU/GPU, days |
| OPD rollout/inference engine + teacher API/top-20 storage | to build — the main OPD work |
| Engram hash parity P0 / nano-Engram implementation | cheap; tokenizer unblocked |

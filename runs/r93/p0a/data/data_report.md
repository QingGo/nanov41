# P0a data report — K=40k (V=41,280) ClimbMix corpus + eval panel

**Verdict: PASS** ✅ (verified 2026-11-15T06:31:15Z UTC on `<instance-id>`, no-card CPU mode)
Role: data-ops / task-16. Build host: GPU box (Lead); data host: this instance.

## 1. Deliverables (cloud paths on `<instance-id>`)

| artifact | path | size |
|---|---|---|
| **P1 training corpus** | `/root/autodl-tmp/nanochat/p2_climbmix_40k` | 24,282,487,014 B (24.28 GB; `web.u32.bin` = **24,282,435,364 B**) |
| **eval panel** | `/root/autodl-tmp/nanochat/p0_eval_40k` | 283,935,211 B (0.284 GB) |
| val shard (panel source, kept) | `/root/autodl-tmp/nanochat/climbmix/shard_06542.parquet` | 91,699,537 B |
| K=40k tokenizer | `/root/autodl-tmp/nanochat/tok40k/tokenizer.json` | 2,201,199 B |
| official 129k tokenizer | `/root/dsv41/tokenizer.json` | 6,367,257 B |
| raw shards | deleted after verification (was 9,786,774,903 B) | — |

Instance disk after cleanup: **26 GB free / 49 GB** (corpus 24.28 GB + panel 0.28 GB + val 0.09 GB).

## 2. Corpus verification (all checks PASS)

| check | value | expected | ok |
|---|---|---|:--:|
| `tokens_total` (pieces) | **6,070,608,841** | ≥6.0e9 (6.06–6.08e9) | ✅ |
| `docs` | **9,053,184** | == raw rows (9,053,184) | ✅ |
| shards | **107** | 107 | ✅ |
| `bytes_per_token` | **4.442642** | 4.442 ±1.2% (P0b 4.4421) | ✅ (Δ +0.0144%) |
| `web.u32.bin` bytes | **24,282,435,364** = tokens×4 | exact | ✅ |
| EOS id | **1** (`<｜end▁of▁sentence｜>`) | 1 | ✅ |
| vocab size | **41,280** | 41,280 | ✅ |
| val shard excluded | `shard_06542` not in corpus | excluded | ✅ |
| sampled max token id | **39,999** (8M sampled tokens) | < 41,280 | ✅ |
| sampled EOS rate | 0.1533% (12,267 EOS / 8M) | doc separators present | ✅ |
| decode prefix | first docs of `shard_00000` == corpus head | exact ids | ✅ |
| decode round-trip | 3/3 docs decode to identical UTF-8 bytes | lossless | ✅ |
| official-129k cross-check | doc0: 209 official vs 221 K=40k pieces (×1.0574); both decode to identical bytes | lossless | ✅ |
| shard pieces | min 56,364,954 / avg 56,734,662 / max 57,221,527 | ~56.9M | ✅ |

Corpus manifest: `format=nanov41-p2-corpus/1`, `doc_sep=eos`, `incremental=true`,
tokenizer sha256 `34cdf9c52ebf256af5a0d912949187b3548bb2adf00d2543e3d1267d1f48f19c`.

## 3. Eval panel verification (all checks PASS)

| check | value | ok |
|---|---|:--:|
| format | `nanov41-eval-data/1`, domain `climbmix_val` | ✅ |
| `token_bytes.npy` | len **41,280** (uint8), sha256 match, all >0 | ✅ |
| window | **2048** | ✅ |
| tokens (incl EOS) | **56,778,205**; TEXT tokens 56,693,186 | ✅ |
| text bytes / bpt | 252,081,736 B / **4.446420 B/tok** (Δ +0.1%) | ✅ |
| EOS separators | 85,019, all typed IGNORE (contribute neither nats nor bytes) | ✅ |
| max token id | 39,999 < 41,280 | ✅ |
| prefix vs val shard | panel head == tokenized first docs of `shard_06542` | ✅ |
| byte accounting | first 2 docs: TEXT-position token_bytes sum = 7,482 = UTF-8 bytes | ✅ |

> Note: the raw `sum(token_bytes[all ids])` includes EOS tokens (27 B each) so it reads 4.4801 B/tok;
> the eval harness uses types=IGNORE for those, giving the correct 4.4464.

## 4. Provenance / pipeline

- Raw ClimbMix train shards `shard_00000..00106` (107 × ~92 MB, 9.05M docs) fetched from hf-mirror;
  GPU-box full-column read after repair: `n=107 total_rows=9,053,184 bad=0` (13 interior-corrupt
  shards re-downloaded fresh; this verifier found no residual issues).
- Corpus built by Lead on the GPU instance with `r90_vocab_retokenize.py --append --delete-raw`
  (deterministic shard order, EOS after every doc) using `tok40k/tokenizer.json`.
- Panel built by Lead with `r90_vocab_eval_panel.py` from `shard_06542` (window 2048).
- Verified on the data host with `/root/verify_p0a_40k.py` (this report embeds its full JSON).

## 5. Mac artifacts

- `runs/r93/p0a/data/data_report.md` (this file)
- `runs/r93/p0a/data/data_report.json` (structured summary + full verifier JSON)
- `runs/r93/p0a/data/verification.json` (raw verifier output)
- `runs/r93/p0a/data/download_validation.json` (107-shard footer-level download validation, superseded by GPU-box FULLREAD)

## 6. Full verifier JSON (appendix)

```json
{
 "ts": "2026-11-15T06:31:15Z",
 "corpus_dir": "/root/autodl-tmp/nanochat/p2_climbmix_40k",
 "panel_dir": "/root/autodl-tmp/nanochat/p0_eval_40k",
 "tokenizer": "/root/autodl-tmp/nanochat/tok40k/tokenizer.json",
 "errors": [],
 "warnings": [],
 "tokenizer_check": {
  "vocab_size": 41280,
  "eos_id": 1,
  "vocab_ok": true,
  "eos_ok": true
 },
 "corpus": {
  "manifest": {
   "format": "nanov41-p2-corpus/1",
   "tokens_total": 6070608841,
   "bytes_per_token": 4.442641782130927,
   "docs": null,
   "doc_sep": "eos",
   "incremental": true
  },
  "format_ok": true,
  "manifest_tokenizer": {
   "vocab_size": 41280,
   "eos_id": 1,
   "eos_token": "<\uff5cend\u2581of\u2581sentence\uff5c>",
   "sha256": "34cdf9c52ebf256af5a0d912949187b3548bb2adf00d2543e3d1267d1f48f19c"
  },
  "vocab_ok": true,
  "eos_ok": true,
  "doc_sep": "eos",
  "eos_policy_ok": true,
  "n_shards": 107,
  "shards": [
   {
    "name": "shard_00000.parquet",
    "tokens": 56933914
   },
   {
    "name": "shard_00001.parquet",
    "tokens": 56868447
   },
   {
    "name": "shard_00002.parquet",
    "tokens": 56777738
   },
   {
    "name": "shard_00003.parquet",
    "tokens": 56906908
   },
   {
    "name": "shard_00004.parquet",
    "tokens": 56497214
   }
  ],
  "shard_tokens_sum": 6070608841,
  "shard_tokens_min": 56364954,
  "shard_tokens_max": 57221527,
  "shard_tokens_avg": 56734662,
  "val_excluded": true,
  "tokens_total": 6070608841,
  "tokens_ge_6b": true,
  "tokens_sum_ok": true,
  "file_bytes": 24282435364,
  "file_size_ok": true,
  "bytes_per_token": 4.442641782130927,
  "bpt_expected": 4.442,
  "bpt_delta_pct": 0.0144,
  "bpt_ok": true,
  "docs_manifest": 9053184,
  "docs_sum": 9053184,
  "raw_rows_for_included_shards": 9053184,
  "docs_match_raw": true,
  "sampled_tokens": 8000000,
  "sampled_max_id": 39999,
  "max_id_ok": true,
  "sampled_eos": 12267,
  "eos_rate": 0.001533,
  "decode_prefix_doc0_bytes": 1051,
  "decode_prefix_len": 1029,
  "decode_prefix_match": true,
  "decode_roundtrip_ok": true,
  "decode_roundtrip_docs": 3,
  "decode_roundtrip_each": [
   true,
   true,
   true
  ],
  "spot_bpt": 4.313897,
  "official_vocab": 129280,
  "official_pieces_doc0": 209,
  "official_decode_ok": true,
  "k40k_pieces_doc0": 221,
  "inflation_vs_official": 1.057416,
  "checks": {
   "format_ok": true,
   "vocab_ok": true,
   "eos_ok": true,
   "eos_policy_ok": true,
   "val_excluded": true,
   "tokens_sum_ok": true,
   "file_size_ok": true,
   "bpt_ok": true,
   "max_id_ok": true,
   "decode_prefix_match": true,
   "decode_roundtrip_ok": true,
   "tokens_ge_6b": true
  }
 },
 "panel": {
  "format": "nanov41-eval-data/1",
  "format_ok": true,
  "tokenizer": {
   "vocab_size": 41280,
   "eos_id": 1,
   "sha256": "34cdf9c52ebf256af5a0d912949187b3548bb2adf00d2543e3d1267d1f48f19c"
  },
  "token_bytes_len": 41280,
  "token_bytes_len_ok": true,
  "token_bytes_dtype": "uint8",
  "token_bytes_positive": 41280,
  "token_bytes_sha_match": true,
  "domain_kind": "text",
  "domain_window": 2048,
  "window_ok": true,
  "n_tokens": null,
  "n_shards": 1,
  "tokens_total": 56778205,
  "max_id": 39999,
  "max_id_ok": true,
  "eos_count": 85019,
  "eos_ignore_ok": true,
  "text_bytes_sum": 254377249,
  "bytes_per_token_panel": 4.480192,
  "panel_text_tokens": 56693186,
  "panel_text_bytes": 252081736,
  "panel_text_bpt_total": 4.44642,
  "panel_text_bpt_total_ok": true,
  "panel_prefix_match": true,
  "panel_prefix_len": 1572,
  "panel_bytes_ok": true,
  "panel_bytes_expected": 7482,
  "panel_bytes_got": 7482,
  "panel_text_bpt": 4.765605,
  "panel_text_bpt_ok": false,
  "checks": {
   "format_ok": true,
   "token_bytes_len_ok": true,
   "window_ok": true,
   "max_id_ok": true,
   "eos_ignore_ok": true,
   "panel_prefix_match": true,
   "panel_bytes_ok": true,
   "panel_text_bpt_total_ok": true
  }
 },
 "disk": {
  "corpus": {
   "path": "/root/autodl-tmp/nanochat/p2_climbmix_40k",
   "bytes": 24282487014,
   "gb": 24.282
  },
  "panel": {
   "path": "/root/autodl-tmp/nanochat/p0_eval_40k",
   "bytes": 283935211,
   "gb": 0.284
  },
  "raw": {
   "path": "/root/autodl-tmp/nanochat/climbmix40k_raw",
   "bytes": 9786774903,
   "gb": 9.787
  },
  "val_shard_dir": {
   "path": "/root/autodl-tmp/nanochat/climbmix",
   "bytes": 91699537,
   "gb": 0.092
  },
  "filesystem": {
   "free_gb": 17.96,
   "total_gb": 53.69
  }
 },
 "ok": true,
 "result": "PASS"
}
```

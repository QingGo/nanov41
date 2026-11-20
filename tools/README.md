# tools/

Small public utilities. Everything else under `tools/` is internal development material
(one-off experiment runners, cloud-instance tooling, SSH sync helpers) and is **not** part of
the public repository — see `.gitignore`.

| File | What it does | Requirements |
|---|---|---|
| `make_patched_ref.py` | Generates `nanov41/_ref_patched_model.py` from `ref/model.py`; `--check` verifies the checked-in file is up to date (run by CI) | Python, no GPU |
| `ab_bench.py` | Interleaved A/B benchmark of head × MoE implementations (median of rounds, robust against host CPU contention) | CUDA |
| `mfu_accounting.py` | Precision-discounted FLOP/MFU accounting (bf16/fp8/fp4 = 1/0.5/0.25, fixed bf16 denominator); R50 presets | Python, no GPU |
| `v41_scaling_check.py` | Compares nano-v41 activated-parameter / decode-FLOPs scaling with the DeepSeek-V4.1-Flash report numbers | Python, no GPU |
| `make_eval_slices.py` | Builds held-out eval shards for the T0 harness (text or agent trajectories): tokenize → exact UTF-8 byte table → `.npy` shards + manifest; renders trajectories with the pinned `chat_template.jinja` | Python + `tokenizers` (+`jinja2` for trajectories); no GPU |
| `bench_engram_gpu.py` | Interleaved A/B of the d12 training step with/without Engram on a 5090 (eager fused-head path): step ms, overhead %, peak memory, table stats; writes JSON evidence | CUDA GPU, tokenizer for Engram |
| `engram_hash_parity.py` | P0 Engram gate: official V4.1 layout rows = sum of prime buckets, nano preset rows, deterministic/in-range/chunk-continuous n-gram hash ids (needs the pinned tokenizer for hash ids) | Python + torch; no GPU |
| `fetch_eval_data.py` | Fetches held-out raw text for the T0 domains from Marin-referenced public sources (DCLM-Baseline shard 10, ProofPile-2 test splits, codeparrot validation, WikiText-103 test, StackExchange, COIG zh); streams JSONL, small parquet only; writes a provenance sidecar with pinned revision sha and holdout rationale | Python + requests/pyarrow (+zstandard); network |
| `eval_t0.py` | T0 evaluator: per-domain / per-token-type bits-per-byte from pre-tokenized shards; `--provider fake` smoke needs no model or GPU; `--tier t1` hooks the reserved CORE-lite interface | Python + torch (model path only); no GPU for fake/smoke |

Notes:

- `make_patched_ref.py --check` is what keeps the vendored `ref/model.py` and the generated
  training module from drifting; run it after any `ref/` update.
- `ab_bench.py` exits with an error on machines without CUDA.
- The eval data pipeline needs the pinned DeepSeek-V4.1-Flash tokenizer (and chat template for
  trajectory domains); evaluation itself reads pre-tokenized shards and needs neither. See
  `docs/eval-harness.md`.
- CPU kernel-parity regression scripts that used to live in `tools/` were promoted to
  `tests/kernel_parity_*.py` (run by CI); `tl_kernels/` experimental code was curated into
  `experiments/tl_kernels/`.
- `tools/` used to contain cloud ops tooling (instance start/stop over a web-console API, SSH sync)
  and ~120 one-off W2 kernel/benchmark scripts. Those are intentionally not versioned.

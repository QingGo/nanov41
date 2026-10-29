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

Notes:

- `make_patched_ref.py --check` is what keeps the vendored `ref/model.py` and the generated
  training module from drifting; run it after any `ref/` update.
- `ab_bench.py` exits with an error on machines without CUDA.
- CPU kernel-parity regression scripts that used to live in `tools/` were promoted to
  `tests/kernel_parity_*.py` (run by CI); `tl_kernels/` experimental code was curated into
  `experiments/tl_kernels/`.
- `tools/` used to contain cloud ops tooling (instance start/stop over a web-console API, SSH sync)
  and ~120 one-off W2 kernel/benchmark scripts. Those are intentionally not versioned.

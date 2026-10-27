# tools/

Small public utilities. Everything else under `tools/` is internal development material
(one-off experiment runners, cloud-instance tooling, SSH sync helpers) and is **not** part of
the public repository — see `.gitignore`.

| File | What it does | Requirements |
|---|---|---|
| `make_patched_ref.py` | Generates `nanov41/_ref_patched_model.py` from `ref/model.py`; `--check` verifies the checked-in file is up to date (run by CI) | Python, no GPU |
| `ab_bench.py` | Interleaved A/B benchmark of head × MoE implementations (median of rounds, robust against host CPU contention) | CUDA |

Notes:

- `make_patched_ref.py --check` is what keeps the vendored `ref/model.py` and the generated
  training module from drifting; run it after any `ref/` update.
- `ab_bench.py` exits with an error on machines without CUDA.
- CPU kernel-parity regression scripts that used to live in `tools/` were promoted to
  `tests/kernel_parity_*.py` (run by CI); `tl_kernels/` experimental code was curated into
  `experiments/tl_kernels/`.
- `tools/` used to contain cloud ops tooling (instance start/stop over a web-console API, SSH sync)
  and ~120 one-off W2 kernel/benchmark scripts. Those are intentionally not versioned.

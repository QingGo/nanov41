# experiments/

Optional, **not installed** by `pip install nanov41` and not needed for `parity.py` / `train.py`.
Everything here is experimental "best-effort" code: it was measured on specific hardware and
torch/TileLang versions. Only its CPU tests run in CI; the GPU kernels are not CI-validated.

| Directory | What | Requirements |
|---|---|---|
| [`tl_kernels/`](tl_kernels/) | sm_120 (RTX 5090) NVFP4 block-scaled GEMM + fused quantizer, fused FP8 cast; pure-torch host quantizer has CPU tests | TileLang + CUDA sm_120 for the kernels; CPU (`torch`) for the host quantizer and its tests |

Run the CPU tests:

```bash
python -m pytest experiments/tl_kernels/tests -q
```

See each experiment's README for constraints, measured numbers and attribution.

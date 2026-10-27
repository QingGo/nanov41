# ref/ — vendored official inference reference (subset)

These files are copied **byte-identically** from the official
[deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
`inference/` directory:

| File | Upstream |
|---|---|
| `model.py` | model definition (`Transformer`, `ModelArgs`) |
| `kernel.py` | TileLang kernels (`sparse_attn`, `hc_split_sinkhorn`, quant, ...) |
| `engram.py` | Engram n-gram hash memory |
| `convert.py` | HF → tensor-parallel checkpoint converter |
| `generate.py` | minimal autoregressive generation script |
| `config.json` | released model config |

**License:** MIT, Copyright (c) 2023 DeepSeek — see [`../NOTICE`](../NOTICE).

**Do not edit these files.** They are kept pristine so the parity harness can compare the
training port against the official code path. `tools/make_patched_ref.py --check` verifies that
the generated `nanov41/_ref_patched_model.py` still matches `ref/model.py` + the five documented
patches.

**This is a subset, not a standalone inference entry point.** The official implementation also
imports `vision`, `image_processor` and the `encoding/` package, which are intentionally not
vendored here. The training port exposes the official model through
`nanov41.patched_env.import_patched()`, which installs torch stubs for those modules and for the
TileLang kernels, allowing parity runs on CPU/Mac. For real inference, use the upstream
repository together with the official weights and tokenizer.

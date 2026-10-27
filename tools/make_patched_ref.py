"""Generate `nanov41/_ref_patched_model.py` from the untouched official reference.

The official `ref/model.py` is inference-first: buffers detach the compression/indexer
paths (fine for serving, fatal for gradients) and `Transformer.forward` is decorated with
`torch.inference_mode()`. This script copies the file and applies exactly five minimal,
documented patches — everything else stays byte-identical to the official code so that
the parity tests (patched vs pristine) are meaningful.

Usage
-----
    python tools/make_patched_ref.py          # regenerate the patched module
    python tools/make_patched_ref.py --check  # verify it is up to date; exit 1 on drift

`--check` is what CI runs: it fails if `ref/model.py` and `nanov41/_ref_patched_model.py`
have drifted apart.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "ref" / "model.py"
DST = ROOT / "nanov41" / "_ref_patched_model.py"


def build_patched_source() -> str:
    """Return the patched module source. Raises AssertionError if a patch anchor moved."""
    src = SRC.read_text(encoding="utf-8")

    # P1: drop inference_mode decorators (training needs autograd).
    src = src.replace("    @torch.inference_mode()\n", "")

    # P2: prefill compression shares the local latent (grad-carrying) instead of the KV buffer.
    old = """        if self.is_kv_source:
            latent = self.compressor(x, start_pos)
            shared_attn.compress_kv = self.compress_kv_cache"""
    new = """        if self.is_kv_source:
            latent = self.compressor(x, start_pos)
            # PATCH: prefill shares the local (grad-carrying) latent; decode reads the ring cache.
            shared_attn.compress_kv = latent if (start_pos == 0 and latent is not None) else self.compress_kv_cache"""
    assert old in src, "compress-kv patch anchor not found"
    src = src.replace(old, new)

    # P3: prefill index keys likewise stay local so the indexer gets gradients.
    old = """        if self.owns_k:
            shared_attn.index_k = self.k_cache"""
    new = """        if self.owns_k:
            # PATCH: prefill shares the local (grad-carrying) index keys; decode reads the ring cache.
            shared_attn.index_k = k if (start_pos == 0 and latent is not None) else self.k_cache"""
    assert old in src, "index-k patch anchor not found"
    src = src.replace(old, new)

    # P4/P5: Transformer.forward gains a full_logits switch (training / parity inspection).
    old = "    def forward(\n        self, input_ids: torch.Tensor, start_pos: int = 0, images=None, token_types: torch.Tensor | None = None\n    ):"
    new = "    def forward(\n        self,\n        input_ids: torch.Tensor,\n        start_pos: int = 0,\n        images=None,\n        token_types: torch.Tensor | None = None,\n        full_logits: bool = False,\n    ):"
    assert old in src, "forward signature anchor not found"
    src = src.replace(old, new)

    old = """        h = layer.hc_pre(h, pre_mix)
        logits = self.head(self.norm(h))
        output_ids = sample(logits, self.temperature)"""
    new = """        h = layer.hc_pre(h, pre_mix)
        logits = self.head(self.norm(h), full_logits=full_logits)
        if full_logits:  # PATCH: training/inspection returns per-position logits
            return logits
        output_ids = sample(logits, self.temperature)"""
    assert old in src, "forward logits patch anchor not found"
    src = src.replace(old, new)

    header = (
        '"""AUTO-GENERATED from ../ref/model.py by tools/make_patched_ref.py. Do not edit.\n\n'
        "Patches (all documented in the script):\n"
        "  P1 inference_mode decorators removed (autograd)\n"
        "  P2 prefill compressed KV shared as a local tensor instead of the detached cache\n"
        "  P3 prefill index keys shared as a local tensor instead of the detached cache\n"
        "  P4 Transformer.forward gains full_logits parameter\n"
        "  P5 forward returns per-position logits when full_logits=True\n\n"
        "Do not import this module directly: it uses the official code's sibling imports\n"
        "(`engram`, `kernel`, ...). Load it through `nanov41.patched_env.import_patched()`.\n"
        '"""\n\n'
    )
    return header + src


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="only verify that the checked-in file is up to date (exit 1 on drift)")
    args = ap.parse_args(argv)

    text = build_patched_source()
    if args.check:
        if not DST.exists():
            print(f"MISSING {DST}", file=sys.stderr)
            return 1
        if DST.read_text(encoding="utf-8") != text:
            print(f"DRIFT {DST} is out of date; run: python tools/make_patched_ref.py", file=sys.stderr)
            return 1
        print(f"OK {DST} is up to date ({len(text.splitlines())} lines)")
        return 0

    DST.write_text(text, encoding="utf-8")
    print(f"wrote {DST} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

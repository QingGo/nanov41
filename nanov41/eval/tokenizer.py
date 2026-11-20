"""Tokenizer helpers for the eval data pipeline (not needed at eval time).

Evaluation itself only reads pre-tokenized shards, so the pinned DeepSeek-V4.1-Flash
tokenizer is needed only when *building* eval data (`tools/make_eval_slices.py`) and when
computing the per-token UTF-8 byte table used by bits-per-byte.  This module also owns the
byte-accounting contract:

    bytes(text) == sum(token_bytes[tok] for tok in encode(text))

The DeepSeek tokenizer is a byte-level BPE (BBPE) with GPT-2 style byte<->unicode mapping,
so the byte length of an ordinary token is recovered by inverting that mapping.  Added
tokens (``<｜begin▁of▁sentence｜>``, ``<｜User｜>`` ...) are *literal* strings rather than
byte-mapped text, so their UTF-8 length is used instead.  `validate_token_bytes` checks the
invariant on real text and is run by the slice builder before any shard is written.

The tokenizer artifact itself is pinned by `docs/tokenizer-contract.md`; this project does
not train or vendor a tokenizer.
"""

from __future__ import annotations

import hashlib
import pathlib
from typing import Iterable, Sequence

import numpy as np

# DeepSeek V4.1-Flash vocabulary/contract (see docs/tokenizer-contract.md).
DS_VOCAB_SIZE = 129_280
DS_PAD_TOKEN = "<｜▁pad▁｜>"
DS_BOS_TOKEN = "<｜begin▁of▁sentence｜>"
DS_EOS_TOKEN = "<｜end▁of▁sentence｜>"


def load_tokenizer(path: str | pathlib.Path):
    """Load the pinned tokenizer with the `tokenizers` backend (lazy import)."""
    tokenizers = _require_tokenizers()
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(f"tokenizer.json not found: {path}")
    return tokenizers.Tokenizer.from_file(str(path))


def _require_tokenizers():
    try:
        import tokenizers  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "the eval data pipeline needs `tokenizers` (pip install tokenizers); "
            "evaluation itself only needs numpy/torch"
        ) from exc
    return tokenizers


def sha256_file(path: str | pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gpt2_byte_decoder() -> dict[str, int]:
    """The GPT-2 byte-level BPE byte<->unicode map, inverted to unicode -> byte."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def token_bytes_from_tokenizer(tok, *, max_bytes: int = 255) -> tuple[np.ndarray, dict]:
    """Per-token UTF-8 byte length for every vocabulary id.

    Returns ``(table, info)`` where table is uint8 (or uint16 if a token exceeds 255 bytes)
    and info records the special/unknown ids encountered.  `id_to_token(i)` returns the
    byte-mapped token string for ordinary BPE tokens; any token whose characters are not all
    in the GPT-2 byte map is treated as a literal added token (its UTF-8 length is exact).
    """
    vocab_size = int(tok.get_vocab_size())
    decoder = gpt2_byte_decoder()
    out = np.zeros(vocab_size, dtype=np.int64)
    literal: list[int] = []
    missing: list[int] = []
    for i in range(vocab_size):
        token = tok.id_to_token(i)
        if token is None:
            missing.append(i)
            continue
        if token == "":
            out[i] = 0
            continue
        if all(ch in decoder for ch in token):
            out[i] = len(bytes(decoder[ch] for ch in token))
        else:
            out[i] = len(token.encode("utf-8"))
            literal.append(i)
    max_len = int(out.max()) if vocab_size else 0
    if max_len <= max_bytes:
        table = out.astype(np.uint8)
    elif max_len <= 65_535:
        table = out.astype(np.uint16)
    else:  # pragma: no cover - not reachable for a sane vocabulary
        table = out.astype(np.uint32)
    info = {
        "vocab_size": vocab_size,
        "max_token_bytes": max_len,
        "dtype": str(table.dtype),
        "literal_token_count": len(literal),
        "missing_token_count": len(missing),
        "missing_token_ids": missing[:16],
    }
    return table, info


def validate_token_bytes(tok, table: np.ndarray, texts: Iterable[str], *, strict: bool = True) -> dict:
    """Check ``sum(table[encode(text)]) == len(text.encode('utf-8'))`` on sample texts."""
    bad: list[dict] = []
    n = 0
    for text in texts:
        n += 1
        ids = np.asarray(tok.encode(text, add_special_tokens=False).ids, dtype=np.int64)
        got = int(table[ids].sum()) if ids.size else 0
        want = len(text.encode("utf-8"))
        if got != want:
            bad.append({"chars": len(text), "tokens": int(ids.size), "got": got, "want": want})
    stats = {"checked": n, "bad": len(bad), "examples": bad[:10]}
    if bad and strict:
        raise ValueError(f"token byte table failed validation on {len(bad)}/{n} texts: {bad[:3]}")
    return stats


def encode_text(tok, text: str) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize text and return ``(ids int64, offsets int64 [n, 2])`` (character offsets)."""
    enc = tok.encode(text, add_special_tokens=False)
    ids = np.asarray(enc.ids, dtype=np.int64)
    offsets = np.asarray(enc.offsets, dtype=np.int64).reshape(-1, 2)
    if ids.size != offsets.shape[0]:
        raise ValueError(f"tokenizer returned {ids.size} ids but {offsets.shape[0]} offsets")
    return ids, offsets


def contract_info(tok, path: str | pathlib.Path | None = None) -> dict:
    """Record the tokenizer contract for the dataset manifest."""
    vocab = tok.get_vocab()
    info = {
        "vocab_size": int(tok.get_vocab_size()),
        "bos_id": vocab.get(DS_BOS_TOKEN),
        "eos_id": vocab.get(DS_EOS_TOKEN),
        "pad_id": vocab.get(DS_PAD_TOKEN),
        "pinned_v41_flash": int(tok.get_vocab_size()) == DS_VOCAB_SIZE,
    }
    if path is not None:
        path = pathlib.Path(path)
        info["path"] = str(path)
        info["sha256"] = sha256_file(path)
        info["bytes"] = path.stat().st_size
    return info


def char_spans_to_token_spans(offsets: np.ndarray, spans: Sequence[tuple[int, int]]) -> list[np.ndarray]:
    """Map character spans to token index arrays (tokens overlapping the span).

    Offsets are assumed monotonic (true for all tokenizers), so this is O(log n) per span.
    """
    starts = offsets[:, 0]
    ends = offsets[:, 1]
    out: list[np.ndarray] = []
    for start, end in spans:
        lo = int(np.searchsorted(ends, start, side="right"))  # first token ending after start
        hi = int(np.searchsorted(starts, end, side="left"))  # first token starting at/after end
        out.append(np.arange(lo, hi, dtype=np.int64))
    return out

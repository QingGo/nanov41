"""Held-out eval shards: the on-disk data contract and a manifest/dataset reader.

The harness never tokenizes at eval time.  Eval data is built once (on CPU, with the pinned
DeepSeek-V4.1-Flash tokenizer) into memory-mappable ``.npy`` arrays and a JSON manifest:

    <dataset>/token_bytes.npy               uint8 / uint16 [vocab_size]  UTF-8 bytes per token
    <dataset>/<domain>/<shard>.tokens.npy   uint32 [n]                   token ids
    <dataset>/<domain>/<shard>.types.npy    uint8  [n] (optional)        token types
    <dataset>/manifest.json

Token types (per *label* position, i.e. the token being predicted):

    0  IGNORE       excluded from every metric (padding / deliberately skipped)
    1  TEXT         prompt text, user turns, system prompt, plain corpora
    2  ASSISTANT    tokens of an assistant turn (header + reasoning + content + tool-call mark-up)
    3  TOOL         content inside ``<tool_result>...</tool_result>`` (tool output / logs)

A domain without a types array is a pure-text domain: every token is TEXT.

Bytes-per-byte (BPB) accounting is exact because ``token_bytes`` maps every vocabulary id to
the number of UTF-8 bytes it represents; the denominator of a metric is the byte length of the
tokens that were actually scored, not the token count and not a dataset-level average.

Windowing rule (see `bpb.py`/`harness.py`): each shard is cut into non-overlapping windows of
``window`` tokens; the first token of a window has no in-window context and is not scored, so
a window contributes ``window - 1`` predictions and the trailing partial window is dropped.
The bias is < 1/window of the tokens and is reported (`tokens_dropped`).
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import time
from typing import Any, Iterable, Sequence

import numpy as np

FORMAT = "nanov41-eval-data/1"

TYPE_IGNORE = 0
TYPE_TEXT = 1
TYPE_ASSISTANT = 2
TYPE_TOOL = 3

TYPE_NAMES: dict[int, str] = {
    TYPE_TEXT: "text",
    TYPE_ASSISTANT: "assistant",
    TYPE_TOOL: "tool_output",
}
ALL_TYPES = (TYPE_TEXT, TYPE_ASSISTANT, TYPE_TOOL)

DEFAULT_WINDOW = 2048


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path: str | pathlib.Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_npy_atomic(path: str | pathlib.Path, array: np.ndarray) -> str:
    """Write a .npy file atomically and return its sha256."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, np.ascontiguousarray(array), allow_pickle=False)
    tmp.replace(path)
    return sha256_file(path)


def _check_tokens(tokens: np.ndarray) -> np.ndarray:
    arr = np.asarray(tokens)
    if arr.ndim != 1:
        raise ValueError(f"tokens must be 1-D, got shape {arr.shape}")
    if arr.dtype.kind not in "iu":
        raise ValueError(f"tokens must be an integer array, got {arr.dtype}")
    if arr.size and int(arr.max()) > 0xFFFFFFFF:
        raise ValueError("token ids exceed uint32; are you feeding the wrong tokenizer?")
    return arr.astype(np.uint32, copy=False)


def _check_types(types: np.ndarray, n_tokens: int) -> np.ndarray:
    arr = np.asarray(types)
    if arr.ndim != 1 or arr.shape[0] != n_tokens:
        raise ValueError(f"types must be 1-D of length {n_tokens}, got {arr.shape}")
    if arr.dtype.kind not in "iu":
        raise ValueError(f"types must be an integer array, got {arr.dtype}")
    if arr.size and int(arr.min()) < 0:
        raise ValueError("types must be non-negative")
    if arr.size and int(arr.max()) > 255:
        raise ValueError("types must fit in uint8")
    return arr.astype(np.uint8, copy=False)


def summarize_types(tokens: np.ndarray, types: np.ndarray | None, token_bytes: np.ndarray) -> dict:
    """Per-type token/byte counts for a freshly built shard (pre-windowing)."""
    if types is None:
        types = np.full(tokens.shape[0], TYPE_TEXT, dtype=np.uint8)
    if tokens.size and int(tokens.max()) >= int(token_bytes.shape[0]):
        raise ValueError(
            f"token id {int(tokens.max())} outside token_bytes table (len {token_bytes.shape[0]})"
        )
    stats: dict[str, Any] = {"n_tokens": int(tokens.size), "by_type": {}}
    for t in ALL_TYPES:
        mask = types == t
        n = int(mask.sum())
        if not n:
            continue
        nbytes = int(token_bytes[np.asarray(tokens[mask], dtype=np.int64)].sum())
        stats["by_type"][TYPE_NAMES[t]] = {"tokens": n, "bytes": nbytes}
    stats["n_scored"] = int(np.count_nonzero(types != TYPE_IGNORE))
    stats["n_bytes"] = int(
        token_bytes[np.asarray(tokens[types != TYPE_IGNORE], dtype=np.int64)].sum()
    )
    return stats


@dataclasses.dataclass(frozen=True)
class ShardSpec:
    tokens: str  # manifest-relative path
    types: str | None
    n_tokens: int
    tokens_sha256: str | None = None
    types_sha256: str | None = None


@dataclasses.dataclass(frozen=True)
class DomainSpec:
    name: str
    kind: str
    shards: tuple[ShardSpec, ...]
    eval: dict
    stats: dict

    @property
    def n_tokens(self) -> int:
        return sum(s.n_tokens for s in self.shards)

    @property
    def window(self) -> int:
        return int(self.eval.get("window") or DEFAULT_WINDOW)

    @property
    def max_tokens(self) -> int | None:
        v = self.eval.get("max_tokens")
        return None if v is None else int(v)

    @property
    def weight(self) -> float:
        return float(self.eval.get("weight", 1.0))


class EvalDataset:
    """A manifest-backed, pre-tokenized held-out dataset (mmap, tokenizer-free)."""

    def __init__(self, root: pathlib.Path, manifest: dict):
        self.root = pathlib.Path(root)
        self.manifest = manifest
        self.manifest_sha256 = sha256_file(self.root / "manifest.json")
        self.token_bytes = self._load_token_bytes()
        self.domains = self._load_domains()
        self.mixes: dict[str, dict] = dict(manifest.get("mixes", {}))

    # ---------------------------------------------------------------- constructors
    @classmethod
    def open(cls, path: str | pathlib.Path) -> "EvalDataset":
        root = pathlib.Path(path)
        if root.is_file() and root.name == "manifest.json":
            root = root.parent
        manifest_path = root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"no eval dataset at {root} (missing manifest.json); build one with "
                "tools/make_eval_slices.py"
            )
        with open(manifest_path) as f:
            manifest = json.load(f)
        if manifest.get("format") != FORMAT:
            raise ValueError(
                f"unsupported eval dataset format {manifest.get('format')!r} (want {FORMAT})"
            )
        return cls(root, manifest)

    # ---------------------------------------------------------------- loading
    def _load_token_bytes(self) -> np.ndarray:
        meta = self.manifest.get("token_bytes")
        if not meta:
            raise ValueError("manifest has no token_bytes table; rebuild it with make_eval_slices.py")
        path = self.root / meta["path"]
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        if arr.ndim != 1 or arr.dtype.kind not in "iu":
            raise ValueError(f"bad token_bytes table {path}: shape={arr.shape} dtype={arr.dtype}")
        declared = self.manifest.get("tokenizer", {}).get("vocab_size")
        if declared is not None and int(declared) != int(arr.shape[0]):
            raise ValueError(
                f"token_bytes length {arr.shape[0]} != tokenizer vocab_size {declared}"
            )
        return arr

    def _load_domains(self) -> dict[str, DomainSpec]:
        domains: dict[str, DomainSpec] = {}
        for name, entry in self.manifest.get("domains", {}).items():
            shards = tuple(
                ShardSpec(
                    tokens=s["tokens"],
                    types=s.get("types"),
                    n_tokens=int(s["n_tokens"]),
                    tokens_sha256=s.get("tokens_sha256"),
                    types_sha256=s.get("types_sha256"),
                )
                for s in entry.get("shards", [])
            )
            domains[name] = DomainSpec(
                name=name,
                kind=entry.get("kind", "text"),
                shards=shards,
                eval=dict(entry.get("eval", {})),
                stats=dict(entry.get("stats", {})),
            )
        return domains

    # ---------------------------------------------------------------- access
    def read_shard(self, spec: ShardSpec) -> tuple[np.ndarray, np.ndarray | None]:
        tokens = np.load(self.root / spec.tokens, mmap_mode="r", allow_pickle=False)
        if tokens.ndim != 1:
            raise ValueError(f"shard {spec.tokens}: tokens must be 1-D, got {tokens.shape}")
        types = None
        if spec.types:
            types = np.load(self.root / spec.types, mmap_mode="r", allow_pickle=False)
            if types.shape != tokens.shape:
                raise ValueError(
                    f"shard {spec.tokens}: types shape {types.shape} != tokens shape {tokens.shape}"
                )
            # manifest stores types as uint8; keep a full-resolution dtype for math
            types = np.asarray(types, dtype=np.uint8)
        if spec.n_tokens and spec.n_tokens != int(tokens.shape[0]):
            raise ValueError(
                f"shard {spec.tokens}: manifest says {spec.n_tokens} tokens, file has {tokens.shape[0]}"
            )
        return np.asarray(tokens), types

    def domain(self, name: str) -> DomainSpec:
        if name not in self.domains:
            raise KeyError(f"unknown domain {name!r}; available: {sorted(self.domains)}")
        return self.domains[name]

    def domain_names(self) -> list[str]:
        return list(self.domains)

    # ---------------------------------------------------------------- validation
    def verify(self, *, domains: Iterable[str] | None = None, check_hashes: bool = False) -> dict:
        """Cheap structural validation; `check_hashes=True` re-hashes every shard."""
        names = list(domains) if domains is not None else self.domain_names()
        out: dict[str, Any] = {"domains": {}, "ok": True}
        for name in names:
            spec = self.domain(name)
            entry: dict[str, Any] = {"shards": len(spec.shards), "tokens": 0}
            for shard in spec.shards:
                tokens, types = self.read_shard(shard)
                entry["tokens"] += int(tokens.shape[0])
                if tokens.size and int(tokens.max()) >= int(self.token_bytes.shape[0]):
                    raise ValueError(
                        f"domain {name}: token id {int(tokens.max())} outside token_bytes "
                        f"(len {self.token_bytes.shape[0]})"
                    )
                if check_hashes and shard.tokens_sha256:
                    got = sha256_file(self.root / shard.tokens)
                    if got != shard.tokens_sha256:
                        raise ValueError(f"sha256 mismatch for {shard.tokens}: {got} != {shard.tokens_sha256}")
            entry["window"] = spec.window
            out["domains"][name] = entry
        return out


class DatasetWriter:
    """Incrementally build/update an eval dataset directory (used by the slice builder)."""

    def __init__(self, root: str | pathlib.Path, *, token_bytes: np.ndarray | None = None,
                 tokenizer_meta: dict | None = None):
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self.manifest = self._load_manifest()
        self.token_bytes: np.ndarray | None = None
        if token_bytes is not None:
            self.set_token_bytes(token_bytes, tokenizer_meta)
        elif self.manifest.get("token_bytes"):
            self.token_bytes = np.load(
                self.root / self.manifest["token_bytes"]["path"], mmap_mode="r", allow_pickle=False
            )
        elif tokenizer_meta is not None:
            self.manifest.setdefault("tokenizer", {}).update(tokenizer_meta)

    # ---------------------------------------------------------------- manifest
    def _load_manifest(self) -> dict:
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                manifest = json.load(f)
            if manifest.get("format") != FORMAT:
                raise ValueError(
                    f"{self.manifest_path}: unsupported format {manifest.get('format')!r} (want {FORMAT})"
                )
            return manifest
        return {"format": FORMAT, "created_utc": utc_now(), "domains": {}, "mixes": {}}

    def set_token_bytes(self, table: np.ndarray, tokenizer_meta: dict | None = None) -> dict:
        arr = np.asarray(table)
        if arr.ndim != 1 or arr.dtype.kind not in "iu":
            raise ValueError(f"token_bytes must be a 1-D integer array, got {arr.shape}/{arr.dtype}")
        rel = "token_bytes.npy"
        path = self.root / rel
        existing = self.manifest.get("token_bytes")
        if existing and path.exists() and existing.get("sha256") == sha256_file(path):
            stored = np.load(path, mmap_mode="r", allow_pickle=False)
            same = (stored.shape == arr.shape and stored.dtype == arr.dtype
                    and np.array_equal(np.asarray(stored), arr))
            if same:
                self.token_bytes = stored
                return self.manifest["token_bytes"]
            raise ValueError(
                "existing dataset uses a different token_bytes table (same file hash but "
                "different values); refusing to mix tokenizers. Use a fresh --out directory."
            )
        # Write to a temp file first: a byte-table mismatch must not corrupt the existing table.
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as f:
            np.save(f, np.ascontiguousarray(arr), allow_pickle=False)
        sha = sha256_file(tmp)
        existing_sha = existing.get("sha256") if existing else None
        if existing_sha and existing_sha != sha:
            tmp.unlink(missing_ok=True)
            raise ValueError(
                "existing dataset uses a different token_bytes table "
                f"(sha {existing_sha[:12]} != {sha[:12]}); refusing to mix tokenizers. "
                "Use a fresh --out directory."
            )
        tmp.replace(path)
        self.manifest["token_bytes"] = {
            "path": rel,
            "sha256": sha,
            "dtype": str(arr.dtype),
            "n": int(arr.shape[0]),
        }
        if tokenizer_meta is not None:
            self.manifest.setdefault("tokenizer", {}).update(tokenizer_meta)
        self.token_bytes = arr
        return self.manifest["token_bytes"]

    # ---------------------------------------------------------------- mixes
    def set_mix(self, name: str, weights: dict[str, float], *, source: str | None = None,
                description: str | None = None, source_weights: dict | None = None) -> dict:
        """Record a named reference mixture (domain -> token proportion) in the manifest."""
        from .mix import validate_mix  # local import avoids a data<->bpb import cycle

        validate_mix(weights)
        self.manifest.setdefault("mixes", {})[name] = {
            "weights": {str(k): float(v) for k, v in weights.items()},
            "source": source,
            "description": description,
            "source_weights": dict(source_weights or {}),
        }
        self.save()
        return self.manifest["mixes"][name]

    def auto_add_reference_mixes(self) -> list[str]:
        """Add built-in reference mixes that the current domains fully cover."""
        from .mix import REFERENCE_MIXES, validate_mix

        added: list[str] = []
        mixes = self.manifest.setdefault("mixes", {})
        for name, mix in REFERENCE_MIXES.items():
            validate_mix(mix["weights"])
            if set(mix["weights"]) <= set(self.manifest.get("domains", {})):
                mixes[name] = {
                    "weights": dict(mix["weights"]),
                    "source": mix.get("source"),
                    "description": mix.get("description"),
                    "source_weights": dict(mix.get("source_weights", {})),
                }
                added.append(name)
        if added:
            self.save()
        return added

    # ---------------------------------------------------------------- domains
    def add_domain(self, name: str, tokens: np.ndarray, types: np.ndarray | None = None, *,
                   kind: str = "text", sources: Sequence[dict] = (), eval: dict | None = None,
                   shard_name: str = "shard-00000", shard_tokens: int | None = None) -> dict:
        """Write one domain (split into ``shard_tokens``-sized shards) and update the manifest."""
        if self.token_bytes is None:
            raise RuntimeError("set_token_bytes() must be called before add_domain()")
        tokens = _check_tokens(tokens)
        if types is not None:
            types = _check_types(types, tokens.shape[0])
        if kind not in ("text", "trajectory"):
            raise ValueError(f"unknown domain kind {kind!r}")
        if shard_tokens is None:
            shard_tokens = max(int(tokens.shape[0]), 1)
        shards: list[dict] = []
        entries: list[tuple[np.ndarray, np.ndarray | None]] = []
        for i in range(0, tokens.shape[0], shard_tokens):
            chunk = tokens[i:i + shard_tokens]
            types_chunk = types[i:i + shard_tokens] if types is not None else None
            entries.append((chunk, types_chunk))
        domain_dir = self.root / name
        for idx, (chunk, types_chunk) in enumerate(entries):
            tag = f"{shard_name}-{idx:05d}" if len(entries) > 1 else shard_name
            tokens_rel = f"{name}/{tag}.tokens.npy"
            tokens_sha = save_npy_atomic(self.root / tokens_rel, chunk)
            entry: dict[str, Any] = {
                "tokens": tokens_rel,
                "types": None,
                "n_tokens": int(chunk.shape[0]),
                "tokens_sha256": tokens_sha,
            }
            if types_chunk is not None and np.any(types_chunk != TYPE_TEXT):
                types_rel = f"{name}/{tag}.types.npy"
                entry["types"] = types_rel
                entry["types_sha256"] = save_npy_atomic(self.root / types_rel, types_chunk)
            elif types_chunk is not None:
                # all-TEXT mask: keep it out of the dataset to save space
                entry["types"] = None
            shards.append(entry)
        all_tokens = tokens
        all_types = types
        stats = summarize_types(all_tokens, all_types, np.asarray(self.token_bytes))
        domain_dir.mkdir(parents=True, exist_ok=True)
        entry_eval = dict(eval or {})
        self.manifest["domains"][name] = {
            "kind": kind,
            "shards": shards,
            "stats": stats,
            "eval": entry_eval,
            "sources": list(sources),
            "created_utc": utc_now(),
        }
        self.manifest["updated_utc"] = utc_now()
        self.save()
        self.auto_add_reference_mixes()
        return self.manifest["domains"][name]

    def save(self) -> None:
        self.manifest["updated_utc"] = utc_now()
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path.with_name(self.manifest_path.name + ".tmp")
        with open(tmp, "w") as f:
            json.dump(self.manifest, f, indent=2, sort_keys=True)
            f.write("\n")
        tmp.replace(self.manifest_path)

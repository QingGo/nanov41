#!/usr/bin/env python3
"""Convert a directory of karpathy ``climbmix`` ``.npy`` shards into the P2 uint32 corpus format.

``tools/p2_train.py`` expects ``<data-dir>/manifest.json`` plus one little-endian uint32
``<domain>.u32.bin`` per domain.  ClimbMix shards are ``uint16``/``uint32`` numpy arrays; we
stream them (never loading all shards at once) into a single ``web.u32.bin``, optionally capped
at ``--tokens`` (the 0.5B-token aligned A/B needs ~0.5B tokens).

Usage::

    python3 tools/climbmix_to_p2.py --src /root/autodl-tmp/climbmix \\
        --out /root/autodl-tmp/climbmix_p2 --tokens 600000000
"""
from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

MANIFEST_FORMAT = "nanov41-p2-corpus/1"


def _shard_files(src: pathlib.Path) -> list[pathlib.Path]:
    files = sorted(p for p in src.iterdir() if p.suffix in (".npy", ".npz"))
    if not files:
        files = sorted(p for p in src.iterdir() if p.suffix in (".bin", ".u32", ".raw"))
    if not files:
        raise SystemExit(f"no .npy/.bin shards under {src}")
    return files


def _load_shard(path: pathlib.Path) -> np.ndarray:
    if path.suffix == ".npy":
        arr = np.load(path, mmap_mode="r")
    elif path.suffix == ".npz":
        with np.load(path) as z:
            key = "arr_0" if "arr_0" in z else list(z.keys())[0]
            arr = z[key]
    else:
        # raw uint16 fallback (climbmix also ships .bin in some mirrors)
        arr = np.fromfile(path, dtype=np.uint16)
    if arr.dtype == np.uint16:
        return arr.astype(np.uint32)
    if arr.dtype == np.uint32:
        return arr.astype(np.uint32, copy=False)
    if arr.dtype.itemsize > 4 or np.issubdtype(arr.dtype, np.signedinteger):
        raise SystemExit(f"{path}: unsupported dtype {arr.dtype}")
    return arr.astype(np.uint32)


def convert(src: pathlib.Path, out: pathlib.Path, tokens: int = 0,
            domain: str = "web", chunk: int = 1 << 25) -> dict:
    """Stream ``src`` shards into ``out/<domain>.u32.bin``; returns the manifest dict."""
    out.mkdir(parents=True, exist_ok=True)
    files = _shard_files(src)
    target = int(tokens) if tokens > 0 else None
    written = 0
    dtype_ok: dict[str, int] = {}
    path = out / f"{domain}.u32.bin"
    with open(path, "wb") as fh:
        for fp in files:
            arr = _load_shard(fp)
            dtype_ok[str(arr.dtype)] = dtype_ok.get(str(arr.dtype), 0) + 1
            if target is not None:
                arr = arr[: max(0, target - written)]
            if arr.size:
                # chunked tofile keeps peak memory bounded when a shard is mmap-backed
                for start in range(0, arr.size, chunk):
                    np.asarray(arr[start:start + chunk], dtype="<u4").tofile(fh)
                written += int(arr.size)
            if target is not None and written >= target:
                break
    manifest = {
        "format": MANIFEST_FORMAT,
        "source": f"climbmix {src.name} ({len(files)} shards)",
        "domains": {domain: {"tokens": int(written)}},
        "tokens_total": int(written),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory with climbmix .npy shards")
    ap.add_argument("--out", required=True, help="output data dir for p2_train --data-dir")
    ap.add_argument("--tokens", type=int, default=0, help="cap total tokens (0 = all)")
    ap.add_argument("--domain", default="web")
    args = ap.parse_args()
    manifest = convert(pathlib.Path(args.src), pathlib.Path(args.out), args.tokens, args.domain)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fetch raw held-out eval text for the Marin-referenced T0 domains.

The T0 harness consumes pre-tokenized shards; this tool only downloads the *raw text* that
`tools/make_eval_slices.py` later tokenizes.  Sources and their role follow Marin-8B (fully open
model that publishes its mixture), so the domain set can be mapped onto Marin's token
distribution and reported as a mix-weighted BPB (`nanov41/eval/mix.py`):

| domain | Marin role (phase-1 / jellyfish) | source used here | holdout |
|---|---|---|---|
| web | DCLM-Baseline 92.6% / Dolmino DCLM HQ 67.8% | `mlfoundations/dclm-baseline-1.0`, **global-shard_10_of_10** | reserved eval shard; never train on global shard 10 |
| code | StarCoder 6.1% / 2.2% | `codeparrot/codeparrot-clean-valid` (official validation split) | StarCoder is gated; documented fallback |
| math | ProofPile-2 1.3% / 9.3% | `EleutherAI/proof-pile-2` **test** splits: algebraic-stack + open-web-math | official test split |
| academic | Dolma peS2o 10.8% + Arxiv 5.2% | `EleutherAI/proof-pile-2` **arxiv/test** | official test split |
| wikipedia | Dolma Wikipedia 0.7% + Megawika 0.8% | `Salesforce/wikitext` wikitext-103-raw test+validation | official test/validation |
| stackexchange | Dolma StackExchange 3.2% | `HuggingFaceH4/stack-exchange-preferences` (small sites) | QA proxy; exclude these sites from training |
| zh | auxiliary (not in Marin) | `m-a-p/COIG-CQIA` wiki/zhihu JSONL | Chinese held-out text; weight 0 |
| books, agent | auxiliary / planned | not wired yet | see docs/eval-harness.md §7 |

Two formats are streamed without saving the whole file: JSONL (plain / gzip / zstd) and small
Parquet files (downloaded once under ``--max-download-mb`` then read with pyarrow).  Streaming
stops as soon as the per-domain character budget is reached, so a 140 MB DCLM shard costs only
the bytes actually read.

Every fetch writes a provenance sidecar next to the JSONL: dataset repo, pinned revision sha,
file names, record/char counts, endpoint, UTC time and the exact holdout rationale.  Keep that
file with the eval data — it is the audit trail for "this slice was never trained on".

Examples
--------
    python tools/fetch_eval_data.py --list-sources
    python tools/fetch_eval_data.py --domain web --out runs/eval/raw/web.jsonl --max-chars 20000000
    python tools/fetch_eval_data.py --domain math --out runs/eval/raw/math.jsonl --max-chars 20000000
    python tools/fetch_eval_data.py --domain web --dry-run          # resolve files, no download
"""

from __future__ import annotations

import argparse
import dataclasses
import fnmatch
import gzip
import html
import io
import json
import os
import pathlib
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_ENDPOINT = os.environ.get("HF_ENDPOINT") or "https://hf-mirror.com"
DEFAULT_MAX_CHARS = 20_000_000  # ~5-6 M tokens for English-ish text
CHUNK = 1 << 20

MARIN_DOI = "https://github.com/marin-community/marin/blob/main/docs/reports/marin-8b-retro.md"


@dataclasses.dataclass(frozen=True)
class Source:
    repo: str
    note: str
    files: tuple[str, ...] = ()          # explicit files (mutually exclusive with globs)
    file_glob: tuple[str, ...] = ()      # fnmatch patterns against the repo file list
    text_fields: tuple[str, ...] = ("text",)
    join: str = "\n\n"
    kind: str = "field"                  # "field" (joined fields) | "qa" (question+answers)
    clean: str = "none"                  # "none" | "html"


SOURCES: dict[str, list[Source]] = {
    "web": [
        Source(
            repo="mlfoundations/dclm-baseline-1.0",
            files=("global-shard_10_of_10/local-shard_9_of_10/shard_00000000_processed.jsonl.zst",),
            note=(
                "DCLM-Baseline; Marin phase-1 web 92.6% (jellyfish web 67.8% via Dolmino HQ). "
                "global-shard_10_of_10 is RESERVED as eval holdout - never train on it."
            ),
        ),
    ],
    "code": [
        Source(
            repo="codeparrot/codeparrot-clean-valid",
            files=("file-000000000054.json.gz",),
            text_fields=("content",),
            note=(
                "Code fallback: StarCoder Data (Marin 6.1% / jellyfish 2.2%) is gated; this is "
                "codeparrot-clean-valid, an official Python validation split."
            ),
        ),
    ],
    "math": [
        Source(
            repo="EleutherAI/proof-pile-2",
            file_glob=("algebraic-stack/test/*.jsonl.zst", "open-web-math/test/test.jsonl.zst"),
            note=(
                "ProofPile-2 test splits: Algebraic Stack + Open Web Math; Marin phase-1 math 1.3%, "
                "jellyfish math 9.3% (FineMath-3+ 6.3 + AlgebraicStack 2.1 + OpenWebMath 0.9)."
            ),
        ),
    ],
    "academic": [
        Source(
            repo="EleutherAI/proof-pile-2",
            file_glob=("arxiv/test/*.jsonl.zst",),
            note=(
                "ProofPile-2 arXiv test split; proxy for Dolma Arxiv 5.2% + peS2o 10.8% "
                "(jellyfish academic 16.0%)."
            ),
        ),
    ],
    "wikipedia": [
        Source(
            repo="Salesforce/wikitext",
            files=(
                "wikitext-103-raw-v1/test-00000-of-00001.parquet",
                "wikitext-103-raw-v1/validation-00000-of-00001.parquet",
            ),
            note=(
                "WikiText-103 raw test+validation (official held-out splits); proxy for "
                "Dolma Wikipedia 0.7% + Megawika 0.8% (jellyfish wikipedia 1.5%)."
            ),
        ),
    ],
    "stackexchange": [
        Source(
            repo="HuggingFaceH4/stack-exchange-preferences",
            files=(
                "data/3dprinting.stackexchange.com/train-00000-of-00001.parquet",
                "data/arduino.stackexchange.com/train-00000-of-00001.parquet",
                "data/bicycles.stackexchange.com/train-00000-of-00001.parquet",
            ),
            kind="qa",
            clean="html",
            note=(
                "StackExchange Q&A (three small sites); proxy for Dolma StackExchange 3.2%. "
                "Exclude these sites (3dprinting/arduino/bicycles) from training."
            ),
        ),
    ],
    "zh": [
        Source(
            repo="m-a-p/COIG-CQIA",
            file_glob=("wiki/*.jsonl", "zhihu/*.jsonl"),
            text_fields=("instruction", "input", "output"),
            join="\n",
            note=(
                "Chinese held-out corpora (COIG-CQIA wiki + zhihu subsets). Auxiliary domain: not "
                "in Marin's mix, weight 0 in mix summaries."
            ),
        ),
    ],
}

DOMAIN_MIX_ROLE = {
    "web": "marin jellyfish web 67.8% / phase1 92.6%",
    "code": "marin jellyfish code 2.2% / phase1 6.1%",
    "math": "marin jellyfish math 9.3% / phase1 1.3%",
    "academic": "marin jellyfish academic 16.0%",
    "wikipedia": "marin jellyfish wikipedia 1.5%",
    "stackexchange": "marin jellyfish stackexchange 3.2%",
    "zh": "aux (weight 0)",
}

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def strip_html(text: str) -> str:
    text = _HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


# --------------------------------------------------------------------------- HF helpers
_filelist_cache: dict[str, dict] = {}


def list_repo_files(repo: str, endpoint: str, revision: str, *, timeout: float = 60.0) -> dict:
    """Return ``{sha, files:[...]}`` for a dataset repo (API via the configured endpoint)."""
    key = f"{endpoint}|{repo}|{revision}"
    if key in _filelist_cache:
        return _filelist_cache[key]
    api_revision = revision if revision != "main" else "main"
    url = f"{endpoint}/api/datasets/{urllib.parse.quote(repo, safe='/')}?revision={api_revision}"
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                data = json.load(resp)
            files = [s["rfilename"] for s in data.get("siblings", [])]
            info = {"sha": data.get("sha"), "files": files}
            _filelist_cache[key] = info
            return info
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:  # pragma: no cover
            last_exc = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"could not list files for {repo}: {last_exc}")


def resolve_files(source: Source, endpoint: str, revision: str) -> list[str]:
    explicit = list(source.files)
    if explicit:
        return explicit
    files = list_repo_files(source.repo, endpoint, revision)["files"]
    matched: list[str] = []
    for pattern in source.file_glob:
        matched.extend(sorted(f for f in files if fnmatch.fnmatch(f, pattern)))
    seen: set[str] = set()
    out = []
    for f in matched:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def file_url(repo: str, path: str, endpoint: str, revision: str) -> str:
    quoted = urllib.parse.quote(path, safe="/")
    return f"{endpoint}/datasets/{urllib.parse.quote(repo, safe='/')}/resolve/{revision}/{quoted}"


def http_stream(url: str, *, timeout: tuple[float, float] = (15.0, 120.0), retries: int = 3):
    import requests

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            # identity avoids HTTP-level gzip; the file formats (.jsonl.gz/.zst) are decoded
            # explicitly, and plain .jsonl must not arrive pre-compressed.
            resp = requests.get(url, stream=True, timeout=timeout,
                                headers={"Accept-Encoding": "identity"})
            if resp.status_code == 200:
                return resp
            if resp.status_code in (401, 403, 404):
                resp.close()
                raise PermissionError(f"HTTP {resp.status_code} for {url} (gated/missing dataset?)")
            resp.close()
            last_exc = RuntimeError(f"HTTP {resp.status_code}")
        except requests.RequestException as exc:  # pragma: no cover - network
            last_exc = exc
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"download failed after {retries} attempts: {url}: {last_exc}")


# --------------------------------------------------------------------------- record extraction
def extract_text(record: dict, source: Source) -> str | None:
    if not isinstance(record, dict):
        return None
    if source.kind == "qa":
        parts = [record.get("question") or ""]
        for answer in record.get("answers") or []:
            if isinstance(answer, dict):
                parts.append(answer.get("text") or "")
        parts = [str(p) for p in parts if p]
        joiner = "\n\n"
    else:
        parts = [str(record.get(f)) for f in source.text_fields
                 if isinstance(record.get(f), str) and record.get(f)]
        joiner = source.join
    if source.clean == "html":
        parts = [strip_html(part) for part in parts]
    parts = [part for part in parts if part]
    text = joiner.join(parts).strip()
    return text or None


# --------------------------------------------------------------------------- iterators
class _HttpRawReader(io.RawIOBase):
    """File-like adapter over ``resp.raw`` that treats EOF-close as EOF.

    urllib3 closes the underlying HTTPResponse as soon as the body is fully read; a bare
    ``TextIOWrapper(resp.raw)`` then raises "I/O operation on closed file" on its final read.
    Returning 0 bytes once the raw stream is closed makes gzip/zstd/plain readers see a clean EOF.
    """

    def __init__(self, resp):
        super().__init__()
        self._resp = resp
        self._raw = resp.raw

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        raw = self._raw
        if raw is None:
            return 0
        try:
            if getattr(raw, "closed", False):
                return 0
            data = raw.read(len(buffer))
        except ValueError:  # body already released/closed at EOF
            return 0
        except AttributeError:  # pragma: no cover - non-urllib3 stand-ins
            return 0
        if not data:
            return 0
        n = len(data)
        buffer[:n] = data
        return n

    def close(self) -> None:
        try:
            if self._resp is not None:
                self._resp.close()
        finally:
            self._resp = None
            self._raw = None
            super().close()


def _iter_jsonl_wrapper(wrapper):
    try:
        for line in wrapper:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
    finally:
        wrapper.close()


def iter_jsonl_lines(resp, fmt: str):
    raw = _HttpRawReader(resp)
    if fmt == "jsonl":
        wrapper = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
    elif fmt == "jsonl.gz":
        wrapper = io.TextIOWrapper(gzip.GzipFile(fileobj=raw), encoding="utf-8", errors="replace")
    elif fmt == "jsonl.zst":
        import zstandard

        wrapper = io.TextIOWrapper(
            zstandard.ZstdDecompressor().stream_reader(raw), encoding="utf-8", errors="replace"
        )
    else:  # pragma: no cover - guarded by detect_format
        raise ValueError(f"unsupported jsonl format {fmt}")
    try:
        yield from _iter_jsonl_wrapper(wrapper)
    finally:
        try:
            raw.close()
        except Exception:  # pragma: no cover - defensive
            pass


def iter_jsonl_file(path: str | pathlib.Path):
    """Read a local JSONL file (plain/gzip/zstd) - the network path reuses the same parser."""
    fmt = detect_format(str(path))
    path = str(path)
    if fmt == "jsonl":
        with open(path, "rb") as raw:
            yield from _iter_jsonl_wrapper(io.TextIOWrapper(raw, encoding="utf-8", errors="replace"))
    elif fmt == "jsonl.gz":
        with open(path, "rb") as raw:
            yield from _iter_jsonl_wrapper(
                io.TextIOWrapper(gzip.GzipFile(fileobj=raw), encoding="utf-8", errors="replace")
            )
    elif fmt == "jsonl.zst":
        import zstandard

        with open(path, "rb") as raw:
            yield from _iter_jsonl_wrapper(
                io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(raw),
                                 encoding="utf-8", errors="replace")
            )
    else:  # pragma: no cover
        raise ValueError(f"not a JSONL file: {path} ({fmt})")


def iter_parquet_file(path: str | pathlib.Path):
    """Yield rows from a local parquet file as dicts."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=256):
        yield from batch.to_pylist()


def detect_format(path: str) -> str:
    name = path.lower()
    if name.endswith(".jsonl.zst"):
        return "jsonl.zst"
    if name.endswith(".jsonl.gz") or name.endswith(".json.gz"):
        return "jsonl.gz"
    if name.endswith(".jsonl"):
        return "jsonl"
    if name.endswith(".parquet"):
        return "parquet"
    raise ValueError(f"cannot infer format for {path!r} (expected .jsonl[.gz|.zst] or .parquet)")


def iter_parquet_records(url: str, *, max_download_mb: float = 512.0, timeout: float = 120.0):
    """Download a (small) parquet file to a temp path and yield its rows as dicts."""
    import pyarrow.parquet as pq

    resp = http_stream(url)
    size = resp.headers.get("Content-Length")
    if size is not None and float(size) > max_download_mb * 1e6:
        resp.close()
        raise ValueError(
            f"parquet file is {float(size) / 1e6:.1f} MB > --max-download-mb {max_download_mb}; "
            "pick a smaller file/source"
        )
    tmpdir = pathlib.Path(tempfile.mkdtemp(prefix="nanov41-eval-raw-"))
    tmp = tmpdir / "data.parquet"
    try:
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=CHUNK):
                if chunk:
                    f.write(chunk)
        resp.close()
        yield from iter_parquet_file(tmp)
    finally:
        resp.close()
        shutil.rmtree(tmpdir, ignore_errors=True)


def iter_source_records(source: Source, files: list[str], endpoint: str, revision: str,
                        *, max_download_mb: float):
    for path in files:
        fmt = detect_format(path)
        url = file_url(source.repo, path, endpoint, revision)
        if fmt == "parquet":
            for record in iter_parquet_records(url, max_download_mb=max_download_mb):
                yield path, record
        else:
            resp = http_stream(url)
            for record in iter_jsonl_lines(resp, fmt):
                yield path, record


# --------------------------------------------------------------------------- fetching
def fetch_domain(domain: str, out: pathlib.Path, *, max_chars: int = DEFAULT_MAX_CHARS,
                 max_files: int | None = None, skip_files: int = 0,
                 endpoint: str = DEFAULT_ENDPOINT, revision: str = "main",
                 max_download_mb: float = 512.0, max_records: int | None = None,
                 log=print) -> dict:
    if domain not in SOURCES:
        raise SystemExit(f"unknown domain {domain!r}; known: {sorted(SOURCES)}")
    out = pathlib.Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    total_records = 0
    total_chars = 0
    provenance_sources = []
    stop = False
    fh = open(out, "w", encoding="utf-8")  # fresh file per fetch (never append duplicates)
    try:
        for source in SOURCES[domain]:
            if stop:
                break
            info = list_repo_files(source.repo, endpoint, revision)
            files = resolve_files(source, endpoint, revision)
            if not files:
                log(f"[warn] {source.repo}: no files matched {source.file_glob or source.files}")
                continue
            files = files[skip_files:]
            if max_files is not None:
                files = files[:max_files]
            entry = {
                "repo": source.repo,
                "revision": revision,
                "resolved_sha": info.get("sha"),
                "files": [],
                "text_fields": list(source.text_fields),
                "kind": source.kind,
                "clean": source.clean,
                "note": source.note,
            }
            log(f"[fetch] {domain}: {source.repo} ({len(files)} file(s), {detect_format(files[0])})")
            for path in files:
                file_records = 0
                file_chars = 0
                try:
                    for path_used, record in iter_source_records(
                        source, [path], endpoint, revision, max_download_mb=max_download_mb
                    ):
                        text = extract_text(record, source)
                        if not text:
                            continue
                        fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                        total_records += 1
                        total_chars += len(text)
                        file_records += 1
                        file_chars += len(text)
                        if total_chars >= max_chars or (max_records and total_records >= max_records):
                            stop = True
                            break
                except PermissionError as exc:
                    log(f"[warn] {source.repo}/{path}: {exc} (skipping gated source)")
                except Exception as exc:  # noqa: BLE001 - keep going on individual file errors
                    log(f"[warn] {source.repo}/{path}: {type(exc).__name__}: {exc}")
                entry["files"].append({"path": path, "records": file_records, "chars": file_chars})
                if stop:
                    break
            provenance_sources.append(entry)
            log(f"[fetch] {domain}: {source.repo} -> {total_records:,} records, {total_chars:,} chars")
    finally:
        fh.close()

    if total_records == 0:
        raise SystemExit(
            f"no text fetched for domain {domain!r}; check network/gating (endpoint {endpoint})"
        )
    provenance = {
        "schema": "nanov41-eval-raw/1",
        "domain": domain,
        "domain_mix_role": DOMAIN_MIX_ROLE.get(domain),
        "marin_reference": MARIN_DOI,
        "created_utc": utc_now(),
        "endpoint": endpoint,
        "revision": revision,
        "max_chars": max_chars,
        "max_files": max_files,
        "skip_files": skip_files,
        "total_records": total_records,
        "total_chars": total_chars,
        "approx_tokens": int(total_chars / 3.5),
        "sources": provenance_sources,
    }
    sidecar = out.with_name(out.name + ".provenance.json")
    sidecar.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    log(f"[fetch] {domain}: {total_records:,} records / {total_chars:,} chars -> {out}")
    log(f"[fetch] provenance -> {sidecar}")
    return provenance


def dry_run(domain: str, *, endpoint: str = DEFAULT_ENDPOINT, revision: str = "main") -> dict:
    out = {}
    for source in SOURCES[domain]:
        files = resolve_files(source, endpoint, revision)
        info = list_repo_files(source.repo, endpoint, revision)
        out[source.repo] = {
            "sha": info.get("sha"),
            "n_files": len(files),
            "files": files[:5],
            "format": detect_format(files[0]) if files else None,
            "note": source.note,
        }
    return out


def list_sources() -> None:
    print(f"Marin reference: {MARIN_DOI}\n")
    header = f"{'domain':<14} {'source':<52} {'format':<11} {'role'}"
    print(header)
    print("-" * len(header))
    for domain, sources in SOURCES.items():
        for source in sources:
            files = source.files or source.file_glob
            fmt = detect_format(files[0]) if files else "?"
            print(f"{domain:<14} {source.repo:<52} {fmt:<11} {DOMAIN_MIX_ROLE.get(domain, '')}")
    print("\naux domains not in any mix: zh (weight 0). Planned but not wired: books, agent.")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", choices=sorted(SOURCES))
    ap.add_argument("--out", help="output JSONL path (required unless --list-sources/--dry-run)")
    ap.add_argument("--list-sources", action="store_true", help="print the source registry and exit")
    ap.add_argument("--dry-run", action="store_true", help="resolve files for --domain without downloading")
    ap.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS,
                    help=f"stop after ~N characters (default {DEFAULT_MAX_CHARS:,} ~= 5-6M tokens)")
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--max-files", type=int, default=None)
    ap.add_argument("--skip-files", type=int, default=0,
                    help="skip the first N resolved files (deterministic holdout offsets)")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT,
                    help=f"HF-compatible endpoint (default {DEFAULT_ENDPOINT})")
    ap.add_argument("--revision", default="main")
    ap.add_argument("--max-download-mb", type=float, default=512.0,
                    help="refuse parquet files larger than this (default 512 MB)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_sources:
        list_sources()
        return 0
    if not args.domain:
        print("error: --domain is required (or use --list-sources)", file=sys.stderr)
        return 2
    if args.dry_run:
        info = dry_run(args.domain, endpoint=args.endpoint, revision=args.revision)
        print(json.dumps({args.domain: info}, indent=2, ensure_ascii=False))
        return 0
    if not args.out:
        print("error: --out is required", file=sys.stderr)
        return 2
    fetch_domain(
        args.domain, pathlib.Path(args.out),
        max_chars=args.max_chars, max_files=args.max_files, skip_files=args.skip_files,
        endpoint=args.endpoint, revision=args.revision,
        max_download_mb=args.max_download_mb, max_records=args.max_records,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

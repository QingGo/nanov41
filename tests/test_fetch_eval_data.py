# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""CPU tests for tools/fetch_eval_data.py - parsers and fetch bookkeeping, no network.

The network paths (HF API listing, HTTP streaming) are monkeypatched; the format parsers are
tested against local files so a broken zstd/gzip/parquet path fails here, not after a 100 MB
download.
"""

from __future__ import annotations

import gzip
import importlib.util
import io
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _load_module():
    path = ROOT / "tools" / "fetch_eval_data.py"
    spec = importlib.util.spec_from_file_location("fetch_eval_data", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["fetch_eval_data"] = module  # dataclasses needs the module to resolve annotations
    spec.loader.exec_module(module)
    return module


fe = _load_module()


# --------------------------------------------------------------------------- parsers
def test_detect_format():
    assert fe.detect_format("a/b.jsonl.zst") == "jsonl.zst"
    assert fe.detect_format("x.jsonl.gz") == "jsonl.gz"
    assert fe.detect_format("x.json.gz") == "jsonl.gz"
    assert fe.detect_format("x.jsonl") == "jsonl"
    assert fe.detect_format("x.parquet") == "parquet"
    with pytest.raises(ValueError, match="cannot infer format"):
        fe.detect_format("x.txt")


def test_strip_html():
    assert fe.strip_html("<p>Hello <b>world</b></p>") == "Hello world"
    assert fe.strip_html("a &amp; b") == "a & b"


def test_extract_text_field_join():
    source = fe.Source(repo="r", note="", text_fields=("a", "b"), join="\n")
    assert fe.extract_text({"a": "x", "b": "y", "c": "z"}, source) == "x\ny"
    assert fe.extract_text({"a": "", "b": None}, source) is None


def test_extract_text_qa_html():
    source = fe.Source(repo="r", note="", kind="qa", clean="html")
    record = {
        "question": "<p>How?</p>",
        "answers": [{"text": "<p>Like <i>this</i></p>"}, {"text": ""}],
    }
    assert fe.extract_text(record, source) == "How?\n\nLike this"


def test_iter_jsonl_file_all_compressions(tmp_path):
    rows = [{"text": "hello"}, {"text": "world"}]
    plain = tmp_path / "x.jsonl"
    plain.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert list(fe.iter_jsonl_file(plain)) == rows

    gz = tmp_path / "x.jsonl.gz"
    with gzip.open(gz, "wt") as f:
        f.write("\n".join(json.dumps(r) for r in rows) + "\n")
    assert list(fe.iter_jsonl_file(gz)) == rows

    zstandard = pytest.importorskip("zstandard")
    zst = tmp_path / "x.jsonl.zst"
    cctx = zstandard.ZstdCompressor()
    with open(zst, "wb") as f:
        f.write(cctx.compress(("\n".join(json.dumps(r) for r in rows) + "\n").encode()))
    assert list(fe.iter_jsonl_file(zst)) == rows


def test_iter_parquet_file(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    import pyarrow as pa

    table = pa.table({"text": ["alpha", "beta", None], "id": [1, 2, 3]})
    path = tmp_path / "x.parquet"
    pq.write_table(table, path)
    rows = list(fe.iter_parquet_file(path))
    assert len(rows) == 3
    assert rows[0]["text"] == "alpha"


def test_http_reader_treats_eof_close_as_eof():
    """urllib3 closes the raw stream at EOF; the reader must not raise 'I/O on closed file'."""

    class _ClosingBytesIO(io.BytesIO):
        def read(self, n=-1):
            data = super().read(n)
            if not data:
                self.close()
            return data

    class _FakeResp:
        def __init__(self, payload: bytes):
            self.raw = _ClosingBytesIO(payload)
            self.closed = False

        def close(self):
            self.closed = True

    payload = b'{"text": "a"}\n{"text": "b"}\n{"text": "c"}\n'
    resp = _FakeResp(payload)
    records = list(fe.iter_jsonl_lines(resp, "jsonl"))
    assert records == [{"text": "a"}, {"text": "b"}, {"text": "c"}]
    assert resp.closed


def test_resolve_explicit_files_needs_no_network():
    source = fe.Source(repo="toy/repo", note="", files=("f1.jsonl", "f2.jsonl"))
    assert fe.resolve_files(source, "http://127.0.0.1:1", "main") == ["f1.jsonl", "f2.jsonl"]


# --------------------------------------------------------------------------- fetch bookkeeping
def test_fetch_domain_monkeypatched(tmp_path, monkeypatch):
    source = fe.Source(repo="toy/repo", note="toy", files=("f1.jsonl",))
    monkeypatch.setitem(fe.SOURCES, "toy", [source])
    monkeypatch.setattr(
        fe, "list_repo_files",
        lambda repo, endpoint, revision, **kw: {"sha": "cafebabe", "files": ["f1.jsonl"]},
    )

    def fake_iter(source_, files, endpoint, revision, *, max_download_mb):
        for _ in range(10):
            yield files[0], {"text": "x" * 500}

    monkeypatch.setattr(fe, "iter_source_records", fake_iter)

    out = tmp_path / "toy.jsonl"
    prov = fe.fetch_domain("toy", out, max_chars=1200, endpoint="http://example.invalid",
                           log=lambda msg: None)
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(lines) == 3  # 500 + 500 + 500 >= 1200
    assert all(len(line["text"]) == 500 for line in lines)
    assert prov["total_records"] == 3
    assert prov["total_chars"] == 1500
    assert prov["sources"][0]["resolved_sha"] == "cafebabe"
    assert prov["sources"][0]["files"][0]["records"] == 3
    assert (tmp_path / "toy.jsonl.provenance.json").exists()

    # a second fetch truncates the output instead of appending duplicates
    fe.fetch_domain("toy", out, max_chars=500, endpoint="http://example.invalid",
                    log=lambda msg: None)
    assert len(out.read_text().splitlines()) == 1


def test_fetch_domain_requires_some_text(tmp_path, monkeypatch):
    source = fe.Source(repo="toy/empty", note="", files=("f1.jsonl",))
    monkeypatch.setitem(fe.SOURCES, "empty", [source])
    monkeypatch.setattr(fe, "list_repo_files", lambda *a, **kw: {"sha": None, "files": []})
    monkeypatch.setattr(fe, "iter_source_records", lambda *a, **kw: iter(()))
    with pytest.raises(SystemExit, match="no text fetched"):
        fe.fetch_domain("empty", tmp_path / "empty.jsonl", endpoint="http://example.invalid",
                        log=lambda msg: None)


def test_cli_list_sources():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "fetch_eval_data.py"), "--list-sources"],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    for domain in ("web", "code", "math", "academic", "wikipedia", "stackexchange", "zh"):
        assert domain in proc.stdout
    assert "marin-8b-retro" in proc.stdout
    assert "books" in proc.stdout  # listed as planned/not wired

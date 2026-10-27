"""Public tools must be usable from a plain checkout (no hardcoded machine paths)."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_ab_bench_help_works_from_repo_root() -> None:
    proc = subprocess.run(
        [sys.executable, "tools/ab_bench.py", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"tools/ab_bench.py --help failed:\n{out}"
    assert "usage:" in proc.stdout.lower(), out

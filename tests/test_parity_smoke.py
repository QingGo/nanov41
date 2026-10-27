"""CPU smoke test for the W1 parity harness (same gate as README quick start)."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_parity_float32_passes() -> None:
    proc = subprocess.run(
        [sys.executable, "parity.py", "--dtype", "float32"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"parity.py exited {proc.returncode}\n{out}"
    assert "ALL PASS" in proc.stdout, out
    # The three gates must be reported as passing.
    assert "T1 True" in out and "T2 True" in out and "T3 True" in out, out

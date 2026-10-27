"""The checked-in generated training module must match ref/model.py + the five patches."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_generated_patched_ref_is_up_to_date() -> None:
    proc = subprocess.run(
        [sys.executable, "tools/make_patched_ref.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"make_patched_ref.py --check failed:\n{out}"
    assert "is up to date" in proc.stdout, out

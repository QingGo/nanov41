# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""CPU kernel-parity regression suite (scripts moved from the internal W2 tooling).

Each ``kernel_parity_*.py`` file is a standalone script (self-printing, ``sys.exit(main())``)
that compares an optimized module from ``nanov41/`` against the reference implementation
on CPU. They are run in subprocesses here so that:
  * pytest does not collect their internal ``test_*`` helper functions as test cases, and
  * a crash/timeout in one script is reported against that script only.

Fast subset (~43 s CPU): run ``pytest -q -m "not slow"``.
Full suite (~133 s CPU): ``pytest -q``.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

CASES = [
    ("fast_attn", 180, False),
    ("fast_moe", 300, False),
    ("fast_mhc", 420, True),
    ("fused_head_ce", 180, False),
    ("fused_rmsnorm", 180, False),
    ("fused_rope", 180, False),
    ("fused_swiglu", 180, False),
    ("moe_stacked", 180, False),
    ("moe_stacked_model", 300, True),
]


@pytest.mark.parametrize(
    "name,timeout,slow",
    [pytest.param(n, t, s, marks=pytest.mark.slow if s else ()) for n, t, s in CASES],
    ids=[n for n, _, _ in CASES],
)
def test_kernel_parity(name: str, timeout: int, slow: bool) -> None:  # noqa: ARG001
    script = ROOT / "tests" / f"kernel_parity_{name}.py"
    assert script.exists(), f"missing parity script {script}"

    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"{script.name} exited {proc.returncode}\n{out[-4000:]}"
    assert "PASS" in proc.stdout, f"{script.name} did not report PASS\n{out[-4000:]}"

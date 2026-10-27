# SPDX-License-Identifier: MIT
# Copyright (c) 2026 QingGo
"""Make the repository root importable for the experiments test package."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

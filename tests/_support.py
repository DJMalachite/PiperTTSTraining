"""Test helpers.

Tests run from a plain checkout with no venv and no dependencies, so they add
``src/`` to the path themselves rather than relying on an install.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

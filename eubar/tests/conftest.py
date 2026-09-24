"""Shared pytest helpers for the small pre-refactor regression suite."""
from __future__ import annotations

import sys
from pathlib import Path

# The handoff zip contains the package directly rather than an installed wheel.
# Make `import eubar` work when users run `pytest` from the repository root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

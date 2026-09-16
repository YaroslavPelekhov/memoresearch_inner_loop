#!/usr/bin/env python3
"""Entrypoint for the adaptive multi-fidelity benchmark."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoresearch.multifidelity.runner import main

if __name__ == "__main__":
    raise SystemExit(main())

"""Shim: keeps ``python run_sqformat_study.py`` working.

Actual logic lives in :mod:`experiments.sqformat.cli`.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.sqformat.cli import main

if __name__ == "__main__":
    sys.exit(main())

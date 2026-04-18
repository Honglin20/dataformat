"""Shim: keeps ``python run_4bit_study.py`` working.

Actual logic lives in :mod:`experiments.fourbit.cli`.
"""
from __future__ import annotations

import os
import sys

# Ensure repo root on path when run from elsewhere
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.fourbit.cli import main

if __name__ == "__main__":
    sys.exit(main())

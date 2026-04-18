"""Shim: keeps `python generate_qsnr_table.py` working.

Actual logic lives in utils/qsnr_table.py.
"""
from __future__ import annotations
import sys
from utils.qsnr_table import main

if __name__ == "__main__":
    sys.exit(main())

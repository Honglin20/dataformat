"""Entry point for the 4-bit data format study.

Usage
-----
    # Full study (Part 1 + Part 2 on the MNIST Transformer)
    python run_4bit_study.py

    # Part 1 only (no real model required)
    python run_4bit_study.py --part 1

    # Part 2 only (requires model checkpoint; trains if missing)
    python run_4bit_study.py --part 2

    # Custom output directory
    python run_4bit_study.py --out results/fourbit

Configuration lives in :mod:`fourbit.config`.  To add or remove formats,
edit ``DEFAULT_CONFIG.formats`` there; the experiment code iterates over the
config without hard-coding any format name.
"""
from __future__ import annotations

import argparse
import os
import sys

# Ensure repo root on path when run from elsewhere
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fourbit.config import DEFAULT_CONFIG
from fourbit import part1, part2, reporter


def main():
    p = argparse.ArgumentParser(description="4-bit data format study")
    p.add_argument("--part", choices=["1", "2", "all"], default="all",
                   help="Which part to run (default: all)")
    p.add_argument("--out", default=None,
                   help="Override config.output_dir (default: results/fourbit)")
    p.add_argument("--model-path", default="results/mnist/model.pt",
                   help="Path to MNIST Transformer checkpoint (Part 2)")
    p.add_argument("--data-dir", default="~/.cache/mnist",
                   help="MNIST data cache directory (Part 2)")
    args = p.parse_args()

    config = DEFAULT_CONFIG
    if args.out:
        config = DEFAULT_CONFIG.__class__(**{**DEFAULT_CONFIG.__dict__, "output_dir": args.out})

    out_dir = config.output_dir
    os.makedirs(out_dir, exist_ok=True)

    part1_res = None
    part2_df = None

    if args.part in ("1", "all"):
        part1_res = part1.run_all(config)

    if args.part in ("2", "all"):
        part2_df, _ = part2.run(config, model_path=args.model_path, data_dir=args.data_dir)

    # Only generate the full report when both parts were run in the same call.
    if args.part == "all" and part1_res is not None and part2_df is not None:
        report_path = os.path.join(out_dir, "report.md")
        reporter.generate_report(config, part1_res, part2_df, report_path)
        print(f"\n[Report] Written → {report_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

"""SQ-Format study CLI entry point.

Mirrors the shape of :mod:`experiments.fourbit.cli` but loads the
SQ-Format :data:`experiments.sqformat.config.DEFAULT_CONFIG` so the
study inherits 16 format cells × 3 transforms × Y-quantisation × the
``QuantizedMHA`` swap.
"""
from __future__ import annotations

import argparse
import os
import sys

from experiments.sqformat.config import DEFAULT_CONFIG
from experiments.fourbit import part1, part2, reporter


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="SQ-Format comparison study")
    p.add_argument("--part", choices=["1", "2", "all"], default="all",
                   help="Which part to run (default: all)")
    p.add_argument("--out", default=None,
                   help="Override config.output_dir (default: results/sqformat)")
    p.add_argument("--model-path", default="results/mnist/model.pt",
                   help="Path to MNIST Transformer checkpoint (Part 2)")
    p.add_argument("--data-dir", default="~/.cache/mnist",
                   help="MNIST data cache directory (Part 2)")
    p.add_argument("--profile-samples", type=int, default=None,
                   help="Override FourBitConfig.profile_samples (default: 128)")
    args = p.parse_args(argv)

    config = DEFAULT_CONFIG
    overrides: dict = {}
    if args.out:
        overrides["output_dir"] = args.out
    if args.profile_samples is not None:
        overrides["profile_samples"] = args.profile_samples
    if overrides:
        config = DEFAULT_CONFIG.__class__(
            **{**DEFAULT_CONFIG.__dict__, **overrides}
        )

    out_dir = config.output_dir
    os.makedirs(out_dir, exist_ok=True)

    part1_res = None
    part2_df = None
    part2_acc_df = None

    if args.part in ("1", "all"):
        part1_res = part1.run_all(config)

    if args.part in ("2", "all"):
        part2_df, _, part2_acc_df = part2.run(
            config, model_path=args.model_path, data_dir=args.data_dir,
        )

    if args.part == "all" and part1_res is not None and part2_df is not None:
        report_path = os.path.join(out_dir, "report.md")
        reporter.generate_report(
            config, part1_res, part2_df, report_path,
            accuracy_df=part2_acc_df,
        )
        print(f"\n[Report] Written → {report_path}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

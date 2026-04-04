"""ExperimentRunner: executes format × distribution sweeps.

The runner is deliberately decoupled from any specific format or distribution —
all such knowledge lives in ExperimentConfig and the format registry.

To add a new format:
    1. Implement it in ``formats/``.
    2. Register it in ``formats/__init__.build_all_formats()``.
    3. Add its name to the relevant ``FormatGroup`` in ``experiments/defaults.py``.
    No changes to this file are needed.

To add a new distribution:
    1. Implement the generator function in ``distributions/generators.py``.
    2. Add a ``DistributionConfig`` entry in ``experiments/defaults.py``.
    No changes to this file are needed.

To add a new metric:
    1. Implement it in ``distributions/metrics.py`` and add it to
       ``evaluate_all()``.
    2. Add the metric key to the ``metrics`` field of your ``ExperimentConfig``.
    No changes to this file are needed.
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np
import pandas as pd

from experiments.config import ExperimentConfig, FormatGroup
from distributions.metrics import evaluate_all


class ExperimentRunner:
    """Runs format × distribution sweeps as defined by an ExperimentConfig.

    Parameters
    ----------
    config : ExperimentConfig
        Full experiment specification (groups, distributions, metrics, etc.).
    registry : dict[str, QuantFormat]
        Format objects keyed by name (from ``build_all_formats()``).
        Unknown format names in a FormatGroup trigger a warning and are skipped.
    """

    def __init__(self, config: ExperimentConfig, registry: dict):
        self.config  = config
        self.registry = registry

    # ── Core sweep ────────────────────────────────────────────────────────────

    def run_group(self, group: FormatGroup) -> pd.DataFrame:
        """Evaluate every (format, distribution) pair in *group*.

        Returns a DataFrame with one row per (format, distribution) pair.
        """
        cfg = self.config
        fmt_names = group.filter_available(self.registry)

        total  = len(fmt_names) * len(cfg.distributions)
        count  = 0
        rows: list[dict] = []

        for dist_cfg in cfg.distributions:
            x, dist_meta = dist_cfg.generate(cfg.n_samples, cfg.seed)

            for fmt_name in fmt_names:
                count += 1
                if cfg.verbose:
                    print(
                        f"  [{group.name}] [{count}/{total}] "
                        f"{fmt_name} × {dist_cfg.name}",
                        end="\r",
                        flush=True,
                    )

                fmt = self.registry[fmt_name]
                try:
                    x_q     = fmt.quantize(x)
                    metrics = evaluate_all(x, x_q)
                except Exception as exc:
                    metrics = {m: np.nan for m in cfg.metrics}
                    if cfg.verbose:
                        print(
                            f"\n  WARNING {fmt_name} × {dist_cfg.name}: {exc}"
                        )

                # Encoding overhead metadata (scalar fields only)
                overhead: dict = {}
                try:
                    raw = fmt.encoding_overhead()
                    overhead = {
                        f"enc_{k}": v for k, v in raw.items()
                        if isinstance(v, (int, float))
                    }
                except Exception:
                    pass

                rows.append({
                    "group":     group.name,
                    "bits":      group.bits,
                    "format":    fmt_name,
                    "dist_name": dist_cfg.name,
                    "dist_tags": ",".join(dist_cfg.tags),
                    **{k: metrics.get(k, np.nan) for k in cfg.metrics},
                    **overhead,
                    **{f"dist_{k}": v for k, v in dist_meta.items()
                       if isinstance(v, (int, float, str))},
                })

        if cfg.verbose:
            print()   # newline after progress line

        return pd.DataFrame(rows)

    # ── Top-level entry point ─────────────────────────────────────────────────

    def run(self) -> Dict[str, pd.DataFrame]:
        """Run all format groups defined in the config.

        Returns
        -------
        dict[str, pd.DataFrame]
            Keys are ``group.name`` (e.g. ``"4bit"``, ``"8bit"``).
        """
        results: Dict[str, pd.DataFrame] = {}
        os.makedirs(self.config.output_dir, exist_ok=True)

        for group in self.config.groups:
            if self.config.verbose:
                print(f"\n=== {group.label} ({len(group.formats)} formats × "
                      f"{len(self.config.distributions)} dists) ===")

            df = self.run_group(group)
            results[group.name] = df

            # Persist to CSV — one file per group
            csv_path = os.path.join(
                self.config.output_dir,
                f"{self.config.name}_{group.name}.csv",
            )
            df.to_csv(csv_path, index=False)

            if self.config.verbose:
                print(f"  Saved → {csv_path}")
                _print_summary(df, self.config.metrics)

        return results

    # ── Convenience: load previously saved results ────────────────────────────

    def load(self) -> Dict[str, pd.DataFrame]:
        """Load results from CSV files that were written by a previous run()."""
        results = {}
        for group in self.config.groups:
            csv_path = os.path.join(
                self.config.output_dir,
                f"{self.config.name}_{group.name}.csv",
            )
            if os.path.exists(csv_path):
                results[group.name] = pd.read_csv(csv_path)
            else:
                raise FileNotFoundError(
                    f"No saved results for group '{group.name}' at {csv_path}. "
                    "Run runner.run() first."
                )
        return results


# ── Internal helper ───────────────────────────────────────────────────────────

def _print_summary(df: pd.DataFrame, metrics: list[str]) -> None:
    primary = "eff_bits" if "eff_bits" in metrics else metrics[0]
    if primary in df.columns:
        summary = (
            df.groupby("format")[primary]
            .mean()
            .sort_values(ascending=False)
            .round(3)
        )
        print(f"  Mean {primary} per format:\n{summary.to_string()}")

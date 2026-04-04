"""Experiment configuration dataclasses.

Design principles:
  - Open/Closed: adding formats/distributions/metrics requires only touching
    the config (or defaults.py), never the ExperimentRunner logic.
  - Dependency Injection: runner receives registry + config; no globals.
  - Single Responsibility: config describes *what* to run; runner handles *how*.

Usage
-----
    from experiments.config import ExperimentConfig, FormatGroup, DistributionConfig
    from experiments.runner import ExperimentRunner
    from formats import build_all_formats

    cfg = ExperimentConfig(
        name="my_study",
        groups=[FormatGroup("4bit", "4-bit Regime", ["INT4", "MXINT4"], bits=4)],
        distributions=[DistributionConfig("Gauss", gaussian_fn, tags=["gaussian"])],
    )
    runner = ExperimentRunner(cfg, build_all_formats())
    results = runner.run()   # -> {"4bit": pd.DataFrame}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Tuple

import numpy as np


# ── Distribution ──────────────────────────────────────────────────────────────

@dataclass
class DistributionConfig:
    """Specification for a single synthetic distribution.

    Parameters
    ----------
    name : str
        Human-readable name shown in plots and CSV output.
    fn : Callable[[int, int], tuple[np.ndarray, dict]]
        Function with signature ``fn(n_samples, seed) -> (x, metadata_dict)``.
    tags : list[str]
        Free-form labels for filtering (e.g. ``["outlier", "channel"]``).
    """
    name: str
    fn: Callable[[int, int], Tuple[np.ndarray, dict]]
    tags: List[str] = field(default_factory=list)

    def generate(self, n: int, seed: int) -> Tuple[np.ndarray, dict]:
        return self.fn(n, seed)

    def has_tag(self, tag: str) -> bool:
        return tag in self.tags


# ── Format group ──────────────────────────────────────────────────────────────

@dataclass
class FormatGroup:
    """A named set of formats evaluated together in one experiment sweep.

    Parameters
    ----------
    name : str
        Short identifier used as CSV suffix and dict key (e.g. ``"4bit"``).
    label : str
        Human-readable display label for titles / legends.
    formats : list[str]
        Format keys from the format registry (``build_all_formats()``).
    bits : int
        Nominal bit-width for this group (used for metadata and axis labels).
    """
    name: str
    label: str
    formats: List[str]
    bits: int

    def filter_available(self, registry: dict) -> List[str]:
        """Return the subset of ``self.formats`` that exist in *registry*."""
        missing = [f for f in self.formats if f not in registry]
        if missing:
            import warnings
            warnings.warn(
                f"FormatGroup '{self.name}': formats not in registry will be "
                f"skipped: {missing}",
                stacklevel=2,
            )
        return [f for f in self.formats if f in registry]


# ── Top-level experiment config ───────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    """Complete specification for one experiment run.

    Parameters
    ----------
    name : str
        Experiment identifier, used as prefix for output CSV files.
    groups : list[FormatGroup]
        Format groups to evaluate (e.g. 4-bit group, 8-bit group).
    distributions : list[DistributionConfig]
        Distributions to sweep over.
    metrics : list[str]
        Metric keys to record (must match keys returned by
        ``distributions.metrics.evaluate_all``).
    n_samples : int
        Tensor length passed to each distribution generator.
    seed : int
        Global random seed.
    output_dir : str
        Directory for CSV outputs (created if absent).
    verbose : bool
        Print per-combination progress lines.
    """
    name: str
    groups: List[FormatGroup]
    distributions: List[DistributionConfig]
    metrics: List[str] = field(default_factory=lambda: [
        "mse", "snr_db", "kl_div", "max_ae", "eff_bits"
    ])
    n_samples: int = 4096
    seed: int = 42
    output_dir: str = "results"
    verbose: bool = True

    # ── Convenience helpers ───────────────────────────────────────────────────

    def filter_distributions(self, tag: str) -> "ExperimentConfig":
        """Return a copy with only distributions that have *tag*."""
        import copy
        cfg = copy.copy(self)
        cfg.distributions = [d for d in self.distributions if d.has_tag(tag)]
        return cfg

    def with_group(self, group: FormatGroup) -> "ExperimentConfig":
        """Return a copy with *group* appended to the groups list."""
        import copy
        cfg = copy.copy(self)
        cfg.groups = list(self.groups) + [group]
        return cfg

    def subset_formats(self, formats: List[str]) -> "ExperimentConfig":
        """Return a copy where every group is filtered to *formats*."""
        import copy
        cfg = copy.copy(self)
        cfg.groups = [
            FormatGroup(g.name, g.label,
                        [f for f in g.formats if f in formats], g.bits)
            for g in self.groups
        ]
        return cfg

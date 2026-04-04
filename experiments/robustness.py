"""Phase 2: Distribution robustness experiment.

Runs all formats × all distributions → 5-metric evaluation matrix.

To add a new format or distribution: edit ``experiments/defaults.py``.
No changes to this file are needed.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from experiments.config import ExperimentConfig
from experiments.runner import ExperimentRunner
from experiments.defaults import ROBUSTNESS_CONFIG
from formats import build_all_formats


def run_robustness_experiment(
    config: ExperimentConfig = ROBUSTNESS_CONFIG,
    verbose: bool = True,
):
    """Run the robustness experiment.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment specification. Defaults to ``ROBUSTNESS_CONFIG``
        (all formats × 16 distributions). Override to run a custom subset::

            from experiments.defaults import ROBUSTNESS_CONFIG, GROUP_4BIT
            cfg = ROBUSTNESS_CONFIG.subset_formats(["INT4", "MXINT4", "HAD+INT4(C)"])
            run_robustness_experiment(config=cfg)

    verbose : bool
        Print per-combination progress.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are group names (``"4bit"``, ``"8bit"``).
    """
    cfg = config
    cfg.verbose = verbose

    registry = build_all_formats(dim=256, seed=cfg.seed)
    runner = ExperimentRunner(cfg, registry)
    return runner.run()


if __name__ == "__main__":
    results = run_robustness_experiment(verbose=True)
    for group_name, df in results.items():
        print(f"\n[{group_name}] mean eff_bits per format:")
        print(df.groupby("format")["eff_bits"].mean().sort_values(ascending=False).round(3))

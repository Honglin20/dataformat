"""Phase 3: 4-bit vs 8-bit ablation experiment.

Compares all 4-bit formats against all 8-bit formats including
SQ-Format (4b dense) vs SQ-Format(8b) (8b dense).

To add a new format or distribution: edit ``experiments/defaults.py``.
No changes to this file are needed.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from experiments.config import ExperimentConfig
from experiments.runner import ExperimentRunner
from experiments.defaults import ABLATION_CONFIG
from formats import build_all_formats


def run_bitwidth_ablation(
    config: ExperimentConfig = ABLATION_CONFIG,
    verbose: bool = True,
):
    """Run 4-bit vs 8-bit ablation.

    Parameters
    ----------
    config : ExperimentConfig
        Defaults to ``ABLATION_CONFIG`` (all 4-bit + 8-bit formats,
        7-distribution set). Override for custom subsets::

            from experiments.defaults import ABLATION_CONFIG, GROUP_HW_4BIT, GROUP_HW_8BIT
            from experiments.config import ExperimentConfig
            hw_cfg = ExperimentConfig(
                name="ablation_hw",
                groups=[GROUP_HW_4BIT, GROUP_HW_8BIT],
                distributions=ABLATION_CONFIG.distributions,
            )
            run_bitwidth_ablation(config=hw_cfg)

    verbose : bool

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: ``"4bit"`` and ``"8bit"`` (or custom group names).
    """
    cfg = config
    cfg.verbose = verbose

    registry = build_all_formats(dim=256, seed=cfg.seed)
    runner = ExperimentRunner(cfg, registry)
    return runner.run()


if __name__ == "__main__":
    results = run_bitwidth_ablation(verbose=True)
    for group_name, df in results.items():
        print(f"\n[{group_name}] mean eff_bits per format:")
        print(df.groupby("format")["eff_bits"].mean().sort_values(ascending=False).round(3))

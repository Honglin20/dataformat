"""Master pipeline: run all experiments and generate all figures.

Usage:
    python run_all.py [--fast] [--skip-hw] [--figs-only] [--hw-focus]

Options:
    --fast      Use FAST_CONFIG (N=512, 5 formats, 3 distributions).
    --skip-hw   Skip hardware PPA evaluation (useful without PyRTL).
    --figs-only Load pre-computed CSVs and regenerate figures only.
    --hw-focus  Run only the 3-paradigm hardware-focus experiment
                (MXINT / BFP / SQ-Format at 4-bit and 8-bit).

To add a new format or distribution to the pipeline:
    → Edit experiments/defaults.py only. No changes needed here.
"""

from __future__ import annotations

import os
import sys
import argparse
import time

sys.path.insert(0, os.path.dirname(__file__))
os.makedirs("results/figures", exist_ok=True)


# ── Phase helpers ─────────────────────────────────────────────────────────────

def run_phase1_check() -> bool:
    """Quick smoke-test: verify all formats can quantize a test tensor."""
    print("\n[Phase 1] Format smoke-test...")
    import numpy as np
    from formats import build_all_formats

    x = np.random.default_rng(42).normal(0, 1, 256).astype("float32")
    x[0] = 50.0
    formats = build_all_formats(dim=256, seed=42)
    failures = []
    for name, fmt in formats.items():
        try:
            q = fmt.quantize(x)
            assert q.shape == x.shape
        except Exception as e:
            failures.append(f"  {name}: {e}")
    if failures:
        print(f"  WARNING: {len(failures)} formats failed:")
        for f in failures:
            print(f)
    else:
        print(f"  All {len(formats)} formats OK.")
    return len(failures) == 0


def run_phase2_robustness(config) -> dict:
    """Phase 2: Distribution robustness sweep."""
    print("\n[Phase 2] Distribution robustness sweep...")
    t0 = time.time()
    from experiments.robustness import run_robustness_experiment
    results = run_robustness_experiment(config=config, verbose=True)
    print(f"  Completed in {time.time() - t0:.1f}s.")
    return results


def run_phase3_ablation(config) -> dict:
    """Phase 3: 4-bit vs 8-bit ablation."""
    print("\n[Phase 3] 4-bit vs 8-bit ablation...")
    t0 = time.time()
    from experiments.bitwidth_ablation import run_bitwidth_ablation
    results = run_bitwidth_ablation(config=config, verbose=True)
    print(f"  Completed in {time.time() - t0:.1f}s.")
    return results


def run_phase4_hardware() -> dict:
    """Phase 4: Hardware PPA evaluation."""
    print("\n[Phase 4] Hardware PPA evaluation...")
    t0 = time.time()
    from hardware.ppa_evaluator import run_full_ppa_evaluation
    results = run_full_ppa_evaluation(use_yosys=True)
    print(f"  Completed in {time.time() - t0:.1f}s.")
    return results


def run_phase5_visualizations(out_dir: str = "results/figures") -> list:
    """Phase 5: Generate all figures."""
    print("\n[Phase 5] Generating all figures...")
    failures = []

    figures = [
        ("Fig 1:  Distribution Evolution",
         "visualization.plot_distributions", "plot_distribution_evolution"),
        ("Fig 2:  Outlier Sensitivity Heatmap",
         "visualization.plot_outlier_heatmap", "plot_outlier_heatmap"),
        ("Fig 3+4: Pareto Frontiers",
         "visualization.plot_pareto", "plot_pareto_charts"),
        ("Fig 5:  HAD vs MXINT vs SQ Comparison",
         "visualization.plot_had_vs_random", "plot_had_vs_mxint"),
        ("Fig 6:  PPA Bubble Chart (4b+8b panels)",
         "visualization.plot_ppa_bubble", "plot_ppa_bubble"),
        ("Fig 7:  Roofline Model",
         "visualization.plot_roofline", "plot_roofline"),
        ("Fig 8:  Channel Heatmap",
         "visualization.plot_channel_heatmap", "plot_channel_heatmap"),
        ("Fig 9:  Encoding Efficiency",
         "visualization.plot_encoding_eff", "plot_encoding_efficiency"),
        ("Fig 10: Pipeline Breakdown",
         "visualization.plot_pipeline", "plot_pipeline_breakdown"),
        ("Fig 11: Hardware Area Breakdown (4b+8b panels)",
         "visualization.plot_area", "plot_area_breakdown"),
        ("Fig 12: Cross-Distribution Robustness",
         "visualization.plot_distribution_robustness", "plot_distribution_robustness"),
        ("Fig 13: Outlier Row Fraction Sweep",
         "visualization.plot_outlier_fraction", "plot_outlier_fraction"),
    ]

    for name, module_path, fn_name in figures:
        print(f"  Generating {name}...")
        try:
            t0 = time.time()
            mod = __import__(module_path, fromlist=[fn_name])
            getattr(mod, fn_name)(out_dir=out_dir)
            print(f"    Done ({time.time() - t0:.1f}s)")
        except Exception as e:
            print(f"    ERROR: {e}")
            failures.append(name)

    if failures:
        print(f"\n  {len(failures)} figures failed: {failures}")
    else:
        print(f"\n  All {len(figures)} figures generated → {out_dir}/")
    return failures


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Outlier format study — full pipeline."
    )
    parser.add_argument("--fast",      action="store_true",
                        help="Use FAST_CONFIG (N=512, minimal formats)")
    parser.add_argument("--skip-hw",   action="store_true",
                        help="Skip hardware PPA evaluation")
    parser.add_argument("--figs-only", action="store_true",
                        help="Skip experiments, regenerate figures from saved CSVs")
    parser.add_argument("--hw-focus",  action="store_true",
                        help="Run hardware-focus experiment (MXINT/BFP/SQ only)")
    args = parser.parse_args()

    # Pick config
    if args.fast:
        from experiments.defaults import FAST_CONFIG as exp_config
    elif args.hw_focus:
        from experiments.defaults import HW_FOCUS_CONFIG as exp_config
    else:
        from experiments.defaults import ABLATION_CONFIG as exp_config

    robustness_config = exp_config
    if not args.fast and not args.hw_focus:
        from experiments.defaults import ROBUSTNESS_CONFIG
        robustness_config = ROBUSTNESS_CONFIG

    print("=" * 60)
    print("Outlier Format Study — Full Pipeline")
    print(f"Config: {exp_config.name}  N={exp_config.n_samples}  Seed={exp_config.seed}")
    print(f"Groups: {[g.name for g in exp_config.groups]}")
    print("=" * 60)

    t_start = time.time()

    if not args.figs_only:
        ok = run_phase1_check()
        if not ok:
            print("Phase 1 failures detected — proceeding anyway.")

        run_phase2_robustness(robustness_config)
        run_phase3_ablation(exp_config)

        if not args.skip_hw:
            try:
                run_phase4_hardware()
            except Exception as e:
                print(f"  Phase 4 failed: {e}")
                print("  Continuing with analytical estimates in figures...")
        else:
            print("\n[Phase 4] Skipped (--skip-hw).")

    run_phase5_visualizations()

    print(f"\nTotal time: {time.time() - t_start:.1f}s")
    print(f"Results in: {os.path.abspath('results/')}")


if __name__ == "__main__":
    main()

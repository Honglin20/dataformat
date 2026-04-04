"""Master pipeline: run all experiments and generate all figures.

Usage:
    python run_all.py [--fast] [--skip-hw] [--figs-only]

Options:
    --fast      Use small N for quick sanity check (N=512).
    --skip-hw   Skip hardware PPA evaluation (useful without PyRTL installed).
    --figs-only Load pre-computed CSVs and regenerate figures only.
"""

import os
import sys
import argparse
import time

# Allow imports from project root
sys.path.insert(0, os.path.dirname(__file__))

os.makedirs("results/figures", exist_ok=True)


def run_phase1_check():
    """Quick smoke-test: verify all formats can quantize a test tensor."""
    print("\n[Phase 1] Format smoke-test...")
    import numpy as np
    from formats import build_all_formats

    x = np.random.default_rng(42).normal(0, 1, 256).astype(np.float32)
    x[0] = 50.0   # inject one outlier
    formats = build_all_formats(dim=256, seed=42)
    failures = []
    for name, fmt in formats.items():
        try:
            q = fmt.quantize(x)
            assert q.shape == x.shape, f"Shape mismatch: {q.shape} vs {x.shape}"
        except Exception as e:
            failures.append(f"  {name}: {e}")

    if failures:
        print(f"  WARNING: {len(failures)} formats failed:")
        for f in failures:
            print(f)
    else:
        print(f"  All {len(formats)} formats OK.")
    return len(failures) == 0


def run_phase2_robustness(n: int, seed: int):
    """Phase 2: Distribution robustness experiment."""
    print("\n[Phase 2] Distribution robustness experiment...")
    t0 = time.time()
    from experiments.robustness import run_robustness_experiment
    df = run_robustness_experiment(n=n, seed=seed, verbose=True)
    print(f"  Completed in {time.time()-t0:.1f}s. Shape: {df.shape}")
    return df


def run_phase3_ablation(n: int, seed: int):
    """Phase 3: Bit-width ablation."""
    print("\n[Phase 3] Bit-width ablation (4-bit vs 8-bit)...")
    t0 = time.time()
    from experiments.bitwidth_ablation import run_bitwidth_ablation
    results = run_bitwidth_ablation(n=n, seed=seed, verbose=True)
    print(f"  Completed in {time.time()-t0:.1f}s.")
    return results


def run_phase4_hardware():
    """Phase 4: Hardware PPA evaluation."""
    print("\n[Phase 4] Hardware PPA evaluation...")
    t0 = time.time()
    from hardware.ppa_evaluator import run_full_ppa_evaluation
    results = run_full_ppa_evaluation(use_yosys=True)
    print(f"  Completed in {time.time()-t0:.1f}s.")
    return results


def run_phase5_visualizations(out_dir: str = "results/figures"):
    """Phase 5: Generate all 10 figures."""
    print("\n[Phase 5] Generating all figures...")
    failures = []

    figures = [
        ("Fig 1: Distribution Evolution",
         lambda: __import__("visualization.plot_distributions", fromlist=["plot_distribution_evolution"]).plot_distribution_evolution(out_dir=out_dir)),
        ("Fig 2: Outlier Sensitivity Heatmap",
         lambda: __import__("visualization.plot_outlier_heatmap", fromlist=["plot_outlier_heatmap"]).plot_outlier_heatmap(out_dir=out_dir)),
        ("Fig 3+4: Pareto Frontiers",
         lambda: __import__("visualization.plot_pareto", fromlist=["plot_pareto_charts"]).plot_pareto_charts(out_dir=out_dir)),
        ("Fig 5: MXINT vs HAD+INT vs SQ Comparison",
         lambda: __import__("visualization.plot_had_vs_random", fromlist=["plot_had_vs_mxint"]).plot_had_vs_mxint(out_dir=out_dir)),
        ("Fig 6: PPA Bubble Chart",
         lambda: __import__("visualization.plot_ppa_bubble", fromlist=["plot_ppa_bubble"]).plot_ppa_bubble(out_dir=out_dir)),
        ("Fig 7: Roofline Model",
         lambda: __import__("visualization.plot_roofline", fromlist=["plot_roofline"]).plot_roofline(out_dir=out_dir)),
        ("Fig 8: Channel Heatmap",
         lambda: __import__("visualization.plot_channel_heatmap", fromlist=["plot_channel_heatmap"]).plot_channel_heatmap(out_dir=out_dir)),
        ("Fig 9: Encoding Efficiency",
         lambda: __import__("visualization.plot_encoding_eff", fromlist=["plot_encoding_efficiency"]).plot_encoding_efficiency(out_dir=out_dir)),
        ("Fig 10: Pipeline Breakdown",
         lambda: __import__("visualization.plot_pipeline", fromlist=["plot_pipeline_breakdown"]).plot_pipeline_breakdown(out_dir=out_dir)),
        ("Fig 11: Hardware Area Breakdown",
         lambda: __import__("visualization.plot_area", fromlist=["plot_area_breakdown"]).plot_area_breakdown(out_dir=out_dir)),
    ]

    for name, fn in figures:
        print(f"  Generating {name}...")
        try:
            t0 = time.time()
            fn()
            print(f"    Done ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"    ERROR: {e}")
            failures.append(name)

    if failures:
        print(f"\n  {len(failures)} figures failed: {failures}")
    else:
        print(f"\n  All {len(figures)} figures generated → {out_dir}/")
    return failures


def main():
    parser = argparse.ArgumentParser(description="Run full outlier-format research pipeline.")
    parser.add_argument("--fast", action="store_true",
                        help="Use N=512 for quick sanity check")
    parser.add_argument("--skip-hw", action="store_true",
                        help="Skip hardware PPA evaluation")
    parser.add_argument("--figs-only", action="store_true",
                        help="Skip experiments, regenerate figures from saved CSVs")
    args = parser.parse_args()

    N = 512 if args.fast else 4096
    SEED = 42

    print("=" * 60)
    print("Outlier Format Study — Full Pipeline")
    print(f"N={N}, Seed={SEED}, fast={args.fast}")
    print("=" * 60)

    t_start = time.time()

    if not args.figs_only:
        ok = run_phase1_check()
        if not ok:
            print("Phase 1 failures detected — proceeding anyway.")

        run_phase2_robustness(N, SEED)
        run_phase3_ablation(N, SEED)

        if not args.skip_hw:
            try:
                run_phase4_hardware()
            except Exception as e:
                print(f"  Phase 4 failed: {e}")
                print("  Continuing with figures using analytical estimates...")
        else:
            print("\n[Phase 4] Skipped (--skip-hw).")

    run_phase5_visualizations()

    print(f"\nTotal time: {time.time()-t_start:.1f}s")
    print(f"Results in: {os.path.abspath('results/')}")


if __name__ == "__main__":
    main()

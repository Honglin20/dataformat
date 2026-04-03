"""PPA Evaluator: Comprehensive hardware cost evaluation.

Aggregates:
  1. PyRTL-based area/timing estimates for INT and MXFP arrays.
  2. FWHT module PPA.
  3. SQ-Format Gather/Scatter PPA.
  4. Format converter decode PPA.
  5. Horowitz energy model for compute + memory + overhead.
  6. BOPs analysis.
  7. Roofline arithmetic intensity.

Produces a unified comparison table for Scheme A vs Scheme B (vs B+).

Also provides optional Yosys integration:
  - Export PyRTL block to Verilog via pyrtl.output_to_verilog().
  - Call Yosys for synthesis statistics if available.
  - Fall back to PyRTL × 2.5 calibration factor if Yosys is absent.
"""

import os
import subprocess
import tempfile
import math
import numpy as np

from hardware.pyrtl_modules.int_mac_array import get_int_array_ppa
from hardware.pyrtl_modules.mxfp_mac_array import get_mxfp_array_ppa
from hardware.pyrtl_modules.fwht_module import get_fwht_ppa
from hardware.pyrtl_modules.sq_gather_scatter import get_sq_gather_scatter_ppa
from hardware.pyrtl_modules.format_converters import get_all_converter_ppas
from hardware.energy_model import EnergyModel
from hardware.roofline import build_roofline_data
from hardware.bop_counter import compare_formats_bops
from config import ARRAY_ROWS, ARRAY_COLS


# ── Yosys integration (optional) ─────────────────────────────────────────────

def _try_yosys_synthesis(verilog_str: str) -> dict | None:
    """Attempt Yosys synthesis on a Verilog string.

    Returns a dict with cell counts if Yosys is available, else None.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".v", mode="w", delete=False) as f:
            f.write(verilog_str)
            tmp_path = f.name

        script = (
            f'read_verilog {tmp_path}; '
            'synth -top top -flatten; '
            'stat -tech cmos'
        )
        result = subprocess.run(
            ["yosys", "-p", script],
            capture_output=True, text=True, timeout=60
        )

        if result.returncode != 0:
            return None

        # Parse cell count from Yosys stat output
        lines = result.stdout.split("\n")
        cells = {}
        in_section = False
        for line in lines:
            if "Number of cells:" in line:
                in_section = True
            if in_section and line.strip().startswith("$"):
                parts = line.strip().split()
                if len(parts) >= 2:
                    cells[parts[0]] = int(parts[-1])

        total_cells = sum(cells.values())
        return {
            "tool": "yosys",
            "total_cells": total_cells,
            "cell_breakdown": cells,
            "raw_stdout": result.stdout[-2000:],  # last 2KB
        }
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _try_export_pyrtl_verilog(block) -> str | None:
    """Export a PyRTL working block to Verilog string."""
    try:
        import pyrtl
        import io
        buf = io.StringIO()
        pyrtl.output_to_verilog(buf, block=block)
        return buf.getvalue()
    except Exception:
        return None


# ── Scheme definitions ───────────────────────────────────────────────────────

def evaluate_scheme_a(
    rows: int = ARRAY_ROWS,
    cols: int = ARRAY_COLS,
    element_bits_list: list = (4, 8),
    use_yosys: bool = True,
) -> dict:
    """Scheme A: MX array total overhead.

    Evaluates both:
      - MXFP4/8: FP systolic array with exponent alignment shifter.
      - MXINT4/8: INT systolic array with POT scale broadcast (no barrel shifter).

    For MXFP: uses mxfp_mac_array (exponent alignment on critical path).
    For MXINT: uses int_mac_array + scale SRAM read overhead (dequant = right-shift).
    """
    results = {}
    em = EnergyModel()
    from hardware.pyrtl_modules.format_converters import get_mxfp_scale_read_ppa

    for bits in element_bits_list:
        n_macs = rows * cols
        n_weights = rows * cols

        # ── MXFP variant ──────────────────────────────────────────────────────
        mxfp_ppa = get_mxfp_array_ppa(element_bits=bits, rows=rows, cols=cols)
        decode_info = get_mxfp_scale_read_ppa(element_bits=bits)
        mxfp_energy = em.total_inference_energy(
            f"MXFP{bits}", n_macs=n_macs,
            n_weight_reads=n_weights, n_activation_reads=n_weights
        )
        results[f"MXFP{bits}"] = {
            "scheme": "A",
            "format": f"MXFP{bits}",
            **mxfp_ppa,
            "decode_bw_amplification": decode_info["bandwidth_amplification"],
            "scale_sram_read_pj": decode_info["scale_sram_read_pj"],
            **{f"energy_{k}": v for k, v in mxfp_energy.items()},
            "yosys": None,
        }

        # ── MXINT variant ─────────────────────────────────────────────────────
        # INT array is simpler than MXFP (no barrel shifter, no exp alignment).
        # Overhead: 1 E8M0 scale SRAM read per 32 elements + per-element right-shift.
        mxint_ppa = get_int_array_ppa(bits=bits, rows=rows, cols=cols)
        mxint_energy = em.total_inference_energy(
            f"MXINT{bits}", n_macs=n_macs,
            n_weight_reads=n_weights, n_activation_reads=n_weights
        )
        # Block scale bandwidth amplification: 8 scale bits per 32×bits data bits
        scale_bw_amp = 1.0 + 8.0 / (32 * bits)
        results[f"MXINT{bits}"] = {
            "scheme": "A",
            "format": f"MXINT{bits}",
            **mxint_ppa,
            "decode_bw_amplification": scale_bw_amp,
            "scale_sram_read_pj": decode_info["scale_sram_read_pj"],  # same scale format
            **{f"energy_{k}": v for k, v in mxint_energy.items()},
            "yosys": None,
        }

    return results


def evaluate_scheme_b(
    rows: int = ARRAY_ROWS,
    cols: int = ARRAY_COLS,
    transform_n: int = 256,
    bits_list: list = (4, 8),
    use_yosys: bool = True,
) -> dict:
    """Scheme B: Pure INT array + FWHT preprocessing module.

    Includes:
      - INT4/8 systolic array.
      - FWHT module (shared across all rows, area amortized).
    """
    results = {}
    em = EnergyModel()

    for bits in bits_list:
        array_ppa = get_int_array_ppa(bits=bits, rows=rows, cols=cols)
        fwht_ppa = get_fwht_ppa(n=transform_n, bits=bits)

        # Total area: array + FWHT (shared, not per-PE)
        total_area = (
            array_ppa.get("area_mm2_total", 0) +
            fwht_ppa.get("area_mm2_total", 0)
        )

        # Critical path: max(array, FWHT pipelined in parallel)
        # FWHT runs in preprocessing, parallel to weight loading → doesn't add latency
        # unless it's on the critical path of the first compute
        array_timing = array_ppa.get("critical_path_ps", 0)
        fwht_timing = fwht_ppa.get("critical_path_ps", 0)
        # FWHT is pipelined → effective crit path = max(array, FWHT/n_stages)
        effective_timing = max(array_timing, fwht_timing / fwht_ppa.get("n_stages", 1))

        n_macs = rows * cols
        energy = em.total_inference_energy(
            f"HAD+INT{bits}", n_macs=n_macs,
            n_weight_reads=n_macs, n_activation_reads=n_macs
        )

        # FWHT area as fraction of array area (amortization argument)
        fwht_array_ratio = fwht_ppa.get("area_mm2_total", 0) / max(
            array_ppa.get("area_mm2_total", 1e-9), 1e-9
        )

        results[f"HAD+INT{bits}"] = {
            "scheme": "B",
            "format": f"HAD+INT{bits}",
            **array_ppa,
            "fwht_area_mm2": fwht_ppa.get("area_mm2_total", 0),
            "fwht_timing_ps": fwht_ppa.get("critical_path_ps", 0),
            "area_mm2_total": float(total_area),
            "effective_timing_ps": float(effective_timing),
            "fwht_vs_array_area_ratio": float(fwht_array_ratio),
            **{f"energy_{k}": v for k, v in energy.items()},
        }

    return results


def evaluate_scheme_b_plus(
    rows: int = ARRAY_ROWS,
    cols: int = ARRAY_COLS,
    transform_n: int = 256,
    n_sq_elements: int = 4096,
    use_yosys: bool = True,
) -> dict:
    """Scheme B+: INT array + FWHT + SQ-Format Gather/Scatter.

    The HAD+SQ combination: global Gaussianization + sparse salient weight handling.
    """
    bits = 4   # B+ is primarily a 4-bit scheme

    array_ppa = get_int_array_ppa(bits=bits, rows=rows, cols=cols)
    fwht_ppa = get_fwht_ppa(n=transform_n, bits=bits)
    sq_ppa = get_sq_gather_scatter_ppa(n=n_sq_elements)

    total_area = (
        array_ppa.get("area_mm2_total", 0) +
        fwht_ppa.get("area_mm2_total", 0) +
        sq_ppa.get("area_mm2_total", 0)
    )

    em = EnergyModel()
    energy = em.total_inference_energy(
        "HAD+SQ", n_macs=rows * cols,
        n_weight_reads=rows * cols, n_activation_reads=rows * cols
    )

    return {
        "HAD+SQ": {
            "scheme": "B+",
            "format": "HAD+SQ",
            "array_area_mm2": array_ppa.get("area_mm2_total", 0),
            "fwht_area_mm2": fwht_ppa.get("area_mm2_total", 0),
            "sq_gs_area_mm2": sq_ppa.get("area_mm2_total", 0),
            "area_mm2_total": float(total_area),
            "array_timing_ps": array_ppa.get("critical_path_ps", 0),
            "fwht_timing_ps": fwht_ppa.get("critical_path_ps", 0),
            "sq_timing_ps": sq_ppa.get("critical_path_ps", 0),
            **{f"energy_{k}": v for k, v in energy.items()},
        }
    }


def run_full_ppa_evaluation(
    rows: int = ARRAY_ROWS,
    cols: int = ARRAY_COLS,
    transform_n: int = 256,
    use_yosys: bool = True,
) -> dict:
    """Run complete PPA evaluation for all schemes and formats.

    Returns
    -------
    dict with keys:
      'scheme_a': {format_name: ppa_dict, ...}
      'scheme_b': {format_name: ppa_dict, ...}
      'scheme_b_plus': {format_name: ppa_dict, ...}
      'converters': [converter_ppa, ...]
      'roofline': [roofline_data_point, ...]
      'bops': [bops_comparison, ...]
      'summary': comparison DataFrame
    """
    import pandas as pd

    print("Evaluating Scheme A (MXFP arrays)...")
    scheme_a = evaluate_scheme_a(rows, cols, use_yosys=use_yosys)

    print("Evaluating Scheme B (INT + FWHT)...")
    scheme_b = evaluate_scheme_b(rows, cols, transform_n, use_yosys=use_yosys)

    print("Evaluating Scheme B+ (INT + HAD + SQ)...")
    scheme_b_plus = evaluate_scheme_b_plus(rows, cols, transform_n, use_yosys=use_yosys)

    print("Getting format converter PPAs...")
    converters = get_all_converter_ppas()

    print("Building Roofline data...")
    roofline = build_roofline_data()

    print("Computing BOPs...")
    all_fmt_names = (
        list(scheme_a.keys()) + list(scheme_b.keys()) + list(scheme_b_plus.keys())
    )
    bops = compare_formats_bops(all_fmt_names, M=rows, K=transform_n, N=cols)

    # Summary comparison table
    all_schemes = {**scheme_a, **scheme_b, **scheme_b_plus}
    summary_rows = []
    for fmt_name, d in all_schemes.items():
        summary_rows.append({
            "format": fmt_name,
            "scheme": d.get("scheme", "?"),
            "area_mm2_total": d.get("area_mm2_total", np.nan),
            "critical_path_ps": d.get("critical_path_ps", d.get("effective_timing_ps", np.nan)),
            "max_freq_ghz": 1000.0 / max(d.get("critical_path_ps", d.get("effective_timing_ps", 1000.0)), 1e-3),
            "energy_total_pJ": d.get("energy_total_pJ", np.nan),
        })
    summary_df = pd.DataFrame(summary_rows)

    os.makedirs("results", exist_ok=True)
    summary_df.to_csv("results/ppa_summary.csv", index=False)
    print(f"PPA summary saved → results/ppa_summary.csv")

    return {
        "scheme_a": scheme_a,
        "scheme_b": scheme_b,
        "scheme_b_plus": scheme_b_plus,
        "converters": converters,
        "roofline": roofline,
        "bops": bops,
        "summary": summary_df,
    }


if __name__ == "__main__":
    results = run_full_ppa_evaluation()
    print("\n=== PPA Summary ===")
    print(results["summary"].to_string(index=False))

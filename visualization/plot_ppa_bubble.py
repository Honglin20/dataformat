"""Figure 6: Hardware PPA Bubble Chart.

X-axis:  Decode latency (critical path, ns).
Y-axis:  End-to-end quantization quality (EffBits, from distribution experiments).
Bubble:  Size = total energy per inference (pJ) — bigger bubble = more energy.
Color:   Format family.

This chart gives the definitive system-level answer:
  - X left, Y high, small bubble → ideal.
  - Shows whether MX's decode complexity costs latency WITHOUT quality benefit.
  - Shows whether HAD's transform overhead is amortized by the INT array.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from hardware.pyrtl_modules.int_mac_array import get_int_array_ppa
from hardware.pyrtl_modules.mxfp_mac_array import get_mxfp_array_ppa
from hardware.pyrtl_modules.fwht_module import get_fwht_ppa
from hardware.pyrtl_modules.sq_gather_scatter import get_sq_gather_scatter_ppa
from hardware.pyrtl_modules.format_converters import (
    get_nf4_decoder_ppa, get_fp6_unpacker_ppa, get_nvfp4_decoder_ppa,
    get_mxfp_scale_read_ppa
)
from hardware.energy_model import EnergyModel
from distributions.generators import channel_outliers
from distributions.metrics import effective_bits
from formats import build_all_formats
from visualization.style import save_fig, PALETTE, get_color, get_marker


_N_ELEMENTS = 4096
_TRANSFORM_N = 256


def _get_format_hardware_data() -> dict:
    """Collect decode latency and energy for each format."""
    em = EnergyModel()
    n_macs = 16 * 16  # 16×16 array
    n_reads = n_macs

    hw_data = {}

    # INT4/8 (Scheme B base)
    for bits in [4, 8]:
        ppa = get_int_array_ppa(bits=bits)
        energy = em.total_inference_energy(
            f"INT{bits}", n_macs=n_macs, n_weight_reads=n_reads, n_activation_reads=n_reads
        )
        hw_data[f"INT{bits}"] = {
            "latency_ns": ppa.get("critical_path_ps", 200) / 1000,
            "energy_pJ": energy["total_pJ"],
        }

    # MXFP4/8 (Scheme A)
    for bits in [4, 8]:
        ppa = get_mxfp_array_ppa(element_bits=bits)
        scale_info = get_mxfp_scale_read_ppa(element_bits=bits)
        energy = em.total_inference_energy(
            f"MXFP{bits}", n_macs=n_macs, n_weight_reads=n_reads, n_activation_reads=n_reads
        )
        hw_data[f"MXFP{bits}"] = {
            "latency_ns": ppa.get("critical_path_ps", ppa.get("effective_timing_ps", 300)) / 1000,
            "energy_pJ": energy["total_pJ"],
        }

    # MXINT4/8
    for bits in [4, 8]:
        # MXINT uses INT array + scale broadcast; slightly higher latency than plain INT
        ppa = get_int_array_ppa(bits=bits)
        energy = em.total_inference_energy(
            f"MXINT{bits}", n_macs=n_macs, n_weight_reads=n_reads, n_activation_reads=n_reads
        )
        hw_data[f"MXINT{bits}"] = {
            "latency_ns": ppa.get("critical_path_ps", 200) / 1000 * 1.15,
            "energy_pJ": energy["total_pJ"],
        }

    # NF4 (LUT decoder overhead)
    nf4_dec = get_nf4_decoder_ppa()
    int4_ppa = get_int_array_ppa(bits=4)
    energy = em.total_inference_energy(
        "NF4", n_macs=n_macs, n_weight_reads=n_reads, n_activation_reads=n_reads
    )
    hw_data["NF4"] = {
        "latency_ns": (int4_ppa.get("critical_path_ps", 200) + nf4_dec["critical_path_ps"]) / 1000,
        "energy_pJ": energy["total_pJ"],
    }

    # FP6
    fp6_dec = get_fp6_unpacker_ppa()
    int8_ppa = get_int_array_ppa(bits=8)
    energy = em.total_inference_energy(
        "FP6", n_macs=n_macs, n_weight_reads=n_reads, n_activation_reads=n_reads
    )
    hw_data["FP6"] = {
        "latency_ns": (int8_ppa.get("critical_path_ps", 200) + fp6_dec["critical_path_ps"]) / 1000,
        "energy_pJ": energy["total_pJ"],
    }

    # NVFP4
    nvfp4_dec = get_nvfp4_decoder_ppa()
    energy = em.total_inference_energy(
        "NVFP4", n_macs=n_macs, n_weight_reads=n_reads, n_activation_reads=n_reads
    )
    hw_data["NVFP4"] = {
        "latency_ns": (int4_ppa.get("critical_path_ps", 200) + nvfp4_dec["critical_path_ps"]) / 1000,
        "energy_pJ": energy["total_pJ"],
    }

    # HAD+INT4/8 (Scheme B)
    fwht_ppa = get_fwht_ppa(n=_TRANSFORM_N, bits=4)
    for bits in [4, 8]:
        array_ppa = get_int_array_ppa(bits=bits)
        array_timing = array_ppa.get("critical_path_ps", 200)
        fwht_timing = fwht_ppa.get("critical_path_ps", 100)
        # FWHT pipelined → effective timing = array (FWHT overlaps with data load)
        effective_timing = max(array_timing, fwht_timing / fwht_ppa.get("n_stages", 8))
        energy = em.total_inference_energy(
            f"HAD+INT{bits}", n_macs=n_macs, n_weight_reads=n_reads, n_activation_reads=n_reads
        )
        hw_data[f"HAD+INT{bits}"] = {
            "latency_ns": effective_timing / 1000,
            "energy_pJ": energy["total_pJ"],
        }

    # HAD+SQ (Scheme B+)
    sq_ppa = get_sq_gather_scatter_ppa(n=_N_ELEMENTS)
    energy = em.total_inference_energy(
        "HAD+SQ", n_macs=n_macs, n_weight_reads=n_reads, n_activation_reads=n_reads
    )
    hw_data["HAD+SQ"] = {
        "latency_ns": (
            int4_ppa.get("critical_path_ps", 200) +
            sq_ppa.get("critical_path_ps", 150) * 0.3   # SQ in parallel
        ) / 1000,
        "energy_pJ": energy["total_pJ"],
    }

    # SmoothQuant+INT4/8
    for bits in [4, 8]:
        array_ppa = get_int_array_ppa(bits=bits)
        # SmoothQuant scale multiply: cheap (per-channel, 1 mul per channel)
        sq_timing = 40.0  # ~1 gate delay for scale multiply
        energy = em.total_inference_energy(
            f"SmoothQuant+INT{bits}", n_macs=n_macs, n_weight_reads=n_reads, n_activation_reads=n_reads
        )
        hw_data[f"SmoothQuant+INT{bits}"] = {
            "latency_ns": (array_ppa.get("critical_path_ps", 200) + sq_timing) / 1000,
            "energy_pJ": energy["total_pJ"],
        }

    # TurboQuant+INT4
    energy = em.total_inference_energy(
        "TurboQuant+INT4", n_macs=n_macs, n_weight_reads=n_reads, n_activation_reads=n_reads
    )
    hw_data["TurboQuant+INT4"] = {
        "latency_ns": int4_ppa.get("critical_path_ps", 200) / 1000 * 1.02,  # XOR: near-free
        "energy_pJ": energy["total_pJ"],
    }

    # FP32/BF16 reference
    for fmt in ["FP32", "BF16"]:
        bits = 32 if fmt == "FP32" else 16
        energy = em.total_inference_energy(
            fmt, n_macs=n_macs, n_weight_reads=n_reads, n_activation_reads=n_reads
        )
        hw_data[fmt] = {
            "latency_ns": bits * 0.01,   # rough reference
            "energy_pJ": energy["total_pJ"],
        }

    return hw_data


def _get_format_quality(seed: int = 42, n: int = 2048) -> dict:
    """Get EffBits quality for each format on channel-outlier distribution."""
    all_formats = build_all_formats(dim=256, seed=seed)
    x, _ = channel_outliers(n=n, outlier_sigma=50.0, seed=seed)
    quality = {}
    for fmt_name, fmt in all_formats.items():
        try:
            x_q = fmt.quantize(x)
            quality[fmt_name] = effective_bits(x, x_q)
        except Exception:
            quality[fmt_name] = np.nan
    return quality


def plot_ppa_bubble(out_dir: str = "results/figures", seed: int = 42):
    """Plot Figure 6: Hardware PPA Bubble Chart."""
    hw_data = _get_format_hardware_data()
    quality = _get_format_quality(seed=seed)

    fig, ax = plt.subplots(figsize=(13, 8))

    # Bubble size: proportional to energy (normalized)
    all_energies = [v["energy_pJ"] for v in hw_data.values() if np.isfinite(v["energy_pJ"])]
    e_min, e_max = min(all_energies), max(all_energies)
    size_scale = 3000  # max bubble area

    plotted = []
    for fmt_name, hw in hw_data.items():
        q = quality.get(fmt_name, np.nan)
        if not np.isfinite(q) or not np.isfinite(hw["latency_ns"]):
            continue

        latency = hw["latency_ns"]
        energy = hw["energy_pJ"]
        # Normalize energy → bubble size
        bubble_size = 50 + (energy - e_min) / max(e_max - e_min, 1e-9) * size_scale

        color = PALETTE.get(fmt_name, "#888888")
        marker = get_marker(fmt_name)

        sc = ax.scatter(latency, q, s=bubble_size, c=color, marker=marker,
                        alpha=0.75, zorder=5, edgecolors="white", linewidths=0.8)

        ax.annotate(
            fmt_name,
            (latency, q),
            xytext=(5, 5), textcoords="offset points",
            fontsize=8, color=color,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none")
        )
        plotted.append((fmt_name, latency, q, energy, bubble_size))

    # Ideal region annotation
    if plotted:
        lat_vals = [p[1] for p in plotted]
        q_vals = [p[2] for p in plotted]
        ax.annotate(
            "← Ideal Region\n(Low latency, High quality)",
            xy=(min(lat_vals) * 1.1, max(q_vals) * 0.95),
            fontsize=9, color="green", style="italic",
            bbox=dict(boxstyle="round", fc="#e8f5e9", ec="green", alpha=0.5)
        )

    # Energy bubble legend
    energy_levels = [e_min, (e_min + e_max) / 2, e_max]
    for e_level in energy_levels:
        bsize = 50 + (e_level - e_min) / max(e_max - e_min, 1e-9) * size_scale
        ax.scatter([], [], s=bsize, c="gray", alpha=0.5,
                   label=f"Energy = {e_level:.0f} pJ")

    ax.set_xlabel("Decode Latency / Critical Path (ns)", fontsize=12)
    ax.set_ylabel("Quantization Quality (EffBits, higher = better)", fontsize=12)
    ax.set_title(
        "Figure 6: Hardware PPA Bubble Chart\n"
        "(Bubble size = inference energy; left-upper small bubble = optimal system design)",
        fontsize=12
    )
    ax.legend(loc="lower right", fontsize=8, title="Energy scale")
    ax.grid(True, alpha=0.3)

    # Draw "Pareto frontier" annotation
    ax.text(0.02, 0.98,
            "Scheme A (MXFP): higher latency due to\nexponent alignment + scale broadcast",
            transform=ax.transAxes, fontsize=8, va="top", color="navy",
            bbox=dict(boxstyle="round", fc="lightyellow", ec="navy", alpha=0.7))

    save_fig(fig, "fig06_ppa_bubble", out_dir)
    return fig


if __name__ == "__main__":
    plot_ppa_bubble()

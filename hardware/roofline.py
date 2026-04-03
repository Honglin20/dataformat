"""Roofline Model Analysis.

The Roofline model characterizes whether a computation is:
  - Memory-Bound: performance limited by memory bandwidth.
  - Compute-Bound: performance limited by peak compute throughput.

Key formula:
  Attainable_Performance(I) = min(Peak_Compute, Peak_BW × I)

where I = Arithmetic Intensity (FLOPs / Byte) is the "ridge point".

For quantized formats, two effects occur:
  1. Reduced compute precision → higher Peak_Compute (more ops/s).
  2. Reduced memory footprint → fewer bytes transferred, increasing I.
  BUT: metadata (MX block scales) increases effective bytes, reducing I.

This module:
  - Computes arithmetic intensity for each format.
  - Plots formats on the Roofline.
  - Identifies whether MX metadata makes it memory-bound again.
"""

import numpy as np
from config import PEAK_COMPUTE_TOPS, PEAK_BW_TB_S


def arithmetic_intensity(
    format_name: str,
    n_ops: float,
    n_elements: int,
    data_bits: int = None,
    metadata_bits_per_element: float = 0.0,
    transform_overhead_ops: float = 0.0,
) -> dict:
    """Compute arithmetic intensity I = FLOPs / Bytes for a given format.

    Parameters
    ----------
    format_name : str
    n_ops : float
        Number of arithmetic operations (MACs count as 2 FLOPs).
    n_elements : int
        Total elements transferred from memory.
    data_bits : int, optional
        Bit-width of data elements. Inferred from format_name if not given.
    metadata_bits_per_element : float
        Extra bits per element for format metadata (e.g., 0.25 for MX block scale).
    transform_overhead_ops : float
        Additional ops from transform (HAD, SmoothQuant, etc.) — adds to numerator.
    """
    fmt_upper = format_name.upper()

    if data_bits is None:
        bit_map = {
            "FP32": 32, "BF16": 16, "INT8": 8, "MXFP8": 8, "MXINT8": 8,
            "FP6": 6, "INT4": 4, "MXFP4": 4, "MXINT4": 4, "NF4": 4,
            "NVFP4": 4, "SQ": 5,   # SQ: effective ~5 bits (4 dense + 1 mask + overhead)
        }
        data_bits = 8
        for k, v in bit_map.items():
            if k in fmt_upper:
                data_bits = v
                break

    # Total bytes transferred
    total_bits = (data_bits + metadata_bits_per_element) * n_elements
    total_bytes = total_bits / 8.0

    # Total ops (MACs = 2 FLOPs, + transform overhead)
    total_ops = n_ops + transform_overhead_ops

    I = total_ops / max(total_bytes, 1e-9)   # FLOPs/Byte

    return {
        "format": format_name,
        "data_bits": data_bits,
        "metadata_bits_per_element": metadata_bits_per_element,
        "n_elements": n_elements,
        "total_bytes": float(total_bytes),
        "total_ops": float(total_ops),
        "arithmetic_intensity": float(I),
    }


def attainable_performance(I: float, format_name: str) -> float:
    """Attainable performance in TOPs given arithmetic intensity I.

    Performance = min(Peak_Compute[format], Peak_BW × I)
    """
    fmt_upper = format_name.upper()
    if "INT4" in fmt_upper or "MXINT4" in fmt_upper or "NF4" in fmt_upper or "NVFP4" in fmt_upper:
        peak_compute = PEAK_COMPUTE_TOPS["int4"]
    elif "INT8" in fmt_upper or "MXINT8" in fmt_upper or "FP6" in fmt_upper:
        peak_compute = PEAK_COMPUTE_TOPS["int8"]
    elif "MXFP8" in fmt_upper or "FP8" in fmt_upper:
        peak_compute = PEAK_COMPUTE_TOPS["fp8"]
    elif "FP16" in fmt_upper or "BF16" in fmt_upper:
        peak_compute = PEAK_COMPUTE_TOPS["fp16"]
    else:
        peak_compute = PEAK_COMPUTE_TOPS["fp32"]

    bw_limited = PEAK_BW_TB_S * 1e12 * I / 1e12   # convert to TOPs
    return min(peak_compute, bw_limited)


def ridge_point(format_name: str) -> float:
    """Arithmetic intensity at the Roofline ridge point (transition BW→Compute bound)."""
    fmt_upper = format_name.upper()
    if "INT4" in fmt_upper or "MXINT4" in fmt_upper or "NF4" in fmt_upper or "NVFP4" in fmt_upper:
        peak_compute = PEAK_COMPUTE_TOPS["int4"]
    elif "INT8" in fmt_upper or "MXINT8" in fmt_upper:
        peak_compute = PEAK_COMPUTE_TOPS["int8"]
    elif "MXFP8" in fmt_upper or "FP8" in fmt_upper:
        peak_compute = PEAK_COMPUTE_TOPS["fp8"]
    elif "FP16" in fmt_upper or "BF16" in fmt_upper:
        peak_compute = PEAK_COMPUTE_TOPS["fp16"]
    else:
        peak_compute = PEAK_COMPUTE_TOPS["fp32"]

    # Ridge: Peak_Compute = Peak_BW × I → I = Peak_Compute / Peak_BW
    bw_tops_per_flop_byte = PEAK_BW_TB_S * 1e12 / 1e12  # TB/s → TOPS/FLOPs·Byte
    return peak_compute / (PEAK_BW_TB_S * 1e12 / 1e12)


def build_roofline_data(
    n_elements: int = 4096,
    matmul_m: int = 16,
    matmul_k: int = 256,
    matmul_n: int = 256,
) -> list:
    """Build Roofline data points for all formats.

    Models a single linear layer: Y[M×N] = X[M×K] @ W[K×N]
    n_ops = 2 × M × K × N (MACs)
    n_elements = M×K (activations) + K×N (weights)

    Parameters
    ----------
    matmul_m, matmul_k, matmul_n : int
        Matrix dimensions for the modeled linear layer.
    """
    n_mac = matmul_m * matmul_k * matmul_n
    n_ops = 2 * n_mac      # MACs → FLOPs
    n_elem_weights = matmul_k * matmul_n
    n_elem_activations = matmul_m * matmul_k
    n_elem_total = n_elem_weights + n_elem_activations

    import math

    formats_config = [
        # (name, data_bits, metadata_bpe, transform_ops)
        ("FP32",          32, 0.0, 0),
        ("BF16",          16, 0.0, 0),
        ("INT8",           8, 0.0, 0),
        ("MXFP8",          8, 0.25, 0),     # 0.25 bpe block scale
        ("MXINT8",         8, 0.25, 0),
        ("INT4",           4, 0.0, 0),
        ("MXFP4",          4, 0.25, 0),
        ("MXINT4",         4, 0.25, 0),
        ("NVFP4",          4, 0.0, 0),
        ("NF4",            4, 0.0, 0),
        ("FP6",            6, 0.0, 0),
        ("SQ-Format",      5, 1.0, 0),      # 4 dense + 1 mask
        ("HAD+INT8",       8, 0.0, n_elem_total * math.log2(max(n_elem_total/matmul_m, 2))),
        ("HAD+INT4",       4, 0.0, n_elem_total * math.log2(max(n_elem_total/matmul_m, 2))),
        ("SmoothQuant+INT8", 8, 0.0, n_elem_activations),
        ("SmoothQuant+INT4", 4, 0.0, n_elem_activations),
        ("RandRot+INT4",   4, 0.0, n_elem_activations ** 2),  # dense matmul
        ("TurboQuant+INT4", 4, 0.0, n_elem_activations),
    ]

    results = []
    for fmt_name, dbits, meta_bpe, t_ops in formats_config:
        ai = arithmetic_intensity(
            fmt_name, n_ops, n_elem_total,
            data_bits=dbits,
            metadata_bits_per_element=meta_bpe,
            transform_overhead_ops=t_ops,
        )
        perf = attainable_performance(ai["arithmetic_intensity"], fmt_name)
        ridge = ridge_point(fmt_name)
        is_memory_bound = ai["arithmetic_intensity"] < ridge

        results.append({
            **ai,
            "attainable_tops": float(perf),
            "ridge_point": float(ridge),
            "is_memory_bound": is_memory_bound,
            "regime": "memory-bound" if is_memory_bound else "compute-bound",
        })

    return results

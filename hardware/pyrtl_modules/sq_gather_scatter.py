"""SQ-Format Gather/Scatter unit in PyRTL.

Hardware decomposition of SQ-Format operations:

  GATHER unit:
    - Input: N-element tensor (dense, mixed precision) + 1-bit bitmask per element.
    - Output: k sparse high-precision elements (INT8/FP16) sent to high-precision ALU.
    - Mechanism: priority encoder + compaction network to extract salient indices.
    - Key cost: arbitration logic (barrel/butterfly compactor).

  SCATTER unit:
    - Input: computed high-precision partial sums + dense INT4 partial sums.
    - Output: merged result at correct output positions.
    - Mechanism: multiplexer array controlled by the bitmask.

Hardware costs:
  - Bitmask: 1 bit per element (stored in SRAM, amortized).
  - Gather compactor: O(N log N) in terms of mux count.
  - Scatter mux: O(N) mux units.
  - Arbitration (priority encoder): O(N log N) gates.
"""

import pyrtl
import math


def build_gather_unit(n: int, dense_bits: int = 4, sparse_bits: int = 8) -> dict:
    """Model the Gather unit for N-element SQ-Format tensor.

    Simplified model: for each element, a 2-to-1 mux selects between
    a "pass through" path and a "gather" path based on the bitmask bit.

    The dominant cost is the compaction network (like a sorting network
    for bitmask-indexed gather).
    """
    pyrtl.reset_working_block()

    # For tractability, model the core per-element gather logic:
    # Each element has: data (dense_bits), mask (1 bit) → output routing mux
    n_model = min(n, 8)   # model small N for PyRTL, scale analytically

    mask_bits   = [pyrtl.Input(1, f"mask_{i}") for i in range(n_model)]
    dense_data  = [pyrtl.Input(dense_bits, f"dense_{i}") for i in range(n_model)]
    sparse_data = [pyrtl.Input(sparse_bits, f"sparse_{i}") for i in range(n_model)]

    # Gather: select sparse or dense path based on mask
    outputs = []
    for i in range(n_model):
        # mux: if mask[i]==1, route sparse_data[i] to high-prec path
        out = pyrtl.Output(sparse_bits, f"gather_out_{i}")
        out <<= pyrtl.select(mask_bits[i],
                             falsecase=dense_data[i].zero_extended(sparse_bits),
                             truecase=sparse_data[i])
        outputs.append(out)

    # Priority encoder: find first set bit (indicates sparse element)
    # Simplified: OR tree to detect any mask bit
    any_sparse = pyrtl.Output(1, "any_sparse")
    or_tree = mask_bits[0]
    for m in mask_bits[1:]:
        or_tree = or_tree | m
    any_sparse <<= or_tree

    return {
        "block": pyrtl.working_block(),
        "n_modeled": n_model,
        "n_actual": n,
    }


def get_sq_gather_scatter_ppa(
    n: int,
    dense_bits: int = 4,
    sparse_bits: int = 8,
    sparsity_ratio: float = 0.01,
) -> dict:
    """Compute PPA for SQ-Format Gather/Scatter unit.

    Parameters
    ----------
    n : int
        Total tensor size.
    dense_bits, sparse_bits : int
        Bit-widths for dense/sparse components.
    sparsity_ratio : float
        Fraction of elements that are sparse/salient (default 1%).
    """
    k = max(1, int(math.ceil(sparsity_ratio * n)))

    try:
        gu_info = build_gather_unit(n, dense_bits, sparse_bits)
        block = gu_info["block"]
        n_modeled = gu_info["n_modeled"]

        per_elem_area = pyrtl.area_estimation(tech_in_nm=45, block=block) / n_modeled
        per_elem_timing = pyrtl.timing_estimation(block=block)

        # Scale to actual N
        gather_area = per_elem_area * n
        # Scatter: simpler than gather (just mux output to correct position)
        scatter_area = gather_area * 0.6   # empirical: scatter ~60% of gather
        # Arbitration (priority encoder) for k sparse elements: O(N log N)
        arbiter_gates = n * math.log2(max(n, 2)) * 1.5   # NAND2-equiv
        arbiter_area = arbiter_gates * 0.614e-6

        total_area = gather_area + scatter_area + arbiter_area
        critical_path = per_elem_timing * (1 + math.log2(n) / 4)

        return {
            "module": "SQ_Gather_Scatter",
            "n": n,
            "k_sparse": k,
            "dense_bits": dense_bits,
            "sparse_bits": sparse_bits,
            "sparsity_ratio": sparsity_ratio,
            "gather_area_mm2": float(gather_area),
            "scatter_area_mm2": float(scatter_area),
            "arbiter_area_mm2": float(arbiter_area),
            "area_mm2_total": float(total_area),
            "critical_path_ps": float(critical_path),
            "max_freq_ghz": 1000.0 / float(critical_path) if critical_path > 0 else 0,
            "method": "pyrtl_scaled",
        }
    except Exception as e:
        return _analytical_sq_ppa(n, dense_bits, sparse_bits, sparsity_ratio, error=str(e))


def _analytical_sq_ppa(
    n: int,
    dense_bits: int,
    sparse_bits: int,
    sparsity_ratio: float,
    error: str = "",
) -> dict:
    """Analytical NAND2 model for SQ Gather/Scatter.

    Gather mux:       n × (sparse_bits × 1.5) NAND2-equiv (n mux units)
    Scatter mux:      n × (sparse_bits × 1.0) NAND2-equiv
    Priority encoder: n × log2(n) × 1.5 NAND2-equiv
    Bitmask register: n × 8 NAND2-equiv (1-bit FF per element)
    """
    NAND2_AREA_UM2 = 0.614
    GATE_DELAY_PS = 40.0
    k = max(1, int(math.ceil(sparsity_ratio * n)))

    gather_gates = n * sparse_bits * 1.5
    scatter_gates = n * sparse_bits * 1.0
    arbiter_gates = n * math.log2(max(n, 2)) * 1.5
    bitmask_gates = n * 8   # 1-bit FFs for bitmask register

    total_gates = gather_gates + scatter_gates + arbiter_gates + bitmask_gates
    total_area_mm2 = total_gates * NAND2_AREA_UM2 * 1e-6

    # Critical path: mux chain (2 stages) + priority encoder (log2 N stages)
    stages = 2 + math.ceil(math.log2(max(n, 2)))
    crit_path_ps = stages * GATE_DELAY_PS

    return {
        "module": "SQ_Gather_Scatter",
        "n": n,
        "k_sparse": k,
        "dense_bits": dense_bits,
        "sparse_bits": sparse_bits,
        "sparsity_ratio": sparsity_ratio,
        "gather_nand2": float(gather_gates),
        "scatter_nand2": float(scatter_gates),
        "arbiter_nand2": float(arbiter_gates),
        "bitmask_nand2": float(bitmask_gates),
        "total_nand2_equiv": float(total_gates),
        "area_mm2_total": float(total_area_mm2),
        "critical_path_ps": float(crit_path_ps),
        "max_freq_ghz": 1000.0 / crit_path_ps if crit_path_ps > 0 else 0,
        "method": "analytical_nand2",
        "pyrtl_error": error,
    }

"""INT4/INT8 16×16 Systolic Array in PyRTL.

Models a weight-stationary systolic MAC array where:
  - Data flows east (activations) and south (partial sums).
  - Weights are pre-loaded and stationary in each PE.
  - Each PE computes: psum_out = psum_in + weight × activation.

For area/timing estimation only — not cycle-accurate for full matrix multiply.
We model a single PE pipeline stage and replicate for 16×16.
"""

import pyrtl


def build_int_mac_pe(bits: int, accum_bits: int = None) -> dict:
    """Build one INT MAC Processing Element in PyRTL.

    PE computes: psum_out = psum_in + weight * activation
    All operands are signed two's complement.

    Parameters
    ----------
    bits : int
        Operand bit-width (4 or 8).
    accum_bits : int
        Accumulator bit-width. Default = 2*bits + 4 to prevent overflow.

    Returns
    -------
    dict with 'inputs', 'outputs', and timing/area estimates.
    """
    if accum_bits is None:
        accum_bits = 2 * bits + 4

    pyrtl.reset_working_block()

    weight  = pyrtl.Input(bits, "weight")
    act     = pyrtl.Input(bits, "activation")
    psum_in = pyrtl.Input(accum_bits, "psum_in")

    # Sign-extend operands to accumulator width before multiply
    # PyRTL's * operator performs unsigned multiply; we handle signed via MSB
    w_ext  = weight.sign_extended(accum_bits)
    a_ext  = act.sign_extended(accum_bits)
    product = (w_ext * a_ext).truncate(accum_bits)
    psum_out = pyrtl.Output(accum_bits, "psum_out")
    psum_out <<= (psum_in + product).truncate(accum_bits)

    # Pipeline register for timing
    reg = pyrtl.Register(accum_bits, "psum_reg")
    reg.next <<= psum_out

    return {
        "bits": bits,
        "accum_bits": accum_bits,
        "block": pyrtl.working_block(),
    }


def build_int_mac_array(bits: int, rows: int = 16, cols: int = 16) -> dict:
    """Build the full INT systolic array and extract PPA metrics.

    Models rows×cols PEs as independent PyRTL blocks (area scales linearly).
    Returns area/timing by estimating one PE and multiplying by array size.

    Parameters
    ----------
    bits : int
        4 for INT4, 8 for INT8.
    rows, cols : int
        Array dimensions.
    """
    pe_info = build_int_mac_pe(bits)
    block = pe_info["block"]

    # Area and timing estimation from PyRTL
    logic_area = pyrtl.area_estimation(tech_in_nm=45, block=block)
    timing = pyrtl.timing_estimation(block=block)

    # Scale to full array (rows × cols PEs)
    n_pes = rows * cols

    # Timing: critical path stays same (PEs are parallel in a given cycle)
    # Area: sum of all PE areas
    area_mm2_per_pe = logic_area
    area_mm2_total = area_mm2_per_pe * n_pes

    return {
        "format": f"INT{bits}",
        "bits": bits,
        "array_size": f"{rows}×{cols}",
        "n_pes": n_pes,
        "area_mm2_per_pe": float(area_mm2_per_pe),
        "area_mm2_total": float(area_mm2_total),
        "critical_path_ps": float(timing),
        "max_freq_ghz": 1e3 / float(timing) if timing > 0 else float("inf"),
    }


def get_int_array_ppa(bits: int, rows: int = 16, cols: int = 16) -> dict:
    """Top-level entry point: build INT array and return PPA dict."""
    try:
        return build_int_mac_array(bits, rows, cols)
    except Exception as e:
        # Fallback: analytical model if PyRTL estimation fails
        return _analytical_int_ppa(bits, rows, cols, error=str(e))


def _analytical_int_ppa(bits: int, rows: int, cols: int, error: str = "") -> dict:
    """Analytical NAND2-equivalent gate count model for INT MAC PE.

    Based on standard cell synthesis estimates:
      - b-bit signed multiplier: ~3.5 × b² NAND2-equivalent gates
      - b-bit adder: ~1.5 × b NAND2-equivalent gates
      - Flip-flops for pipeline register: ~8 × b NAND2-equivalent gates

    NAND2 in 45nm TSMC: area ≈ 0.614 μm² (from FreePDK45)
    Timing: critical path ≈ (log2(b) + 3) × gate_delay, gate_delay ≈ 40ps
    """
    NAND2_AREA_UM2 = 0.614  # μm² per NAND2 in 45nm
    GATE_DELAY_PS = 40.0    # ps per logic stage (45nm FO4 ≈ 40ps)

    mul_gates = 3.5 * bits * bits
    add_gates = 1.5 * (2 * bits + 4)   # accumulator adder
    reg_gates = 8.0 * (2 * bits + 4)    # pipeline register FFs
    pe_gates = mul_gates + add_gates + reg_gates

    n_pes = rows * cols
    total_gates = pe_gates * n_pes
    total_area_um2 = total_gates * NAND2_AREA_UM2
    total_area_mm2 = total_area_um2 * 1e-6

    # Critical path: multiplier tree depth ≈ log2(b) + 2 adder stages
    import math
    stages = math.ceil(math.log2(bits)) + 3
    crit_path_ps = stages * GATE_DELAY_PS

    return {
        "format": f"INT{bits}",
        "bits": bits,
        "array_size": f"{rows}×{cols}",
        "n_pes": n_pes,
        "pe_nand2_equiv": float(pe_gates),
        "total_nand2_equiv": float(total_gates),
        "area_mm2_per_pe": float(pe_gates * NAND2_AREA_UM2 * 1e-6),
        "area_mm2_total": float(total_area_mm2),
        "critical_path_ps": float(crit_path_ps),
        "max_freq_ghz": 1000.0 / crit_path_ps,
        "method": "analytical_nand2",
        "pyrtl_error": error,
    }

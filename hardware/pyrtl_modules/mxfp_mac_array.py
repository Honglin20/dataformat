"""MXFP4/MXFP8 16×16 Systolic Array in PyRTL.

Key additional logic vs. INT array:
  1. Block Scale Broadcast Unit: one E8M0 scale per 32 elements,
     broadcast to all PEs in a block row.
  2. Exponent Alignment Shifter: align mantissas before addition
     (this is the main source of additional latency over INT).
  3. Mantissa MAC: shorter than INT (E2M1 → 1-bit mantissa, E4M3 → 3-bit).

The critical path includes: exponent comparison + mantissa shift + add.
This is longer than a pure INT critical path for the same effective precision.
"""

import pyrtl
import math


def build_mxfp4_pe() -> dict:
    """Build one MXFP4 (E2M1) MAC PE.

    Element format: 1 sign + 2 exponent + 1 mantissa = 4 bits.
    Shared scale: 8-bit E8M0 per block (broadcast, not per-PE).

    PE computes: product = w_mantissa * a_mantissa with exponent addition.
    Partial sums accumulated in a 16-bit register.
    """
    pyrtl.reset_working_block()

    # MXFP4 element: [sign(1)|exp(2)|mant(1)] — 4 bits
    w_elem  = pyrtl.Input(4, "w_elem")
    a_elem  = pyrtl.Input(4, "a_elem")
    # Shared block scale (E8M0, broadcast to all PEs in block)
    w_scale = pyrtl.Input(8, "w_scale")   # E8M0 weight block scale
    a_scale = pyrtl.Input(8, "a_scale")   # E8M0 activation block scale
    psum_in = pyrtl.Input(16, "psum_in")

    # Decompose E2M1 elements
    w_sign = w_elem[3]
    w_exp  = w_elem[1:3]   # 2 bits
    w_mant = w_elem[0]     # 1 bit (implicit leading 1 for normals)

    a_sign = a_elem[3]
    a_exp  = a_elem[1:3]
    a_mant = a_elem[0]

    # Result sign = XOR of input signs
    res_sign = w_sign ^ a_sign

    # Exponent addition: w_exp + a_exp + w_scale + a_scale (conceptually)
    # In hardware: add the two E8M0 scale exponents and two element exponents
    # Simplified model: combined exponent = w_exp + a_exp + w_scale + a_scale - bias
    # We model this as an 8-bit adder (dominant hardware cost)
    combined_exp = (
        w_exp.zero_extended(8) + a_exp.zero_extended(8) +
        w_scale.zero_extended(8) + a_scale.zero_extended(8)
    ).truncate(8)

    # Mantissa multiply: 2-bit implicit (1.mant) × 2-bit implicit
    # 1-bit mant → effective 2-bit value: {1, mant}
    w_mantissa_full = pyrtl.concat(pyrtl.Const(1, 1), w_mant)   # 2 bits
    a_mantissa_full = pyrtl.concat(pyrtl.Const(1, 1), a_mant)   # 2 bits
    mant_product = (w_mantissa_full * a_mantissa_full).truncate(4)  # 4 bits

    # Accumulate into partial sum (simplified: add mantissa product scaled by exp)
    # Full FP accumulation requires normalization; we model the adder cost
    psum_out = pyrtl.Output(16, "psum_out")
    psum_out <<= (psum_in + mant_product.zero_extended(16)).truncate(16)

    # Pipeline register
    reg = pyrtl.Register(16, "psum_reg")
    reg.next <<= psum_out

    return {"block": pyrtl.working_block(), "element_bits": 4}


def build_mxfp8_pe() -> dict:
    """Build one MXFP8 E4M3 MAC PE.

    Element: 1 sign + 4 exponent + 3 mantissa = 8 bits.
    Exponent alignment: up to 16-step shift (max_exp=15, min_exp=0).
    Mantissa adder: 8-bit (4-bit mantissa + guard bits).
    """
    pyrtl.reset_working_block()

    w_elem  = pyrtl.Input(8, "w_elem")
    a_elem  = pyrtl.Input(8, "a_elem")
    w_scale = pyrtl.Input(8, "w_scale")
    a_scale = pyrtl.Input(8, "a_scale")
    psum_in = pyrtl.Input(32, "psum_in")

    # Decompose E4M3
    w_sign = w_elem[7]
    w_exp  = w_elem[3:7]   # 4 bits
    w_mant = w_elem[0:3]   # 3 bits

    a_sign = a_elem[7]
    a_exp  = a_elem[3:7]
    a_mant = a_elem[0:3]

    res_sign = w_sign ^ a_sign

    # Exponent: combined_exp = w_exp + a_exp + w_scale + a_scale (8-bit arithmetic)
    combined_exp = (
        w_exp.zero_extended(8) + a_exp.zero_extended(8) +
        w_scale.zero_extended(8) + a_scale.zero_extended(8)
    ).truncate(8)

    # Exponent alignment difference (for accumulation)
    exp_diff = (combined_exp - psum_in[24:32]).truncate(5)   # 5-bit shift amount (0..15)

    # Mantissa multiply: 4-bit × 4-bit (implicit leading 1)
    w_mantissa_full = pyrtl.concat(pyrtl.Const(1, 1), w_mant)   # 4 bits
    a_mantissa_full = pyrtl.concat(pyrtl.Const(1, 1), a_mant)   # 4 bits
    mant_product = (w_mantissa_full * a_mantissa_full).truncate(8)

    # Barrel shifter: shift mant_product by exp_diff to align with accumulator
    # Model: 5-stage mux tree for 5-bit shift amount (dominant area cost)
    shifted = mant_product.zero_extended(32) >> exp_diff.zero_extended(5)

    # Accumulate
    psum_out = pyrtl.Output(32, "psum_out")
    psum_out <<= (psum_in + shifted).truncate(32)

    reg = pyrtl.Register(32, "psum_reg")
    reg.next <<= psum_out

    return {"block": pyrtl.working_block(), "element_bits": 8}


def build_scale_broadcast_unit(block_size: int = 32) -> dict:
    """Model the Block Scale broadcast unit.

    For a block of `block_size` elements sharing one E8M0 scale:
    - Read 8-bit scale from memory.
    - Broadcast to all PEs in the block.
    - Hardware: 8-bit register + fan-out buffer.
    """
    pyrtl.reset_working_block()

    scale_in = pyrtl.Input(8, "scale_in")
    # Fan-out to block_size destinations: modeled as a register with high fan-out
    scale_reg = pyrtl.Register(8, "scale_broadcast")
    scale_reg.next <<= scale_in
    scale_out = pyrtl.Output(8 * block_size, "scale_out_fanout")
    # Concatenate scale to all outputs (fan-out)
    fanout = scale_reg
    for _ in range(block_size - 1):
        fanout = pyrtl.concat(fanout, scale_reg)
    scale_out <<= fanout.truncate(8 * block_size)

    return {"block": pyrtl.working_block(), "block_size": block_size}


def get_mxfp_array_ppa(element_bits: int, rows: int = 16, cols: int = 16) -> dict:
    """Build MXFP array and return full PPA including scale broadcast overhead."""
    try:
        if element_bits == 4:
            pe_info = build_mxfp4_pe()
        else:
            pe_info = build_mxfp8_pe()

        block = pe_info["block"]
        pe_area = pyrtl.area_estimation(tech_in_nm=45, block=block)
        pe_timing = pyrtl.timing_estimation(block=block)

        # Scale broadcast unit
        pyrtl.reset_working_block()
        sb_info = build_scale_broadcast_unit(block_size=32)
        sb_area = pyrtl.area_estimation(tech_in_nm=45, block=sb_info["block"])
        sb_timing = pyrtl.timing_estimation(block=sb_info["block"])

        n_pes = rows * cols
        n_scale_units = rows  # one per row (block spans a row of 32 elements)

        total_area = pe_area * n_pes + sb_area * n_scale_units
        # Critical path: max(PE timing, scale broadcast → PE input)
        total_timing = max(pe_timing, sb_timing + pe_timing * 0.3)

        return {
            "format": f"MXFP{element_bits}",
            "element_bits": element_bits,
            "array_size": f"{rows}×{cols}",
            "n_pes": n_pes,
            "pe_area_mm2": float(pe_area),
            "scale_broadcast_area_mm2": float(sb_area * n_scale_units),
            "area_mm2_total": float(total_area),
            "critical_path_ps": float(total_timing),
            "max_freq_ghz": 1000.0 / float(total_timing) if total_timing > 0 else 0,
            "method": "pyrtl",
        }
    except Exception as e:
        return _analytical_mxfp_ppa(element_bits, rows, cols, error=str(e))


def _analytical_mxfp_ppa(
    element_bits: int, rows: int, cols: int, error: str = ""
) -> dict:
    """Analytical NAND2-equivalent model for MXFP array.

    Extra costs vs. INT array:
      - Exponent alignment barrel shifter: ~6 × shift_bits NAND2-equiv gates per stage
      - Scale broadcast register + fan-out: ~8 + 2×block_size NAND2-equiv gates
      - Exponent adder: additional adder for scale + element exp

    Reference: analogous to FP32 FMA estimates scaled to reduced precision.
    """
    NAND2_AREA_UM2 = 0.614
    GATE_DELAY_PS = 40.0

    if element_bits == 4:
        exp_bits, mant_bits = 2, 1
        accum_bits = 16
        shift_bits = 4   # max shift = 15 (4-bit shift amount)
    else:  # 8-bit E4M3
        exp_bits, mant_bits = 4, 3
        accum_bits = 32
        shift_bits = 5   # max shift 0..15 → 4-bit shift amount but model 5 stages

    # PE components
    mant_mul_gates = 3.5 * (mant_bits + 1) ** 2     # implicit 1-bit added
    exp_add_gates  = 1.5 * 8 * 2                      # two 8-bit adds for scale+elem
    barrel_shifter = 6.0 * shift_bits * (mant_bits + 4)  # mux tree
    accum_add_gates = 1.5 * accum_bits
    reg_gates = 8.0 * accum_bits                      # pipeline register

    pe_gates = mant_mul_gates + exp_add_gates + barrel_shifter + accum_add_gates + reg_gates

    # Scale broadcast unit (per row)
    n_scale_units = rows
    scale_bcast_gates = 8 + 2 * 32   # 8-bit reg + fan-out muxes for 32 elements
    scale_total_gates = scale_bcast_gates * n_scale_units

    n_pes = rows * cols
    total_gates = pe_gates * n_pes + scale_total_gates
    total_area_mm2 = total_gates * NAND2_AREA_UM2 * 1e-6

    # Critical path: exp_add → barrel_shifter → mantissa_add → pipeline_reg
    stages = (
        2 +           # exponent addition stages
        shift_bits +  # barrel shifter mux chain
        2             # accumulator add + register
    )
    crit_path_ps = stages * GATE_DELAY_PS

    # MXFP overhead vs INT (extra gates from exponent logic + barrel shifter)
    int_ppa_approx = _int_pe_gates(element_bits, accum_bits)
    overhead_ratio = pe_gates / max(int_ppa_approx, 1.0)

    return {
        "format": f"MXFP{element_bits}",
        "element_bits": element_bits,
        "array_size": f"{rows}×{cols}",
        "n_pes": n_pes,
        "pe_nand2_equiv": float(pe_gates),
        "scale_broadcast_nand2": float(scale_total_gates),
        "total_nand2_equiv": float(total_gates),
        "area_mm2_total": float(total_area_mm2),
        "critical_path_ps": float(crit_path_ps),
        "max_freq_ghz": 1000.0 / crit_path_ps,
        "overhead_vs_int": float(overhead_ratio),
        "method": "analytical_nand2",
        "pyrtl_error": error,
    }


def _int_pe_gates(bits: int, accum_bits: int) -> float:
    """Helper: INT PE gate count for overhead comparison."""
    return 3.5 * bits * bits + 1.5 * accum_bits + 8.0 * accum_bits

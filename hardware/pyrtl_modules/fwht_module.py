"""Fast Walsh-Hadamard Transform (FWHT) hardware module in PyRTL.

Models the butterfly network for 1D FWHT of N elements.
Each butterfly stage computes N/2 pairs: (a+b, a-b).
Total stages = log2(N).

Key insight for area amortization:
  - FWHT module is shared across all rows processed by the MAC array.
  - Its area is added once, then amortized over all tokens.
  - Each butterfly operation requires only ADD and SUB — no multipliers.
  - Compared to the barrel shifter in MXFP, FWHT is significantly cheaper.

Hardware model:
  - Input width: N × element_bits (parallel input, all elements at once).
  - Each butterfly: 1 adder + 1 subtractor.
  - Total butterflies per stage: N/2.
  - Pipeline: 1 register stage per butterfly stage → log2(N) pipeline stages.
"""

import pyrtl
import math


def build_fwht_butterfly_stage(n: int, bits: int, stage: int) -> dict:
    """Model one butterfly stage of FWHT for N elements.

    In stage k, butterflies connect elements at distance 2^k.
    N/2 butterfly units, each doing: (a+b, a-b).

    Returns area/timing for one stage (to be summed over all stages).
    """
    pyrtl.reset_working_block()

    n_butterflies = n // 2
    accum_bits = bits + int(math.ceil(math.log2(n))) + 1  # prevent overflow

    inputs_a = [pyrtl.Input(bits, f"in_a_{i}") for i in range(n_butterflies)]
    inputs_b = [pyrtl.Input(bits, f"in_b_{i}") for i in range(n_butterflies)]

    outputs_sum = []
    outputs_diff = []

    for i in range(n_butterflies):
        a_ext = inputs_a[i].sign_extended(accum_bits)
        b_ext = inputs_b[i].sign_extended(accum_bits)
        s = (a_ext + b_ext).truncate(accum_bits)
        d = (a_ext - b_ext).truncate(accum_bits)
        out_s = pyrtl.Output(accum_bits, f"out_sum_{i}")
        out_d = pyrtl.Output(accum_bits, f"out_diff_{i}")
        out_s <<= s
        out_d <<= d
        outputs_sum.append(out_s)
        outputs_diff.append(out_d)

    # Pipeline register for this stage
    regs = [pyrtl.Register(accum_bits, f"reg_{i}") for i in range(n_butterflies * 2)]
    for idx, reg in enumerate(regs[:n_butterflies]):
        reg.next <<= outputs_sum[idx]
    for idx, reg in enumerate(regs[n_butterflies:]):
        reg.next <<= outputs_diff[idx]

    return {"block": pyrtl.working_block(), "n_butterflies": n_butterflies}


def get_fwht_ppa(n: int, bits: int = 8) -> dict:
    """Compute full FWHT hardware PPA for N-point transform.

    Parameters
    ----------
    n : int
        Transform size (must be power of 2).
    bits : int
        Input element bit-width.
    """
    n_stages = int(math.log2(n))
    n_butterflies_per_stage = n // 2
    accum_bits = bits + n_stages + 1

    try:
        # Build one stage for PyRTL estimation
        stage_info = build_fwht_butterfly_stage(min(n, 16), bits, 0)
        block = stage_info["block"]
        stage_area = pyrtl.area_estimation(tech_in_nm=45, block=block)
        stage_timing = pyrtl.timing_estimation(block=block)

        # Scale to actual N (linear in N/2 butterflies per stage)
        scale = n_butterflies_per_stage / stage_info["n_butterflies"]
        total_stage_area = stage_area * scale
        total_area = total_stage_area * n_stages
        # Critical path: n_stages pipeline stages
        critical_path = stage_timing * n_stages

        return {
            "module": "FWHT",
            "n": n,
            "bits": bits,
            "n_stages": n_stages,
            "n_butterflies_total": n_stages * n_butterflies_per_stage,
            "area_mm2_per_stage": float(total_stage_area),
            "area_mm2_total": float(total_area),
            "critical_path_ps": float(critical_path),
            "max_freq_ghz": 1000.0 / float(critical_path) if critical_path > 0 else 0,
            "method": "pyrtl_scaled",
        }
    except Exception as e:
        return _analytical_fwht_ppa(n, bits, error=str(e))


def _analytical_fwht_ppa(n: int, bits: int, error: str = "") -> dict:
    """Analytical NAND2-equivalent model for FWHT.

    One N/2-butterfly stage:
      - N/2 adders (bit-width = bits + log2(N) + 1 for overflow)
      - N/2 subtractors (same bit-width)
      - N pipeline registers (accum_bits each)

    Adder gate count ≈ 1.5 × adder_bits NAND2-equiv
    Register: 8 × bits NAND2-equiv
    """
    NAND2_AREA_UM2 = 0.614
    GATE_DELAY_PS = 40.0

    n_stages = int(math.log2(n))
    n_butterflies = n // 2
    accum_bits = bits + n_stages + 1

    # Per stage: N/2 adders + N/2 subtractors + N registers
    per_stage_gates = (
        n_butterflies * 1.5 * accum_bits +    # adders
        n_butterflies * 1.5 * accum_bits +    # subtractors
        n * 8 * accum_bits                    # registers
    )
    total_gates = per_stage_gates * n_stages
    total_area_mm2 = total_gates * NAND2_AREA_UM2 * 1e-6

    # Critical path: 1 adder per stage → n_stages stages
    adder_stages = math.ceil(math.log2(accum_bits)) + 1
    crit_path_ps = adder_stages * n_stages * GATE_DELAY_PS

    # Compare to INT MAC array: FWHT should be much smaller
    # (no multipliers, only add/sub)
    return {
        "module": "FWHT",
        "n": n,
        "bits": bits,
        "n_stages": n_stages,
        "n_butterflies_total": n_stages * n_butterflies,
        "total_nand2_equiv": float(total_gates),
        "area_mm2_per_stage": float(per_stage_gates * NAND2_AREA_UM2 * 1e-6),
        "area_mm2_total": float(total_area_mm2),
        "critical_path_ps": float(crit_path_ps),
        "max_freq_ghz": 1000.0 / crit_path_ps if crit_path_ps > 0 else 0,
        "method": "analytical_nand2",
        "pyrtl_error": error,
    }

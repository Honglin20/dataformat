"""Format encode/decode overhead modules in PyRTL.

Every format requires encoding (write to memory) and decoding (read before compute).
This module models the decode latency and area cost of each format's
encode/decode logic, which is critical for accurate critical-path analysis.

Formats modeled:
  - NF4: LUT lookup (16-entry table, 4-bit address → FP32 output)
  - FP6: bit-unpacking (non-standard 6-bit → 8-bit aligned)
  - NVFP4: E2M1 → FP32 (simple 3-case decode)
  - MXFP4/8: E8M0 scale read + element decode
  - SQ-Format: bitmask read + routing (modeled in sq_gather_scatter.py)
  - INT4/8: trivial sign-extend (essentially free, no separate module needed)
"""

import pyrtl
import math


# ── NF4 LUT Decoder ──────────────────────────────────────────────────────────

def build_nf4_lut_decoder() -> dict:
    """4-bit address → 32-bit FP32 output LUT for NF4.

    16-entry table: maps NF4 index to NF4 quantized level (FP32).
    Hardware: ROM implemented as mux tree (4-stage binary mux).
    """
    pyrtl.reset_working_block()

    nf4_idx = pyrtl.Input(4, "nf4_idx")
    fp32_out = pyrtl.Output(32, "fp32_out")

    # NF4 levels encoded as uint32 (FP32 bit patterns)
    # Pre-computed from NF4_LEVELS in config.py
    import struct
    import numpy as np
    from config import NF4_LEVELS
    lut = [int.from_bytes(struct.pack("f", float(v)), "little") for v in NF4_LEVELS]

    # Build mux tree: 4-bit → 16-way mux (2 stages of 4-way mux)
    # First stage: select between groups of 4
    sel_low  = nf4_idx[0:2]   # lower 2 bits
    sel_high = nf4_idx[2:4]   # upper 2 bits

    # 16 constant entries
    entries = [pyrtl.Const(v, 32) for v in lut]

    # Stage 1: 4 × 4-to-1 muxes (lower 2 bits)
    stage1 = []
    for grp in range(4):
        grp_entries = entries[grp * 4: grp * 4 + 4]
        m = pyrtl.select(sel_low[1],
                falsecase=pyrtl.select(sel_low[0],
                              falsecase=grp_entries[0], truecase=grp_entries[1]),
                truecase=pyrtl.select(sel_low[0],
                              falsecase=grp_entries[2], truecase=grp_entries[3]))
        stage1.append(m)

    # Stage 2: 4-to-1 mux (upper 2 bits)
    result = pyrtl.select(sel_high[1],
                falsecase=pyrtl.select(sel_high[0],
                              falsecase=stage1[0], truecase=stage1[1]),
                truecase=pyrtl.select(sel_high[0],
                              falsecase=stage1[2], truecase=stage1[3]))
    fp32_out <<= result

    return {"block": pyrtl.working_block(), "format": "NF4"}


def get_nf4_decoder_ppa() -> dict:
    try:
        info = build_nf4_lut_decoder()
        area = pyrtl.area_estimation(tech_in_nm=45, block=info["block"])
        timing = pyrtl.timing_estimation(block=info["block"])
        return {
            "format": "NF4",
            "module": "LUT_Decoder",
            "area_mm2": float(area),
            "critical_path_ps": float(timing),
            "max_freq_ghz": 1000.0 / float(timing) if timing > 0 else 0,
            "method": "pyrtl",
        }
    except Exception as e:
        return _analytical_lut_decoder_ppa("NF4", addr_bits=4, data_bits=32, error=str(e))


# ── FP6 Bit-Unpacker ──────────────────────────────────────────────────────────

def build_fp6_unpacker() -> dict:
    """FP6 (6-bit) → FP32 decode.

    FP6 E3M2 decode: extract sign/exp/mant bits, compute FP32 value.
    Key cost: bit-packing requires special alignment (3 FP6 elements per 18 bits),
    and decode needs exponent un-biasing + mantissa zero-extension.
    """
    pyrtl.reset_working_block()

    fp6_in = pyrtl.Input(6, "fp6_in")
    fp32_out = pyrtl.Output(32, "fp32_out")

    sign = fp6_in[5]
    exp  = fp6_in[2:5]   # 3 bits, bias=3
    mant = fp6_in[0:2]   # 2 bits

    # Un-bias exponent: FP6_exp(3b) → FP32_exp(8b, bias=127)
    # FP6 bias=3 → FP32 bias=127 → delta=124
    # Result biased exp = fp6_exp - 3 + 127 = fp6_exp + 124
    fp32_exp = (exp.zero_extended(8) + pyrtl.Const(124, 8)).truncate(8)

    # Mantissa: 2 FP6 mant bits → 23 FP32 mant bits (zero-extend lower 21 bits)
    fp32_mant = pyrtl.concat(pyrtl.Const(0, 21), mant)  # 23 bits

    # Assemble FP32
    fp32_out <<= pyrtl.concat(sign, pyrtl.concat(fp32_exp, fp32_mant))

    return {"block": pyrtl.working_block(), "format": "FP6"}


def get_fp6_unpacker_ppa() -> dict:
    try:
        info = build_fp6_unpacker()
        area = pyrtl.area_estimation(tech_in_nm=45, block=info["block"])
        timing = pyrtl.timing_estimation(block=info["block"])
        return {
            "format": "FP6",
            "module": "Bit_Unpacker",
            "area_mm2": float(area),
            "critical_path_ps": float(timing),
            "max_freq_ghz": 1000.0 / float(timing) if timing > 0 else 0,
            "method": "pyrtl",
        }
    except Exception as e:
        return _analytical_fp6_ppa(error=str(e))


# ── NVFP4 E2M1 Decoder ───────────────────────────────────────────────────────

def build_nvfp4_decoder() -> dict:
    """NVFP4 E2M1 → FP32 decode.

    8 positive values: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
    Decode: 3-bit unsigned code (drop sign) → 3-stage comparator tree → FP32.
    Small LUT: 8-entry, 32-bit output.
    """
    pyrtl.reset_working_block()

    nvfp4_in = pyrtl.Input(4, "nvfp4_in")
    fp32_out = pyrtl.Output(32, "fp32_out")

    sign = nvfp4_in[3]
    code = nvfp4_in[0:3]   # 3-bit code → 8 positive levels

    import struct
    pos_levels = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    lut = [int.from_bytes(struct.pack("f", v), "little") for v in pos_levels]
    entries = [pyrtl.Const(v, 32) for v in lut]

    # 3-to-8 decode via mux tree (3 stages of 2-to-1 mux)
    sel0 = code[0]
    sel1 = code[1]
    sel2 = code[2]

    m00 = pyrtl.select(sel0, falsecase=entries[0], truecase=entries[1])
    m01 = pyrtl.select(sel0, falsecase=entries[2], truecase=entries[3])
    m10 = pyrtl.select(sel0, falsecase=entries[4], truecase=entries[5])
    m11 = pyrtl.select(sel0, falsecase=entries[6], truecase=entries[7])

    m0 = pyrtl.select(sel1, falsecase=m00, truecase=m01)
    m1 = pyrtl.select(sel1, falsecase=m10, truecase=m11)
    m  = pyrtl.select(sel2, falsecase=m0, truecase=m1)

    # Apply sign: flip bit 31 if sign==1
    fp32_out <<= pyrtl.concat(m[0:31], sign)

    return {"block": pyrtl.working_block(), "format": "NVFP4"}


def get_nvfp4_decoder_ppa() -> dict:
    try:
        info = build_nvfp4_decoder()
        area = pyrtl.area_estimation(tech_in_nm=45, block=info["block"])
        timing = pyrtl.timing_estimation(block=info["block"])
        return {
            "format": "NVFP4",
            "module": "E2M1_Decoder",
            "area_mm2": float(area),
            "critical_path_ps": float(timing),
            "max_freq_ghz": 1000.0 / float(timing) if timing > 0 else 0,
            "method": "pyrtl",
        }
    except Exception as e:
        return _analytical_lut_decoder_ppa("NVFP4", addr_bits=3, data_bits=32, error=str(e))


# ── MXFP Scale Read Unit ─────────────────────────────────────────────────────

def get_mxfp_scale_read_ppa(element_bits: int, block_size: int = 32) -> dict:
    """PPA of reading and applying the E8M0 block scale.

    The scale read introduces extra SRAM access per block_size elements.
    Memory bandwidth overhead: 8 bits per 32 × element_bits = 8/(32×eb) overhead.
    """
    extra_bits_per_element = 8 / block_size   # 0.25 bits/element for block=32
    bw_amplification = (element_bits + extra_bits_per_element) / element_bits

    # Scale multiply unit: 8-bit exponent + → convert to shift amount → barrel shifter
    # This is already modeled in mxfp_mac_array.py; here we just report the overhead
    return {
        "format": f"MXFP{element_bits}",
        "module": "Scale_Read",
        "extra_bits_per_element": extra_bits_per_element,
        "bandwidth_amplification": bw_amplification,
        "scale_sram_read_pj": 1.56 * 8 / 8,   # 8-bit SRAM read, amortized over block
        "note": "Timing modeled in mxfp_mac_array.py scale broadcast unit",
    }


# ── Analytical fallbacks ─────────────────────────────────────────────────────

def _analytical_lut_decoder_ppa(
    fmt: str, addr_bits: int, data_bits: int, error: str = ""
) -> dict:
    NAND2_AREA_UM2 = 0.614
    GATE_DELAY_PS = 40.0
    n_entries = 2 ** addr_bits
    # Mux tree: (n_entries - 1) × data_bits / 2 NAND2-equiv
    gates = (n_entries - 1) * data_bits
    stages = addr_bits  # log2(n_entries)
    return {
        "format": fmt,
        "module": "LUT_Decoder",
        "total_nand2_equiv": float(gates),
        "area_mm2": float(gates * NAND2_AREA_UM2 * 1e-6),
        "critical_path_ps": float(stages * GATE_DELAY_PS),
        "max_freq_ghz": 1000.0 / (stages * GATE_DELAY_PS),
        "method": "analytical_nand2",
        "pyrtl_error": error,
    }


def _analytical_fp6_ppa(error: str = "") -> dict:
    NAND2_AREA_UM2 = 0.614
    GATE_DELAY_PS = 40.0
    # Adder (8-bit exp add) + mux (sign) + concat (zero-extend)
    gates = 1.5 * 8 + 1.5 + 23   # exp adder + mux + mant zero-extend
    stages = 4   # adder chain
    return {
        "format": "FP6",
        "module": "Bit_Unpacker",
        "total_nand2_equiv": float(gates),
        "area_mm2": float(gates * NAND2_AREA_UM2 * 1e-6),
        "critical_path_ps": float(stages * GATE_DELAY_PS),
        "max_freq_ghz": 1000.0 / (stages * GATE_DELAY_PS),
        "method": "analytical_nand2",
        "pyrtl_error": error,
    }


def get_all_converter_ppas() -> list:
    """Return PPA estimates for all format decoders."""
    return [
        get_nf4_decoder_ppa(),
        get_fp6_unpacker_ppa(),
        get_nvfp4_decoder_ppa(),
        get_mxfp_scale_read_ppa(element_bits=4),
        get_mxfp_scale_read_ppa(element_bits=8),
        # INT4/8: sign-extend only (trivial, ~bits NAND2 gates)
        {
            "format": "INT4", "module": "SignExtend",
            "area_mm2": 4 * 0.614e-6, "critical_path_ps": 40.0,
        },
        {
            "format": "INT8", "module": "SignExtend",
            "area_mm2": 8 * 0.614e-6, "critical_path_ps": 40.0,
        },
    ]

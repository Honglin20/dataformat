"""Global configuration and format registry."""

# ── Quantization bit-widths under test ────────────────────────────────────────
BITWIDTHS = [4, 8]

# ── Block size for MX formats (OCP standard = 32) ─────────────────────────────
MX_BLOCK_SIZE = 32

# ── NF4 quantization levels (from QLoRA paper) ───────────────────────────────
import numpy as np

NF4_LEVELS = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
     0.07958029955625534,  0.16093020141124725,  0.24611230194568634,
     0.33791524171829224,  0.44070982933044434,  0.5626170039176941,
     0.7229568362236023,   1.0,
], dtype=np.float32)

# ── FP6 E3M2 lookup table (representative quantization levels) ────────────────
# E3M2: 3-bit exponent, 2-bit mantissa, 1 sign bit
# Positive values: exponent bias = 3
FP6_E3M2_MAX = 28.0   # max representable value
FP6_E3M2_MIN_NORMAL = 0.25  # min normal value (exp=001, mant=00)

# ── Systolic array dimensions for PyRTL hardware models ──────────────────────
ARRAY_ROWS = 16
ARRAY_COLS = 16

# ── Horowitz 45nm energy constants (pJ per operation) ────────────────────────
# Source: Horowitz, ISSCC 2014 "Computing's Energy Problem"
ENERGY = {
    "int8_add":      0.03,   # pJ
    "int8_mul":      0.20,   # pJ
    "int4_add":      0.01,   # pJ
    "int4_mul":      0.05,   # pJ
    "fp8_add":       0.40,   # pJ
    "fp8_mul":       0.50,   # pJ
    "fp16_add":      0.90,   # pJ
    "fp16_mul":      1.10,   # pJ
    "fp32_add":      3.70,   # pJ
    "fp32_mul":      4.60,   # pJ
    "sram_read_8b":  1.56,   # pJ per 8-bit read (256KB SRAM, 45nm)
    "sram_read_32b": 6.25,   # pJ per 32-bit read
    "dram_read_8b": 40.00,   # pJ per 8-bit DRAM read
    "lut_access":    2.00,   # pJ per LUT lookup (16-entry, 8-bit output)
    "mux_1bit":      0.005,  # pJ per 1-bit mux
    "shift_1b":      0.01,   # pJ per 1-bit shift
}

# ── Roofline model parameters (example accelerator) ──────────────────────────
PEAK_COMPUTE_TOPS = {
    "int4":  256,   # TOPS (tera operations per second)
    "int8":  128,
    "fp8":   64,
    "fp16":  32,
    "fp32":  8,
}
PEAK_BW_TB_S = 4.0   # TB/s memory bandwidth (HBM3-class)

# ── Distribution test configurations ─────────────────────────────────────────
N_SAMPLES = 4096       # tensor size for distribution tests
RANDOM_SEED = 42

# ── Experiment output directory ───────────────────────────────────────────────
OUTPUT_DIR = "results"

# ── Format registry: maps name -> (module, class) for dynamic loading ─────────
FORMAT_NAMES = [
    "FP32", "BF16",
    "NVFP4",
    "MXFP4", "MXFP8",
    "MXINT4", "MXINT8",
    "NF4", "FP6",
    "SmoothQuant+INT4", "SmoothQuant+INT8",
    "HAD+INT4", "HAD+INT8",
    "HAD+LUT",
    "HAD+SQ",
    "RandRot+INT4", "RandRot+INT8",
    "TurboQuant+INT4", "TurboQuant+INT8",
]

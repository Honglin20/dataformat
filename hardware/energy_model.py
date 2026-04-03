"""Horowitz 45nm Energy Model.

Reference: M. Horowitz, "Computing's Energy Problem (and what we can do about it)",
           ISSCC 2014.

Provides per-operation energy estimates (pJ) for:
  - Integer arithmetic (ADD, MUL) at various bit-widths.
  - Floating-point arithmetic at various precisions.
  - Memory access (SRAM, DRAM) at various data widths.
  - Special operations: LUT access, MUX, shift, XOR.

Usage:
    from hardware.energy_model import EnergyModel
    em = EnergyModel()
    energy = em.mac_energy("INT4", n_macs=256)
"""

from config import ENERGY


class EnergyModel:
    """Horowitz 45nm energy estimation for MAC operations.

    All energies in picojoules (pJ).
    """

    def __init__(self, tech_nm: int = 45):
        self.tech_nm = tech_nm
        # Technology scaling: energy scales roughly as (tech_nm / 45)^2
        self._scale = (tech_nm / 45.0) ** 2
        self._e = {k: v * self._scale for k, v in ENERGY.items()}

    # ── Per-operation energy ──────────────────────────────────────────────────

    def add_energy(self, bits: int, is_float: bool = False) -> float:
        """Energy for one ADD/SUB operation."""
        if is_float:
            if bits <= 8:
                return self._e["fp8_add"]
            elif bits <= 16:
                return self._e["fp16_add"]
            else:
                return self._e["fp32_add"]
        else:
            if bits <= 4:
                return self._e["int4_add"]
            else:
                return self._e["int8_add"]

    def mul_energy(self, bits: int, is_float: bool = False) -> float:
        """Energy for one MUL operation."""
        if is_float:
            if bits <= 8:
                return self._e["fp8_mul"]
            elif bits <= 16:
                return self._e["fp16_mul"]
            else:
                return self._e["fp32_mul"]
        else:
            if bits <= 4:
                return self._e["int4_mul"]
            else:
                return self._e["int8_mul"]

    def mac_energy(self, format_name: str, n_macs: int = 1) -> float:
        """Total energy for n MAC (multiply-accumulate) operations in given format."""
        fmt = format_name.upper()
        if "INT4" in fmt:
            e = self._e["int4_mul"] + self._e["int4_add"]
        elif "INT8" in fmt:
            e = self._e["int8_mul"] + self._e["int8_add"]
        elif "MXFP4" in fmt or "NVFP4" in fmt or "NF4" in fmt:
            # FP4: modeled as FP8-ish (exponent logic adds overhead)
            e = self._e["fp8_mul"] + self._e["fp8_add"]
        elif "MXFP8" in fmt or "FP8" in fmt:
            e = self._e["fp8_mul"] + self._e["fp8_add"]
        elif "FP6" in fmt:
            # FP6: between FP8 and INT8
            e = (self._e["fp8_mul"] + self._e["int8_mul"]) / 2
        elif "FP16" in fmt or "BF16" in fmt:
            e = self._e["fp16_mul"] + self._e["fp16_add"]
        elif "FP32" in fmt:
            e = self._e["fp32_mul"] + self._e["fp32_add"]
        else:
            e = self._e["int8_mul"] + self._e["int8_add"]  # default
        return e * n_macs

    # ── Memory access energy ──────────────────────────────────────────────────

    def sram_read_energy(self, bits: int, n_reads: int = 1) -> float:
        """Energy to read `bits` from SRAM."""
        # Linear scaling by bit-width (256KB SRAM reference)
        base = self._e["sram_read_8b"]
        return base * (bits / 8.0) * n_reads

    def dram_read_energy(self, bits: int, n_reads: int = 1) -> float:
        """Energy to read `bits` from DRAM."""
        base = self._e["dram_read_8b"]
        return base * (bits / 8.0) * n_reads

    def lut_energy(self, n_lookups: int = 1) -> float:
        """Energy for 16-entry LUT access (NF4/NVFP4 decode)."""
        return self._e["lut_access"] * n_lookups

    # ── Format-specific overhead energy ──────────────────────────────────────

    def format_overhead_energy(self, format_name: str, n_elements: int) -> dict:
        """Compute additional decode/encode energy beyond bare MAC operations.

        Returns dict with itemized energy costs (pJ).
        """
        fmt = format_name.upper()
        overhead = {}

        if "MXFP" in fmt or "MXINT" in fmt:
            # E8M0 scale read: 1 SRAM read per 32 elements (8-bit scale)
            n_scale_reads = max(1, n_elements // 32)
            overhead["scale_sram_read_pJ"] = self.sram_read_energy(8, n_scale_reads)
            if "MXFP" in fmt:
                # Exponent alignment shift: modeled as 1 shift op per element
                overhead["exponent_align_pJ"] = (
                    self._e["shift_1b"] * 4 * n_elements  # 4-bit shift
                )

        elif "NF4" in fmt or "NVFP4" in fmt:
            # LUT decode per element
            overhead["lut_decode_pJ"] = self.lut_energy(n_elements)

        elif "FP6" in fmt:
            # Bit unpack: 2 shift ops per element (6-bit alignment)
            overhead["bit_unpack_pJ"] = self._e["shift_1b"] * 2 * n_elements

        elif "SQ" in fmt:
            # Gather/scatter: 1 mux + 1 mask read per element
            overhead["gather_mux_pJ"] = self._e["mux_1bit"] * 8 * n_elements
            overhead["mask_read_pJ"] = self.sram_read_energy(1, n_elements)

        elif "HAD" in fmt:
            # FWHT: N log2(N) add/sub operations
            import math
            n_ops = n_elements * math.log2(max(n_elements, 2))
            bits = 8 if "INT8" in fmt else 4
            overhead["fwht_pJ"] = self.add_energy(bits) * n_ops

        elif "SMOOTHQUANT" in fmt:
            # Per-channel scale multiply
            overhead["scale_mul_pJ"] = self.mul_energy(8, False) * n_elements

        elif "RANDROT" in fmt:
            # Dense matrix-vector multiply (N² ops)
            bits = 8 if "INT8" in fmt else 4
            overhead["rotation_mac_pJ"] = (
                self.mul_energy(bits) + self.add_energy(bits)
            ) * n_elements * n_elements

        elif "TURBOQUANT" in fmt:
            # Diagonal sign flip: XOR per element
            overhead["sign_flip_pJ"] = self._e["mux_1bit"] * n_elements

        overhead["total_overhead_pJ"] = sum(overhead.values())
        return overhead

    # ── Total system energy per inference ────────────────────────────────────

    def total_inference_energy(
        self,
        format_name: str,
        n_macs: int,
        n_weight_reads: int,
        n_activation_reads: int,
        memory_type: str = "sram",
    ) -> dict:
        """Compute total energy for one linear layer inference.

        Parameters
        ----------
        format_name : str
        n_macs : int
            Total MAC operations (e.g., M × K × N for a MxK @ KxN matmul).
        n_weight_reads : int
            Total weight elements read from memory.
        n_activation_reads : int
            Total activation elements read.
        memory_type : str
            'sram' or 'dram'.
        """
        # Determine effective bit-width for memory reads
        bits_map = {
            "FP32": 32, "BF16": 16, "FP16": 16,
            "MXFP8": 8, "MXFP4": 4, "MXINT8": 8, "MXINT4": 4,
            "INT8": 8, "INT4": 4, "NF4": 4, "FP6": 6, "NVFP4": 4,
        }
        fmt_upper = format_name.upper()
        data_bits = 8   # default
        for k, v in bits_map.items():
            if k in fmt_upper:
                data_bits = v
                break

        compute_pJ = self.mac_energy(format_name, n_macs)

        if memory_type == "dram":
            mem_pJ = (
                self.dram_read_energy(data_bits, n_weight_reads) +
                self.dram_read_energy(data_bits, n_activation_reads)
            )
        else:
            mem_pJ = (
                self.sram_read_energy(data_bits, n_weight_reads) +
                self.sram_read_energy(data_bits, n_activation_reads)
            )

        overhead_pJ = self.format_overhead_energy(
            format_name, n_weight_reads
        ).get("total_overhead_pJ", 0)

        return {
            "format": format_name,
            "compute_pJ": float(compute_pJ),
            "memory_pJ": float(mem_pJ),
            "overhead_pJ": float(overhead_pJ),
            "total_pJ": float(compute_pJ + mem_pJ + overhead_pJ),
            "compute_fraction": compute_pJ / max(compute_pJ + mem_pJ + overhead_pJ, 1e-9),
        }

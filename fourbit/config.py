"""User-editable config for the 4-bit study.

To add a new format: append an entry to ``FourBitConfig.formats`` referencing
a factory name registered in :mod:`fourbit.formats.FORMAT_FACTORIES`.

To add a new transform: append to ``FourBitConfig.transforms`` with a
factory key from :mod:`fourbit.transforms.TRANSFORM_FACTORIES`.

No other file needs to change – ``Pipeline`` iterates over the config at
run time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class FormatSpec:
    """Declarative format entry."""
    display_name: str              # e.g. "INT4"
    factory: str                   # key into FORMAT_FACTORIES
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransformSpec:
    """Declarative transform entry."""
    display_name: str              # e.g. "base", "smooth", "had"
    factory: str                   # key into TRANSFORM_FACTORIES
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FourBitConfig:
    """Top-level experiment config.

    ``formats`` and ``transforms`` determine the Cartesian product of
    Pipelines that every experiment sweeps over.  The study's fixed scope
    is 4-bit INT / FP / NF / NV / MX families, but the config lets you
    evaluate any subset (or add new formats) without touching experiment
    code.
    """
    formats: List[FormatSpec] = field(default_factory=list)
    transforms: List[TransformSpec] = field(default_factory=list)

    # Sweep parameters
    n_samples: int = 4096          # Part 1.1 tensor size
    batch_size: int = 128          # Part 1.2/1.3 batch
    in_features: int = 256         # power of 2 for HAD
    out_features: int = 128
    seed: int = 42

    # Part 2 (real-model) profiling
    profile_samples: int = 256
    smooth_alpha: float = 0.5

    # Output
    output_dir: str = "results/fourbit"


# ── Default configuration (matches the study specification) ──────────────────

DEFAULT_CONFIG = FourBitConfig(
    formats=[
        # Per-channel, power-of-two scale — cheapest INT decode.
        FormatSpec("INT4",    "int4_per_channel"),
        # Per-channel, full-FP scale — matches NF4's scale cost.
        FormatSpec("INT4_FP", "int4_fp_per_channel"),
        # Per-channel FP4 (E2M1 levels, POT scale).
        FormatSpec("FP4",     "fp4_per_channel"),
        # QLoRA NF4 per-channel.  FP scale (not HW-optimal).
        FormatSpec("NF4",     "nf4_per_channel"),
        # HW-realistic NF4 with FP8-E4M3 scale (QLoRA double-quant).
        FormatSpec("NF4_FP8", "nf4_fp8_per_channel"),
        # Additive Power-of-Two (Li 2020) per-channel.
        FormatSpec("APoT4",   "apot4_per_channel"),
        # Logarithmic 4-bit, shift-only decode.
        FormatSpec("LOG4",    "log4_per_channel"),
        # Per-tensor Blackwell NVFP4.
        FormatSpec("NVFP4",   "nvfp4"),
        # Block-scaled OCP-MX variants.
        FormatSpec("MXINT4",  "mxint4", kwargs={"block_size": 32}),
        FormatSpec("MXFP4",   "mxfp4",  kwargs={"block_size": 32}),
    ],
    transforms=[
        TransformSpec("base",   "identity"),
        TransformSpec("smooth", "smoothquant", kwargs={"alpha": 0.5}),
        TransformSpec("had",    "hadamard"),
    ],
)

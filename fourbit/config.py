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
        FormatSpec("INT4",   "int4_per_channel"),
        FormatSpec("FP4",    "fp4_per_channel"),
        FormatSpec("NF4",    "nf4_per_channel"),
        FormatSpec("NVFP4",  "nvfp4"),
        FormatSpec("MXINT4", "mxint4", kwargs={"block_size": 32}),
        FormatSpec("MXFP4",  "mxfp4",  kwargs={"block_size": 32}),
    ],
    transforms=[
        TransformSpec("base",   "identity"),
        TransformSpec("smooth", "smoothquant", kwargs={"alpha": 0.5}),
        TransformSpec("had",    "hadamard"),
    ],
)

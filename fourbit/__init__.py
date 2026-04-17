"""4-bit data format study package.

Provides a config-driven infrastructure for comparing 4-bit quantization
formats (INT4, FP4, NF4, NVFP4, MXINT4, MXFP4) under three transforms:

  - base   : direct scaling + quantization
  - smooth : SmoothQuant per-channel scaling
  - had    : Hadamard (Walsh-Hadamard) rotation

Public entry points:

  fourbit.registry.build_formats(config)   -> dict[str, QuantFormat]
  fourbit.registry.build_transforms(config)-> dict[str, Transform]
  fourbit.pipeline.Pipeline                -> one (transform, format) pair
  fourbit.part1.run                        -> run all Part 1 experiments
  fourbit.part2.run                        -> run Part 2 (model profiling)
"""
from fourbit.config import FourBitConfig, DEFAULT_CONFIG
from fourbit.pipeline import Pipeline
from fourbit.registry import build_formats, build_transforms

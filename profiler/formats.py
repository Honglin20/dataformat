"""Format schedule for the model profiler.

Builds the ordered list of 14 quantization formats to profile.
All format objects are reused from formats/ — no new implementations.
"""
from __future__ import annotations

import numpy as np

from formats.baseline import FP32Format, BF16Format, FP16Format
from formats.mxint import MXINTFormat
from formats.sq_format import SQFormat, SQFormatActivations
from formats.transforms.hadamard import HADTransform
from formats import _POTINTQuantizer, ComposedFormat

PROFILER_FORMAT_NAMES: list[str] = [
    "FP32",
    "FP16",
    "SQ-FORMAT-INT",
    "SQ-FORMAT-FP",
    "INT4(CHANNEL)",
    "INT8(CHANNEL)",
    "INT4(TENSOR)",
    "INT8(TENSOR)",
    "HAD+INT4(C)",
    "HAD+INT8(C)",
    "HAD+INT4(T)",
    "HAD+INT8(T)",
    "MXINT4",
    "MXINT8",
]


def build_profiler_formats() -> list[tuple[str, object]]:
    """Return ordered list of (name, format_object) for the profiler.

    HADTransform uses normalize=False (hardware model: no sqrt(N) division,
    scale absorbed by quantizer). Same setting as the main format registry.
    """
    had = HADTransform(normalize=False)

    formats = [
        ("FP32",          FP32Format()),
        ("FP16",          FP16Format()),
        ("SQ-FORMAT-INT", SQFormat(dense_bits=4, sparse_bits=8, sparsity_ratio=0.01)),
        ("SQ-FORMAT-FP",  SQFormatActivations(bank_size=128, sparsity=0.5,
                                               high_bits=8, low_bits=4)),
        ("INT4(CHANNEL)", _POTINTQuantizer(4, per_channel=True)),
        ("INT8(CHANNEL)", _POTINTQuantizer(8, per_channel=True)),
        ("INT4(TENSOR)",  _POTINTQuantizer(4, per_channel=False)),
        ("INT8(TENSOR)",  _POTINTQuantizer(8, per_channel=False)),
        ("HAD+INT4(C)",   ComposedFormat("HAD+INT4(C)", had, _POTINTQuantizer(4, per_channel=True),  4)),
        ("HAD+INT8(C)",   ComposedFormat("HAD+INT8(C)", had, _POTINTQuantizer(8, per_channel=True),  8)),
        ("HAD+INT4(T)",   ComposedFormat("HAD+INT4(T)", had, _POTINTQuantizer(4, per_channel=False), 4)),
        ("HAD+INT8(T)",   ComposedFormat("HAD+INT8(T)", had, _POTINTQuantizer(8, per_channel=False), 8)),
        ("MXINT4",        MXINTFormat(element_bits=4)),
        ("MXINT8",        MXINTFormat(element_bits=8)),
    ]

    assert [n for n, _ in formats] == PROFILER_FORMAT_NAMES, "Names out of sync"
    return formats

"""Format schedule for the model profiler.

Builds the ordered list of 14 quantization formats to profile.
All format objects are reused from formats/ — no new implementations.

Also provides simulate_linear_output() for end-to-end layer SQNR measurement.
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


def simulate_linear_output(
    fmt_obj,
    W: np.ndarray,
    x: np.ndarray,
    bias: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray]:
    """Simulate a quantized nn.Linear forward pass for end-to-end SQNR measurement.

    For HAD+INT formats the computation is done in the Hadamard domain — the
    mathematically correct hardware model:

        y_quant = Q(H(x)) @ Q(H(W))ᵀ / N

    where H is the un-normalized FWHT (H² = N·I) applied along the last axis of
    each matrix row.  Dividing by N corrects for the double H amplification so
    that y_quant ≈ y_fp32 = x @ Wᵀ.

    For all other formats both weight and activation are quantized independently:

        y_quant = Q(x) @ Q(W)ᵀ + bias

    This models weight + activation quantization as a pair, giving the true
    combined quantization error on the layer output.

    Parameters
    ----------
    fmt_obj : any format object
    W       : weight array, shape (out_features, in_features)
    x       : input array,  shape (..., in_features)
    bias    : optional bias, shape (out_features,)

    Returns
    -------
    y_quant : np.ndarray or None
        Simulated quantized output.  None if simulation fails (e.g. in_features
        is not a power of 2 for HAD formats).
    y_fp32  : np.ndarray
        FP32 reference output  x @ Wᵀ + bias.
    """
    y_fp32 = x @ W.T
    if bias is not None:
        y_fp32 = y_fp32 + bias

    is_had = isinstance(fmt_obj, ComposedFormat) and isinstance(
        fmt_obj._transform, HADTransform
    )

    try:
        if is_had:
            # HAD+INT path — quantize W and x in HAD domain, multiply, correct by 1/N
            N = W.shape[-1]
            W_had = fmt_obj._transform.forward(W)    # (out, in): each row HAD-transformed
            x_had = fmt_obj._transform.forward(x)    # (..., in): each row HAD-transformed
            Q_W = fmt_obj._quantizer.quantize(W_had)
            Q_x = fmt_obj._quantizer.quantize(x_had)
            # H(W) @ H(x)ᵀ = W H Hᵀ xᵀ = N W xᵀ  →  divide by N to recover W xᵀ
            y_quant = (Q_x @ Q_W.T) / float(N)
        else:
            # Standard path — quantize W and x, multiply
            Q_W = fmt_obj.quantize(W)
            Q_x = fmt_obj.quantize(x)
            y_quant = Q_x @ Q_W.T

        if bias is not None:
            y_quant = y_quant + bias

    except AssertionError:
        # HAD requires power-of-2 last dim; skip silently
        return None, y_fp32
    except Exception:
        return None, y_fp32

    return y_quant, y_fp32

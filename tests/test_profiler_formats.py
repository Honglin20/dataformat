import numpy as np
from profiler.formats import build_profiler_formats, PROFILER_FORMAT_NAMES


def test_all_format_names_present():
    expected = [
        "FP32", "FP16",
        "SQ-FORMAT-INT", "SQ-FORMAT-FP",
        "INT4(CHANNEL)", "INT8(CHANNEL)",
        "INT4(TENSOR)", "INT8(TENSOR)",
        "HAD+INT4(C)", "HAD+INT8(C)",
        "HAD+INT4(T)", "HAD+INT8(T)",
        "MXINT4", "MXINT8",
    ]
    assert PROFILER_FORMAT_NAMES == expected


def test_all_formats_have_quantize():
    fmts = build_profiler_formats()
    assert len(fmts) == 14
    for name, fmt in fmts:
        assert hasattr(fmt, "quantize"), f"{name} missing quantize()"


def test_formats_quantize_1d_array():
    fmts = build_profiler_formats()
    x = np.random.randn(256).astype(np.float32)
    for name, fmt in fmts:
        try:
            q = fmt.quantize(x)
            assert q.shape == x.shape, f"{name}: shape mismatch"
        except Exception as e:
            raise AssertionError(f"{name} failed: {e}")


def test_formats_quantize_2d_array():
    fmts = build_profiler_formats()
    x = np.random.randn(32, 64).astype(np.float32)
    for name, fmt in fmts:
        try:
            q = fmt.quantize(x)
            assert q.shape == x.shape, f"{name}: shape mismatch"
        except Exception as e:
            raise AssertionError(f"{name} failed: {e}")

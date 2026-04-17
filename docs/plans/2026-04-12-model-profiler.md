# Model Profiler Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a lightweight PyTorch model profiler that captures per-layer tensor distributions across 14 quantization formats using `start()`/`stop()` flags, with zero intrusion into existing model code.

**Architecture:** Hook-based using `register_forward_hook` / `register_forward_pre_hook` on all leaf `nn.Module` instances. Online statistics (Welford + RunningHistogram) accumulate across batches within a `start`/`stop` window. After `stop()`, the profiler auto-advances to the next format. Results export as CSV aligned with the existing `ExperimentRunner` output.

**Tech Stack:** PyTorch (hooks), NumPy (statistics), existing `formats/` objects (quantization), pandas (CSV export).

> **Note:** `torch` is not in `requirements.txt`. Add it before running tests: `pip install torch --index-url https://download.pytorch.org/whl/cpu`

---

## Task 1: Online Statistics Module

**Files:**
- Create: `profiler/stats.py`
- Test: `tests/test_profiler_stats.py`

### Step 1: Write failing tests

```python
# tests/test_profiler_stats.py
import numpy as np
import pytest
from profiler.stats import WelfordStats, RunningHistogram, QuantStats


class TestWelfordStats:
    def test_single_batch(self):
        s = WelfordStats()
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s.update(x)
        r = s.finalize()
        assert abs(r["mean"] - 3.0) < 1e-9
        assert abs(r["std"] - np.std(x)) < 1e-6
        assert r["abs_max"] == 5.0
        assert r["n_elements"] == 5

    def test_two_batches_equivalent_to_one(self):
        s1 = WelfordStats()
        s1.update(np.array([1.0, 2.0, 3.0]))
        s1.update(np.array([4.0, 5.0]))

        s2 = WelfordStats()
        s2.update(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

        r1, r2 = s1.finalize(), s2.finalize()
        assert abs(r1["mean"] - r2["mean"]) < 1e-9
        assert abs(r1["std"] - r2["std"]) < 1e-6

    def test_empty_raises(self):
        s = WelfordStats()
        with pytest.raises(RuntimeError):
            s.finalize()


class TestRunningHistogram:
    def test_basic(self):
        h = RunningHistogram()
        h.update(np.linspace(-3, 3, 1000))
        r = h.finalize()
        assert len(r["hist_counts"]) == 256
        assert r["outlier_ratio"] == 0.0

    def test_outliers_counted(self):
        h = RunningHistogram()
        h.update(np.linspace(0, 1, 100))   # sets range [0, 1]
        h.update(np.array([-5.0, 2.0]))    # both outside range → outliers
        r = h.finalize()
        assert r["outlier_ratio"] > 0.0

    def test_two_batches_same_total_as_one(self):
        h1 = RunningHistogram()
        h1.update(np.linspace(0, 1, 50))
        h1.update(np.linspace(0, 1, 50))

        h2 = RunningHistogram()
        h2.update(np.linspace(0, 1, 100))

        r1, r2 = h1.finalize(), h2.finalize()
        assert sum(r1["hist_counts"]) == sum(r2["hist_counts"])


class TestQuantStats:
    def test_perfect_reconstruction(self):
        s = QuantStats()
        x = np.array([1.0, 2.0, 3.0])
        s.update(x, x.copy())
        r = s.finalize()
        assert r["mse"] == 0.0
        assert r["max_ae"] == 0.0

    def test_known_mse(self):
        s = QuantStats()
        orig = np.array([0.0, 0.0, 0.0, 0.0])
        quant = np.array([1.0, 1.0, 1.0, 1.0])
        s.update(orig, quant)
        r = s.finalize()
        assert abs(r["mse"] - 1.0) < 1e-9
        assert r["max_ae"] == 1.0

    def test_incremental_equals_batch(self):
        s1 = QuantStats()
        s1.update(np.array([1.0, 2.0]), np.array([1.1, 2.1]))
        s1.update(np.array([3.0, 4.0]), np.array([3.1, 4.1]))

        s2 = QuantStats()
        s2.update(
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([1.1, 2.1, 3.1, 4.1]),
        )
        r1, r2 = s1.finalize(), s2.finalize()
        assert abs(r1["mse"] - r2["mse"]) < 1e-9
```

### Step 2: Run tests to verify they fail

```bash
cd /Users/mozzie/Desktop/Projects/formatresearch/dataformat
pytest tests/test_profiler_stats.py -v
```

Expected: `ModuleNotFoundError: No module named 'profiler'`

### Step 3: Implement profiler/stats.py

```python
# profiler/stats.py
"""Online statistics for per-layer tensor distribution analysis.

Three classes:
  WelfordStats      — incremental mean, std, abs_max across batches
  RunningHistogram  — fixed 256-bin histogram, range set by first batch
  QuantStats        — incremental MSE, SNR, EffBits, max_ae for quant error
"""
from __future__ import annotations
import numpy as np


class WelfordStats:
    """Welford's online algorithm for mean and variance.

    Supports batch updates (parallel Welford combination).
    Tracks absolute maximum value (not quantization error).
    """

    def __init__(self):
        self._n: int = 0
        self._mean: float = 0.0
        self._M2: float = 0.0       # sum of squared deviations from mean
        self._abs_max: float = 0.0

    def update(self, x: np.ndarray) -> None:
        x = x.astype(np.float64).ravel()
        n_b = len(x)
        if n_b == 0:
            return
        mean_b = float(np.mean(x))
        M2_b = float(np.sum((x - mean_b) ** 2))

        n_new = self._n + n_b
        delta = mean_b - self._mean
        self._M2 = (
            self._M2
            + M2_b
            + delta ** 2 * self._n * n_b / n_new
        )
        self._mean = (self._mean * self._n + mean_b * n_b) / n_new
        self._n = n_new
        self._abs_max = max(self._abs_max, float(np.max(np.abs(x))))

    def finalize(self) -> dict:
        if self._n == 0:
            raise RuntimeError("No data recorded — call update() before finalize().")
        return {
            "mean": self._mean,
            "std": float(np.sqrt(self._M2 / self._n)) if self._n > 1 else 0.0,
            "abs_max": self._abs_max,
            "n_elements": self._n,
        }


class RunningHistogram:
    """Fixed 256-bin histogram accumulated across batches.

    Range is determined by the first call to update(). Subsequent batches
    use the same bin edges; out-of-range values are counted as outliers
    but clamped into the edge bins for the histogram counts.
    """

    N_BINS: int = 256

    def __init__(self):
        self._counts: np.ndarray | None = None
        self._edges: np.ndarray | None = None
        self._n_total: int = 0
        self._n_outliers: int = 0

    def update(self, x: np.ndarray) -> None:
        x = x.astype(np.float64).ravel()
        if len(x) == 0:
            return

        if self._edges is None:
            lo, hi = float(np.min(x)), float(np.max(x))
            if lo == hi:
                hi = lo + 1e-10
            self._edges = np.linspace(lo, hi, self.N_BINS + 1)
            self._counts = np.zeros(self.N_BINS, dtype=np.int64)

        lo, hi = self._edges[0], self._edges[-1]
        self._n_outliers += int(np.sum((x < lo) | (x > hi)))
        self._n_total += len(x)

        clipped = np.clip(x, lo, hi)
        counts, _ = np.histogram(clipped, bins=self._edges)
        self._counts += counts

    def finalize(self) -> dict:
        if self._counts is None:
            return {"outlier_ratio": 0.0, "hist_counts": [], "hist_edges": []}
        return {
            "outlier_ratio": self._n_outliers / max(self._n_total, 1),
            "hist_counts": self._counts.tolist(),
            "hist_edges": self._edges.tolist(),
        }


class QuantStats:
    """Incremental quantization error statistics.

    Tracks MSE, SNR (dB), EffBits, and max absolute error between
    an original tensor and its quantized reconstruction.
    """

    def __init__(self):
        self._sum_sq_err: float = 0.0
        self._sum_sq_orig: float = 0.0
        self._sum_orig: float = 0.0
        self._n: int = 0
        self._max_ae: float = 0.0

    def update(self, x_orig: np.ndarray, x_quant: np.ndarray) -> None:
        x_orig = x_orig.astype(np.float64).ravel()
        x_quant = x_quant.astype(np.float64).ravel()
        err = x_orig - x_quant
        self._sum_sq_err += float(np.sum(err ** 2))
        self._sum_sq_orig += float(np.sum(x_orig ** 2))
        self._sum_orig += float(np.sum(x_orig))
        self._n += len(x_orig)
        self._max_ae = max(self._max_ae, float(np.max(np.abs(err))))

    def finalize(self) -> dict:
        if self._n == 0:
            return {
                "mse": float("nan"),
                "snr_db": float("nan"),
                "eff_bits": float("nan"),
                "max_ae": float("nan"),
            }
        mse = self._sum_sq_err / self._n
        mean = self._sum_orig / self._n
        var = max(self._sum_sq_orig / self._n - mean ** 2, 0.0)
        if mse == 0.0:
            snr_db = float("inf")
            eff_bits = float("inf")
        elif var <= 0.0 or var <= mse:
            snr_db = 0.0
            eff_bits = 0.0
        else:
            snr_db = float(10.0 * np.log10(var / mse))
            eff_bits = float(0.5 * np.log2(var / mse))
        return {
            "mse": mse,
            "snr_db": snr_db,
            "eff_bits": eff_bits,
            "max_ae": self._max_ae,
        }
```

Also create `profiler/__init__.py` (empty for now):

```python
# profiler/__init__.py
```

### Step 4: Run tests

```bash
pytest tests/test_profiler_stats.py -v
```

Expected: all tests PASS.

### Step 5: Commit

```bash
git add profiler/__init__.py profiler/stats.py tests/test_profiler_stats.py
git commit -m "feat: add profiler online statistics (Welford, Histogram, QuantStats)"
```

---

## Task 2: Format Schedule

**Files:**
- Create: `profiler/formats.py`
- Test: `tests/test_profiler_formats.py`

### Step 1: Write failing tests

```python
# tests/test_profiler_formats.py
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
```

### Step 2: Run tests to verify they fail

```bash
pytest tests/test_profiler_formats.py -v
```

Expected: `ModuleNotFoundError: No module named 'profiler.formats'`

### Step 3: Implement profiler/formats.py

```python
# profiler/formats.py
"""Format schedule for the model profiler.

Builds the ordered list of 14 quantization formats to profile.
All format objects are reused from formats/__init__.py — no new implementations.

Import note: _POTINTQuantizer, ComposedFormat are private but accessible
within the same package. They are imported directly here.
"""
from __future__ import annotations

import numpy as np

from formats.baseline import FP32Format, BF16Format
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

    HADTransform uses normalize=False (hardware model: no √N division,
    scale absorbed by quantizer). Same setting as the main format registry.
    """
    had = HADTransform(normalize=False)

    formats = [
        ("FP32",          FP32Format()),
        ("FP16",          BF16Format()),                              # BF16 ≈ FP16 simulation
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
```

### Step 4: Run tests

```bash
pytest tests/test_profiler_formats.py -v
```

Expected: all PASS.

### Step 5: Commit

```bash
git add profiler/formats.py tests/test_profiler_formats.py
git commit -m "feat: add profiler format schedule (14 formats)"
```

---

## Task 3: ModelProfiler — Hook Management & start/stop

**Files:**
- Create: `profiler/profiler.py`
- Test: `tests/test_profiler_core.py`

### Step 1: Write failing tests

```python
# tests/test_profiler_core.py
"""Tests for ModelProfiler hook management, start/stop, and format iteration."""
import numpy as np
import pytest
import torch
import torch.nn as nn

from profiler.profiler import ModelProfiler


def _make_model():
    return nn.Sequential(
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
    )


def _run_batches(model, profiler, n=3):
    for _ in range(n):
        x = torch.randn(2, 16)
        with torch.no_grad():
            model(x)


class TestProfilerInit:
    def test_done_false_initially(self):
        p = ModelProfiler(_make_model())
        assert not p.done

    def test_done_true_after_all_formats(self):
        model = _make_model()
        p = ModelProfiler(model)
        while not p.done:
            p.start()
            _run_batches(model, p, n=1)
            p.stop()
        assert p.done

    def test_current_format_name(self):
        p = ModelProfiler(_make_model())
        assert p.current_format_name == "FP32"

    def test_format_advances_after_stop(self):
        model = _make_model()
        p = ModelProfiler(model)
        p.start()
        _run_batches(model, p, n=1)
        p.stop()
        assert p.current_format_name == "FP16"


class TestProfilerHooks:
    def test_hooks_registered_after_start(self):
        model = _make_model()
        p = ModelProfiler(model)
        p.start()
        # Each leaf module gets 2 hooks (pre + post)
        n_leaf = sum(1 for m in model.modules() if len(list(m.children())) == 0)
        assert len(p._hooks) == n_leaf * 2

    def test_hooks_removed_after_stop(self):
        model = _make_model()
        p = ModelProfiler(model)
        p.start()
        _run_batches(model, p, n=1)
        p.stop()
        assert len(p._hooks) == 0

    def test_no_hooks_before_start(self):
        p = ModelProfiler(_make_model())
        assert len(p._hooks) == 0


class TestProfilerStats:
    def test_stats_populated_after_stop(self):
        model = _make_model()
        p = ModelProfiler(model)
        p.start()
        _run_batches(model, p, n=2)
        p.stop()

        fmt_name = "FP32"
        assert fmt_name in p._data
        layer_data = p._data[fmt_name]
        # At least the two Linear layers should be captured
        layer_names = list(layer_data.keys())
        assert len(layer_names) >= 2

    def test_input_output_captured(self):
        model = _make_model()
        p = ModelProfiler(model)
        p.start()
        _run_batches(model, p, n=1)
        p.stop()

        fmt_data = p._data["FP32"]
        for layer_name, tensors in fmt_data.items():
            assert "input" in tensors or "output" in tensors

    def test_weight_captured_for_linear(self):
        model = _make_model()
        p = ModelProfiler(model)
        p.start()
        _run_batches(model, p, n=3)
        p.stop()

        fmt_data = p._data["FP32"]
        linear_layers = [n for n, m in model.named_modules()
                         if isinstance(m, nn.Linear)]
        for layer_name in linear_layers:
            assert "weight" in fmt_data[layer_name]

    def test_target_layers_filter(self):
        model = _make_model()
        p = ModelProfiler(model, target_layers=[nn.Linear])
        p.start()
        _run_batches(model, p, n=1)
        p.stop()

        fmt_data = p._data["FP32"]
        # Only Linear layers captured, not ReLU
        for layer_name in fmt_data:
            module = dict(model.named_modules())[layer_name]
            assert isinstance(module, nn.Linear)

    def test_n_batches_tracked(self):
        model = _make_model()
        p = ModelProfiler(model)
        p.start()
        _run_batches(model, p, n=5)
        p.stop()
        assert p._n_batches["FP32"] == 5
```

### Step 2: Run tests to verify they fail

```bash
pytest tests/test_profiler_core.py -v
```

Expected: `ModuleNotFoundError: No module named 'profiler.profiler'`

### Step 3: Implement profiler/profiler.py

```python
# profiler/profiler.py
"""ModelProfiler: hooks into a PyTorch model to record per-layer tensor
distributions across a sequence of quantization formats.

Usage:
    profiler = ModelProfiler(model)
    while not profiler.done:
        profiler.start()
        for batch in loader:
            model(batch)          # normal inference, unmodified
        profiler.stop()
    profiler.export_csv("results/")
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Type

import numpy as np
import torch
import torch.nn as nn

from profiler.formats import build_profiler_formats
from profiler.stats import WelfordStats, RunningHistogram, QuantStats


class _TensorStats:
    """Stats bundle for one (format, layer, tensor_type) triple."""

    __slots__ = ("welford", "hist", "quant")

    def __init__(self):
        self.welford = WelfordStats()
        self.hist = RunningHistogram()
        self.quant = QuantStats()

    def update(self, x: np.ndarray, fmt_obj) -> None:
        self.welford.update(x)
        self.hist.update(x)
        try:
            x_q = fmt_obj.quantize(x)
            self.quant.update(x, x_q)
        except Exception:
            pass  # silently skip if format cannot handle this tensor shape


class ModelProfiler:
    """Non-intrusive PyTorch model profiler.

    Parameters
    ----------
    model : nn.Module
        The model to profile. Not modified in any way.
    formats : list[tuple[str, object]] | None
        Ordered list of (name, format_object) pairs. Defaults to
        ``build_profiler_formats()`` (14 formats).
    target_layers : list[type | str] | None
        If provided, only hook modules whose type or class name matches.
        None (default) hooks all leaf modules.
    """

    def __init__(
        self,
        model: nn.Module,
        formats: Optional[List[tuple]] = None,
        target_layers: Optional[Sequence[Type | str]] = None,
    ):
        self._model = model
        self._formats: List[tuple] = formats or build_profiler_formats()
        self._target_layers = target_layers
        self._format_idx: int = 0

        self._hooks: list = []
        # _data[fmt_name][layer_name][tensor_type] = _TensorStats
        self._data: dict[str, dict[str, dict[str, _TensorStats]]] = {}
        # tracks which (fmt_name, layer_name) weights have been captured
        self._weight_captured: set[tuple[str, str]] = set()
        self._n_batches: dict[str, int] = {}
        # hook handle for batch counting (on root model)
        self._batch_counter_hook = None

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def done(self) -> bool:
        return self._format_idx >= len(self._formats)

    @property
    def current_format_name(self) -> str:
        if self.done:
            raise RuntimeError("All formats have been profiled.")
        return self._formats[self._format_idx][0]

    def start(self) -> None:
        """Begin recording for the current format. Call before inference."""
        if self.done:
            raise RuntimeError("All formats have been profiled. Nothing to start.")
        fmt_name, fmt_obj = self._formats[self._format_idx]
        if fmt_name not in self._data:
            self._data[fmt_name] = {}
            self._n_batches[fmt_name] = 0
        self._register_hooks(fmt_name, fmt_obj)

    def stop(self) -> None:
        """Finalize current format and advance to the next. Call after inference."""
        self._deregister_hooks()
        self._format_idx += 1

    def wrap(self, fn: Callable, name: str) -> Callable:
        """Wrap a functional op (e.g. torch.matmul) for profiling.

        Usage:
            # Replace: output = torch.matmul(a, b)
            # With:    output = profiler.wrap(torch.matmul, 'matmul')(a, b)
        """
        profiler = self

        def wrapped(*args, **kwargs):
            output = fn(*args, **kwargs)
            if profiler.done:
                return output
            fmt_name, fmt_obj = profiler._formats[profiler._format_idx]
            # Capture first positional arg as "input", output as "output"
            for tensor_type, tensor in [("input", args[0] if args else None),
                                         ("output", output)]:
                if tensor is not None and isinstance(tensor, torch.Tensor):
                    x = tensor.detach().cpu().numpy()
                    layer_stats = profiler._data.get(fmt_name, {})
                    ts = layer_stats.setdefault(name, {}).setdefault(
                        tensor_type, _TensorStats()
                    )
                    ts.update(x, fmt_obj)
                    profiler._data.setdefault(fmt_name, {})[name] = layer_stats[name]
            return output

        return wrapped

    # ── Internal ───────────────────────────────────────────────────────────────

    def _is_leaf(self, module: nn.Module) -> bool:
        return len(list(module.children())) == 0

    def _should_hook(self, module: nn.Module) -> bool:
        if self._target_layers is None:
            return True
        for t in self._target_layers:
            if isinstance(t, type) and isinstance(module, t):
                return True
            if isinstance(t, str) and type(module).__name__ == t:
                return True
        return False

    def _register_hooks(self, fmt_name: str, fmt_obj) -> None:
        for layer_name, module in self._model.named_modules():
            if not self._is_leaf(module):
                continue
            if not self._should_hook(module):
                continue

            # Pre-hook: input activation + weight
            def make_pre_hook(lname: str, fname: str, fobj):
                def pre_hook(mod, args):
                    layer_stats = self._data[fname].setdefault(lname, {})
                    # Input activation
                    if args and isinstance(args[0], torch.Tensor):
                        x = args[0].detach().cpu().numpy()
                        ts = layer_stats.setdefault("input", _TensorStats())
                        ts.update(x, fobj)
                    # Weight (static: capture only on first batch)
                    key = (fname, lname)
                    if key not in self._weight_captured:
                        if hasattr(mod, "weight") and mod.weight is not None:
                            w = mod.weight.detach().cpu().numpy()
                            ts = layer_stats.setdefault("weight", _TensorStats())
                            ts.update(w, fobj)
                            self._weight_captured.add(key)
                return pre_hook

            # Post-hook: output activation
            def make_post_hook(lname: str, fname: str, fobj):
                def post_hook(mod, args, output):
                    if isinstance(output, torch.Tensor):
                        y = output.detach().cpu().numpy()
                        layer_stats = self._data[fname].setdefault(lname, {})
                        ts = layer_stats.setdefault("output", _TensorStats())
                        ts.update(y, fobj)
                return post_hook

            h1 = module.register_forward_pre_hook(
                make_pre_hook(layer_name, fmt_name, fmt_obj)
            )
            h2 = module.register_forward_hook(
                make_post_hook(layer_name, fmt_name, fmt_obj)
            )
            self._hooks.extend([h1, h2])

        # Batch counter: hook on root model
        def _count_batch(mod, args):
            self._n_batches[fmt_name] = self._n_batches.get(fmt_name, 0) + 1

        self._batch_counter_hook = self._model.register_forward_pre_hook(_count_batch)

    def _deregister_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        if self._batch_counter_hook is not None:
            self._batch_counter_hook.remove()
            self._batch_counter_hook = None
```

### Step 4: Run tests

```bash
pytest tests/test_profiler_core.py -v
```

Expected: all PASS.

### Step 5: Commit

```bash
git add profiler/profiler.py tests/test_profiler_core.py
git commit -m "feat: add ModelProfiler with hook management and start/stop"
```

---

## Task 4: CSV Export

**Files:**
- Create: `profiler/export.py`
- Test: `tests/test_profiler_export.py`

### Step 1: Write failing tests

```python
# tests/test_profiler_export.py
import os
import tempfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from profiler.profiler import ModelProfiler


def _profile_tiny_model(n_formats=2):
    model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 2))
    p = ModelProfiler(model)
    for _ in range(n_formats):
        if p.done:
            break
        p.start()
        for _ in range(2):
            with torch.no_grad():
                model(torch.randn(4, 8))
        p.stop()
    return p


class TestExportCSV:
    def test_export_creates_csv(self):
        p = _profile_tiny_model()
        with tempfile.TemporaryDirectory() as d:
            p.export_csv(d)
            files = os.listdir(d)
            assert any(f.endswith(".csv") for f in files)

    def test_csv_has_required_columns(self):
        p = _profile_tiny_model()
        with tempfile.TemporaryDirectory() as d:
            p.export_csv(d)
            csv_path = [os.path.join(d, f) for f in os.listdir(d)
                        if f.endswith(".csv")][0]
            df = pd.read_csv(csv_path)
            required = {
                "format", "layer_name", "layer_type", "tensor_type", "bits",
                "mse", "snr_db", "eff_bits", "max_ae",
                "mean", "std", "outlier_ratio", "n_batches", "n_elements",
            }
            assert required.issubset(set(df.columns)), \
                f"Missing: {required - set(df.columns)}"

    def test_csv_has_one_row_per_format_layer_tensortype(self):
        p = _profile_tiny_model(n_formats=2)
        with tempfile.TemporaryDirectory() as d:
            p.export_csv(d)
            csv_path = [os.path.join(d, f) for f in os.listdir(d)
                        if f.endswith(".csv")][0]
            df = pd.read_csv(csv_path)
            # No duplicate (format, layer_name, tensor_type) rows
            dupes = df.duplicated(["format", "layer_name", "tensor_type"])
            assert not dupes.any(), f"Duplicate rows found:\n{df[dupes]}"

    def test_fp32_mse_is_zero(self):
        p = _profile_tiny_model(n_formats=1)
        with tempfile.TemporaryDirectory() as d:
            p.export_csv(d)
            csv_path = [os.path.join(d, f) for f in os.listdir(d)
                        if f.endswith(".csv")][0]
            df = pd.read_csv(csv_path)
            fp32_rows = df[df["format"] == "FP32"]
            assert (fp32_rows["mse"].fillna(0.0) == 0.0).all(), \
                "FP32 should have zero quantization error"
```

### Step 2: Run tests to verify they fail

```bash
pytest tests/test_profiler_export.py -v
```

Expected: `AttributeError: 'ModelProfiler' object has no attribute 'export_csv'`

### Step 3: Implement profiler/export.py and add export_csv to profiler.py

```python
# profiler/export.py
"""CSV export for ModelProfiler results.

Output schema (one row per format × layer × tensor_type):
  format, layer_name, layer_type, tensor_type, bits,
  mse, snr_db, eff_bits, max_ae,
  mean, std, outlier_ratio, n_batches, n_elements

Compatible with ExperimentRunner CSV output for use with existing
visualization scripts.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pandas as pd
import torch.nn as nn

if TYPE_CHECKING:
    from profiler.profiler import ModelProfiler

# Nominal bit-width for each format name
_FORMAT_BITS: dict[str, int] = {
    "FP32": 32, "FP16": 16,
    "SQ-FORMAT-INT": 4, "SQ-FORMAT-FP": 4,
    "INT4(CHANNEL)": 4, "INT8(CHANNEL)": 8,
    "INT4(TENSOR)": 4,  "INT8(TENSOR)": 8,
    "HAD+INT4(C)": 4,   "HAD+INT8(C)": 8,
    "HAD+INT4(T)": 4,   "HAD+INT8(T)": 8,
    "MXINT4": 4,        "MXINT8": 8,
}


def export_csv(profiler: "ModelProfiler", output_dir: str, filename: str = "profiler_results.csv") -> str:
    """Materialize all recorded stats into a single CSV file.

    Parameters
    ----------
    profiler : ModelProfiler
        Profiler instance after one or more start/stop cycles.
    output_dir : str
        Directory to write the CSV into.
    filename : str
        Output filename (default: profiler_results.csv).

    Returns
    -------
    str
        Absolute path to the written CSV file.
    """
    # Build a lookup: layer_name → layer_type string
    layer_types: dict[str, str] = {
        name: type(mod).__name__
        for name, mod in profiler._model.named_modules()
    }

    rows = []
    for fmt_name, layer_dict in profiler._data.items():
        n_batches = profiler._n_batches.get(fmt_name, 0)
        bits = _FORMAT_BITS.get(fmt_name, -1)

        for layer_name, tensor_dict in layer_dict.items():
            layer_type = layer_types.get(layer_name, "unknown")

            for tensor_type, ts in tensor_dict.items():
                try:
                    w_stats = ts.welford.finalize()
                    h_stats = ts.hist.finalize()
                    q_stats = ts.quant.finalize()
                except RuntimeError:
                    continue  # skip if no data

                rows.append({
                    "format":        fmt_name,
                    "layer_name":    layer_name,
                    "layer_type":    layer_type,
                    "tensor_type":   tensor_type,
                    "bits":          bits,
                    "mse":           q_stats["mse"],
                    "snr_db":        q_stats["snr_db"],
                    "eff_bits":      q_stats["eff_bits"],
                    "max_ae":        q_stats["max_ae"],
                    "mean":          w_stats["mean"],
                    "std":           w_stats["std"],
                    "outlier_ratio": h_stats["outlier_ratio"],
                    "n_batches":     n_batches,
                    "n_elements":    w_stats["n_elements"],
                })

    df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False)
    return path
```

Then add to `profiler/profiler.py` (import + method):

At the top of `profiler/profiler.py`, add:
```python
from profiler.export import export_csv as _export_csv
```

Add method to `ModelProfiler`:
```python
    def export_csv(self, output_dir: str, filename: str = "profiler_results.csv") -> str:
        """Export all recorded stats to a CSV file.

        Returns the absolute path to the written file.
        """
        return _export_csv(self, output_dir, filename)
```

### Step 4: Run tests

```bash
pytest tests/test_profiler_export.py -v
```

Expected: all PASS.

### Step 5: Commit

```bash
git add profiler/export.py profiler/profiler.py tests/test_profiler_export.py
git commit -m "feat: add CSV export for ModelProfiler results"
```

---

## Task 5: Package Init + Integration Test

**Files:**
- Modify: `profiler/__init__.py`
- Create: `tests/test_profiler_integration.py`

### Step 1: Write failing integration test

```python
# tests/test_profiler_integration.py
"""End-to-end test: profile a small model through all 14 formats."""
import os
import tempfile

import pandas as pd
import torch
import torch.nn as nn

from profiler import ModelProfiler


class TinyTransformerLayer(nn.Module):
    """Minimal model with Linear + LayerNorm + Softmax (common in LLMs)."""

    def __init__(self):
        super().__init__()
        self.q = nn.Linear(16, 16)
        self.k = nn.Linear(16, 16)
        self.norm = nn.LayerNorm(16)
        self.out = nn.Linear(16, 8)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        scores = torch.softmax(q * k, dim=-1)
        return self.out(self.norm(scores))


class TestFullProfileRun:
    def test_all_14_formats_profiled(self):
        model = TinyTransformerLayer()
        p = ModelProfiler(model)
        while not p.done:
            p.start()
            for _ in range(3):
                with torch.no_grad():
                    model(torch.randn(4, 16))
            p.stop()
        assert p.done
        assert len(p._data) == 14

    def test_csv_contains_all_formats(self):
        model = TinyTransformerLayer()
        p = ModelProfiler(model)
        while not p.done:
            p.start()
            with torch.no_grad():
                model(torch.randn(4, 16))
            p.stop()

        with tempfile.TemporaryDirectory() as d:
            path = p.export_csv(d)
            df = pd.read_csv(path)

        from profiler.formats import PROFILER_FORMAT_NAMES
        assert set(df["format"].unique()) == set(PROFILER_FORMAT_NAMES)

    def test_quantization_error_increases_with_lower_bits(self):
        """FP32 MSE == 0; lower-bit formats have higher MSE."""
        model = nn.Linear(64, 64)
        p = ModelProfiler(model, target_layers=[nn.Linear])
        while not p.done:
            p.start()
            for _ in range(5):
                with torch.no_grad():
                    model(torch.randn(8, 64))
            p.stop()

        with tempfile.TemporaryDirectory() as d:
            path = p.export_csv(d)
            df = pd.read_csv(path)

        weight_df = df[df["tensor_type"] == "weight"]
        fp32_mse = weight_df[weight_df["format"] == "FP32"]["mse"].mean()
        int4_mse = weight_df[weight_df["format"] == "INT4(TENSOR)"]["mse"].mean()
        assert fp32_mse == 0.0
        assert int4_mse > fp32_mse

    def test_wrap_functional(self):
        """profiler.wrap() captures stats for torch.matmul."""
        model = nn.Linear(16, 16)
        p = ModelProfiler(model)
        p.start()
        with torch.no_grad():
            x = torch.randn(4, 16)
            a = model(x)
            b = p.wrap(torch.matmul, "custom_matmul")(a, a.T)
        p.stop()

        assert "custom_matmul" in p._data.get("FP32", {})
```

### Step 2: Run test to verify it fails

```bash
pytest tests/test_profiler_integration.py -v
```

Expected: `ImportError` from `from profiler import ModelProfiler`

### Step 3: Update profiler/__init__.py

```python
# profiler/__init__.py
"""Model profiler for per-layer tensor distribution analysis.

Quick start:
    from profiler import ModelProfiler

    profiler = ModelProfiler(model)
    while not profiler.done:
        profiler.start()
        for batch in loader:
            model(batch)
        profiler.stop()
    profiler.export_csv("results/")
"""
from profiler.profiler import ModelProfiler

__all__ = ["ModelProfiler"]
```

### Step 4: Run all profiler tests

```bash
pytest tests/test_profiler_stats.py tests/test_profiler_formats.py tests/test_profiler_core.py tests/test_profiler_export.py tests/test_profiler_integration.py -v
```

Expected: all PASS.

### Step 5: Run existing tests to confirm no regressions

```bash
pytest tests/ -v
```

Expected: all previously passing tests still PASS.

### Step 6: Commit

```bash
git add profiler/__init__.py tests/test_profiler_integration.py
git commit -m "feat: complete model profiler — integration tested across 14 formats"
```

---

## Summary

| Task | Files Created | Tests |
|---|---|---|
| 1 | `profiler/stats.py` | `tests/test_profiler_stats.py` |
| 2 | `profiler/formats.py` | `tests/test_profiler_formats.py` |
| 3 | `profiler/profiler.py` | `tests/test_profiler_core.py` |
| 4 | `profiler/export.py` + update profiler.py | `tests/test_profiler_export.py` |
| 5 | `profiler/__init__.py` | `tests/test_profiler_integration.py` |

**New dependency:** `torch` (not in requirements.txt). Install with:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

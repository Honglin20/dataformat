"""ModelProfiler: hooks into a PyTorch model to record per-layer tensor
distributions across a sequence of quantization formats.

Usage (basic):
    profiler = ModelProfiler(model)
    while not profiler.done:
        profiler.start()
        for batch in loader:
            model(batch)
        profiler.stop()
    profiler.export_csv("results/")

Usage (layer-mode selection):
    # Profile only Linear layers, including end-to-end matmul SQNR
    profiler = ModelProfiler(model, layer_mode="linear")

    # Profile only non-linear layers (LayerNorm, GELU, …)
    profiler = ModelProfiler(model, layer_mode="nonlinear")

End-to-end SQNR:
    For every nn.Linear layer and every format, the profiler simulates the
    quantized matmul and compares it to the actual FP32 output.
    - HAD+INT formats: y_q = Q(H(x)) @ Q(H(W))ᵀ / N  (HAD-domain multiply)
    - All other formats: y_q = Q(x) @ Q(W)ᵀ + b        (both W and x quantized)
    The e2e SNR data is stored in profiler._e2e_data and exported separately.
"""
from __future__ import annotations

import collections
from typing import Callable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from profiler.formats import build_profiler_formats, simulate_linear_output
from profiler.stats import WelfordStats, RunningHistogram, QuantStats
from profiler.export import export_csv as _export_csv


# ── Layer-mode type sets ───────────────────────────────────────────────────────

_LINEAR_TYPES = (nn.Linear,)

_NONLINEAR_TYPES = (
    nn.ReLU, nn.GELU, nn.SiLU, nn.Mish, nn.ELU, nn.LeakyReLU,
    nn.Sigmoid, nn.Tanh, nn.Hardswish, nn.Hardtanh,
    nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
    nn.Softmax, nn.LogSoftmax,
)


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a torch.Tensor to a float32 numpy array."""
    try:
        return tensor.detach().cpu().numpy()
    except RuntimeError:
        arr = np.array(tensor.detach().cpu().tolist())
        return arr.astype(np.float32) if arr.dtype.kind == "f" else arr


# ── Per-tensor stats bundle ────────────────────────────────────────────────────

class _TensorStats:
    """Stats bundle for one (format, layer, tensor_type) triple.

    welford        — original-domain statistics (mean, std, kurtosis, skewness, …)
    hist           — running histogram for outlier detection
    quant          — quantization error (MSE, SNR, EffBits, MaxAE, MARE, saturation)
    domain_welford — post-transform-domain statistics (e.g. HAD domain for HAD+INT);
                     populated only for formats that expose 'transformed' in their
                     quantize_with_metadata() return dict.
    """

    __slots__ = ("welford", "hist", "quant", "domain_welford")

    def __init__(self):
        self.welford = WelfordStats()
        self.hist = RunningHistogram()
        self.quant = QuantStats()
        self.domain_welford: WelfordStats | None = None

    def update(self, x: np.ndarray, fmt_obj) -> None:
        self.welford.update(x)
        self.hist.update(x)
        try:
            if hasattr(fmt_obj, "quantize_with_metadata"):
                meta = fmt_obj.quantize_with_metadata(x)
                x_q = meta["quantized"]
                smask = meta.get("saturation_mask")
                self.quant.update(x, x_q, saturation_mask=smask)
                # Record post-transform domain stats (e.g. HAD domain)
                x_t = meta.get("transformed")
                if x_t is not None:
                    if self.domain_welford is None:
                        self.domain_welford = WelfordStats()
                    self.domain_welford.update(x_t)
            else:
                x_q = fmt_obj.quantize(x)
                self.quant.update(x, x_q)
        except Exception:
            pass  # silently skip if format cannot handle this tensor shape


# ── ModelProfiler ──────────────────────────────────────────────────────────────

class ModelProfiler:
    """Non-intrusive PyTorch model profiler.

    Parameters
    ----------
    model : nn.Module
        The model to profile. Not modified in any way.
    formats : list[tuple[str, object]] | None
        Ordered list of (name, format_object) pairs.
        Defaults to build_profiler_formats() (14 formats).
    target_layers : list[type | str] | None
        Explicit allowlist — only hook modules whose type or class name matches.
        When set, overrides layer_mode.
    layer_mode : str
        "all"        — hook all leaf modules (default).
        "linear"     — hook only nn.Linear.
        "nonlinear"  — hook all leaf modules except nn.Linear.
    """

    def __init__(
        self,
        model: nn.Module,
        formats: Optional[List[tuple]] = None,
        target_layers: Optional[Sequence] = None,
        layer_mode: str = "all",
    ):
        assert layer_mode in ("all", "linear", "nonlinear"), (
            f"layer_mode must be 'all', 'linear', or 'nonlinear'; got {layer_mode!r}"
        )
        self._model = model
        self._formats: List[tuple] = formats or build_profiler_formats()
        self._target_layers = target_layers
        self._layer_mode = layer_mode
        self._format_idx: int = 0

        # Per-tensor stats hooks
        self._hooks: list = []
        # End-to-end (layer output) hooks — kept separate so hook count tests still pass
        self._e2e_hooks: list = []

        # _data[fmt_name][layer_name][tensor_type] = _TensorStats
        self._data: dict = {}
        # _e2e_data[fmt_name][layer_name] = QuantStats  (layer output SQNR)
        self._e2e_data: dict = {}
        # _pending_e2e[layer_name] = deque of y_quant arrays (pre-hook stash for post-hook)
        self._pending_e2e: dict = {}

        self._weight_captured: set = set()
        self._n_batches: dict = {}
        self._batch_counter_hook = None

    # ── Public interface ───────────────────────────────────────────────────────

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

    def cleanup(self) -> None:
        """Remove all registered hooks. Safe to call multiple times."""
        self._deregister_hooks()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False

    def stop(self) -> None:
        """Finalize current format and advance to the next. Call after inference."""
        self.cleanup()
        self._format_idx += 1

    def wrap(self, fn: Callable, name: str) -> Callable:
        """Wrap a functional op for profiling.

        Usage:
            output = profiler.wrap(torch.matmul, 'matmul')(a, b)
        """
        profiler = self

        def wrapped(*args, **kwargs):
            output = fn(*args, **kwargs)
            if profiler.done:
                return output
            fmt_name, fmt_obj = profiler._formats[profiler._format_idx]
            fmt_data = profiler._data.setdefault(fmt_name, {})
            layer_data = fmt_data.setdefault(name, {})
            for tensor_type, tensor in [
                ("input", args[0] if args else None),
                ("output", output),
            ]:
                if tensor is not None and isinstance(tensor, torch.Tensor):
                    x = _to_numpy(tensor)
                    ts = layer_data.setdefault(tensor_type, _TensorStats())
                    ts.update(x, fmt_obj)
            return output

        return wrapped

    def export_csv(self, output_dir: str, filename: str = "profiler_results.csv") -> str:
        """Export all recorded stats to a CSV file. Returns the path."""
        return _export_csv(self, output_dir, filename)

    def export_histograms(self, output_dir: str, filename: str = "profiler_histograms.json") -> str:
        """Export per-layer histogram data as JSON. Returns the path."""
        from profiler.export import export_histograms as _export_histograms
        return _export_histograms(self, output_dir, filename)

    # ── Internal: hook management ──────────────────────────────────────────────

    def _is_leaf(self, module: nn.Module) -> bool:
        return len(list(module.children())) == 0

    def _should_hook(self, module: nn.Module) -> bool:
        # Explicit target_layers allowlist takes priority over layer_mode
        if self._target_layers is not None:
            for t in self._target_layers:
                if isinstance(t, type) and isinstance(module, t):
                    return True
                if isinstance(t, str) and type(module).__name__ == t:
                    return True
            return False
        # layer_mode filter
        is_linear = isinstance(module, _LINEAR_TYPES)
        if self._layer_mode == "linear":
            return is_linear
        if self._layer_mode == "nonlinear":
            return not is_linear
        return True  # "all"

    def _register_hooks(self, fmt_name: str, fmt_obj) -> None:
        for layer_name, module in self._model.named_modules():
            if not self._is_leaf(module):
                continue
            if not self._should_hook(module):
                continue

            # ── Per-tensor stats hooks ─────────────────────────────────────────

            def make_pre_hook(lname: str, fname: str, fobj):
                def pre_hook(mod, args):
                    layer_stats = self._data[fname].setdefault(lname, {})
                    if args and isinstance(args[0], torch.Tensor):
                        x = _to_numpy(args[0])
                        ts = layer_stats.setdefault("input", _TensorStats())
                        ts.update(x, fobj)
                    key = (fname, lname)
                    if key not in self._weight_captured:
                        if hasattr(mod, "weight") and mod.weight is not None:
                            w = _to_numpy(mod.weight)
                            ts = layer_stats.setdefault("weight", _TensorStats())
                            ts.update(w, fobj)
                            self._weight_captured.add(key)
                return pre_hook

            def make_post_hook(lname: str, fname: str, fobj):
                def post_hook(mod, args, output):
                    if isinstance(output, torch.Tensor):
                        y = _to_numpy(output)
                        layer_stats = self._data[fname].setdefault(lname, {})
                        ts = layer_stats.setdefault("output", _TensorStats())
                        ts.update(y, fobj)
                return post_hook

            h1 = module.register_forward_pre_hook(make_pre_hook(layer_name, fmt_name, fmt_obj))
            h2 = module.register_forward_hook(make_post_hook(layer_name, fmt_name, fmt_obj))
            self._hooks.extend([h1, h2])

            # ── End-to-end simulation hooks (Linear layers only) ───────────────

            if isinstance(module, nn.Linear):

                def make_e2e_pre_hook(lname: str, fname: str, fobj):
                    def e2e_pre(mod, args):
                        if not (args and isinstance(args[0], torch.Tensor)):
                            return
                        x_np = _to_numpy(args[0])
                        W_np = _to_numpy(mod.weight)
                        b_np = _to_numpy(mod.bias) if mod.bias is not None else None
                        try:
                            y_q, _ = simulate_linear_output(fobj, W_np, x_np, b_np)
                        except Exception:
                            y_q = None
                        queue = self._pending_e2e.setdefault(lname, collections.deque())
                        queue.append(y_q)   # None signals "skip this batch"
                    return e2e_pre

                def make_e2e_post_hook(lname: str, fname: str, fobj):
                    def e2e_post(mod, args, output):
                        queue = self._pending_e2e.get(lname)
                        if not queue:
                            return
                        y_q = queue.popleft()
                        if y_q is None or not isinstance(output, torch.Tensor):
                            return
                        y_fp32 = _to_numpy(output)
                        e2e_layer = self._e2e_data.setdefault(fname, {})
                        qs = e2e_layer.setdefault(lname, QuantStats())
                        qs.update(y_fp32.ravel(), y_q.ravel())
                    return e2e_post

                h3 = module.register_forward_pre_hook(make_e2e_pre_hook(layer_name, fmt_name, fmt_obj))
                h4 = module.register_forward_hook(make_e2e_post_hook(layer_name, fmt_name, fmt_obj))
                self._e2e_hooks.extend([h3, h4])

        # Global batch counter on the model root
        def _count_batch(mod, args):
            self._n_batches[fmt_name] = self._n_batches.get(fmt_name, 0) + 1

        self._batch_counter_hook = self._model.register_forward_pre_hook(_count_batch)

    def _deregister_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        for h in self._e2e_hooks:
            h.remove()
        self._e2e_hooks.clear()
        if self._batch_counter_hook is not None:
            self._batch_counter_hook.remove()
            self._batch_counter_hook = None
        # Discard any stale pre-hook stashes (e.g. from exceptions during inference)
        self._pending_e2e.clear()

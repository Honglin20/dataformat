"""ModelProfiler: hooks into a PyTorch model to record per-layer tensor
distributions across a sequence of quantization formats.

Usage:
    profiler = ModelProfiler(model)
    while not profiler.done:
        profiler.start()
        for batch in loader:
            model(batch)
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


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a torch Tensor to numpy array, compatible with numpy 2.x."""
    try:
        return tensor.detach().cpu().numpy()
    except RuntimeError:
        # Fallback when torch was built against numpy 1.x but numpy 2.x is installed
        return np.array(tensor.detach().cpu().tolist(), dtype=np.float32)


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
        Ordered list of (name, format_object) pairs.
        Defaults to build_profiler_formats() (14 formats).
    target_layers : list[type | str] | None
        If provided, only hook modules whose type or class name matches.
        None (default) hooks all leaf modules.
    """

    def __init__(
        self,
        model: nn.Module,
        formats: Optional[List[tuple]] = None,
        target_layers: Optional[Sequence] = None,
    ):
        self._model = model
        self._formats: List[tuple] = formats or build_profiler_formats()
        self._target_layers = target_layers
        self._format_idx: int = 0

        self._hooks: list = []
        # _data[fmt_name][layer_name][tensor_type] = _TensorStats
        self._data: dict = {}
        # tracks which (fmt_name, layer_name) weights have already been captured
        self._weight_captured: set = set()
        self._n_batches: dict = {}
        self._batch_counter_hook = None

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
            for tensor_type, tensor in [("input", args[0] if args else None),
                                         ("output", output)]:
                if tensor is not None and isinstance(tensor, torch.Tensor):
                    x = _to_numpy(tensor)
                    ts = layer_data.setdefault(tensor_type, _TensorStats())
                    ts.update(x, fmt_obj)
            return output

        return wrapped

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

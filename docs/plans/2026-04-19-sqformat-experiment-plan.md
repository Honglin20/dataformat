# SQ-Format Comparison Study — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deliver the SQ-Format comparison study as an extension of
`experiments/fourbit/` — once the six code gaps are closed, every future
SQ-Format sweep is a `FormatSpec.kwargs` change with no experiment-code edits.

**Architecture:** Element-encoder registry turns `base: "int"|"fp"` into a
configurable parameter on the existing `SQFormat` / `SQFormatActivations`
classes (no duplicated banking code). Metrics become a config-driven
registry. A single `experiments/sqformat/` subpackage reuses the 4-bit
pipeline (`part1.run_all` / `part2.run` / `accuracy.accuracy_sweep` /
`reporter.generate_report`) via a new `FourBitConfig` instance.

**Tech Stack:** NumPy ≥ 1.24, PyTorch ≥ 2.0, pandas, pytest. No new
third-party dependencies.

**Design doc:** `docs/plans/2026-04-19-sqformat-experiment-design.md`

**PR topological order:** A → B → C → D → E → F. Every PR must pass
`pytest tests/test_regression.py -v` byte-identical against the existing
4-bit goldens before merging. PR F adds SQ-Format-specific goldens.

---

## PR A · `base="int"|"fp"` parameter on SQ-Format classes

**Files:**
- Modify: `formats/sq_format.py` (add element encoder registry + `base`
  parameter on `SQFormat` and `SQFormatActivations`)
- Test: `tests/test_sq_format_base_fp.py` (new)

### Task A.1 — Add element-encoder registry

**Files:**
- Modify: `formats/sq_format.py` (top of file, near line 100 before the
  `SQFormat` class)
- Test: `tests/test_sq_format_base_fp.py` (new)

**Step 1: Write the failing test**

```python
# tests/test_sq_format_base_fp.py
import numpy as np
import pytest
from formats.sq_format import _ELEMENT_ENCODERS


def test_int4_encoder_round_clip():
    fn, q_max = _ELEMENT_ENCODERS[("int", 4)]
    assert q_max == 7
    x = np.array([0.2, 7.6, -8.3, 0.51], dtype=np.float32)
    # scale=1.0 → round(x) clipped to [-7, 7]
    got = fn(x, scale=1.0)
    np.testing.assert_array_equal(got, np.array([0.0, 7.0, -7.0, 1.0], dtype=np.float32))


def test_fp4_e2m1_encoder_uses_level_set():
    fn, q_max = _ELEMENT_ENCODERS[("fp", 4)]
    assert q_max == 6.0
    # scale=1.0 → nearest E2M1 level (0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6)
    x = np.array([0.3, 1.4, 5.5, -6.5], dtype=np.float32)
    got = fn(x, scale=1.0)
    np.testing.assert_allclose(got, np.array([0.5, 1.5, 6.0, -6.0], dtype=np.float32))


def test_fp8_e4m3_encoder_saturates_at_448():
    fn, q_max = _ELEMENT_ENCODERS[("fp", 8)]
    assert q_max == 448.0
    x = np.array([0.0, 448.0, 500.0, -600.0], dtype=np.float32)
    got = fn(x, scale=1.0)
    assert got[0] == 0.0
    assert got[1] == 448.0
    assert abs(got[2]) <= 448.0
    assert abs(got[3]) <= 448.0
```

Run: `pytest tests/test_sq_format_base_fp.py -v`
Expected: FAIL with `ImportError: cannot import name '_ELEMENT_ENCODERS'`.

**Step 2: Write minimal implementation**

Add to `formats/sq_format.py` after existing imports:

```python
from formats.mxfp import _E2M1_POS, _fp8_e4m3_vec


def _int_encode(x: np.ndarray, scale: float, q_max: int) -> np.ndarray:
    q = np.clip(np.round(x / np.maximum(scale, 1e-38)), -q_max, q_max)
    return (q * scale).astype(np.float32)


def _fp_e2m1_encode(x: np.ndarray, scale: float) -> np.ndarray:
    x_scaled = x / np.maximum(scale, 1e-38)
    sign = np.where(x_scaled < 0, -1.0, 1.0).astype(np.float32)
    x_abs = np.clip(np.abs(x_scaled), 0.0, float(_E2M1_POS[-1]))
    dists = np.abs(x_abs[..., None] - _E2M1_POS)
    idx = np.argmin(dists, axis=-1)
    return (sign * _E2M1_POS[idx] * scale).astype(np.float32)


def _fp_e4m3_encode(x: np.ndarray, scale: float) -> np.ndarray:
    return (_fp8_e4m3_vec(x / np.maximum(scale, 1e-38)) * scale).astype(np.float32)


_ELEMENT_ENCODERS = {
    ("int", 8): (lambda x, s: _int_encode(x, s, 127), 127),
    ("int", 4): (lambda x, s: _int_encode(x, s,   7),   7),
    ("int", 2): (lambda x, s: _int_encode(x, s,   1),   1),
    ("fp",  8): (_fp_e4m3_encode, 448.0),
    ("fp",  4): (_fp_e2m1_encode,   6.0),
}
```

Run: `pytest tests/test_sq_format_base_fp.py -v`
Expected: PASS.

**Step 3: Commit**

```bash
git add formats/sq_format.py tests/test_sq_format_base_fp.py
git commit -m "feat: add _ELEMENT_ENCODERS registry to sq_format"
```

### Task A.2 — `base` parameter on `SQFormat` (Algorithm 1)

**Files:**
- Modify: `formats/sq_format.py` — `SQFormat.__init__`, `_quantize_2d`,
  `_quantize_1d`
- Test: `tests/test_sq_format_base_fp.py` (extend)

**Step 1: Write the failing tests (append to file)**

```python
from formats.sq_format import SQFormat
from formats.nvfp4 import NVFP4Format


def test_sqformat_base_int_default_unchanged():
    # Regression pin: default SQFormat() behaviour is identical to
    # SQFormat(base="int") — existing golden CSVs depend on this.
    rng = np.random.default_rng(0)
    W = rng.standard_normal((128, 64)).astype(np.float32)
    a = SQFormat()
    b = SQFormat(base="int")
    np.testing.assert_array_equal(a.quantize(W), b.quantize(W))


def test_sqformat_base_fp_produces_fp_levels():
    rng = np.random.default_rng(1)
    W = rng.standard_normal((128, 64)).astype(np.float32)
    fmt = SQFormat(base="fp", high_bits=8, low_bits=4, sparsity=0.5)
    W_q = fmt.quantize(W)
    assert W_q.shape == W.shape
    # FP output must not be on an integer grid — this distinguishes it
    # from base="int" at a coarse scale.
    a_int = SQFormat(base="int", high_bits=8, low_bits=4, sparsity=0.5).quantize(W)
    assert not np.array_equal(a_int, W_q)


def test_sqformat_rejects_unsupported_cell():
    with pytest.raises(ValueError, match="Unsupported SQ-Format cell"):
        SQFormat(base="fp", high_bits=2, low_bits=2)  # no FP2 in registry
```

Run: `pytest tests/test_sq_format_base_fp.py -v`
Expected: FAIL (the new tests, not yet wired).

**Step 2: Minimal implementation**

Update `SQFormat.__init__` to accept `base="int"`, look up
`_ELEMENT_ENCODERS[(base, high_bits)]` / `(base, low_bits)` and store
`self._enc_high, self._qmax_high, self._enc_low, self._qmax_low`. Raise
`ValueError("Unsupported SQ-Format cell: base=..., bits=...")` if key
missing.

Replace the `int32 round / clip` blocks in `_quantize_2d` (lines ~320-345)
and `_quantize_1d` (via `_int_quantize_pot_with_scale` helper) with:

```python
# high stream
absmax_h = np.where(mask_b, np.abs(W_b), 0.0).max(axis=0)
scale_h  = _pot_scale_vec(absmax_h, self._qmax_high)
dq_h = np.where(mask_b, self._enc_high(W_b, scale_h[np.newaxis, :]), 0.0)

# low stream — mirror with self._enc_low, self._qmax_low
```

For the 1-D path, define a thin helper `_encode_subset(vals, enc, qmax)`
that computes `scale = _pot_scale(absmax(vals), qmax)` then calls `enc`.

Run: `pytest tests/test_sq_format_base_fp.py -v`
Expected: PASS.

Run: `pytest tests/test_regression.py -v`
Expected: PASS (byte-identical — `base="int"` default unchanged).

**Step 3: Commit**

```bash
git add formats/sq_format.py tests/test_sq_format_base_fp.py
git commit -m "feat: add base={int,fp} parameter to SQFormat (Algorithm 1)"
```

### Task A.3 — `base` parameter on `SQFormatActivations` (Algorithm 2)

Mirror Task A.2 against `SQFormatActivations` (~line 417 onward). Reuse
the same `_ELEMENT_ENCODERS` table. Test with:

```python
def test_sqformat_activations_base_int_default_unchanged():
    from formats.sq_format import SQFormatActivations
    rng = np.random.default_rng(2)
    W = rng.standard_normal((128, 64)).astype(np.float32)
    A = rng.standard_normal((32, 128)).astype(np.float32)
    a = SQFormatActivations()
    b = SQFormatActivations(base="int")
    np.testing.assert_array_equal(a.quantize_weights(W, A.mean(0)),
                                  b.quantize_weights(W, A.mean(0)))


def test_sqformat_activations_base_fp_runs():
    from formats.sq_format import SQFormatActivations
    rng = np.random.default_rng(3)
    W = rng.standard_normal((128, 64)).astype(np.float32)
    A = rng.standard_normal((32, 128)).astype(np.float32)
    fmt = SQFormatActivations(base="fp", high_bits=8, low_bits=4)
    out = fmt.quantize_weights(W, A.mean(0))
    assert out.shape == W.shape
```

Run: `pytest tests/test_sq_format_base_fp.py tests/test_regression.py -v`
Expected: PASS.

```bash
git add formats/sq_format.py tests/test_sq_format_base_fp.py
git commit -m "feat: add base={int,fp} parameter to SQFormatActivations (Algorithm 2)"
```

---

## PR B · Metrics registry + config-driven columns

**Files:**
- Modify: `distributions/metrics.py` (add registries + public
  `register_metric`)
- Modify: `experiments/fourbit/config.py` (add `MetricSpec`, `metrics`,
  `tensor_stats` fields)
- Modify: `experiments/fourbit/profiler_v2.py` (loop over
  `config.metrics` instead of hard-coding columns)
- Modify: `experiments/fourbit/part1.py` (same)
- Test: `tests/test_metrics_registry.py` (new)

### Task B.1 — Metrics registries + `register_metric` entry

**Step 1: Write failing test**

```python
# tests/test_metrics_registry.py
import numpy as np
import pytest
from distributions.metrics import (
    METRIC_REGISTRY,
    TENSOR_STAT_REGISTRY,
    register_metric,
)


def test_default_metrics_present():
    for key in ("qsnr_db", "snr_db", "mse", "fp16_qsnr_db"):
        assert key in METRIC_REGISTRY


def test_default_tensor_stats_present():
    for key in ("crest", "kurtosis"):
        assert key in TENSOR_STAT_REGISTRY


def test_register_metric_pair_kind():
    register_metric("custom_mse", lambda r, q: float(np.mean((r - q) ** 2)), kind="pair")
    assert "custom_mse" in METRIC_REGISTRY


def test_register_metric_rejects_unknown_kind():
    with pytest.raises(ValueError):
        register_metric("nope", lambda *_: 0.0, kind="bogus")
```

Run: `pytest tests/test_metrics_registry.py -v`
Expected: FAIL (names not importable).

**Step 2: Implementation (`distributions/metrics.py`)**

```python
from typing import Callable, Dict


METRIC_REGISTRY: Dict[str, Callable] = {
    "qsnr_db":      qsnr_db,
    "snr_db":       snr_db,
    "mse":          mse,
    "fp16_qsnr_db": lambda ref, _q: fp16_qsnr_db(ref),   # single-tensor shim
}

TENSOR_STAT_REGISTRY: Dict[str, Callable[[np.ndarray], float]] = {
    "crest":    crest_factor,
    "kurtosis": kurtosis,
}


def register_metric(name: str, fn: Callable, kind: str = "pair") -> None:
    if kind == "pair":
        METRIC_REGISTRY[name] = fn
    elif kind == "tensor_stat":
        TENSOR_STAT_REGISTRY[name] = fn
    else:
        raise ValueError(f"kind must be 'pair' or 'tensor_stat', got {kind!r}")
```

Run tests: PASS.

Commit:

```bash
git add distributions/metrics.py tests/test_metrics_registry.py
git commit -m "feat: add metrics registry and register_metric entry"
```

### Task B.2 — `MetricSpec` in `FourBitConfig`

**Step 1: Failing test**

```python
# tests/test_metrics_registry.py (append)
def test_fourbit_config_default_metrics_match_current_columns():
    from experiments.fourbit.config import DEFAULT_CONFIG
    names = {(m.name, tuple(m.roles)) for m in DEFAULT_CONFIG.metrics}
    # Exact set must equal the four columns that profiler_v2 currently emits
    assert names == {
        ("qsnr_db",      ("W","X","Y")),
        ("fp16_qsnr_db", ("W","X","Y")),
    }
```

Run: `pytest tests/test_metrics_registry.py::test_fourbit_config_default_metrics_match_current_columns -v`
Expected: FAIL.

**Step 2: Implementation (`experiments/fourbit/config.py`)**

Add:

```python
@dataclass
class MetricSpec:
    name: str
    func: str
    roles: List[str] = field(default_factory=lambda: ["W","X","Y"])
    kind: str = "pair"
```

Add to `FourBitConfig`:

```python
metrics: List[MetricSpec] = field(default_factory=lambda: [
    MetricSpec("qsnr_db",      "qsnr_db"),
    MetricSpec("fp16_qsnr_db", "fp16_qsnr_db"),
])
tensor_stats: List[str] = field(default_factory=lambda:
    ["mean","std","min","max","crest","kurtosis"]
)
```

Run: PASS.

```bash
git add experiments/fourbit/config.py tests/test_metrics_registry.py
git commit -m "feat: add MetricSpec and metrics/tensor_stats fields to FourBitConfig"
```

### Task B.3 — `profiler_v2.analyse_layer` drives columns from config

**Step 1: Failing test**

```python
# tests/test_metrics_registry.py (append)
def test_profiler_emits_custom_metric_when_configured():
    import numpy as np
    import pandas as pd
    from experiments.fourbit.config import DEFAULT_CONFIG, MetricSpec
    from experiments.fourbit.profiler_v2 import analyse_layer
    from distributions.metrics import register_metric

    register_metric("unit_metric", lambda r, q: 42.0, kind="pair")

    cfg = DEFAULT_CONFIG.__class__(**{**DEFAULT_CONFIG.__dict__,
                                       "metrics": [MetricSpec("unit_metric", "unit_metric",
                                                              roles=["W"])]})
    rec = {"W": np.ones((8, 4), dtype=np.float32),
           "X": np.ones((4, 8), dtype=np.float32),
           "Y": np.ones((4, 4), dtype=np.float32),
           "bias": None}
    df = analyse_layer(rec, "layer0", cfg)
    assert "unit_metric_w" in df.columns
    assert (df["unit_metric_w"] == 42.0).all()
```

**Step 2: Implementation**

Refactor `analyse_layer` in `experiments/fourbit/profiler_v2.py` so rows
are built by iterating `config.metrics`:

```python
row = {"layer": layer_name, "format": ..., "transform": t_name, "reason": ""}

tensors = {"W": (W, W_q), "X": (X, X_q), "Y": (Y_ref, Y_q)}
for m in config.metrics:
    fn = METRIC_REGISTRY[m.func]
    for role in m.roles:
        ref, quant = tensors[role]
        row[f"{m.name}_{role.lower()}"] = fn(ref, quant)

for role, (tensor, _) in tensors.items():
    for key in config.tensor_stats:
        row[f"{role}_{key}"] = tensor_summary(tensor).get(key, float("nan"))
```

Preserve the **exact column order** of the current CSV so the golden
regression stays byte-identical.

Run:

```bash
pytest tests/test_metrics_registry.py -v
pytest tests/test_regression.py -v
```

Expected: PASS (all) + byte-identical goldens.

```bash
git add experiments/fourbit/profiler_v2.py
git commit -m "feat: drive profiler_v2 CSV schema from config.metrics"
```

### Task B.4 — `part1` loops over `config.metrics`

Same refactor on `experiments/fourbit/part1.py`: identify any hard-coded
`qsnr_db` column writes, replace with a `config.metrics` loop. Keep the
default output byte-identical.

Run both regression suites. Commit:

```bash
git add experiments/fourbit/part1.py
git commit -m "feat: drive part1 CSV schema from config.metrics"
```

---

## PR C · Y quantisation at GEMM output

**Files:**
- Modify: `experiments/fourbit/pipeline.py` (`Pipeline.output_fmt`,
  `simulate_linear`)
- Modify: `experiments/fourbit/accuracy.py` (`QuantLinear.forward`)
- Modify: `experiments/fourbit/config.py` (`quantize_output: bool = False`)
- Modify: `experiments/fourbit/registry.py` (propagate flag into
  pipelines)
- Test: `tests/test_pipeline_output_quant.py` (new)

### Task C.1 — `quantize_output` config flag (default False)

**Step 1: Failing test**

```python
# tests/test_pipeline_output_quant.py
import numpy as np
from experiments.fourbit.config import DEFAULT_CONFIG


def test_quantize_output_default_false():
    assert DEFAULT_CONFIG.quantize_output is False
```

**Step 2:** Add `quantize_output: bool = False` to `FourBitConfig`.
Tests pass. Commit:

```bash
git add experiments/fourbit/config.py tests/test_pipeline_output_quant.py
git commit -m "feat: add quantize_output flag to FourBitConfig (default False)"
```

### Task C.2 — `Pipeline.output_fmt` + Y quantisation

**Step 1: Failing test**

```python
def test_pipeline_output_fmt_none_keeps_y_fp32():
    import numpy as np
    from experiments.fourbit.registry import build_formats
    from experiments.fourbit.pipeline import Pipeline
    from experiments.fourbit.transforms import make_transform
    from experiments.fourbit.config import DEFAULT_CONFIG

    fmt = build_formats(DEFAULT_CONFIG)["INT4"]
    pipe = Pipeline(transform=make_transform("identity"), fmt=fmt, output_fmt=None)
    X = np.random.default_rng(0).standard_normal((8, 64)).astype(np.float32)
    W = np.random.default_rng(1).standard_normal((32, 64)).astype(np.float32)
    Y = pipe.simulate_linear(X, W)
    # FP32 accumulator semantics — Y must not be quantised
    assert Y.dtype == np.float32
    # Should match the current behaviour byte-for-byte
    pipe_legacy = Pipeline(transform=make_transform("identity"), fmt=fmt)
    np.testing.assert_array_equal(Y, pipe_legacy.simulate_linear(X, W))


def test_pipeline_output_fmt_set_quantises_y():
    import numpy as np
    from experiments.fourbit.registry import build_formats
    from experiments.fourbit.pipeline import Pipeline
    from experiments.fourbit.transforms import make_transform
    from experiments.fourbit.config import DEFAULT_CONFIG

    fmt = build_formats(DEFAULT_CONFIG)["INT4"]
    pipe_q = Pipeline(transform=make_transform("identity"), fmt=fmt, output_fmt=fmt)
    pipe_n = Pipeline(transform=make_transform("identity"), fmt=fmt, output_fmt=None)
    X = np.random.default_rng(0).standard_normal((8, 64)).astype(np.float32)
    W = np.random.default_rng(1).standard_normal((32, 64)).astype(np.float32)
    Y_q = pipe_q.simulate_linear(X, W)
    Y_n = pipe_n.simulate_linear(X, W)
    assert not np.array_equal(Y_q, Y_n)
```

**Step 2: Implementation**

`experiments/fourbit/pipeline.py`:

```python
@dataclass
class Pipeline:
    transform: Transform
    fmt: object
    output_fmt: object | None = None
    name: str = ""

    def simulate_linear(self, X, W, bias=None):
        ...
        y = X_q @ W_q.T
        y = y * self.transform.output_correction()
        if self.output_fmt is not None:
            y = self.output_fmt.quantize(y)
        if bias is not None:
            y = y + bias
        return y.astype(np.float32)
```

Update `build_pipelines` in `registry.py` to pass `output_fmt=formats[...]`
when `config.quantize_output` is True.

Run regression + new tests; PASS.

Commit.

### Task C.3 — `QuantLinear.forward` quantises Y

Mirror: `experiments/fourbit/accuracy.py::QuantLinear.__init__` stores
`self.output_fmt = pipe.output_fmt`; `forward` quantises `y` before bias
if `output_fmt` is not None.

Test: check that `accuracy_sweep` with `quantize_output=False` reproduces
current accuracy CSV byte-for-byte. Commit.

---

## PR D · QuantizedMHA + optional flag on `MNISTTransformer`

**Files:**
- Create: `examples/model_quantizable.py`
- Modify: `examples/model.py` (accept `use_quantizable_mha: bool = False`)
- Modify: `experiments/fourbit/accuracy.py` (honour flag when skipping
  MHA children)
- Modify: `experiments/fourbit/config.py` (add `use_quantizable_mha: bool = False`)
- Test: `tests/test_quantizable_mha.py` (new)

### Task D.1 — Hand-written `QuantizedMHA`

**Step 1: Failing test**

```python
# tests/test_quantizable_mha.py
import torch
from examples.model_quantizable import QuantizedMHA


def test_quantizable_mha_matches_nn_mha_in_fp32():
    torch.manual_seed(0)
    embed_dim = 32
    num_heads = 4
    batch = 5
    seq_len = 9

    # Build both with identical weights
    ref = torch.nn.MultiheadAttention(embed_dim, num_heads, bias=True, batch_first=True)
    ours = QuantizedMHA(embed_dim, num_heads, bias=True)
    ours.load_from_nn(ref)

    x = torch.randn(batch, seq_len, embed_dim)
    y_ref, _ = ref(x, x, x)
    y_ours  = ours(x)
    torch.testing.assert_close(y_ours, y_ref, atol=1e-5, rtol=1e-5)
```

**Step 2: Implementation**

```python
# examples/model_quantizable.py
import math
import torch
import torch.nn as nn


class QuantizedMHA(nn.Module):
    """Self-attention implemented with four independent nn.Linear layers.

    Drop-in replacement for nn.MultiheadAttention in the forward path of
    MNISTTransformer. Unlike nn.MultiheadAttention, every internal matmul
    goes through Module.forward, so QuantLinear can intercept them.
    """

    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def load_from_nn(self, ref: nn.MultiheadAttention) -> None:
        # in_proj_weight is (3*embed_dim, embed_dim)
        W = ref.in_proj_weight.detach()
        b = ref.in_proj_bias.detach() if ref.in_proj_bias is not None else None
        e = self.embed_dim
        self.q_proj.weight.data.copy_(W[0*e:1*e])
        self.k_proj.weight.data.copy_(W[1*e:2*e])
        self.v_proj.weight.data.copy_(W[2*e:3*e])
        if b is not None:
            self.q_proj.bias.data.copy_(b[0*e:1*e])
            self.k_proj.bias.data.copy_(b[1*e:2*e])
            self.v_proj.bias.data.copy_(b[2*e:3*e])
        self.out_proj.weight.data.copy_(ref.out_proj.weight.detach())
        if ref.out_proj.bias is not None:
            self.out_proj.bias.data.copy_(ref.out_proj.bias.detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, E = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = scores.softmax(dim=-1)
        ctx = attn @ v
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, E)
        return self.out_proj(ctx)
```

Run test: PASS.

Commit.

### Task D.2 — Wire through `MNISTTransformer` + accuracy sweep

**Step 1: Failing test**

```python
def test_mnist_transformer_accepts_quantizable_mha_flag():
    from examples.model import MNISTTransformer
    m = MNISTTransformer(use_quantizable_mha=True)
    import torch
    x = torch.randn(2, 1, 28, 28)
    y = m(x)
    assert y.shape == (2, 10)
```

**Step 2:** In `examples/model.py::MNISTTransformer.__init__`, after the
nn.TransformerEncoderLayer build, if `use_quantizable_mha` is True, walk
the encoder layers and replace each `self_attn` (nn.MultiheadAttention)
with a `QuantizedMHA` copy. Expose flag via `__init__` parameter.

In `experiments/fourbit/accuracy.py::_build_quantised_model`, keep the
current "skip MHA children" only when
`isinstance(parent, nn.MultiheadAttention)` — since `QuantizedMHA` is
not a subclass, its four child Linears (q/k/v/out) will be wrapped
normally. No explicit flag check needed here.

Run regression suites; PASS. Commit.

---

## PR E · SQ-Format factory entries + new experiment subpackage

**Files:**
- Modify: `experiments/fourbit/formats.py` (add 16 SQ-Format factory
  entries + legacy `SQFormatFP`)
- Create: `experiments/sqformat/__init__.py`
- Create: `experiments/sqformat/config.py`
- Create: `experiments/sqformat/cli.py`
- Create: `run_sqformat_study.py` (shim)
- Test: `tests/test_sqformat_smoke.py` (new)

### Task E.1 — Register SQ-Format factories

Add to `experiments/fourbit/formats.py::FORMAT_FACTORIES`:

```python
from formats.sq_format import SQFormat, SQFormatActivations, SQFormatFP

# Algorithm 1 — SQFormat
"sqformat_alg1":            SQFormat,
# Algorithm 2 — SQFormatActivations
"sqformat_alg2":            SQFormatActivations,
# Legacy FP8/INT hybrid (reference cell)
"sqformat_fp_hybrid":       SQFormatFP,
```

`FormatSpec.kwargs` supplies `{"base": ..., "high_bits": ..., "low_bits": ...}`.

Unit test: each of the 16 SQ-Format cells instantiates successfully.

### Task E.2 — `SQFormatConfig` with all 16 cells

```python
# experiments/sqformat/config.py
from experiments.fourbit.config import FourBitConfig, FormatSpec, TransformSpec, MetricSpec


def _cell(alg: str, base: str, hi: int, lo: int) -> FormatSpec:
    algo_factory = {"alg1": "sqformat_alg1", "alg2": "sqformat_alg2"}[alg]
    name = f"SQ-{base.upper()}-{hi}{lo}-{alg.upper()}"
    return FormatSpec(name, algo_factory, kwargs={"base": base, "high_bits": hi, "low_bits": lo})


INT_PAIRS = [(8, 8), (4, 8), (4, 4), (4, 2), (2, 2)]
FP_PAIRS  = [(8, 8), (4, 8), (4, 4)]

SQFORMAT_CELLS = (
    [_cell("alg1", "int", hi, lo) for hi, lo in INT_PAIRS] +
    [_cell("alg1", "fp",  hi, lo) for hi, lo in FP_PAIRS]  +
    [_cell("alg2", "int", hi, lo) for hi, lo in INT_PAIRS] +
    [_cell("alg2", "fp",  hi, lo) for hi, lo in FP_PAIRS]  +
    # Legacy hybrid reference
    [FormatSpec("SQ-FP8-INT4-hybrid", "sqformat_fp_hybrid", kwargs={"low_bits": 4})]
)


SQ_METRICS = [
    MetricSpec("qsnr_db",      "qsnr_db"),
    MetricSpec("snr_db",       "snr_db"),
    MetricSpec("mse",          "mse"),
    MetricSpec("fp16_qsnr_db", "fp16_qsnr_db"),
]


DEFAULT_CONFIG = FourBitConfig(
    formats=SQFORMAT_CELLS,
    transforms=[
        TransformSpec("base",   "identity"),
        TransformSpec("smooth", "smoothquant", kwargs={"alpha": 0.5}),
        TransformSpec("had",    "hadamard"),
    ],
    metrics=SQ_METRICS,
    quantize_output=True,
    use_quantizable_mha=True,
    output_dir="results/sqformat",
    profile_samples=128,  # smaller for run-time budget
)
```

### Task E.3 — `experiments/sqformat/cli.py`

Copy the shape of `experiments/fourbit/cli.py`. Reuse `part1.run_all`
and `part2.run` (they accept `FourBitConfig` subclasses). Pass
`DEFAULT_CONFIG` from this module.

```python
# experiments/sqformat/cli.py
import argparse, os, sys
from experiments.sqformat.config import DEFAULT_CONFIG
from experiments.fourbit import part1, part2, reporter


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--part", choices=["1","2","all"], default="all")
    p.add_argument("--out", default=None)
    p.add_argument("--profile-samples", type=int, default=None)
    args = p.parse_args(argv)

    cfg = DEFAULT_CONFIG
    if args.out:
        cfg = DEFAULT_CONFIG.__class__(**{**DEFAULT_CONFIG.__dict__, "output_dir": args.out})
    if args.profile_samples is not None:
        cfg = cfg.__class__(**{**cfg.__dict__, "profile_samples": args.profile_samples})

    os.makedirs(cfg.output_dir, exist_ok=True)
    part1_res = part1.run_all(cfg) if args.part in ("1","all") else None
    part2_df = None
    acc_df = None
    if args.part in ("2","all"):
        part2_df, _, acc_df = part2.run(cfg)

    if args.part == "all" and part1_res is not None and part2_df is not None:
        reporter.generate_report(cfg, part1_res, part2_df,
                                 os.path.join(cfg.output_dir, "report.md"),
                                 accuracy_df=acc_df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### Task E.4 — Top-level shim

```python
# run_sqformat_study.py
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments.sqformat.cli import main

if __name__ == "__main__":
    sys.exit(main())
```

### Task E.5 — Smoke test

```python
# tests/test_sqformat_smoke.py
import subprocess, sys, os


def test_sqformat_part1_smoke(tmp_path):
    out = tmp_path / "sqformat"
    res = subprocess.run(
        [sys.executable, "run_sqformat_study.py",
         "--part", "1", "--out", str(out)],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        capture_output=True, text=True, timeout=300,
    )
    assert res.returncode == 0, res.stderr
    assert (out / "part1").is_dir()
```

Run `pytest tests/test_sqformat_smoke.py -v` — PASS.

Commit.

---

## PR F · Golden fixture + regression protection for SQ-Format Part 1

### Task F.1 — Generate canonical SQ-Format Part 1 CSVs

Run:

```bash
rm -rf results/sqformat
python run_sqformat_study.py --part 1 --out results/sqformat
```

Copy the deterministic CSVs:

```bash
mkdir -p tests/fixtures/golden
cp results/sqformat/part1/sqnr_by_dist.csv \
   tests/fixtures/golden/sqformat_part1_sqnr_by_dist.csv
cp results/sqformat/part1/encoding_overhead.csv \
   tests/fixtures/golden/sqformat_part1_encoding_overhead.csv
```

### Task F.2 — Add byte-identical regression test

Append to `tests/test_regression.py`:

```python
def test_sqformat_part1_sqnr_by_dist_golden(tmp_path):
    out = tmp_path / "sqformat"
    res = subprocess.run(
        [sys.executable, "run_sqformat_study.py",
         "--part", "1", "--out", str(out)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600,
    )
    assert res.returncode == 0, res.stderr
    _assert_csv_byte_identical(
        out / "part1" / "sqnr_by_dist.csv",
        REPO_ROOT / "tests/fixtures/golden/sqformat_part1_sqnr_by_dist.csv",
    )


def test_sqformat_part1_encoding_overhead_golden(tmp_path):
    ...  # same shape
```

Run: `pytest tests/test_regression.py -v` — all five tests pass.

Commit fixtures + test together.

---

## Final verification checklist

After PR F:

```bash
pytest tests/ -q                                        # full suite
pytest tests/test_regression.py -v                      # 5 regression tests pass
python run_4bit_study.py --part 1                       # 4-bit CLI still runs
python run_sqformat_study.py --part all                 # full SQ-Format run
```

The generated `results/sqformat/report.md` should contain all five
top-level report sections (§4.3 of the design doc).

# Model Profiler Design

**Date:** 2026-04-12  
**Goal:** Add a lightweight PyTorch model profiler that captures per-layer tensor distributions across 14 quantization formats, with zero intrusion into existing model code.

---

## Problem

The existing library compares quantization formats on synthetic distributions (numpy arrays). We need to extend this to **real PyTorch model inference**: capture actual weight, input activation, and output activation tensors from each layer, analyze their distributions, and evaluate quantization quality for all target formats.

---

## Architecture

New `profiler/` directory, parallel to existing `formats/`, `experiments/`:

```
profiler/
├── __init__.py
├── profiler.py    # ModelProfiler: start/stop/wrap/export
├── stats.py       # RunningHistogram + WelfordStats (online statistics)
├── formats.py     # PROFILER_FORMATS format schedule
└── export.py      # CSV export aligned with ExperimentRunner
```

No changes to existing code.

---

## Usage

```python
from profiler import ModelProfiler

profiler = ModelProfiler(model)

while not profiler.done:
    profiler.start()
    for batch in loader:
        model(batch)          # normal inference, zero intrusion
    profiler.stop()

profiler.export_csv("results/")

# For functional ops (optional, one-line change):
# torch.matmul(a, b)  →  profiler.wrap(torch.matmul, 'matmul')(a, b)
```

---

## Data Collection

**Hook registration** via `register_forward_pre_hook` and `register_forward_hook` on all leaf `nn.Module` instances. Default targets: all leaf modules. Configurable via `target_layers` (list of types or names).

**Per forward pass, per hooked layer:**

1. `pre_hook` → capture `input[0]` (input activation) + `module.weight` (weight, first batch only — static)
2. `post_hook` → capture `output`
3. For each tensor type (`weight`, `input`, `output`):
   - Update **RunningHistogram** (256 fixed bins, accumulated across batches) → KL divergence, distribution shape, outlier ratio
   - Update **Welford online stats** (mean, variance, max_ae) → SNR, EffBits
   - Apply current format quantization → accumulate `sum_sq_err` / `n_elements` → MSE

**Memory cost:** `O(layers × 3 tensor types × 256 bins)` — independent of batch count and model size.

**Functional ops:** Not hookable via standard `nn.Module` hooks. Solution: `profiler.wrap(fn, name)` wraps any callable. User replaces `torch.matmul(a, b)` with `profiler.wrap(torch.matmul, 'matmul')(a, b)` — one line change, no other modification needed.

---

## Format Schedule

14 formats, all reusing existing format objects from `build_all_formats()`:

| Format Name      | Existing Object                                      | Bits |
|------------------|------------------------------------------------------|------|
| FP32             | `FP32Format()`                                       | 32   |
| FP16             | `BF16Format()` (simulated FP16)                      | 16   |
| SQ-FORMAT-INT    | `SQFormat(dense_bits=4, sparse_bits=8)`              | 4/8  |
| SQ-FORMAT-FP     | `SQFormatActivations(bank_size=128, sparsity=0.5)`   | 4/8  |
| INT4(CHANNEL)    | `_POTINTQuantizer(4, per_channel=True)`              | 4    |
| INT8(CHANNEL)    | `_POTINTQuantizer(8, per_channel=True)`              | 8    |
| INT4(TENSOR)     | `_POTINTQuantizer(4, per_channel=False)`             | 4    |
| INT8(TENSOR)     | `_POTINTQuantizer(8, per_channel=False)`             | 8    |
| HAD+INT4(C)      | `ComposedFormat(had, _POTINTQuantizer(4, True))`     | 4    |
| HAD+INT8(C)      | `ComposedFormat(had, _POTINTQuantizer(8, True))`     | 8    |
| HAD+INT4(T)      | `ComposedFormat(had, _POTINTQuantizer(4, False))`    | 4    |
| HAD+INT8(T)      | `ComposedFormat(had, _POTINTQuantizer(8, False))`    | 8    |
| MXINT4           | `MXINTFormat(element_bits=4)`                        | 4    |
| MXINT8           | `MXINTFormat(element_bits=8)`                        | 8    |

No new format implementations needed.

---

## Online Statistics

### WelfordStats
Incremental mean and variance (Welford's algorithm), running max absolute error, element count.

```
update(x: np.ndarray)    # add one batch
finalize() -> dict       # returns mean, std, max_ae, n_elements
```

### RunningHistogram
Fixed 256-bin histogram. First batch sets the bin range; subsequent batches use the same range (clamp outliers into edge bins).

```
update(x: np.ndarray)
merge(other: RunningHistogram)   # combine across formats if needed
finalize() -> dict               # returns bin counts, outlier_ratio
```

### QuantStats
Per-format quantization error accumulator.

```
update(x_orig, x_quant)   # accumulates sum_sq_err, n_elements, max_ae
finalize() -> dict         # returns mse, snr_db, eff_bits, max_ae
```

---

## CSV Output

One row per `(format, layer_name, tensor_type)` triple. Fields aligned with `ExperimentRunner`:

```
format, layer_name, layer_type, tensor_type, bits,
mse, snr_db, kl_div, max_ae, eff_bits,
mean, std, outlier_ratio,
n_batches, n_elements
```

- `tensor_type` ∈ `{weight, input, output}`
- `layer_name` = PyTorch `named_modules()` path (e.g. `transformer.h.0.attn.c_proj`)
- Compatible with existing visualization scripts

---

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Hook mechanism | `register_forward_hook` | Zero model intrusion, fully reversible |
| Functional ops | `wrap_functional` utility | One-line user change, no global monkey-patching |
| Statistics | Welford + RunningHistogram | Fixed memory, correct incremental updates |
| Weight capture | First batch only | Weights are static, no need to re-capture |
| Format reuse | Existing `build_all_formats()` objects | No new code, consistent with research |
| Output format | CSV aligned with ExperimentRunner | Directly usable by existing visualization |

---

## Out of Scope

- Gradient/backward pass profiling
- Multi-GPU distributed inference
- learnable rotation transforms (SpinQuant)
- Automatic torch.fx tracing (dynamic models not supported)

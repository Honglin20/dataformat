# Quantization Format Research — Data Format Study

Benchmarks quantization formats for deep learning across common weight/activation
distributions. Covers INT4/8, MXINT4/8, MXFP4/8, HAD+INT, and SQ-Format
(Alg 1 / Alg 2, INT and FP bases), evaluated on 8 distribution families
(24 variants). Dedicated studies: 4-bit (`run_4bit_study.py`) and
17-cell SQ-Format matrix (`run_sqformat_study.py`).

> Extended analysis: see `analysis.md`

---

## 4-bit Format Study (`fourbit/`)

A self-contained, config-driven study of six 4-bit formats (**INT4, FP4, NF4,
NVFP4, MXINT4, MXFP4**) under three transforms (**base, smooth, had**). Code
lives under `fourbit/`; the entry point is `run_4bit_study.py`.

### Running

```bash
# Full study (Part 1 synthetic + Part 2 real MNIST Transformer)
python run_4bit_study.py

# Part 1 only (no model required)
python run_4bit_study.py --part 1

# Part 2 only (loads or trains results/mnist/model.pt)
python run_4bit_study.py --part 2

# Custom output directory
python run_4bit_study.py --out results/fourbit_custom

# Custom MNIST checkpoint / data cache
python run_4bit_study.py --model-path path/to/model.pt --data-dir ~/.cache/mnist
```

### Outputs (`results/fourbit/`)

| Path | Description |
|------|-------------|
| `part1/exp11_direct_quant.csv` | 1.1 — direct quantization of common distributions |
| `part1/exp12_linear_wa.csv` | 1.2 — base-quantized linear Y = X Wᵀ on typical W/X pairs |
| `part1/exp13_smooth_transforms.csv` | 1.3 — base/smooth/had on smooth-friendly pairs |
| `part2/per_layer_metrics.csv` | flat per-layer × format × transform QSNR + tensor stats |
| `figures/scatter_{input,output,weight}_{base,smooth,had}.png` | crest-factor vs QSNR scatter |
| `report.md` | full markdown report (Part 1 tables + Part 2 figures/tables) |

### Adding / removing formats

All experiment code iterates over `fourbit/config.py → DEFAULT_CONFIG.formats`.
To add or remove a format, edit that list — no other file needs to change,
provided the format key is registered in `fourbit/formats.py → FORMAT_FACTORIES`:

```python
# fourbit/config.py
DEFAULT_CONFIG = FourBitConfig(
    formats=[
        FormatSpec(display_name="INT4",   factory="int4_per_channel"),
        FormatSpec(display_name="FP4",    factory="fp4_per_channel"),
        FormatSpec(display_name="MyFmt4", factory="my_format_key"),  # ← add here
        ...
    ],
    transforms=[
        TransformSpec(display_name="base",   factory="identity"),
        TransformSpec(display_name="smooth", factory="smoothquant", kwargs={"alpha": 0.5}),
        TransformSpec(display_name="had",    factory="hadamard"),
    ],
    ...
)
```

Transforms work the same way — add an entry to `TRANSFORM_FACTORIES` in
`experiments/fourbit/transforms.py` and a `TransformSpec` in the config.

---

## Using the Pipeline with a Custom Model

Both the 4-bit and SQ-Format studies expose a model-agnostic Python API.
You can profile and sweep **any** trained `nn.Module` — no MNIST dependency.

### Minimal example (layer-level QSNR only)

```python
import torch
from torch.utils.data import DataLoader, TensorDataset
from experiments.fourbit.config import DEFAULT_CONFIG
from experiments.fourbit.part2 import profile_model

# 1. Load your model
model = torch.load("my_model.pt").eval()

# 2. Build any DataLoader — only the first element of each batch is used
ds = TensorDataset(torch.randn(256, 512))   # (samples, in_features)
loader = DataLoader(ds, batch_size=32)

# 3. Run the pipeline  (writes results/fourbit/part2/per_layer_metrics.csv)
metrics_df, layers, _ = profile_model(DEFAULT_CONFIG, model, loader)
print(metrics_df[["layer", "format", "transform", "qsnr_y_db"]].head())
```

### With an accuracy / quality sweep

Supply an `eval_fn(model, loader) -> float` to also produce
`accuracy_sweep.csv`:

```python
def my_eval_fn(m, ldr):
    m.eval()
    total = correct = 0
    with torch.no_grad():
        for (x,), y in zip(ldr, labels):
            logits = m(x)
            correct += (logits.argmax(1) == y).sum().item()
            total   += y.numel()
    return correct / total

metrics_df, layers, acc_df = profile_model(
    DEFAULT_CONFIG, model, loader, eval_fn=my_eval_fn
)
print(acc_df)
```

### SQ-Format study on a custom model

```python
from experiments.sqformat.config import DEFAULT_CONFIG as SQ_CONFIG
from experiments.fourbit.part2 import profile_model

metrics_df, layers, acc_df = profile_model(
    SQ_CONFIG, model, loader, eval_fn=my_eval_fn
)
# outputs go to results/sqformat/part2/
```

### Output files

| File | Description |
|------|-------------|
| `<output_dir>/part2/per_layer_metrics.csv` | One row per layer × format × transform; QSNR for W/X/Y and tensor stats |
| `<output_dir>/part2/accuracy_sweep.csv` | FP32 / FP16 baselines + per-(format, transform) score (only when `eval_fn` is provided) |

`output_dir` is `config.output_dir` (e.g. `results/fourbit` or `results/sqformat`).

---

## SQ-Format Study (`experiments/sqformat/`)

17-cell SQ-Format comparison matrix (Alg 1 / Alg 2 × {INT, FP} bases × 5 INT / 3
FP bit-split pairs + one legacy FP-hybrid cell), reusing the 4-bit study
Part 1/Part 2 infrastructure with two toggles flipped on:

- `quantize_output=True` — `simulate_linear` also quantises Y (transposed so the
  bank axis aligns with `out_features`; the cell auto-adapts `bank_size` when
  `out_features < bank_size`).
- `use_quantizable_mha=True` — MNIST-Transformer attention is rewritten as
  four `nn.Linear`s (`QuantizedMHA`) so Q/K/V/out_proj are visible to
  `QuantLinear`. Requires a model trained (or re-loadable) under the flag.

```bash
python run_sqformat_study.py --part 1
python run_sqformat_study.py --part 2 --model-path results/mnist/model.pt
python run_sqformat_study.py --profile-samples 256   # override default 128
```

Outputs mirror the 4-bit study at `results/sqformat/`. Regression protection:
`tests/test_regression.py::test_sqformat_part1` pins the three Part-1 CSVs.

---

## Project Structure

```
dataformat/
├── config.py                        # Global constants (block size, bit-widths, energy)
├── run_all.py                       # Master pipeline (all phases)
├── run_4bit_study.py                # Shim → experiments.fourbit.cli.main
├── run_sqformat_study.py            # Shim → experiments.sqformat.cli.main
├── generate_qsnr_table.py           # Shim → utils.qsnr_table.main
│
├── formats/                         # Canonical format primitives
│   ├── __init__.py                  # Format registry (build_all_formats), _pot_scale alias
│   ├── _pot.py                      # Canonical POT scale helpers (floor + ceil variants)
│   ├── baseline.py                  # FP32 / BF16 / FP16
│   ├── mxint.py                     # MXINT4 / MXINT8
│   ├── mxfp.py                      # MXFP4 / MXFP8
│   ├── sq_format.py                 # SQ-Format (sparse-quantized)
│   ├── int_variants.py              # INT4-FP / APoT4 / LOG4 / NF4-FP8 (was in fourbit)
│   └── transforms/
│       ├── hadamard.py              # HAD transform (normalize=False)
│       └── smoothquant.py           # SmoothQuant baseline
│
├── distributions/
│   ├── generators.py                # Gaussian, Laplace, bimodal, outlier, …
│   ├── linear_pairs.py              # Paired (weight, activation) generators
│   └── metrics.py                   # SQNR, MSE, QSNR, FP16 baseline, crest, tensor_summary
│
├── profiler/                        # PyTorch runtime profiler (streaming hooks)
│   ├── profiler.py                  # ModelProfiler (forward-hook based)
│   └── stats.py                     # WelfordStats / RunningHistogram / QuantStats
│
├── utils/
│   └── qsnr_table.py                # HTML QSNR table from results/exp1/*.csv
│
├── experiments/
│   ├── defaults.py                  # ← Edit here to add formats / distributions
│   ├── config.py                    # ExperimentConfig / FormatGroup dataclasses
│   ├── robustness.py                # Phase 2: distribution robustness sweep
│   ├── bitwidth_ablation.py         # Phase 3: 4-bit vs 8-bit ablation
│   ├── exp1_common_distributions.py # Standalone Exp 1 (9 formats × 24 distributions)
│   ├── exp2_crest_factor.py         # Standalone Exp 2 (SQNR vs crest factor)
│   ├── fourbit/                     # 4-bit data format study (Part 1 + Part 2)
│   │   ├── cli.py                   # Entry point (called by run_4bit_study.py shim)
│   │   ├── config.py / registry.py / pipeline.py
│   │   ├── part1.py / part2.py / accuracy.py
│   │   ├── reporter.py / profiler_v2.py
│   │   ├── formats.py / transforms.py
│   │   └── distribution_sets.py     # Curated DistSpec / LinearSpec lists
│   └── sqformat/                    # SQ-Format study (17 cells, reuses fourbit infra)
│       ├── cli.py                   # Entry point (called by run_sqformat_study.py shim)
│       └── config.py                # 17-cell DEFAULT_CONFIG (quantize_output + QuantizedMHA on)
│
├── examples/
│   ├── train_mnist.py               # Train MNISTTransformer
│   ├── profile_mnist.py             # Profile all formats on trained model
│   ├── generate_report.py           # Build HTML report from profiler output
│   ├── model.py                     # MNISTTransformer (use_quantizable_mha flag)
│   └── model_quantizable.py         # QuantizedMHA: nn.MHA drop-in with visible q/k/v/out Linears
│
├── visualization/                   # Figure generators (called by run_all.py Phase 5)
│
├── tests/
│   ├── test_regression.py           # Golden CLI regression (exp1/exp2/fourbit/sqformat part1 = 5)
│   ├── test_pot_scale_equivalence.py
│   ├── test_sq_format_base_fp.py    # Element-encoder registry + base={int,fp} semantics
│   ├── test_pipeline_output_quant.py # Y quantisation + R2 auto-adapt for SQ-Format
│   ├── test_quantizable_mha.py      # QuantizedMHA ≈ nn.MultiheadAttention in FP32
│   ├── test_sqformat_smoke.py test_metrics_registry.py
│   ├── fixtures/golden/             # Committed reference CSVs for regression
│   └── …                            # unit tests for each format/distribution
│
└── results/                         # All output (git-ignored large files)
    ├── robustness.csv
    ├── ablation_4bit.csv / ablation_8bit.csv
    ├── qsnr_summary.html            # Cross-format SQNR table (all distributions)
    ├── exp1/
    │   ├── results_4bit.csv         # Raw SQNR / MSE / KL / MaxAE per format × dist
    │   ├── results_8bit.csv
    │   └── fig1_heatmap_*.png … fig6_best_format_*.png
    ├── figures/                     # Figures from run_all.py Phase 5
    └── mnist/
        ├── model.pt
        ├── profiler_results.csv
        └── report.html
```

---

## Quick Start

### 1. Experiment 1 — SQNR across distributions (standalone)

Runs 9 quantization formats against 24 distribution variants. Fastest entry point.

```bash
# Both 4-bit and 8-bit (default)
python experiments/exp1_common_distributions.py

# 4-bit only
python experiments/exp1_common_distributions.py --bits 4

# 8-bit only
python experiments/exp1_common_distributions.py --bits 8
```

**Outputs** → `results/exp1/`

| File | Description |
|------|-------------|
| `results_4bit.csv` | SQNR / MSE / KL-div / MaxAE per format × distribution |
| `results_8bit.csv` | Same for 8-bit |
| `fig1_heatmap_{4,8}bit.png` | SQNR heatmap (format × distribution) |
| `fig2_family_bars_{4,8}bit.png` | Per-family bar chart |
| `fig3_gaussian_sweep_{4,8}bit.png` | Gaussian σ sweep |
| `fig4_outlier_sweep_{4,8}bit.png` | Channel outlier σ sweep |
| `fig5_overall_ranking_{4,8}bit.png` | Average ranking |
| `fig6_best_format_{4,8}bit.png` | Best format per distribution |

### 2. SQNR Summary HTML table

Reads `results/exp1/results_{4,8}bit.csv` and produces a color-coded interactive table.

```bash
python generate_qsnr_table.py
```

**Output** → `results/qsnr_summary.html`  
Open in any browser. Color scale: red (low SQNR) → green (high SQNR). Includes 4-bit and 8-bit sections, grouped by distribution family, with per-format average column.

### 3. Full pipeline

```bash
# Full run (all phases, all figures)
python run_all.py

# Skip hardware PPA evaluation (faster, no PyRTL required)
python run_all.py --skip-hw

# Quick smoke-test with minimal config (N=512, 5 formats)
python run_all.py --fast

# Regenerate figures only from existing CSVs (no re-running experiments)
python run_all.py --figs-only

# Hardware-focus experiment (MXINT / HAD+INT / SQ-Format only)
python run_all.py --hw-focus
```

**Pipeline phases:**

| Phase | Description | Output |
|-------|-------------|--------|
| 1 | Format smoke-test | console |
| 2 | Distribution robustness sweep | `results/robustness.csv` |
| 3 | Bitwidth ablation (4b vs 8b) | `results/ablation_{4,8}bit.csv` |
| 4 | Hardware PPA evaluation | console / hardware models |
| 5 | Generate 13 figures (Fig 1–13) | `results/figures/*.png` |

### 4. MNIST end-to-end example

```bash
# Step 1: Train
python examples/train_mnist.py [--epochs 10] [--batch-size 256] [--out-dir results/mnist]

# Step 2: Profile all formats
python examples/profile_mnist.py [--model-path results/mnist/model.pt] \
                                  [--out-dir results/mnist] \
                                  [--n-samples 256]

# Step 3: Generate HTML report
python examples/generate_report.py [--results-dir results/mnist]
```

**Outputs** → `results/mnist/`

| File | Description |
|------|-------------|
| `model.pt` | Trained weights |
| `training_log.json` | Per-epoch loss / accuracy |
| `profiler_results.csv` | Per-layer SQNR for all formats |
| `profiler_histograms.json` | Weight/activation histograms (optional) |
| `report.html` | Self-contained HTML report (opens automatically) |

---

## Profiler API

`ModelProfiler` (`profiler/profiler.py`) hooks into any PyTorch model non-intrusively and records per-layer tensor statistics (mean, std, SNR, EffBits, outlier ratio) for each of 14 quantization formats.

### Basic usage

```python
from profiler import ModelProfiler

profiler = ModelProfiler(model)           # wraps all leaf modules

while not profiler.done:
    profiler.start()                      # install hooks for current format
    with torch.no_grad():
        for x, _ in dataloader:
            model(x)                      # runs forward pass; hooks capture tensors
    profiler.stop()                       # remove hooks, advance to next format

csv_path = profiler.export_csv("results/")
```

### Context manager (single format)

```python
profiler = ModelProfiler(model, formats=[("INT4(C)", my_fmt)])
with profiler:
    model(x)
profiler.stop()
```

### Targeting specific layer types

```python
profiler = ModelProfiler(model, target_layers=[nn.Linear])
```

### Profiling functional ops

```python
output = profiler.wrap(torch.matmul, "attn_qk")(q, k)
```

### Output schema — `profiler_results.csv`

| Column | Description |
|--------|-------------|
| `format` | Format name (e.g. `HAD+INT4(C)`) |
| `layer_name` | Named module path (e.g. `encoder.0.fc1`) |
| `layer_type` | PyTorch class name (e.g. `Linear`) |
| `tensor_type` | `weight`, `input`, or `output` |
| `bits` | Nominal bit-width |
| `snr_db` | Signal-to-quantization-noise ratio (dB) |
| `eff_bits` | Effective bits = 0.5 log₂(signal\_var / MSE) |
| `mse` | Mean squared quantization error |
| `max_ae` | Maximum absolute error per layer |
| `mean` / `std` | Raw tensor statistics (pre-quantization) |
| `outlier_ratio` | Fraction of elements outside the initial histogram range |

### Generating the HTML report

```bash
python examples/generate_report.py --results-dir results/mnist
```

The report includes 11 sections: training curves, FP32 distributions, outlier analysis, linear vs non-linear SQNR gap, per-layer sensitivity heatmaps (weight / input / output), format efficiency scatter, EffBits ranking, SNR comparison, and a full summary table with per-tensor-type SNR breakdown.

When a model has more than 30 layers, sensitivity heatmaps automatically select the 30 most sensitive layers (highest cross-format SNR variance) and label the figure accordingly.

---

## Configuration

### Global constants — `config.py`

| Constant | Default | Description |
|----------|---------|-------------|
| `MX_BLOCK_SIZE` | `32` | OCP MX block size (elements per scale) |
| `N_SAMPLES` | `4096` | Tensor size for distribution tests |
| `RANDOM_SEED` | `42` | Global RNG seed |
| `BITWIDTHS` | `[4, 8]` | Bit-widths under test |

### Exp 1 constants — top of `experiments/exp1_common_distributions.py`

| Constant | Default | Description |
|----------|---------|-------------|
| `N_SAMPLES` | `4096` | Elements per distribution sample |
| `BANK_SIZE` | `128` | SQ-Format bank size |
| `SPARSITY` | `0.5` | SQ-Format sparsity (fraction of low-prec elements) |
| `SEED` | `42` | RNG seed |

Per-channel group size is derived as `N_SAMPLES // 64` (simulates 64 output channels).

### Adding formats or distributions — `experiments/defaults.py`

This is the **only file** that needs editing to extend the pipeline:

```python
# Add a new format name to an existing FormatGroup
FOCUS_4BIT = FormatGroup("4-bit", [
    "INT4", "MXINT4", ..., "MyNewFormat4",   # ← add here
])

# Add a new distribution
ABLATION_DISTRIBUTIONS.append(DistributionConfig(
    "MyDist",
    lambda n, s: my_generator(n, seed=s),
    tags=["custom"],
))
```

The format must also be registered in `formats/__init__.py → build_all_formats()`.

---

## Formats

| Label | Class / Key | Granularity | Notes |
|-------|-------------|-------------|-------|
| `FP16` | `FP16Format` | — | Reference baseline |
| `INT4(TENSOR)` / `INT8(TENSOR)` | `_POTINTQuantizer(per_channel=False)` | 1 scale / tensor | POT scale |
| `INT4(CHANNEL)` / `INT8(CHANNEL)` | `_POTINTQuantizer(per_channel=True)` | 1 scale / 64 elem | Simulates per-output-channel |
| `MXINT4` / `MXINT8` | `MXINTFormat` | 1 scale / 32 elem | OCP MX E8M0 block scale |
| `MXFP4` / `MXFP8` | `MXFPFormat` | 1 scale / 32 elem | OCP MX floating-point |
| `HAD+INT4(C)` / `HAD+INT8(C)` | `_HADQuantizer(per_channel)` | HAD + 64 elem | Hadamard + per-channel INT |
| `HAD+INT4(T)` / `HAD+INT8(T)` | `_HADQuantizer(per_tensor)` | HAD + tensor | Hadamard + per-tensor INT |
| `SQ-FORMAT-INT` | `SQFormat` | bank=128, s=0.5 | 50% high-prec INT per bank |
| `SQ-FORMAT-FP` | `SQFormatFP` | bank=128, s=0.5 | 50% high-prec FP per bank |

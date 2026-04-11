# MNIST Transformer Analysis Design

**Date:** 2026-04-12  
**Goal:** Train a minimal Transformer on MNIST, profile it across 14 quantization formats using ModelProfiler, and generate an HTML accuracy report.

---

## Architecture

Three independent scripts in `examples/`, each re-runnable without re-running the others.

```
examples/
├── train_mnist.py        # Step 1: train, save model.pt + training_log.json
├── profile_mnist.py      # Step 2: load model.pt, run 14-format profiling
└── generate_report.py    # Step 3: read CSV, generate report.html

results/mnist/
├── model.pt              # trained weights
├── training_log.json     # per-epoch loss/acc curves
├── profiler_results.csv  # 14 formats × all layers × 3 tensor types
└── report.html           # final HTML report
```

---

## Model: MNISTTransformer

```
Input: MNIST image (28×28 = 784 pixels, flattened)
  → Linear embedding (784 → 128)
  → Reshape to sequence: 28 tokens × 128 dims
  → Prepend learnable [CLS] token → sequence length 29
  → Learnable positional encoding (29 × 128)
  → 2× TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, dropout=0.1)
  → CLS token output (128,)
  → Linear classifier (128 → 10)
```

~300k parameters. Target: ~97% test accuracy. CPU training: ~3-5 min (10 epochs).

---

## Step 1: train_mnist.py

- Dataset: torchvision MNIST, full 60k train / 10k test
- Optimizer: AdamW, lr=1e-3, weight_decay=1e-4
- Schedule: CosineAnnealingLR (T_max=10)
- Batch size: 256, epochs: 10
- Per-epoch output: `Epoch 5/10 | loss=0.082 | train_acc=97.5% | test_acc=97.2%`
- Saves: `results/mnist/model.pt`, `results/mnist/training_log.json`
- `training_log.json` schema: `{epoch: [1..10], train_loss: [...], test_loss: [...], train_acc: [...], test_acc: [...]}`

---

## Step 2: profile_mnist.py

- Loads `results/mnist/model.pt`, sets model to eval()
- Randomly samples `--n-samples 256` images from MNIST test set (default 256, configurable)
- Creates `ModelProfiler(model)`, loops all 14 formats:
  ```
  Profiling format  1/14: FP32 ...
  Profiling format  2/14: FP16 ...
  ...
  Profiling format 14/14: MXINT8 ... done
  ```
- Saves: `results/mnist/profiler_results.csv`
- Prints summary on completion: total rows, formats covered

---

## Step 3: generate_report.py

Reads `profiler_results.csv` + `training_log.json`, writes `results/mnist/report.html`.

### Report sections (using matplotlib → base64 inline PNG, no external dependencies):

1. **Training curves** — dual-axis: loss (left) + accuracy (right) vs epoch
2. **EffBits ranking** — horizontal bar chart, formats sorted by mean EffBits, 4-bit / 8-bit color-coded
3. **Per-layer MSE heatmap** — x=layer_name, y=format, color=log10(MSE), tensor_type=weight
4. **SNR comparison** — grouped bar: mean SNR per format, grouped by tensor_type (weight/input/output)
5. **Summary table** — HTML table: format | bits | mean_mse | mean_snr_db | mean_eff_bits | outlier_ratio, sorted by eff_bits desc

### HTML structure:
- Self-contained single file (all images inline as base64)
- Minimal CSS, no external CDN dependencies
- Sections separated by `<hr>`, auto-opens in default browser on completion

---

## Key Design Decisions

| Decision | Choice | Reason |
|---|---|---|
| Three separate scripts | ✓ | Training takes minutes; report regeneration should be instant |
| n-samples configurable | `--n-samples 256` | Balance between coverage and speed |
| Self-contained HTML | base64 inline images | No server needed, email-safe |
| matplotlib only | No plotly/bokeh | Already in requirements.txt |
| Profiling in eval mode | `model.eval()` | Disable dropout for deterministic activations |

---

## Out of Scope

- Training from the profiling script (always loads pre-trained weights)
- Per-class accuracy breakdown
- Quantization-aware training
- GPU support (CPU sufficient for this model size)

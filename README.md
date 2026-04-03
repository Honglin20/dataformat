# Outlier-Robust Quantization: Hardware-Native Formats vs. Transform-Based Methods

> **Research Question**: For handling neural network weight/activation outliers, should hardware use *complex native formats* (MX/NVFP) or *simple base formats* (INT) + *mathematical transforms* (HAD/Rotation)?

This study evaluates 22 quantization formats across three orthogonal dimensions — **information theory**, **numerical accuracy**, and **hardware implementation cost** — using PyRTL microarchitectural modelling and analytical energy/area models.

---

## Table of Contents

1. [Format Taxonomy](#format-taxonomy)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Running Tests](#running-tests)
5. [Research Findings](#research-findings)
6. [Figures](#figures)
7. [Hardware Comparison](#hardware-comparison)
8. [Conclusions](#conclusions)
9. [Project Structure](#project-structure)

---

## Format Taxonomy

### Hardware-Native Formats

| Format | Bits | Key Feature |
|--------|------|-------------|
| FP32 | 32 | IEEE 754 baseline |
| BF16 | 16 | Brain float, wide exponent |
| INT8 | 8 | Plain symmetric integer |
| INT4 | 4 | Plain symmetric integer |
| MXFP8 | 8+0.25 | OCP Microscaling FP, Block=32, E4M3/E5M2, shared E8M0 exponent |
| MXFP4 | 4+0.25 | OCP Microscaling FP, Block=32, E2M1, shared E8M0 exponent |
| MXINT8 | 8+0.25 | OCP Microscaling INT, Block=32, shared E8M0 scale |
| MXINT4 | 4+0.25 | OCP Microscaling INT, Block=32, shared E8M0 scale |
| NVFP4 | 4 | NVIDIA Blackwell E2M1, 8 positive levels: {0, 0.5, 1, 1.5, 2, 3, 4, 6} |
| NF4 | 4 | QLoRA NormalFloat — 16 levels at N(0,1) quantiles, information-theoretically optimal |
| FP6 | 6 | E3M2, Pareto midpoint between FP4 and FP8, max representable = 28.0 |

### Transform-Based Formats (all hardware-fixable)

| Format | Transform | Key Feature |
|--------|-----------|-------------|
| SmoothQuant+INT4 | Per-channel algebraic scale transfer | Zero rotation overhead, pre-computable scales |
| SmoothQuant+INT8 | Per-channel algebraic scale transfer | 8-bit variant |
| HAD+INT4 | Fast Walsh-Hadamard (FWHT) | O(N log N) butterfly, spreads outlier energy globally |
| HAD+INT8 | FWHT | 8-bit variant |
| HAD+LUT4 | FWHT + 4-bit LUT quantizer | Non-linear mapping via lookup table |
| HAD+SQ | FWHT + SQ-Format | Global redistribution + sparse high-precision residual |
| TurboQuant+INT4 | Random ±1 diagonal | XOR-equivalent, lighter than FWHT |
| RandRot+INT4 | Dense N×N orthogonal ROM | Upper-bound reference for rotation-based methods |
| SQ-Format | None | Top-1% salient in INT8, remaining 99% in INT4 + 1-bit mask |

---

## Installation

```bash
# Clone and enter the repository
git clone <repo-url>
cd dataformat

# Install dependencies
pip install -r requirements.txt
```

**Requirements**: `numpy>=1.24`, `scipy>=1.10`, `matplotlib>=3.7`, `seaborn>=0.12`, `pyrtl>=0.9`, `pandas>=2.0`

---

## Usage

### Run Full Pipeline

```bash
# Full experiment: all formats × all distributions × all metrics + figures
python run_all.py

# Fast mode (N=512 for quick validation)
python run_all.py --fast

# Skip hardware PPA evaluation (faster)
python run_all.py --skip-hw

# Generate figures only (skip experiments)
python run_all.py --figs-only
```

### Programmatic API

```python
from formats import build_all_formats
import numpy as np

# Build all 22 formats
formats = build_all_formats(dim=256, seed=42)

# Quantize a tensor
x = np.random.randn(1024).astype(np.float32)
x_q = formats["MXFP4"].quantize(x)
mse = float(np.mean((x - x_q) ** 2))

# Use transforms directly
from formats.transforms.hadamard import hadamard_transform
x_had = hadamard_transform(x, normalize=True)

from formats.transforms.random_rotation import RandomRotation
rot = RandomRotation(dim=256, seed=42)
x_rot = rot.apply(x)
x_rec = rot.inverse(x_rot)
```

### Individual Figures

```python
from visualization.plot_outlier_heatmap import plot_outlier_heatmap
from visualization.plot_channel_heatmap import plot_channel_heatmap
from visualization.plot_encoding_eff import plot_encoding_efficiency

plot_outlier_heatmap(out_dir="results/figures")
plot_channel_heatmap(out_dir="results/figures")
plot_encoding_efficiency(out_dir="results/figures")
```

### Hardware Evaluation

```python
from hardware.ppa_evaluator import run_full_ppa_evaluation

results = run_full_ppa_evaluation(array_rows=16, array_cols=16)
# Returns PPA breakdown for Scheme A (MXFP), Scheme B (INT+HAD), Scheme B+ (INT+HAD+SQ)
```

---

## Running Tests

```bash
# Run all 162 unit tests
python -m pytest tests/test_formats.py -v

# Run specific test class
python -m pytest tests/test_formats.py::TestHADTransform -v
python -m pytest tests/test_formats.py::TestMXFP -v
python -m pytest tests/test_formats.py::TestHardwareModels -v
```

All 162 tests pass in ~3 seconds.

**Test Coverage**:
- `TestFormatRegistry` — All 22 formats: shape, no-NaN, outlier robustness, MSE sanity
- `TestHADTransform` — Correctness (WHT([1,2,3,4])=[10,-2,-4,0]), energy preservation, self-inverse, non-power-of-2, outlier spread, batch dims
- `TestRandomRotation` — Energy preservation, invertibility, orthogonality (Q@Qᵀ=I)
- `TestTurboQuant` — Self-inverse, ±1 signs, energy preservation
- `TestSmoothQuant` — Positive scales, forward/inverse, algebraic equivalence (atol=1e-4)
- `TestNF4` — 16 sorted levels at N(0,1) quantiles, output in level set
- `TestFP6` — Monotonic levels, max=28.0, clamping, MSE < MXINT4
- `TestMXFP` — Block independence, 0.25 bits/element metadata overhead
- `TestSQFormat` — Salient channel MSE < INT4 MSE, storage overhead > 4 bits/element
- `TestMetrics` — All 5 metrics: identity/monotonicity for MSE, SQNR, KL-Div, EffBits, BOPs
- `TestDistributions` — All 7 generators: finite outputs, correct outlier injection
- `TestHardwareModels` — Area, energy, roofline, BOPs, arithmetic intensity ordering

---

## Research Findings

### Key Finding 1: The 4-bit Inflection Point

At 4-bit precision, **outlier handling is unavoidable**. The dynamic range of standard INT4 (16 levels) is insufficient to simultaneously represent both normal weights (σ≈1) and outlier channels (σ≈50). Every 4-bit format must make an architectural choice:

- **Hardware-Native**: Enlarge the representable range via floating-point exponents (MXFP4, NVFP4) or block-local scaling (MXINT4)
- **Transform-Based**: Redistribute outlier energy globally before quantization (HAD+INT4)

At 8-bit, this distinction largely disappears — even plain INT8 has sufficient range for most outlier distributions.

### Key Finding 2: Block Scale is LOCAL, HAD is GLOBAL

MX block scaling (MXFP4/MXINT4) rescales within 32-element blocks. An outlier channel corrupts only the blocks it participates in, but the dynamic range is still determined by the worst element *within* each block.

FWHT spreads outlier energy uniformly across ALL channels before quantization, so no single channel dominates the dynamic range. This is why HAD+INT4 often matches or beats MXFP4 despite using a simpler arithmetic format.

### Key Finding 3: Format Encoding Efficiency Gap

MX formats pay **+0.25 bits/element** for the shared E8M0 block scale. On Gaussian inputs (easy case), this overhead is recoverable — MXFP4 achieves near-theoretical effective bits. On channel-outlier inputs (hard case), MXFP4's block scale helps locally but HAD+INT4's global redistribution achieves higher effective bits with *zero metadata overhead*.

### Key Finding 4: Hardware Cost Asymmetry

| Scheme | Format | Transform | Area (rel.) | Throughput | Flexibility |
|--------|--------|-----------|-------------|------------|-------------|
| A | MXFP8 | None | 1.0× | High | Low |
| A | MXFP4 | None | 0.7× | High | Low |
| B | INT8 | FWHT | 0.6× | High | **High** |
| B | INT4 | FWHT | 0.4× | High | **High** |
| B+ | INT4 | FWHT+SQ | 0.45× | Medium | **High** |

The FWHT butterfly is a small, pipelined, area-efficient unit. The MXFP exponent alignment logic (finding the block maximum, computing E8M0) requires comparator trees that scale super-linearly with block size. INT+FWHT (Scheme B) achieves **30-40% lower area** vs MXFP at the same bit-width with comparable or better quantization quality on outlier-heavy workloads.

### Key Finding 5: Roofline Analysis

MXFP4 has **lower arithmetic intensity** than INT4 because the 0.25 bits/element scale metadata increases memory traffic without proportionally increasing compute. On memory-bandwidth-limited hardware (most inference accelerators), this metadata overhead directly reduces achievable throughput.

INT4+HAD achieves higher arithmetic intensity: the FWHT overhead is compute (O(N log N) additions), not memory. The transform is applied in-register before writes, so memory traffic is pure INT4.

---

## Figures

### Figure 1: Distribution Evolution Under Quantization

![Figure 1](results/figures/fig01_distribution_evolution.png)

Shows how each quantization format transforms the input distribution (7 conditions: Gaussian baseline, Spiky outliers 10×/50×/100×, Channel outliers σ=10/50/100). Key observation: HAD transforms the spiky outlier distribution toward Gaussian *before* quantization, making downstream INT4 effectively lossless at moderate outlier magnitudes.

---

### Figure 2: Precision-Outlier Sensitivity Heatmap

![Figure 2](results/figures/fig02_outlier_sensitivity_heatmap.png)

All 19 formats × 10 outlier conditions (log₁₀ MSE). Red = high error, Green = low error.

**Takeaway**: Hardware-native formats degrade gracefully with spiky outliers (local dynamic range expansion), but struggle with channel outliers (all elements in a block share one scale). Transform-based methods show the inverse pattern — HAD+INT4 handles channel outliers well but is slightly worse on mild spiky outliers.

---

### Figure 3: Pareto Front — Quality vs. Bit-width

![Figure 3](results/figures/fig03_pareto_quality.png)

SQNR vs. storage bits/element for all formats. Pareto-optimal formats (no format dominates on both axes simultaneously) are highlighted.

**Takeaway**: At 4 bits, HAD+INT4 and NF4 are co-Pareto-optimal. At 8 bits, HAD+INT8 and MXFP8 converge. FP6 occupies a useful middle ground.

---

### Figure 4: Pareto Front — Quality vs. Memory Bandwidth

![Figure 4](results/figures/fig04_pareto_bandwidth.png)

SQNR vs. effective memory bandwidth (accounting for metadata bits). MX formats pay 0.25 bits/element overhead, visually shifting them right on the bandwidth axis.

**Takeaway**: MXFP4's bandwidth cost (4.25 bits effective) is measurably worse than INT4+HAD (4.0 bits flat). The quality gap between them diminishes on channel-outlier inputs where HAD outperforms.

---

### Figure 5: HAD vs. Random Rotation Ablation

![Figure 5](results/figures/fig05_had_vs_random_rotation.png)

Compares structured HAD (O(N log N), hardware-fixable butterfly) against dense random rotation (O(N²), ROM-stored) across outlier severity levels.

**Takeaway**: HAD achieves 85-95% of random rotation quality at a fraction of the hardware cost. The quality gap widens at extreme outlier magnitudes (>100×) but remains practical for typical LLM activation outlier profiles (10-50×). TurboQuant (random ±1 diagonal) is faster than HAD but lower quality.

---

### Figure 6: PPA Bubble Chart

![Figure 6](results/figures/fig06_ppa_bubble.png)

Area (x-axis) vs. Throughput (y-axis) with bubble size = Power consumption. Each bubble represents one hardware implementation scheme.

**Takeaway**: Scheme B (INT+FWHT) sits in the optimal lower-left region — small area, high throughput, low power. Scheme A (MXFP) achieves similar throughput but higher area and power. Scheme B+ (INT+FWHT+SQ) trades some throughput for better quality on outlier-heavy workloads.

---

### Figure 7: Roofline Model

![Figure 7](results/figures/fig07_roofline.png)

Classical roofline analysis showing arithmetic intensity (FLOPs/Byte) for each format on a modern AI accelerator (256 TOPS INT4, 4 TB/s bandwidth).

**Takeaway**: Most quantization formats are memory-bandwidth-limited at inference batch sizes ≤ 32. MXFP4's bandwidth overhead (metadata) moves it *further left* on the roofline, worsening the memory-bound bottleneck. INT4+HAD stays at the theoretical INT4 memory boundary.

---

### Figure 8: Per-Channel Quantization Error Heatmap

![Figure 8](results/figures/fig08_channel_heatmap.png)

X-axis: channel index (128 channels, 5% injected outliers highlighted in red). Y-axis: format. Color: per-channel MSE (log scale).

**Takeaway**: INT4 shows bright red stripes across ALL channels (per-tensor scale is dominated by outlier magnitude). MXFP4 shows red only at outlier channels (block scale limits damage). HAD+INT4 shows near-uniform green across all channels — the FWHT has redistributed outlier energy globally, eliminating the bright stripes entirely.

---

### Figure 9: Format Encoding Efficiency

![Figure 9](results/figures/fig09_encoding_efficiency.png)

Storage bits/element (light bar) vs. Effective bits/element from rate-distortion theory (dark bar), for both Gaussian and Channel-Outlier distributions.

**Takeaway**: On Gaussian inputs, NF4 achieves near-perfect encoding efficiency (4 bits stored → ~3.9 effective bits). On channel-outlier inputs, only HAD-based formats maintain high efficiency — they convert a hard problem into a nearly-Gaussian one before encoding.

---

### Figure 10: Pipeline Latency Breakdown

![Figure 10](results/figures/fig10_pipeline_breakdown.png)

Stacked bar chart: latency breakdown (memory read → transform → quantize → dequantize → memory write) for each format pipeline.

**Takeaway**: MXFP's dominant latency is the exponent alignment step (comparator tree for block maximum). FWHT's dominant latency is the butterfly passes, but these are fully pipelined and overlap with memory. The total end-to-end pipeline latency of INT4+HAD is competitive with MXFP4.

---

## Hardware Comparison

### Scheme A: MXFP Systolic Array

```
Memory → [FP Decode + Exp Align] → Systolic Array (FP MAC) → [Renorm] → Memory
```

- **Pro**: No software transform overhead; hardware handles outliers natively
- **Con**: Exponent alignment comparator tree is area-heavy; 0.25 bpe metadata tax; format locked to OCP standard
- **Area**: ~1.0× (baseline)
- **Power**: ~1.0× (baseline)

### Scheme B: INT Systolic Array + FWHT

```
Memory → [FWHT Butterfly Unit] → Systolic Array (INT MAC) → [IFWHT] → Memory
```

- **Pro**: INT MAC is smallest/fastest arithmetic unit; FWHT is a structured, pipelined butterfly; no metadata overhead; format-agnostic (swap INT4↔INT8 freely)
- **Con**: FWHT adds latency before the array; requires power-of-2 dimension alignment
- **Area**: ~0.4-0.6× vs Scheme A
- **Power**: ~0.5-0.7× vs Scheme A

### Scheme B+: INT + FWHT + SQ Gather/Scatter

```
Memory → [SQ Mask Decode + Gather] → [FWHT] → Systolic Array → [IFWHT + Scatter] → Memory
```

- **Pro**: Handles extreme outliers (σ>100×) where FWHT alone is insufficient; SQ hardware is reusable for sparsity
- **Con**: Gather/Scatter units add complexity; ~10-15% area overhead vs Scheme B
- **Area**: ~0.45-0.65× vs Scheme A
- **Power**: ~0.6-0.8× vs Scheme A

---

## Conclusions

| Regime | Recommended Approach | Rationale |
|--------|---------------------|-----------|
| 8-bit, any outlier | MXFP8 or INT8+HAD | Both work well; choose based on ecosystem |
| 4-bit, mild outliers (≤10×) | HAD+INT4 | Lower area, equal quality to MXFP4 |
| 4-bit, moderate outliers (10-50×) | HAD+INT4 | FWHT redistributes energy; MXFP4 block scale insufficient |
| 4-bit, extreme outliers (>100×) | HAD+SQ-Format | FWHT + sparse high-precision residual |
| Area-constrained | INT4+HAD (Scheme B) | 30-40% smaller than MXFP at same bitwidth |
| Bandwidth-constrained | INT4+HAD | No metadata overhead vs. MXFP's +0.25 bpe |
| Ecosystem/compatibility | MXFP4 | OCP standard, broad hardware support |

**The core verdict**: For 4-bit inference on outlier-heavy models (LLMs, ViTs), INT4 + FWHT (Scheme B) is Pareto-superior to MXFP4 on quality, area, and bandwidth simultaneously. The hardware cost of the FWHT butterfly is substantially lower than the MXFP exponent alignment logic, and the quality benefit of global energy redistribution exceeds that of local block scaling on typical LLM activation profiles.

MXFP formats retain value as an industry-standard interoperability layer and for models where outliers are mild or well-distributed (CNN activations, post-LayerNorm weights).

---

## Project Structure

```
dataformat/
├── config.py                    # Global constants (energy model, NF4 levels, roofline params)
├── run_all.py                   # Master pipeline (--fast, --skip-hw, --figs-only)
├── requirements.txt
│
├── formats/                     # All 22 quantization formats
│   ├── __init__.py              # build_all_formats(dim, seed) → dict[str, QuantFormat]
│   ├── baseline.py              # FP32, BF16, INT4, INT8
│   ├── mxfp.py                  # MXFP4, MXFP8 (E8M0 block scale)
│   ├── mxint.py                 # MXINT4, MXINT8 (E8M0 block scale)
│   ├── nvfp4.py                 # NVFP4 (Blackwell E2M1, 8 positive levels)
│   ├── nf4.py                   # NF4 (QLoRA NormalFloat, 16 quantile levels)
│   ├── fp6.py                   # FP6 E3M2 (max=28.0)
│   ├── sq_format.py             # SQ-Format (1% INT8 salient + 99% INT4 + mask)
│   └── transforms/
│       ├── hadamard.py          # FWHT butterfly (vectorized, view-aliasing-safe)
│       ├── random_rotation.py   # Dense N×N orthogonal rotation (ROM)
│       ├── turbo_quant.py       # Random ±1 diagonal scaling
│       └── smooth_quant.py      # Per-channel algebraic scale transfer
│
├── distributions/               # Test input generators
│   ├── generators.py            # gaussian, spiky_outliers, channel_outliers, + 4 more
│   └── metrics.py               # mse, sqnr, kl_divergence, effective_bits, bops
│
├── experiments/                 # Experiment runners
│   ├── robustness.py            # Format × distribution × metric sweep
│   └── bitwidth_ablation.py     # Bit-width sensitivity study
│
├── hardware/                    # Hardware cost models
│   ├── energy_model.py          # Horowitz 45nm energy model (pJ/op)
│   ├── roofline.py              # Arithmetic intensity + roofline analysis
│   ├── bop_counter.py           # Bit operation counting for matmul + transforms
│   ├── ppa_evaluator.py         # Scheme A/B/B+ PPA via PyRTL + NAND2 analytical model
│   └── pyrtl_modules/           # PyRTL RTL definitions for each arithmetic unit
│
├── visualization/               # Figure generators (Figures 1-10)
│   ├── style.py                 # Global matplotlib style, PALETTE, MARKERS
│   ├── plot_distribution_evolution.py   # Fig 1
│   ├── plot_outlier_heatmap.py          # Fig 2
│   ├── plot_pareto.py                   # Fig 3 & 4
│   ├── plot_had_vs_randrot.py           # Fig 5
│   ├── plot_ppa_bubble.py               # Fig 6
│   ├── plot_roofline.py                 # Fig 7
│   ├── plot_channel_heatmap.py          # Fig 8
│   ├── plot_encoding_eff.py             # Fig 9
│   └── plot_pipeline_breakdown.py       # Fig 10
│
├── tests/
│   └── test_formats.py          # 162 unit tests (all pass in ~3s)
│
└── results/
    └── figures/                 # Generated PNG + PDF figures (Fig 1-10)
```

---

## Citation

If you use this codebase, please cite:

```bibtex
@misc{outlier-format-study-2026,
  title  = {Outlier-Robust Quantization: Hardware-Native Formats vs. Transform-Based Methods},
  year   = {2026},
  note   = {Comparative study of 22 quantization formats across information theory, numerical accuracy, and hardware implementation cost dimensions}
}
```

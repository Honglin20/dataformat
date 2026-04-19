# SQ-Format Comparison Study — Design Document

**Date:** 2026-04-19
**Status:** Approved — implementation plan in `2026-04-19-sqformat-experiment-plan.md`

## 1 · Motivation & Scope

Compare **SQ-Format** variants along four axes to characterise its trade-off
surface and produce a reference study mirroring the structure of the 4-bit
study (`experiments/fourbit/`):

- **Base format** — INT vs FP
- **Bit-width pair** `(h_high, h_low)` — INT has 5 pairs, FP has 3
- **Algorithm** — Algorithm 1 (weight importance `W²/diag(H⁻¹)²`) vs
  Algorithm 2 (activation-aware `|Āⱼ · Σᵢ W'ⱼᵢ|`)
- **Transform** — `{base, smooth, had}` (reused from 4-bit study)

The study is delivered as an extension of `experiments/fourbit/` rather
than as a separate pipeline — all SQ-Format–specific code lives in a new
`experiments/sqformat/` subpackage that reuses `part1.run_all`,
`part2.run`, `accuracy.accuracy_sweep`, `profiler_v2.analyse_all`, and
`reporter.generate_report` by passing in a SQ-Format–specific
`FourBitConfig`.

**Goal:** once the one-time code gap (§4) is closed, adding or varying
any (algorithm × base × bit-pair × transform × distribution × metric)
cell is a pure-config change.

### Out of scope / Non-goals

- Re-running or changing the existing 4-bit study's outputs
  (`results/fourbit/*`).
- Modifying the legacy `SQFormatFP` class (which is an FP8-high /
  INT-low hybrid, not a pure-FP SQ-Format). It is kept as an independent
  reference cell in the new study.
- **Unified W&A mask.** Per user direction, W and A get **independent**
  masks. Inference is expressed as four sub-matmuls
  `(W_h × A_h) + (W_h × A_l) + (W_l × A_h) + (W_l × A_l)` — each is a
  standard mixed-precision GEMM and can be accumulated in FP32/INT32.
- **FP2.** There is no industry-standard FP2 level set; FP pairs
  limited to `{8&8, 4&8, 4&4}`. The INT family covers the 2-bit regime.

## 2 · Key design decisions

| Ref | Decision | Rationale |
|-----|----------|-----------|
| §1  | SQ-Format applied independently to W (offline, per-output-column mask) and to A (runtime, per-token mask). Current `SQFormat` banking stays as-is. | User direction: "保持原来的实现" |
| §2  | FP pairs limited to `{8&8, 4&8, 4&4}`; INT pairs `{8&8, 4&8, 4&4, 4&2, 2&2}` | FP2 has no standard level set; keeps comparison fair |
| §3  | 8&8 and 2&2 are kept as lower-/upper-bound anchors | Dual per-bank scale still measurable at equal bits |
| §4  | W, A, **and Y** are all quantised at every GEMM (decision A2) | User's rule: "所有涉及到 W 和 A 都必须量化" — extended to Y in downstream GEMMs, realised by also quantising the current GEMM's Y output |
| §5  | MHA internal Linears are covered by a custom `QuantizedMHA` module (decision B1) | PyTorch's `F.multi_head_attention_forward` bypasses `Module.forward`, so `QuantLinear` is invisible to it; a hand-written attention fixes this |
| §6  | **Option B** for INT/FP: add `base: "int" \| "fp"` parameter to `SQFormat` / `SQFormatActivations`; share banking / mask / sentinel logic; swap only the element encoder | Avoids ~500 lines of duplicate code; element encoder is a pluggable registry (`_ELEMENT_ENCODERS`); future formats (NF, log) fit the same pattern; `base="int"` default preserves byte-identical golden CSVs |
| §7  | Metrics become a config-driven registry (`distributions.metrics.METRIC_REGISTRY` + `register_metric(...)`); profiler/Part 1 loop over `config.metrics` instead of hard-coding columns | User's request for freely pluggable metrics; default list reproduces current hard-coded columns → golden regression unaffected |
| §8  | Y bank policy: `output_fmt = fmt`, `bank_size = out_features` (one bank per token) | "Y uses same format as W/A" + banks must make hardware sense on Y's axis |

## 3 · Experiment matrix

```
Algorithms       : {Alg1, Alg2}                                    (2)
Bases            : {int, fp}                                       (—)
Bit-width pairs  : INT × {8&8, 4&8, 4&4, 4&2, 2&2}                 (5)
                 + FP  × {8&8, 4&8, 4&4}                            (3)
SQ-Format cells  = 2 × (5 + 3) = 16

Transforms       : {base, smooth, had}                             (3)

Part 1 distributions  = 24 (common) + linear pairs
Part 1 scan size      ≈ 16 × 3 × 24 ≈ 1150 cells
Part 2 accuracy scan  = 16 × 3 = 48 cells
```

All cells use `bank_size=128, sparsity=0.5` (SQ-Format default; 2:4-style
when applicable); these stay configurable via `FormatSpec.kwargs` for
future sweeps.

## 4 · Code gaps (one-time engineering)

All six gaps are one-shot; every further axis sweep is `kwargs`-only.

| ID | Gap | Closed by |
|----|-----|-----------|
| G1 | No pure-FP SQ-Format (legacy `SQFormatFP` is FP8/INT hybrid) | `SQFormat(base="fp", ...)` via element-encoder registry |
| G2 | `SQFormatActivations` is INT-only | Same `base` parameter on Algorithm 2 class |
| G3 | Y is not quantised at GEMM output | Optional `quantize_output=True` in `FourBitConfig`; `Pipeline` / `QuantLinear` quantise `y` with `self.output_fmt = self.fmt` before bias; default stays `False` so 4-bit report unaffected |
| G4 | MHA internal Linears skipped in accuracy sweep | `examples/model_quantizable.py::QuantizedMHA`; flag `use_quantizable_mha` on `MNISTTransformer` (default False) |
| G5 | Metrics hard-coded in profiler / Part 1 | `MetricSpec` list in `FourBitConfig.metrics`; registries in `distributions/metrics.py`; loops in `profiler_v2.analyse_layer` and `part1.py` drive CSV schema from config |
| G6 | No SQ-Format factory entries | 16 new entries in `experiments/fourbit/formats.py::FORMAT_FACTORIES`; declarative `SQFormatConfig` in new `experiments/sqformat/config.py` |

## 5 · PR plan (topologically ordered)

Each PR is independently mergeable and each must pass
`pytest tests/test_regression.py -v` byte-identical before merging.

| PR | Scope | Golden regression impact |
|----|-------|--------------------------|
| **A** | Element-encoder registry + `base` param on `SQFormat` & `SQFormatActivations` | `base="int"` default keeps all existing SQFormat outputs byte-identical |
| **B** | `distributions.metrics` registry + `FourBitConfig.metrics` list + profiler/Part 1 loop | Default `metrics` list = current hard-coded columns → no diff |
| **C** | `Pipeline.output_fmt`, `QuantLinear` Y quantisation, `FourBitConfig.quantize_output = False` default | Default keeps 4-bit pipeline unchanged; SQ-Format config turns it on |
| **D** | `examples/model_quantizable.py::QuantizedMHA`; `MNISTTransformer(use_quantizable_mha=...)`; accuracy sweep honours flag | New flag defaults False; legacy MNIST model unchanged |
| **E** | SQ-Format factory entries + `experiments/sqformat/` subpackage + `run_sqformat_study.py` shim | Introduces new CSV directory; no existing artefacts touched |
| **F** | Tests + golden fixture for SQ-Format Part 1 | Extends regression harness; pre-existing goldens untouched |

## 6 · Metrics default set & report template

### Default `SQ_METRICS`

```python
MetricSpec("qsnr_db",      "qsnr_db",      roles=["W","X","Y"], kind="pair"),
MetricSpec("snr_db",       "snr_db",       roles=["W","X","Y"], kind="pair"),
MetricSpec("mse",          "mse",          roles=["W","X","Y"], kind="pair"),
MetricSpec("fp16_qsnr_db", "fp16_qsnr_db", roles=["W","X","Y"], kind="pair"),
```

`SQ_TENSOR_STATS = ["mean","std","min","max","crest","kurtosis"]` mirrors
`tensor_summary`.

### Report outline (`results/sqformat/report.md`)

```
1. Experiment matrix table
2. Format overview
   2.1 INT 5-pair curves (Alg1 vs Alg2)
   2.2 FP 3-pair curves
   2.3 INT vs FP at common pairs
3. Per-distribution heatmaps (Part 1)
4. MNIST end-to-end (Part 2)
   4.1 Accuracy vs data_bits_per_element
   4.2 Per-layer W/X/Y QSNR by transform
   4.3 Alg1 vs Alg2 accuracy delta histogram
5. Engineering notes (Y bank policy, MHA replacement, legacy SQFormatFP caveat)
```

## 7 · Test plan

| Scope | File | Gate |
|-------|------|------|
| `SQFormat(base="int")` byte-identical vs current | `tests/test_sq_format_base_fp.py` | PR A |
| `SQFormat(base="fp")` vs NVFP4 in degenerate config | `tests/test_sq_format_base_fp.py` | PR A |
| Metrics registry pickup + default list byte-identical | `tests/test_metrics_registry.py` | PR B |
| `quantize_output=False` byte-identical; True actually quantises Y | `tests/test_pipeline_output_quant.py` | PR C |
| `QuantizedMHA` FP32 ≈ `nn.MultiheadAttention` | `tests/test_quantizable_mha.py` | PR D |
| SQ-Format Part 1 golden CSV | `tests/test_regression.py` + `tests/fixtures/golden/sqformat_*.csv` | PR F |
| End-to-end smoke: `run_sqformat_study.py --part all --profile-samples 32` exits 0 | `tests/test_sqformat_smoke.py` | PR E onward |
| 4-bit golden regression continues to pass | `tests/test_regression.py` | **every PR** |

## 8 · Open questions deferred to implementation time

- **R1 (PR A):** Verify FP8 E4M3 `q_max=448.0` inside `pot_scale_vec`
  yields the expected E8M0 scale semantics. Write a unit test pinning
  behaviour against a hand-computed example before doing the broader
  equivalence test.
- **R2 (PR C):** Does Y quantisation with `bank_size = out_features`
  stay numerically stable when `out_features` is small (e.g. MNIST
  classifier head = 10)? If not, fall back to `bank_size = min(128,
  out_features)` and document.
- **R3 (PR E):** Part 2 runtime with all 48 cells × three transforms +
  runtime activation quantisation. If the run exceeds ~15 minutes on
  the dev box, reduce `profile_samples` to 128 for the default config;
  keep the 256-sample setting as an opt-in `--profile-samples 256`.

## 9 · Next step

Follow-up document: `docs/plans/2026-04-19-sqformat-experiment-plan.md`
expands each PR into step-by-step TDD tasks per the `writing-plans`
skill conventions. Execution uses `subagent-driven-development`.

"""Default experiment configurations.

**This is the primary file to edit when:**
  - Adding a new format → add its name to the appropriate FormatGroup.
  - Adding a new distribution → add a DistributionConfig entry.
  - Defining a new experiment → create a new ExperimentConfig.

No other experiment files need to change.

Format naming conventions
-------------------------
  4-bit dense formats : INT4, MXINT4, MXFP4, NVFP4, NF4,
                        HAD+INT4(C), HAD+SQ, SQ-Format
  8-bit dense formats : INT8, MXINT8, MXFP8, FP6,
                        HAD+INT8(C), SQ-Format(8b)
  Hardware focus (BFP = Butterfly FP = HAD+INT paradigm):
                        MXINT4/8, HAD+INT4(C)/HAD+INT8(C), SQ-Format/SQ-Format(8b)
"""

from __future__ import annotations

from experiments.config import DistributionConfig, ExperimentConfig, FormatGroup
from distributions.generators import (
    gaussian, channel_outliers, spiky_outliers, student_t_dist, bimodal,
    laplace, log_normal,
)


# ════════════════════════════════════════════════════════════════════════════
# Distribution library
# Add new distributions here; they become available to all configs below.
# ════════════════════════════════════════════════════════════════════════════

#: Compact 7-distribution set used in the main ablation study.
ABLATION_DISTRIBUTIONS: list[DistributionConfig] = [
    DistributionConfig(
        "Gaussian(σ=1)",
        lambda n, s: gaussian(n, sigma=1.0, seed=s),
        tags=["gaussian", "baseline"],
    ),
    DistributionConfig(
        "Student-t(ν=3)",
        lambda n, s: student_t_dist(n, nu=3.0, seed=s),
        tags=["heavy-tail"],
    ),
    DistributionConfig(
        "Bimodal",
        lambda n, s: bimodal(n, seed=s),
        tags=["multimodal"],
    ),
    DistributionConfig(
        "ChannelOutlier(σ=50)",
        lambda n, s: channel_outliers(n, outlier_sigma=50.0, seed=s),
        tags=["outlier", "channel"],
    ),
    DistributionConfig(
        "Spiky(10×)",
        lambda n, s: spiky_outliers(n, spike_multiplier=10.0, seed=s),
        tags=["outlier", "spiky"],
    ),
    DistributionConfig(
        "Spiky(50×)",
        lambda n, s: spiky_outliers(n, spike_multiplier=50.0, seed=s),
        tags=["outlier", "spiky"],
    ),
    DistributionConfig(
        "Spiky(100×)",
        lambda n, s: spiky_outliers(n, spike_multiplier=100.0, seed=s),
        tags=["outlier", "spiky"],
    ),
]

#: Full 14-distribution set used in the robustness sweep.
ROBUSTNESS_DISTRIBUTIONS: list[DistributionConfig] = [
    DistributionConfig(
        "Gaussian(σ=1)",
        lambda n, s: gaussian(n, sigma=1.0, seed=s),
        tags=["gaussian", "baseline"],
    ),
    DistributionConfig(
        "Laplace(b=0.5)",
        lambda n, s: laplace(n, b=0.5, seed=s),
        tags=["heavy-tail", "laplace"],
    ),
    DistributionConfig(
        "Laplace(b=1.0)",
        lambda n, s: laplace(n, b=1.0, seed=s),
        tags=["heavy-tail", "laplace"],
    ),
    DistributionConfig(
        "Laplace(b=2.0)",
        lambda n, s: laplace(n, b=2.0, seed=s),
        tags=["heavy-tail", "laplace"],
    ),
    DistributionConfig(
        "Student-t(ν=3)",
        lambda n, s: student_t_dist(n, nu=3.0, seed=s),
        tags=["heavy-tail"],
    ),
    DistributionConfig(
        "Student-t(ν=5)",
        lambda n, s: student_t_dist(n, nu=5.0, seed=s),
        tags=["heavy-tail"],
    ),
    DistributionConfig(
        "Student-t(ν=10)",
        lambda n, s: student_t_dist(n, nu=10.0, seed=s),
        tags=["heavy-tail"],
    ),
    DistributionConfig(
        "Bimodal(μ=±3)",
        lambda n, s: bimodal(n, mu1=-3.0, mu2=3.0, sigma=0.5, seed=s),
        tags=["multimodal"],
    ),
    DistributionConfig(
        "ChannelOutlier(σ=30)",
        lambda n, s: channel_outliers(n, outlier_sigma=30.0, seed=s),
        tags=["outlier", "channel"],
    ),
    DistributionConfig(
        "ChannelOutlier(σ=50)",
        lambda n, s: channel_outliers(n, outlier_sigma=50.0, seed=s),
        tags=["outlier", "channel"],
    ),
    DistributionConfig(
        "ChannelOutlier(σ=100)",
        lambda n, s: channel_outliers(n, outlier_sigma=100.0, seed=s),
        tags=["outlier", "channel"],
    ),
    DistributionConfig(
        "Spiky(10×)",
        lambda n, s: spiky_outliers(n, spike_multiplier=10.0, seed=s),
        tags=["outlier", "spiky"],
    ),
    DistributionConfig(
        "Spiky(50×)",
        lambda n, s: spiky_outliers(n, spike_multiplier=50.0, seed=s),
        tags=["outlier", "spiky"],
    ),
    DistributionConfig(
        "Spiky(100×)",
        lambda n, s: spiky_outliers(n, spike_multiplier=100.0, seed=s),
        tags=["outlier", "spiky"],
    ),
    DistributionConfig(
        "LogNormal(σ=1)",
        lambda n, s: log_normal(n, sigma=1.0, seed=s),
        tags=["lognormal"],
    ),
    DistributionConfig(
        "LogNormal(σ=2)",
        lambda n, s: log_normal(n, sigma=2.0, seed=s),
        tags=["lognormal"],
    ),
]


# ════════════════════════════════════════════════════════════════════════════
# Format groups
# Add new format names here when they are added to build_all_formats().
# ════════════════════════════════════════════════════════════════════════════

#: All 4-bit formats — accuracy / Pareto comparison
GROUP_4BIT = FormatGroup(
    name="4bit",
    label="4-bit Survival Regime",
    formats=[
        "INT4",
        "MXFP4", "MXINT4",
        "NVFP4",
        "NF4",
        "SQ-Format",          # 4-bit dense, 8-bit sparse, 1-bit mask
        "SmoothQuant+INT4",
        "HAD+INT4(C)",        # BFP paradigm (Butterfly FP)
        "HAD+INT4(T)",
        "HAD+LUT4",
        "HAD+SQ",
        "RandRot+INT4",
        "FP32",
    ],
    bits=4,
)

#: All 8-bit formats — accuracy / Pareto comparison
GROUP_8BIT = FormatGroup(
    name="8bit",
    label="8-bit Efficiency Regime",
    formats=[
        "INT8",
        "MXFP8", "MXINT8",
        "FP6",
        "SQ-Format(8b)",      # 8-bit dense, 8-bit sparse — ablation vs 4-bit SQ
        "SmoothQuant+INT8",
        "HAD+INT8(C)",
        "HAD+INT8(T)",
        "RandRot+INT8",
        "FP32",
    ],
    bits=8,
)

#: Hardware focus — 4-bit: three key paradigms only (MXINT / BFP / SQ)
GROUP_HW_4BIT = FormatGroup(
    name="hw_4bit",
    label="Hardware Focus: 4-bit (MXINT vs BFP vs SQ)",
    formats=["MXINT4", "HAD+INT4(C)", "SQ-Format"],
    bits=4,
)

#: Hardware focus — 8-bit: three key paradigms only
GROUP_HW_8BIT = FormatGroup(
    name="hw_8bit",
    label="Hardware Focus: 8-bit (MXINT vs BFP vs SQ)",
    formats=["MXINT8", "HAD+INT8(C)", "SQ-Format(8b)"],
    bits=8,
)


# ════════════════════════════════════════════════════════════════════════════
# Experiment configs
# ════════════════════════════════════════════════════════════════════════════

#: Main 4-bit vs 8-bit ablation (compact 7-distribution set).
ABLATION_CONFIG = ExperimentConfig(
    name="ablation",
    groups=[GROUP_4BIT, GROUP_8BIT],
    distributions=ABLATION_DISTRIBUTIONS,
    n_samples=4096,
    seed=42,
    output_dir="results",
)

#: Full robustness sweep (all 16 distributions, all formats).
ROBUSTNESS_CONFIG = ExperimentConfig(
    name="robustness",
    groups=[GROUP_4BIT, GROUP_8BIT],
    distributions=ROBUSTNESS_DISTRIBUTIONS,
    n_samples=4096,
    seed=42,
    output_dir="results",
)

#: Hardware-focus accuracy experiment (3 paradigms × 7 distributions).
HW_FOCUS_CONFIG = ExperimentConfig(
    name="hw_focus",
    groups=[GROUP_HW_4BIT, GROUP_HW_8BIT],
    distributions=ABLATION_DISTRIBUTIONS,
    n_samples=4096,
    seed=42,
    output_dir="results",
)

#: Fast sanity-check config (small N, minimal formats).
FAST_CONFIG = ExperimentConfig(
    name="fast_check",
    groups=[
        FormatGroup("4bit", "4-bit (fast)", ["INT4", "MXINT4", "HAD+INT4(C)", "SQ-Format", "FP32"], bits=4),
        FormatGroup("8bit", "8-bit (fast)", ["INT8", "MXINT8", "HAD+INT8(C)", "SQ-Format(8b)", "FP32"], bits=8),
    ],
    distributions=ABLATION_DISTRIBUTIONS[:3],   # first 3 only
    n_samples=512,
    seed=42,
    output_dir="results",
)

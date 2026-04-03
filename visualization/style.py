"""Global matplotlib style configuration for all research figures.

Defines:
  - Color palette per format family.
  - Figure sizes and DPI for publication quality.
  - Consistent marker shapes, line styles.
  - LaTeX-compatible font settings.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


# ── Publication-quality settings ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "legend.framealpha": 0.9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 2.0,
    "lines.markersize": 7,
    "figure.constrained_layout.use": True,
})

# ── Color palette by format family ───────────────────────────────────────────
PALETTE = {
    # Baselines (grey tones)
    "FP32":    "#6B7280",
    "BF16":    "#9CA3AF",

    # Hardware-native: blue family
    "NVFP4":   "#1D4ED8",
    "MXFP4":   "#3B82F6",
    "MXFP8":   "#60A5FA",
    "MXINT4":  "#1E40AF",
    "MXINT8":  "#2563EB",
    "NF4":     "#7C3AED",  # purple (information-optimal)
    "FP6":     "#8B5CF6",

    # Transform-based: orange/red/green family
    "SmoothQuant+INT4": "#F97316",
    "SmoothQuant+INT8": "#FB923C",
    "HAD+INT4":   "#16A34A",
    "HAD+INT8":   "#22C55E",
    "HAD+LUT4":   "#4ADE80",
    "HAD+SQ":     "#15803D",
    "RandRot+INT4": "#DC2626",
    "RandRot+INT8": "#EF4444",
    "TurboQuant+INT4": "#EA580C",
    "TurboQuant+INT8": "#F97316",

    # SQ-Format
    "SQ-Format": "#0D9488",

    # Plain INT (no transform)
    "INT4":    "#D97706",
    "INT8":    "#B45309",
}

# Marker shapes by format family
MARKERS = {
    "FP32": "s",     # square
    "BF16": "s",
    "NVFP4": "D",    # diamond
    "MXFP4": "D",
    "MXFP8": "D",
    "MXINT4": "P",   # plus-filled
    "MXINT8": "P",
    "NF4": "^",      # triangle
    "FP6": "^",
    "HAD+INT4": "o", # circle
    "HAD+INT8": "o",
    "HAD+LUT4": "o",
    "HAD+SQ": "*",   # star
    "SmoothQuant+INT4": "v",
    "SmoothQuant+INT8": "v",
    "RandRot+INT4": "X",
    "RandRot+INT8": "X",
    "TurboQuant+INT4": "h",
    "TurboQuant+INT8": "h",
    "SQ-Format": "8",
    "INT4": ">",
    "INT8": "<",
}

# Line styles for line plots (e.g., HAD vs RandRot ablation)
LINESTYLES = {
    "HAD+INT4":    "-",
    "HAD+INT8":    "-",
    "RandRot+INT4": "--",
    "RandRot+INT8": "--",
    "TurboQuant+INT4": "-.",
    "TurboQuant+INT8": "-.",
    "MXFP4":       ":",
    "MXFP8":       ":",
    "INT4":        "--",   # plain dashed
    "INT8":        "--",
}

# Format groupings for legend ordering
FAMILY_GROUPS = {
    "Baselines": ["FP32", "BF16"],
    "Hardware-Native 4-bit": ["NVFP4", "MXFP4", "MXINT4", "NF4"],
    "Hardware-Native 8-bit": ["MXFP8", "MXINT8", "FP6"],
    "Transform + INT": ["HAD+INT4", "HAD+INT8", "HAD+LUT4", "HAD+SQ"],
    "SmoothQuant": ["SmoothQuant+INT4", "SmoothQuant+INT8"],
    "Rotation": ["RandRot+INT4", "RandRot+INT8", "TurboQuant+INT4", "TurboQuant+INT8"],
    "Sparse-Quantized": ["SQ-Format"],
}


def get_color(fmt_name: str) -> str:
    """Get color for a format, falling back to a generated color."""
    if fmt_name in PALETTE:
        return PALETTE[fmt_name]
    # Generate a deterministic hex color from hash (matplotlib-compatible)
    h_val = (hash(fmt_name) & 0xFFFFFF)
    return f"#{h_val:06x}"


def get_marker(fmt_name: str) -> str:
    return MARKERS.get(fmt_name, "o")


def get_linestyle(fmt_name: str) -> object:
    return LINESTYLES.get(fmt_name, "-")


def format_family(fmt_name: str) -> str:
    """Return the format family name for grouping in legends."""
    for family, fmts in FAMILY_GROUPS.items():
        if fmt_name in fmts:
            return family
    return "Other"


def fig_and_ax(w: float = 8, h: float = 5, nrows: int = 1, ncols: int = 1):
    """Create figure and axes with publication settings."""
    return plt.subplots(nrows, ncols, figsize=(w, h))


def save_fig(fig, name: str, out_dir: str = "results/figures"):
    """Save figure to PNG and PDF."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{name}.png"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{name}.pdf"), bbox_inches="tight")
    print(f"Saved → {out_dir}/{name}.{{png,pdf}}")
    plt.close(fig)

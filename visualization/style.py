"""Global matplotlib style configuration for all research figures.

Focus format set (for main comparison figures):
  Baselines:        FP32, INT4, INT8
  Hardware-native:  MXINT4, MXINT8, NVFP4, NF4
  Transform-based:  HAD+INT4(C), HAD+INT4(T), HAD+INT8(C), HAD+INT8(T), HAD+SQ
  Sparse-quant:     SQ-Format
  Upper-bound ref:  RandRot+INT4, RandRot+INT8
"""

import matplotlib.pyplot as plt
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
    # Baselines (grey / brown tones)
    "FP32":    "#6B7280",
    "BF16":    "#9CA3AF",
    "INT4":    "#92400E",
    "INT8":    "#B45309",

    # Hardware-native INT: blue family (PRIMARY focus)
    "MXINT4":  "#1E40AF",
    "MXINT8":  "#2563EB",

    # Hardware-native FP: lighter blue (secondary)
    "MXFP4":   "#60A5FA",
    "MXFP8":   "#93C5FD",

    # Other 4-bit hardware-native
    "NVFP4":   "#7C3AED",
    "NF4":     "#A855F7",
    "FP6":     "#C084FC",

    # HAD + INT per-channel (C): green family (PRIMARY focus)
    "HAD+INT4(C)": "#15803D",
    "HAD+INT8(C)": "#16A34A",
    "HAD+INT4":    "#15803D",   # alias
    "HAD+INT8":    "#16A34A",   # alias

    # HAD + INT per-tensor (T): lighter green (ablation)
    "HAD+INT4(T)": "#4ADE80",
    "HAD+INT8(T)": "#86EFAC",

    # HAD + SQ: teal
    "HAD+SQ":  "#0D9488",

    # HAD + LUT (secondary)
    "HAD+LUT4": "#34D399",

    # SQ-Format: amber
    "SQ-Format": "#D97706",

    # Random rotation: red (upper-bound reference)
    "RandRot+INT4": "#DC2626",
    "RandRot+INT8": "#EF4444",

    # SmoothQuant (secondary)
    "SmoothQuant+INT4": "#F97316",
    "SmoothQuant+INT8": "#FB923C",
}

# Marker shapes
MARKERS = {
    "FP32": "s",
    "BF16": "s",
    "INT4":  ">",
    "INT8":  "<",
    "MXINT4": "P",
    "MXINT8": "P",
    "MXFP4":  "D",
    "MXFP8":  "D",
    "NVFP4":  "D",
    "NF4":    "^",
    "FP6":    "^",
    "HAD+INT4(C)": "o",
    "HAD+INT8(C)": "o",
    "HAD+INT4(T)": "v",
    "HAD+INT8(T)": "v",
    "HAD+INT4":    "o",
    "HAD+INT8":    "o",
    "HAD+SQ":   "*",
    "HAD+LUT4": "h",
    "SQ-Format": "8",
    "RandRot+INT4": "X",
    "RandRot+INT8": "X",
    "SmoothQuant+INT4": "v",
    "SmoothQuant+INT8": "v",
}

# Line styles for line plots
LINESTYLES = {
    "FP32":           ":",
    "INT4":           "--",
    "INT8":           "--",
    "MXINT4":         "-",
    "MXINT8":         "-",
    "MXFP4":          "-.",
    "MXFP8":          "-.",
    "NVFP4":          "-.",
    "NF4":            "-.",
    "HAD+INT4(C)":    "-",
    "HAD+INT8(C)":    "-",
    "HAD+INT4(T)":    "--",
    "HAD+INT8(T)":    "--",
    "HAD+INT4":       "-",
    "HAD+INT8":       "-",
    "HAD+SQ":         "-",
    "SQ-Format":      "-",
    "RandRot+INT4":   (0, (3, 1, 1, 1)),
    "RandRot+INT8":   (0, (3, 1, 1, 1)),
    "SmoothQuant+INT4": "-.",
    "SmoothQuant+INT8": "-.",
}

# Format groupings for legend ordering
FAMILY_GROUPS = {
    "Baselines": ["FP32", "INT4", "INT8"],
    "HW-Native INT": ["MXINT4", "MXINT8"],
    "HW-Native 4-bit": ["NVFP4", "NF4"],
    "HAD+INT (focus)": ["HAD+INT4(C)", "HAD+INT4(T)", "HAD+INT8(C)", "HAD+INT8(T)"],
    "HAD+SQ / SQ": ["HAD+SQ", "SQ-Format"],
    "Upper-bound Ref": ["RandRot+INT4", "RandRot+INT8"],
    "Secondary": ["MXFP4", "MXFP8", "FP6", "BF16",
                  "SmoothQuant+INT4", "SmoothQuant+INT8", "HAD+LUT4"],
}


def get_color(fmt_name: str) -> str:
    if fmt_name in PALETTE:
        return PALETTE[fmt_name]
    h_val = (hash(fmt_name) & 0xFFFFFF)
    return f"#{h_val:06x}"


def get_marker(fmt_name: str) -> str:
    return MARKERS.get(fmt_name, "o")


def get_linestyle(fmt_name: str):
    return LINESTYLES.get(fmt_name, "-")


def format_family(fmt_name: str) -> str:
    for family, fmts in FAMILY_GROUPS.items():
        if fmt_name in fmts:
            return family
    return "Other"


def fig_and_ax(w: float = 8, h: float = 5, nrows: int = 1, ncols: int = 1):
    return plt.subplots(nrows, ncols, figsize=(w, h))


def save_fig(fig, name: str, out_dir: str = "results/figures"):
    import os
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"{name}.png"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{name}.pdf"), bbox_inches="tight")
    print(f"Saved → {out_dir}/{name}.{{png,pdf}}")
    plt.close(fig)

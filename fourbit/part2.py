"""Part 2 – real-model analysis on the MNIST Transformer.

Pipeline
--------
1. Load (or train, if missing) the MNIST Transformer checkpoint that already
   ships in ``examples/``.
2. Run a small test-set slice through the model inside a ``LayerCollector``
   context so every ``nn.Linear`` layer's (W, X, Y) tensors are recorded.
3. Call ``analyse_all`` to compute, for every layer and every
   (format × transform) pair, the QSNR on the weight / activation / output.
4. Write a flat per-layer CSV and return the DataFrame for the reporter.

Because HAD requires power-of-2 in_features, layers that don't satisfy this
are reported with ``qsnr_* = NaN`` and a ``reason`` string rather than being
silently dropped.
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.utils.data as tdata
from torchvision import datasets, transforms

from fourbit.config import FourBitConfig
from fourbit.profiler_v2 import LayerCollector, analyse_all


# ── Model loading ────────────────────────────────────────────────────────────

def _load_or_train_model(model_path: str, data_dir: str) -> torch.nn.Module:
    """Load ``model.pt`` if it exists; otherwise train a quick MNIST Transformer."""
    from examples.model import MNISTTransformer
    model = MNISTTransformer()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        return model.eval()

    print(f"[Part 2] No checkpoint at {model_path}; training 2 quick epochs ...")
    import torch.nn as nn
    from torch.optim import AdamW

    tf = transforms.ToTensor()
    data_dir = os.path.expanduser(data_dir)
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=tf)
    # Use a small subset to keep smoke-test time bounded.
    subset = tdata.Subset(train_ds, list(range(4000)))
    loader = tdata.DataLoader(subset, batch_size=128, shuffle=True, num_workers=0)

    crit = nn.CrossEntropyLoss()
    opt  = AdamW(model.parameters(), lr=1e-3)
    for epoch in range(2):
        for x, y in loader:
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    return model.eval()


def _make_loader(data_dir: str, n_samples: int, batch_size: int = 32):
    tf = transforms.ToTensor()
    data_dir = os.path.expanduser(data_dir)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
    g = torch.Generator().manual_seed(0)
    indices = torch.randperm(len(test_ds), generator=g)[:n_samples].tolist()
    subset = tdata.Subset(test_ds, indices)
    return tdata.DataLoader(subset, batch_size=batch_size, shuffle=False)


# ── Runner ───────────────────────────────────────────────────────────────────

def run(
    config: FourBitConfig,
    model_path: str = "results/mnist/model.pt",
    data_dir: str = "~/.cache/mnist",
) -> Tuple[pd.DataFrame, dict]:
    """Collect and analyse a real model.

    Returns
    -------
    metrics_df : pd.DataFrame
        One row per (layer, format, transform); columns include qsnr_w_db,
        qsnr_x_db, qsnr_y_db and the raw tensor statistics (std, crest,
        kurtosis, max_abs) prefixed by ``W_``/``X_``/``Y_``.
    layers : dict[str, LayerRecord]
        Raw collected tensors, handed on to the reporter for figure-level
        aggregation.
    """
    model = _load_or_train_model(model_path, data_dir)
    loader = _make_loader(data_dir, config.profile_samples)

    print(f"[Part 2] Collecting tensors from {config.profile_samples} samples ...")
    with LayerCollector(model) as collector:
        model.eval()
        with torch.no_grad():
            for x, _ in loader:
                model(x)

    print(f"[Part 2] Recorded {len(collector.layers)} Linear layers. Analysing ...")
    df = analyse_all(collector.layers, config)

    out_dir = os.path.join(config.output_dir, "part2")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "per_layer_metrics.csv"), index=False)
    return df, collector.layers

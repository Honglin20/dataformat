"""Part 2 – real-model analysis.

Public API
----------
``QuantProfiler`` (in :mod:`experiments.fourbit.profiler_v2`)
    The primary entry point for any model.  Specify the experiment
    pipeline via a :class:`FourBitConfig`, then control inference freely
    inside a ``with`` block::

        from experiments.fourbit.profiler_v2 import QuantProfiler
        from experiments.fourbit.config import DEFAULT_CONFIG   # 4-bit study
        # from experiments.sqformat.config import DEFAULT_CONFIG  # SQ-Format

        profiler = QuantProfiler(model, DEFAULT_CONFIG)

        with profiler:
            for batch in dataloader:
                model(**batch)   # any call signature

        metrics_df = profiler.analyse()
        profiler.save(metrics_df)

        # Optional accuracy / quality sweep
        def eval_fn(m):
            ...
            return metric

        acc_df = profiler.run_eval_sweep(eval_fn)
        profiler.save_accuracy(acc_df)

``run`` (this module)
    Backward-compatible MNIST wrapper used by ``cli.py``.  Loads (or
    trains) the MNIST Transformer, runs inference, and delegates to
    ``QuantProfiler``.  No public use intended — call ``QuantProfiler``
    directly for any other model.
"""
from __future__ import annotations

import os
from typing import Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as tdata

from experiments.fourbit.config import FourBitConfig
from experiments.fourbit.profiler_v2 import QuantProfiler
from experiments.fourbit.accuracy import _eval_accuracy


# ── MNIST model / data helpers (private) ─────────────────────────────────────

def _load_or_train_model(
    model_path: str,
    data_dir: str,
    use_quantizable_mha: bool = False,
) -> torch.nn.Module:
    """Load ``model.pt`` if it exists; otherwise train a quick MNIST Transformer."""
    from examples.model import MNISTTransformer
    model = MNISTTransformer()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model = model.eval()
        if use_quantizable_mha:
            from examples.model_quantizable import QuantizedMHA
            for layer in model.encoder.layers:
                ref = layer.self_attn
                q = QuantizedMHA(
                    embed_dim=ref.embed_dim,
                    num_heads=ref.num_heads,
                    bias=ref.in_proj_bias is not None,
                )
                q.load_from_nn(ref)
                layer.self_attn = q
        return model

    print(f"[Part 2] No checkpoint at {model_path}; training 2 quick epochs ...")
    from torch.optim import AdamW
    from torchvision import datasets, transforms as T

    tf = T.ToTensor()
    data_dir = os.path.expanduser(data_dir)
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=tf)
    subset = tdata.Subset(train_ds, list(range(4000)))
    loader = tdata.DataLoader(subset, batch_size=128, shuffle=True, num_workers=0)

    crit = nn.CrossEntropyLoss()
    opt  = AdamW(model.parameters(), lr=1e-3)
    for _ in range(2):
        for x, y in loader:
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    return model.eval()


def _make_loader(data_dir: str, n_samples: int, batch_size: int = 32):
    from torchvision import datasets, transforms as T
    tf = T.ToTensor()
    data_dir = os.path.expanduser(data_dir)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
    g = torch.Generator().manual_seed(0)
    indices = torch.randperm(len(test_ds), generator=g)[:n_samples].tolist()
    subset = tdata.Subset(test_ds, indices)
    return tdata.DataLoader(subset, batch_size=batch_size, shuffle=False)


# ── MNIST wrapper (backward-compatible entry point for cli.py) ────────────────

def run(
    config: FourBitConfig,
    model_path: str = "results/mnist/model.pt",
    data_dir: str = "~/.cache/mnist",
) -> Tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Load (or train) the MNIST Transformer and run Part 2 via ``QuantProfiler``.

    This is the entry point used by ``cli.py``; it preserves the exact
    same behaviour and return types as before.  For any other model use
    :class:`~experiments.fourbit.profiler_v2.QuantProfiler` directly.
    """
    model = _load_or_train_model(
        model_path, data_dir,
        use_quantizable_mha=getattr(config, "use_quantizable_mha", False),
    )
    loader = _make_loader(data_dir, config.profile_samples)

    profiler = QuantProfiler(model, config)

    print(f"[Part 2] Collecting tensors from {config.profile_samples} samples ...")
    with profiler:
        model.eval()
        with torch.no_grad():
            for x, _ in loader:
                model(x)

    print(f"[Part 2] Recorded {len(profiler.layers)} Linear layers. Analysing ...")
    metrics_df = profiler.analyse()
    profiler.save(metrics_df)

    def _mnist_eval(m: nn.Module) -> float:
        # Detect fp16 model and pass matching input dtype.
        try:
            w = next(iter(m.parameters()))
            return _eval_accuracy(m, loader, dtype=w.dtype)
        except StopIteration:
            return _eval_accuracy(m, loader)

    print("[Part 2] Running accuracy sweep over all (format, transform) ...")
    acc_df = profiler.run_eval_sweep(_mnist_eval)
    profiler.save_accuracy(acc_df)

    return metrics_df, profiler.layers, acc_df

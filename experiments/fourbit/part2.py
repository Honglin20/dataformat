"""Part 2 – real-model analysis.

Two entry points
----------------
``profile_model``
    Model-agnostic API.  Accepts any trained ``nn.Module`` and a
    ``DataLoader``, collects per-layer (W, X, Y) tensors from every
    ``nn.Linear``, runs the (format × transform) QSNR sweep, and
    optionally calls a user-supplied *eval_fn* for an accuracy /
    quality sweep.  No MNIST dependency whatsoever.

``run``
    Backward-compatible MNIST wrapper.  Loads (or trains) the MNIST
    Transformer, builds the test-set loader, and delegates to
    ``profile_model``.  Existing CLI flags (``--model-path``,
    ``--data-dir``) continue to work unchanged.

Because HAD requires power-of-2 in_features, layers that don't satisfy this
are reported with ``qsnr_* = NaN`` and a ``reason`` string rather than being
silently dropped.
"""
from __future__ import annotations

import os
from typing import Callable, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as tdata

from experiments.fourbit.config import FourBitConfig
from experiments.fourbit.profiler_v2 import LayerCollector, analyse_all
from experiments.fourbit.accuracy import accuracy_sweep, _eval_accuracy


# ── Model loading ────────────────────────────────────────────────────────────

def _load_or_train_model(
    model_path: str,
    data_dir: str,
    use_quantizable_mha: bool = False,
) -> torch.nn.Module:
    """Load ``model.pt`` if it exists; otherwise train a quick MNIST Transformer.

    ``use_quantizable_mha`` (default False) applies AFTER loading the
    checkpoint so a model trained with legacy ``nn.MultiheadAttention``
    can still be swapped over to :class:`QuantizedMHA` for the accuracy
    sweep without changing the on-disk state_dict layout.
    """
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
    from torchvision import datasets, transforms as T
    tf = T.ToTensor()
    data_dir = os.path.expanduser(data_dir)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
    g = torch.Generator().manual_seed(0)
    indices = torch.randperm(len(test_ds), generator=g)[:n_samples].tolist()
    subset = tdata.Subset(test_ds, indices)
    return tdata.DataLoader(subset, batch_size=batch_size, shuffle=False)


# ── Model-agnostic API ────────────────────────────────────────────────────────

def profile_model(
    config: FourBitConfig,
    model: nn.Module,
    loader: tdata.DataLoader,
    *,
    eval_fn: Optional[Callable[[nn.Module, tdata.DataLoader], float]] = None,
) -> Tuple[pd.DataFrame, dict, Optional[pd.DataFrame]]:
    """Collect and analyse any PyTorch model with ``nn.Linear`` layers.

    Parameters
    ----------
    config : FourBitConfig
        Experiment config (formats, transforms, metrics, output_dir, …).
    model : nn.Module
        A trained model in eval mode.  Every ``nn.Linear`` encountered in
        a forward pass is profiled.  The model is not mutated.
    loader : DataLoader
        Yields ``(x, ...)`` batches.  Only the first element ``x`` is
        forwarded through the model during tensor collection.  The same
        loader is reused for the accuracy sweep (if ``eval_fn`` is given),
        so it should be re-iterable (e.g. a standard ``DataLoader``).
    eval_fn : Callable[[nn.Module, DataLoader], float] | None
        Optional metric function with signature::

            def eval_fn(model: nn.Module, loader: DataLoader) -> float:
                ...

        When provided, the accuracy sweep runs FP32 + FP16 baselines and
        every (format × transform) combination, passing the (possibly
        quantised) model and the same loader to ``eval_fn``.  The returned
        float is stored in the ``accuracy`` column of
        ``results/<output_dir>/part2/accuracy_sweep.csv``.
        When ``None`` the sweep is skipped and the third return value is
        ``None``.

    Returns
    -------
    metrics_df : pd.DataFrame
        One row per (layer × format × transform).  Columns include
        ``qsnr_{w,x,y}_db``, ``fp16_qsnr_{w,x,y}_db``, raw tensor stats
        prefixed ``W_`` / ``X_`` / ``Y_``, and a ``reason`` diagnostic.
    layers : dict[str, LayerRecord]
        Raw collected tensors, usable by the reporter for figure-level
        aggregation.
    acc_df : pd.DataFrame | None
        FP32 / FP16 baseline + every (format × transform) score returned
        by ``eval_fn``, or ``None`` when ``eval_fn`` was not provided.
    """
    model = model.eval()
    print(f"[Part 2] Collecting tensors from {config.profile_samples} samples ...")
    with LayerCollector(model) as collector:
        with torch.no_grad():
            for batch in loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                model(x)

    print(f"[Part 2] Recorded {len(collector.layers)} Linear layers. Analysing ...")
    df = analyse_all(collector.layers, config)

    out_dir = os.path.join(config.output_dir, "part2")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "per_layer_metrics.csv"), index=False)

    acc_df: Optional[pd.DataFrame] = None
    if eval_fn is not None:
        print("[Part 2] Running accuracy sweep over all (format, transform) ...")
        acc_rows = accuracy_sweep(
            model, collector.layers, loader, config, eval_fn=eval_fn,
        )
        acc_df = pd.DataFrame(acc_rows)
        acc_df.to_csv(os.path.join(out_dir, "accuracy_sweep.csv"), index=False)

    return df, collector.layers, acc_df


# ── MNIST wrapper (backward-compatible) ──────────────────────────────────────

def run(
    config: FourBitConfig,
    model_path: str = "results/mnist/model.pt",
    data_dir: str = "~/.cache/mnist",
) -> Tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Load (or train) the MNIST Transformer and delegate to ``profile_model``.

    This is the original entry point used by ``cli.py``; it preserves the
    exact same behaviour and return types as before.  For arbitrary models
    call :func:`profile_model` directly.
    """
    model = _load_or_train_model(
        model_path, data_dir,
        use_quantizable_mha=getattr(config, "use_quantizable_mha", False),
    )
    loader = _make_loader(data_dir, config.profile_samples)

    def _mnist_eval_fn(m: nn.Module, ldr: tdata.DataLoader) -> float:
        # Pass fp16 inputs when the model was cast to half precision.
        try:
            w = next(iter(m.parameters()))
            return _eval_accuracy(m, ldr, dtype=w.dtype)
        except StopIteration:
            return _eval_accuracy(m, ldr)

    df, layers, acc_df = profile_model(
        config, model, loader, eval_fn=_mnist_eval_fn,
    )
    # run() always returns a DataFrame (never None) for backward compat.
    if acc_df is None:  # pragma: no cover
        acc_df = pd.DataFrame()
    return df, layers, acc_df

# MNIST Transformer Analysis Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Three runnable scripts — train a minimal Transformer on MNIST, profile it with ModelProfiler across 14 quantization formats, and generate a self-contained HTML accuracy report.

**Architecture:** `examples/model.py` defines `MNISTTransformer`; three independent scripts (`train_mnist.py`, `profile_mnist.py`, `generate_report.py`) each read/write `results/mnist/`. All scripts add the project root to `sys.path` so they can import `profiler/` and `formats/` without installation.

**Tech Stack:** PyTorch (model + training), torchvision (MNIST), ModelProfiler (existing `profiler/` package), matplotlib (charts → base64 inline PNG), pandas (CSV), Python `json` + `html` stdlib (report).

---

## Task 1: MNISTTransformer Model

**Files:**
- Create: `examples/__init__.py` (empty)
- Create: `examples/model.py`
- Test: `tests/test_mnist_model.py`

### Step 1: Write failing tests

```python
# tests/test_mnist_model.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
import pytest
from examples.model import MNISTTransformer


class TestMNISTTransformer:
    def test_output_shape(self):
        model = MNISTTransformer()
        x = torch.randn(4, 1, 28, 28)
        out = model(x)
        assert out.shape == (4, 10), f"Expected (4, 10), got {out.shape}"

    def test_batch_size_1(self):
        model = MNISTTransformer()
        x = torch.randn(1, 1, 28, 28)
        out = model(x)
        assert out.shape == (1, 10)

    def test_param_count_reasonable(self):
        model = MNISTTransformer()
        n_params = sum(p.numel() for p in model.parameters())
        assert 100_000 < n_params < 1_000_000, f"Unexpected param count: {n_params}"

    def test_no_nan_output(self):
        model = MNISTTransformer()
        x = torch.randn(8, 1, 28, 28)
        out = model(x)
        assert not torch.isnan(out).any(), "NaN in output"

    def test_gradients_flow(self):
        model = MNISTTransformer()
        x = torch.randn(2, 1, 28, 28)
        loss = model(x).sum()
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
```

### Step 2: Run to verify FAIL

```bash
cd /Users/mozzie/Desktop/Projects/formatresearch/dataformat
pytest tests/test_mnist_model.py -v
```
Expected: `ModuleNotFoundError: No module named 'examples'`

### Step 3: Create examples/__init__.py and examples/model.py

```python
# examples/__init__.py
# (empty)
```

```python
# examples/model.py
"""Minimal Transformer for MNIST classification.

Architecture:
  - Treat each row of 28 pixels as one token → sequence of 28 tokens
  - Prepend a learnable [CLS] token → sequence length 29
  - 2× TransformerEncoderLayer (d_model=128, nhead=4, dim_ff=256)
  - Classify from CLS token output
"""
import torch
import torch.nn as nn


class MNISTTransformer(nn.Module):
    """Transformer classifier for MNIST (28×28 grayscale images).

    Parameters
    ----------
    d_model : int
        Embedding dimension (default 128).
    nhead : int
        Number of attention heads (default 4).
    num_layers : int
        Number of TransformerEncoder layers (default 2).
    dim_ff : int
        Feedforward dimension (default 256).
    dropout : float
        Dropout rate (default 0.1).
    """

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Embed each row of 28 pixels into d_model dimensions
        self.embed = nn.Linear(28, d_model)
        # Learnable [CLS] token and positional encoding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 29, d_model))  # 28 rows + CLS
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Classifier head
        self.classifier = nn.Linear(d_model, 10)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, 1, 28, 28)

        Returns
        -------
        torch.Tensor of shape (B, 10) — class logits
        """
        B = x.size(0)
        x = x.squeeze(1)                              # (B, 28, 28)
        x = self.embed(x)                             # (B, 28, d_model)
        cls = self.cls_token.expand(B, -1, -1)        # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                # (B, 29, d_model)
        x = x + self.pos_embed                        # add positional encoding
        x = self.encoder(x)                           # (B, 29, d_model)
        return self.classifier(x[:, 0])               # CLS token → (B, 10)
```

### Step 4: Run tests

```bash
pytest tests/test_mnist_model.py -v
```
Expected: all 5 PASS.

### Step 5: Commit

```bash
git add examples/__init__.py examples/model.py tests/test_mnist_model.py
git commit -m "feat: add MNISTTransformer model"
```

---

## Task 2: Training Script

**Files:**
- Create: `examples/train_mnist.py`
- Test: `tests/test_train_mnist.py`

### Step 1: Write failing tests

```python
# tests/test_train_mnist.py
"""Smoke tests for training utilities — does NOT run full training."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
import torch.nn as nn
import pytest
from examples.model import MNISTTransformer


def _dummy_loader(n_batches=2, batch_size=4):
    """Returns a tiny fake dataloader (no torchvision required)."""
    for _ in range(n_batches):
        yield torch.randn(batch_size, 1, 28, 28), torch.randint(0, 10, (batch_size,))


class TestTrainingUtilities:
    def test_run_epoch_train_returns_loss_and_acc(self):
        from examples.train_mnist import run_epoch
        model = MNISTTransformer()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss, acc = run_epoch(model, _dummy_loader(), criterion, optimizer)
        assert 0.0 < loss < 100.0
        assert 0.0 <= acc <= 1.0

    def test_run_epoch_eval_no_grad(self):
        from examples.train_mnist import run_epoch
        model = MNISTTransformer()
        criterion = nn.CrossEntropyLoss()
        # optimizer=None → eval mode
        loss, acc = run_epoch(model, _dummy_loader(), criterion, optimizer=None)
        assert loss >= 0.0
        # weights unchanged (no grad applied)
        for p in model.parameters():
            assert p.grad is None

    def test_run_epoch_updates_weights(self):
        from examples.train_mnist import run_epoch
        model = MNISTTransformer()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = [p.clone().detach() for p in model.parameters()]
        run_epoch(model, _dummy_loader(), criterion, optimizer)
        after = list(model.parameters())
        changed = any(not torch.equal(b, a.detach()) for b, a in zip(before, after))
        assert changed, "Weights did not change after training step"
```

### Step 2: Run to verify FAIL

```bash
pytest tests/test_train_mnist.py -v
```
Expected: `ModuleNotFoundError: No module named 'examples.train_mnist'`

### Step 3: Implement examples/train_mnist.py

```python
# examples/train_mnist.py
"""Train MNISTTransformer on the full MNIST dataset.

Usage:
    python examples/train_mnist.py [--epochs 10] [--batch-size 256] [--out-dir results/mnist]

Saves:
    results/mnist/model.pt           — trained weights (state_dict)
    results/mnist/training_log.json  — per-epoch loss/accuracy curves
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from examples.model import MNISTTransformer


def get_loaders(batch_size: int = 256, data_dir: str = "~/.cache/mnist"):
    """Return (train_loader, test_loader) for MNIST."""
    from torchvision import datasets, transforms
    tf = transforms.ToTensor()
    train_ds = datasets.MNIST(data_dir, train=True,  download=True, transform=tf)
    test_ds  = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=512,        shuffle=False, num_workers=0)
    return train_loader, test_loader


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer=None,
    device: str = "cpu",
) -> tuple[float, float]:
    """Run one epoch. If optimizer is None, runs in eval mode without grad.

    Returns
    -------
    (mean_loss, accuracy)  — accuracy in [0, 1]
    """
    if optimizer is not None:
        model.train()
    else:
        model.eval()

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct    += (logits.argmax(1) == y).sum().item()
            total      += len(y)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train MNISTTransformer")
    parser.add_argument("--epochs",     type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--out-dir",    default="results/mnist")
    parser.add_argument("--data-dir",   default="~/.cache/mnist")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cpu"

    print("Loading MNIST...")
    train_loader, test_loader = get_loaders(args.batch_size, args.data_dir)

    model     = MNISTTransformer().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    log: dict = {"epoch": [], "train_loss": [], "test_loss": [],
                 "train_acc": [], "test_acc": []}

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        te_loss, te_acc = run_epoch(model, test_loader,  criterion, None,      device)
        scheduler.step()

        log["epoch"].append(epoch)
        log["train_loss"].append(round(tr_loss, 4))
        log["test_loss"].append(round(te_loss, 4))
        log["train_acc"].append(round(tr_acc * 100, 2))
        log["test_acc"].append(round(te_acc * 100, 2))

        print(
            f"Epoch {epoch:2d}/{args.epochs} | "
            f"loss={tr_loss:.4f} | train_acc={tr_acc*100:.1f}% | test_acc={te_acc*100:.1f}%"
        )

    model_path = os.path.join(args.out_dir, "model.pt")
    log_path   = os.path.join(args.out_dir, "training_log.json")
    torch.save(model.state_dict(), model_path)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nSaved model      → {model_path}")
    print(f"Saved training log → {log_path}")
    print(f"Final test accuracy: {log['test_acc'][-1]:.1f}%")


if __name__ == "__main__":
    main()
```

### Step 4: Run tests

```bash
pytest tests/test_train_mnist.py -v
```
Expected: all 3 PASS.

### Step 5: Commit

```bash
git add examples/train_mnist.py tests/test_train_mnist.py
git commit -m "feat: add MNIST training script with run_epoch utility"
```

---

## Task 3: Profiling Script

**Files:**
- Create: `examples/profile_mnist.py`
- Test: `tests/test_profile_mnist.py`

### Step 1: Write failing tests

```python
# tests/test_profile_mnist.py
"""Smoke tests for profiling script — uses a tiny dummy model, no MNIST download."""
import sys, os, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
import torch.nn as nn
import torch.utils.data as data
import pandas as pd
import pytest

from profiler import ModelProfiler


def _make_tiny_model():
    return nn.Sequential(nn.Flatten(), nn.Linear(784, 32), nn.ReLU(), nn.Linear(32, 10))


def _make_tiny_loader(n=16, batch_size=8):
    xs = torch.randn(n, 1, 28, 28)
    ys = torch.randint(0, 10, (n,))
    ds = data.TensorDataset(xs, ys)
    return data.DataLoader(ds, batch_size=batch_size)


class TestProfileMnist:
    def test_profiler_runs_and_produces_csv(self):
        from examples.profile_mnist import run_profiling
        model = _make_tiny_model()
        loader = _make_tiny_loader()
        with tempfile.TemporaryDirectory() as d:
            csv_path = run_profiling(model, loader, out_dir=d)
            assert os.path.exists(csv_path)
            df = pd.read_csv(csv_path)
            assert len(df) > 0
            assert "format" in df.columns
            assert "mse" in df.columns

    def test_all_14_formats_in_csv(self):
        from examples.profile_mnist import run_profiling
        from profiler.formats import PROFILER_FORMAT_NAMES
        model = _make_tiny_model()
        loader = _make_tiny_loader()
        with tempfile.TemporaryDirectory() as d:
            csv_path = run_profiling(model, loader, out_dir=d)
            df = pd.read_csv(csv_path)
            assert set(df["format"].unique()) == set(PROFILER_FORMAT_NAMES)

    def test_fp32_mse_is_zero(self):
        from examples.profile_mnist import run_profiling
        model = _make_tiny_model()
        loader = _make_tiny_loader()
        with tempfile.TemporaryDirectory() as d:
            csv_path = run_profiling(model, loader, out_dir=d)
            df = pd.read_csv(csv_path)
            fp32 = df[df["format"] == "FP32"]["mse"].dropna()
            assert (fp32 == 0.0).all(), "FP32 should have zero MSE"
```

### Step 2: Run to verify FAIL

```bash
pytest tests/test_profile_mnist.py -v
```
Expected: `ImportError: cannot import name 'run_profiling' from 'examples.profile_mnist'`

### Step 3: Implement examples/profile_mnist.py

```python
# examples/profile_mnist.py
"""Profile a trained MNISTTransformer across all 14 quantization formats.

Usage:
    python examples/profile_mnist.py [--model-path results/mnist/model.pt]
                                     [--out-dir results/mnist]
                                     [--n-samples 256]

Requires: results/mnist/model.pt (run train_mnist.py first)
Saves:    results/mnist/profiler_results.csv
"""
from __future__ import annotations
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.utils.data as data

from profiler import ModelProfiler


def run_profiling(
    model: nn.Module,
    loader,
    out_dir: str = "results/mnist",
) -> str:
    """Run ModelProfiler on *model* using data from *loader*.

    Parameters
    ----------
    model : nn.Module
        Model in eval mode (set externally or here).
    loader : DataLoader
        Yields (x, y) batches; only x is used.
    out_dir : str
        Directory to write profiler_results.csv into.

    Returns
    -------
    str
        Absolute path to the written CSV file.
    """
    model.eval()
    profiler = ModelProfiler(model)
    total = len(profiler._formats)

    while not profiler.done:
        idx  = profiler._format_idx + 1
        name = profiler.current_format_name
        print(f"  Profiling format {idx:2d}/{total}: {name:<20s}", end="\r", flush=True)

        profiler.start()
        with torch.no_grad():
            for batch in loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                model(x)
        profiler.stop()

    print(f"  Done — {total} formats profiled.{' ' * 20}")
    return profiler.export_csv(out_dir)


def main():
    parser = argparse.ArgumentParser(description="Profile MNISTTransformer quantization")
    parser.add_argument("--model-path", default="results/mnist/model.pt")
    parser.add_argument("--out-dir",    default="results/mnist")
    parser.add_argument("--n-samples",  type=int, default=256,
                        help="Number of test images to use for profiling (default 256)")
    parser.add_argument("--data-dir",   default="~/.cache/mnist")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"ERROR: model not found at {args.model_path}")
        print("Run: python examples/train_mnist.py first")
        sys.exit(1)

    from examples.model import MNISTTransformer
    from torchvision import datasets, transforms

    print(f"Loading model from {args.model_path} ...")
    model = MNISTTransformer()
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.eval()

    print(f"Loading {args.n_samples} test images ...")
    test_ds = datasets.MNIST(args.data_dir, train=False, download=True,
                              transform=transforms.ToTensor())
    indices = torch.randperm(len(test_ds))[:args.n_samples]
    subset  = data.Subset(test_ds, indices.tolist())
    loader  = data.DataLoader(subset, batch_size=32, shuffle=False)

    print(f"Running profiling across all 14 formats ...")
    csv_path = run_profiling(model, loader, out_dir=args.out_dir)
    print(f"Saved → {csv_path}")


if __name__ == "__main__":
    main()
```

### Step 4: Run tests

```bash
pytest tests/test_profile_mnist.py -v
```
Expected: all 3 PASS.

### Step 5: Commit

```bash
git add examples/profile_mnist.py tests/test_profile_mnist.py
git commit -m "feat: add MNIST profiling script"
```

---

## Task 4: HTML Report Generator

**Files:**
- Create: `examples/generate_report.py`
- Test: `tests/test_generate_report.py`

### Step 1: Write failing tests

```python
# tests/test_generate_report.py
"""Smoke tests for HTML report generation using synthetic CSV + log data."""
import sys, os, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import numpy as np
import pytest


def _make_sample_csv(path):
    """Write a minimal profiler_results.csv with 2 formats × 2 layers × 2 tensor types."""
    from profiler.formats import PROFILER_FORMAT_NAMES
    rows = []
    for fmt in ["FP32", "INT4(TENSOR)", "MXINT4", "HAD+INT4(C)"]:
        bits = 32 if fmt == "FP32" else 4
        for layer in ["0", "2"]:
            for tt in ["weight", "input"]:
                mse = 0.0 if fmt == "FP32" else np.random.uniform(1e-4, 1e-2)
                rows.append({
                    "format": fmt, "layer_name": layer, "layer_type": "Linear",
                    "tensor_type": tt, "bits": bits,
                    "mse": mse, "snr_db": 0.0 if mse == 0 else 10 * np.log10(0.1 / mse),
                    "eff_bits": 0.0 if mse == 0 else 0.5 * np.log2(0.1 / mse),
                    "max_ae": np.sqrt(mse), "mean": 0.0, "std": 0.3,
                    "outlier_ratio": 0.01, "n_batches": 5, "n_elements": 1000,
                })
    pd.DataFrame(rows).to_csv(path, index=False)


def _make_sample_log(path):
    log = {
        "epoch": list(range(1, 6)),
        "train_loss": [1.2, 0.5, 0.3, 0.2, 0.15],
        "test_loss":  [0.6, 0.4, 0.25, 0.18, 0.14],
        "train_acc":  [60.0, 85.0, 90.0, 94.0, 96.0],
        "test_acc":   [75.0, 87.0, 91.0, 94.5, 96.5],
    }
    with open(path, "w") as f:
        json.dump(log, f)


class TestGenerateReport:
    def test_report_html_created(self):
        from examples.generate_report import generate_report
        with tempfile.TemporaryDirectory() as d:
            csv_path = os.path.join(d, "profiler_results.csv")
            log_path = os.path.join(d, "training_log.json")
            _make_sample_csv(csv_path)
            _make_sample_log(log_path)
            out_path = generate_report(csv_path, log_path, d, open_browser=False)
            assert os.path.exists(out_path)
            assert out_path.endswith(".html")

    def test_report_is_self_contained_html(self):
        from examples.generate_report import generate_report
        with tempfile.TemporaryDirectory() as d:
            csv_path = os.path.join(d, "profiler_results.csv")
            log_path = os.path.join(d, "training_log.json")
            _make_sample_csv(csv_path)
            _make_sample_log(log_path)
            out_path = generate_report(csv_path, log_path, d, open_browser=False)
            content = open(out_path).read()
            assert "<!DOCTYPE html>" in content
            assert "data:image/png;base64," in content   # at least one inline image
            assert "FP32" in content                      # format names present
            assert "href" not in content or "http" not in content  # no external links

    def test_report_contains_all_sections(self):
        from examples.generate_report import generate_report
        with tempfile.TemporaryDirectory() as d:
            csv_path = os.path.join(d, "profiler_results.csv")
            log_path = os.path.join(d, "training_log.json")
            _make_sample_csv(csv_path)
            _make_sample_log(log_path)
            out_path = generate_report(csv_path, log_path, d, open_browser=False)
            content = open(out_path).read()
            for section in ["Training Curves", "EffBits", "Heatmap", "SNR", "Summary"]:
                assert section in content, f"Missing section: {section}"
```

### Step 2: Run to verify FAIL

```bash
pytest tests/test_generate_report.py -v
```
Expected: `ImportError: cannot import name 'generate_report'`

### Step 3: Implement examples/generate_report.py

```python
# examples/generate_report.py
"""Generate a self-contained HTML report from profiler results.

Usage:
    python examples/generate_report.py [--results-dir results/mnist]

Requires:
    results/mnist/profiler_results.csv  (from profile_mnist.py)
    results/mnist/training_log.json     (from train_mnist.py)

Saves:
    results/mnist/report.html  — opens automatically in default browser
"""
from __future__ import annotations
import argparse
import base64
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Chart helpers ──────────────────────────────────────────────────────────────

def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _plot_training_curves(log: dict) -> str:
    epochs = log["epoch"]
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(epochs, log["train_loss"], "b-",  label="Train Loss")
    ax1.plot(epochs, log["test_loss"],  "b--", label="Test Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax2 = ax1.twinx()
    ax2.plot(epochs, log["train_acc"], "r-",  label="Train Acc")
    ax2.plot(epochs, log["test_acc"],  "r--", label="Test Acc")
    ax2.set_ylabel("Accuracy (%)", color="r")
    ax2.tick_params(axis="y", labelcolor="r")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="center right")
    ax1.set_title("Training Curves")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_effbits_ranking(df: pd.DataFrame) -> str:
    summary   = df.groupby("format")["eff_bits"].mean().sort_values(ascending=True)
    bits_map  = df.groupby("format")["bits"].first()
    colors    = ["#2196F3" if bits_map.get(f, 8) <= 4 else "#4CAF50" for f in summary.index]
    fig, ax   = plt.subplots(figsize=(8, max(4, len(summary) * 0.4)))
    ax.barh(summary.index, summary.values, color=colors)
    ax.axvline(x=4, color="gray", linestyle="--", alpha=0.5, label="4-bit target")
    ax.axvline(x=8, color="gray", linestyle=":",  alpha=0.5, label="8-bit target")
    ax.set_xlabel("Mean EffBits")
    ax.set_title("EffBits Ranking  (blue = 4-bit formats, green = 8-bit+)")
    ax.legend()
    fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_layer_mse_heatmap(df: pd.DataFrame) -> str | None:
    weight_df = df[(df["tensor_type"] == "weight") & (df["format"] != "FP32")]
    if weight_df.empty:
        return None
    pivot = weight_df.pivot_table(
        values="mse", index="format", columns="layer_name", aggfunc="mean"
    )
    log_vals = np.log10(np.clip(pivot.values, 1e-12, None))
    fig, ax  = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.2), max(4, len(pivot) * 0.5)))
    im = ax.imshow(log_vals, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    plt.colorbar(im, ax=ax, label="log₁₀(MSE)")
    ax.set_title("Per-Layer Weight MSE Heatmap (log scale, FP32 excluded)")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_snr_comparison(df: pd.DataFrame) -> str:
    tensor_types = sorted(df["tensor_type"].dropna().unique())
    formats      = df["format"].unique()
    x     = np.arange(len(formats))
    width = 0.8 / max(len(tensor_types), 1)
    fig, ax = plt.subplots(figsize=(max(10, len(formats) * 0.8), 5))
    for i, tt in enumerate(tensor_types):
        sub  = df[df["tensor_type"] == tt].groupby("format")["snr_db"].mean()
        vals = [sub.get(f, float("nan")) for f in formats]
        ax.bar(x + i * width, vals, width, label=tt)
    ax.set_xticks(x + width * (len(tensor_types) - 1) / 2)
    ax.set_xticklabels(formats, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean SNR (dB)")
    ax.set_title("SNR by Format and Tensor Type")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.legend()
    fig.tight_layout()
    return _fig_to_b64(fig)


def _build_summary_table(df: pd.DataFrame) -> str:
    summary = (
        df.groupby("format")
        .agg(bits=("bits", "first"), mean_mse=("mse", "mean"),
             mean_snr_db=("snr_db", "mean"), mean_eff_bits=("eff_bits", "mean"),
             outlier_ratio=("outlier_ratio", "mean"))
        .sort_values("mean_eff_bits", ascending=False)
        .reset_index()
    )
    rows = []
    for _, r in summary.iterrows():
        bits_str = str(int(r["bits"])) if not pd.isna(r["bits"]) else "-"
        rows.append(
            f"<tr><td>{r['format']}</td><td>{bits_str}</td>"
            f"<td>{r['mean_mse']:.2e}</td><td>{r['mean_snr_db']:.1f}</td>"
            f"<td>{r['mean_eff_bits']:.2f}</td><td>{r['outlier_ratio']:.4f}</td></tr>"
        )
    return "\n".join(rows)


# ── Main ───────────────────────────────────────────────────────────────────────

def generate_report(
    csv_path: str,
    log_path: str,
    out_dir: str,
    open_browser: bool = True,
) -> str:
    """Generate self-contained HTML report.

    Returns absolute path to the written report.html.
    """
    df  = pd.read_csv(csv_path)
    with open(log_path) as f:
        log = json.load(f)

    final_acc  = log["test_acc"][-1]
    final_loss = log["test_loss"][-1]

    print("  Rendering training curves ...")
    img_training = _plot_training_curves(log)
    print("  Rendering EffBits ranking ...")
    img_effbits  = _plot_effbits_ranking(df)
    print("  Rendering MSE heatmap ...")
    img_heatmap  = _plot_layer_mse_heatmap(df)
    print("  Rendering SNR comparison ...")
    img_snr      = _plot_snr_comparison(df)
    table_rows   = _build_summary_table(df)

    heatmap_html = (
        f'<h2>3. Per-Layer Weight MSE Heatmap</h2>'
        f'<img src="data:image/png;base64,{img_heatmap}" style="max-width:100%"><hr>'
        if img_heatmap else ""
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MNIST Transformer Quantization Report</title>
<style>
  body  {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; color: #333; }}
  h1   {{ color: #222; }}
  h2   {{ color: #444; border-bottom: 1px solid #ddd; padding-bottom: 6px; margin-top: 30px; }}
  hr   {{ border: none; border-top: 1px solid #eee; margin: 20px 0; }}
  img  {{ max-width: 100%; }}
  .summary {{ background: #f5f5f5; padding: 14px 18px; border-radius: 6px; margin-bottom: 24px; font-size: 0.95em; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.9em; }}
  th, td {{ border: 1px solid #ddd; padding: 7px 10px; text-align: right; }}
  th {{ background: #f0f0f0; font-weight: bold; }}
  td:first-child, th:first-child {{ text-align: left; }}
  tr:hover {{ background: #fafafa; }}
</style>
</head>
<body>
<h1>MNIST Transformer — Quantization Analysis Report</h1>
<div class="summary">
  <strong>Model:</strong> MNISTTransformer (2 layers, d_model=128, nhead=4) &nbsp;|&nbsp;
  <strong>Final Test Accuracy:</strong> {final_acc:.1f}% &nbsp;|&nbsp;
  <strong>Final Test Loss:</strong> {final_loss:.4f} &nbsp;|&nbsp;
  <strong>Formats analysed:</strong> {df['format'].nunique()}
</div>

<h2>1. Training Curves</h2>
<img src="data:image/png;base64,{img_training}" style="max-width:100%">
<hr>

<h2>2. EffBits Ranking by Format</h2>
<img src="data:image/png;base64,{img_effbits}" style="max-width:100%">
<hr>

{heatmap_html}

<h2>4. SNR by Format and Tensor Type</h2>
<img src="data:image/png;base64,{img_snr}" style="max-width:100%">
<hr>

<h2>5. Summary Table (sorted by EffBits ↓)</h2>
<table>
<thead>
  <tr>
    <th>Format</th><th>Bits</th><th>Mean MSE</th>
    <th>Mean SNR (dB)</th><th>Mean EffBits</th><th>Outlier Ratio</th>
  </tr>
</thead>
<tbody>
{table_rows}
</tbody>
</table>

</body>
</html>"""

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    abs_path = os.path.abspath(out_path)
    print(f"  Report saved → {abs_path}")

    if open_browser:
        import webbrowser, pathlib
        webbrowser.open(pathlib.Path(abs_path).resolve().as_uri())

    return abs_path


def main():
    parser = argparse.ArgumentParser(description="Generate HTML quantization report")
    parser.add_argument("--results-dir", default="results/mnist")
    args = parser.parse_args()

    csv_path = os.path.join(args.results_dir, "profiler_results.csv")
    log_path = os.path.join(args.results_dir, "training_log.json")

    for p in [csv_path, log_path]:
        if not os.path.exists(p):
            print(f"ERROR: missing {p}")
            print("Run train_mnist.py and profile_mnist.py first.")
            sys.exit(1)

    print("Generating report ...")
    generate_report(csv_path, log_path, args.results_dir, open_browser=True)


if __name__ == "__main__":
    main()
```

### Step 4: Run tests

```bash
pytest tests/test_generate_report.py -v
```
Expected: all 3 PASS.

### Step 5: Run all tests

```bash
pytest tests/test_mnist_model.py tests/test_train_mnist.py tests/test_profile_mnist.py tests/test_generate_report.py -v
```
Expected: all 14 PASS.

### Step 6: Commit

```bash
git add examples/generate_report.py tests/test_generate_report.py
git commit -m "feat: add HTML report generator for quantization analysis"
```

---

## End-to-End Verification

After all tasks pass, run the full pipeline manually:

```bash
# Step 1: Train (3-5 min on CPU)
python examples/train_mnist.py --epochs 10

# Step 2: Profile (2-3 min on CPU)
python examples/profile_mnist.py --n-samples 256

# Step 3: Generate report (opens browser)
python examples/generate_report.py
```

Expected outputs:
```
results/mnist/model.pt
results/mnist/training_log.json
results/mnist/profiler_results.csv
results/mnist/report.html         ← opens automatically in browser
```

---

## Summary

| Task | Files | Tests |
|---|---|---|
| 1 | `examples/model.py` | `tests/test_mnist_model.py` (5 tests) |
| 2 | `examples/train_mnist.py` | `tests/test_train_mnist.py` (3 tests) |
| 3 | `examples/profile_mnist.py` | `tests/test_profile_mnist.py` (3 tests) |
| 4 | `examples/generate_report.py` | `tests/test_generate_report.py` (3 tests) |

**New dependency:** `torchvision` — install with:
```bash
pip install torchvision
```

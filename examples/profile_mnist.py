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
    data_dir = os.path.expanduser(args.data_dir)
    test_ds = datasets.MNIST(data_dir, train=False, download=True,
                              transform=transforms.ToTensor())
    indices = torch.randperm(len(test_ds))[:args.n_samples]
    subset  = data.Subset(test_ds, indices.tolist())
    loader  = data.DataLoader(subset, batch_size=32, shuffle=False)

    print(f"Running profiling across all 14 formats ...")
    csv_path = run_profiling(model, loader, out_dir=args.out_dir)
    print(f"Saved → {csv_path}")


if __name__ == "__main__":
    main()

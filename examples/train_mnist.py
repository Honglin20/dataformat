"""Train MNISTTransformer on the full MNIST dataset.

Usage:
    python examples/train_mnist.py [--epochs 10] [--batch-size 256] [--out-dir results/mnist]

Saves:
    results/mnist/model.pt           — trained weights (state_dict)
    results/mnist/training_log.json  — per-epoch loss/accuracy curves
"""
from __future__ import annotations
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from examples.model import MNISTTransformer


def get_loaders(batch_size: int = 256, data_dir: str = "~/.cache/mnist"):
    from torchvision import datasets, transforms
    tf = transforms.ToTensor()
    train_ds = datasets.MNIST(data_dir, train=True,  download=True, transform=tf)
    test_ds  = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0),
            DataLoader(test_ds,  batch_size=512,        shuffle=False, num_workers=0))


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    """Run one epoch. optimizer=None → eval mode, no grad update.

    Returns (mean_loss, accuracy) where accuracy is in [0, 1].
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

    print("Loading MNIST ...")
    train_loader, test_loader = get_loaders(args.batch_size, args.data_dir)

    model     = MNISTTransformer().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    log = {"epoch": [], "train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        te_loss, te_acc = run_epoch(model, test_loader,  criterion, None,      device)
        scheduler.step()

        log["epoch"].append(epoch)
        log["train_loss"].append(round(tr_loss, 4))
        log["test_loss"].append(round(te_loss, 4))
        log["train_acc"].append(round(tr_acc * 100, 2))
        log["test_acc"].append(round(te_acc * 100, 2))

        print(f"Epoch {epoch:2d}/{args.epochs} | loss={tr_loss:.4f} | "
              f"train_acc={tr_acc*100:.1f}% | test_acc={te_acc*100:.1f}%")

    model_path = os.path.join(args.out_dir, "model.pt")
    log_path   = os.path.join(args.out_dir, "training_log.json")
    torch.save(model.state_dict(), model_path)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nSaved model       → {model_path}")
    print(f"Saved training log → {log_path}")
    print(f"Final test accuracy: {log['test_acc'][-1]:.1f}%")


if __name__ == "__main__":
    main()

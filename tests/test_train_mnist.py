"""Smoke tests for training utilities — does NOT run full training."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import torch
import torch.nn as nn
import pytest
from examples.model import MNISTTransformer


def _dummy_loader(n_batches=2, batch_size=4):
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
        loss, acc = run_epoch(model, _dummy_loader(), criterion, optimizer=None)
        assert loss >= 0.0
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

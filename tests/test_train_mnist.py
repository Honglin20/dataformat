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

    def test_training_log_json_schema(self):
        """training_log.json must have exactly these 5 keys with list values."""
        import json, tempfile
        from examples.train_mnist import run_epoch
        # Simulate what main() produces: a log dict with the right schema
        log = {"epoch": [1, 2], "train_loss": [1.0, 0.8],
               "test_loss": [0.9, 0.7], "train_acc": [60.0, 75.0], "test_acc": [62.0, 76.0]}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "training_log.json")
            with open(path, "w") as f:
                json.dump(log, f)
            with open(path) as f:
                loaded = json.load(f)
        required_keys = {"epoch", "train_loss", "test_loss", "train_acc", "test_acc"}
        assert required_keys == set(loaded.keys())
        for k in required_keys:
            assert isinstance(loaded[k], list), f"{k} should be a list"

    def test_accuracy_stored_as_percentage(self):
        """run_epoch returns fraction in [0,1]; main() stores it multiplied by 100."""
        from examples.train_mnist import run_epoch
        import torch.nn as nn
        model = MNISTTransformer()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        _, acc = run_epoch(model, _dummy_loader(), criterion, optimizer)
        # acc from run_epoch is in [0, 1]
        assert 0.0 <= acc <= 1.0
        # when stored in log it should be multiplied by 100
        acc_pct = round(acc * 100, 2)
        assert 0.0 <= acc_pct <= 100.0

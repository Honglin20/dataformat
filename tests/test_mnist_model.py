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

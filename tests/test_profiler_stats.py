import numpy as np
import pytest
from profiler.stats import WelfordStats, RunningHistogram, QuantStats


class TestWelfordStats:
    def test_single_batch(self):
        s = WelfordStats()
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        s.update(x)
        r = s.finalize()
        assert abs(r["mean"] - 3.0) < 1e-9
        assert abs(r["std"] - np.std(x)) < 1e-6
        assert r["abs_max"] == 5.0
        assert r["n_elements"] == 5

    def test_two_batches_equivalent_to_one(self):
        s1 = WelfordStats()
        s1.update(np.array([1.0, 2.0, 3.0]))
        s1.update(np.array([4.0, 5.0]))
        s2 = WelfordStats()
        s2.update(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        r1, r2 = s1.finalize(), s2.finalize()
        assert abs(r1["mean"] - r2["mean"]) < 1e-9
        assert abs(r1["std"] - r2["std"]) < 1e-6

    def test_empty_raises(self):
        s = WelfordStats()
        with pytest.raises(RuntimeError):
            s.finalize()


class TestRunningHistogram:
    def test_basic(self):
        h = RunningHistogram()
        h.update(np.linspace(-3, 3, 1000))
        r = h.finalize()
        assert len(r["hist_counts"]) == 256
        assert r["outlier_ratio"] == 0.0

    def test_outliers_counted(self):
        h = RunningHistogram()
        h.update(np.linspace(0, 1, 100))   # sets range [0, 1]
        h.update(np.array([-5.0, 2.0]))    # both outside range → outliers
        r = h.finalize()
        assert r["outlier_ratio"] > 0.0

    def test_two_batches_same_total_as_one(self):
        h1 = RunningHistogram()
        h1.update(np.linspace(0, 1, 50))
        h1.update(np.linspace(0, 1, 50))
        h2 = RunningHistogram()
        h2.update(np.linspace(0, 1, 100))
        r1, r2 = h1.finalize(), h2.finalize()
        assert sum(r1["hist_counts"]) == sum(r2["hist_counts"])


class TestQuantStats:
    def test_perfect_reconstruction(self):
        s = QuantStats()
        x = np.array([1.0, 2.0, 3.0])
        s.update(x, x.copy())
        r = s.finalize()
        assert r["mse"] == 0.0
        assert r["max_ae"] == 0.0

    def test_known_mse(self):
        s = QuantStats()
        orig = np.array([0.0, 0.0, 0.0, 0.0])
        quant = np.array([1.0, 1.0, 1.0, 1.0])
        s.update(orig, quant)
        r = s.finalize()
        assert abs(r["mse"] - 1.0) < 1e-9
        assert r["max_ae"] == 1.0

    def test_incremental_equals_batch(self):
        s1 = QuantStats()
        s1.update(np.array([1.0, 2.0]), np.array([1.1, 2.1]))
        s1.update(np.array([3.0, 4.0]), np.array([3.1, 4.1]))
        s2 = QuantStats()
        s2.update(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.1, 2.1, 3.1, 4.1]))
        r1, r2 = s1.finalize(), s2.finalize()
        assert abs(r1["mse"] - r2["mse"]) < 1e-9

"""Model profiler for per-layer tensor distribution analysis.

Quick start:
    from profiler import ModelProfiler

    profiler = ModelProfiler(model)
    while not profiler.done:
        profiler.start()
        for batch in loader:
            model(batch)
        profiler.stop()
    profiler.export_csv("results/")
"""
from profiler.profiler import ModelProfiler

__all__ = ["ModelProfiler"]

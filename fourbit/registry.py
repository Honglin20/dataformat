"""Build format / transform instances from :class:`FourBitConfig`."""
from __future__ import annotations

from typing import Dict, List

from fourbit.config import FourBitConfig
from fourbit.formats import make_format
from fourbit.transforms import Transform, make_transform
from fourbit.pipeline import Pipeline


def build_formats(config: FourBitConfig) -> Dict[str, object]:
    """Return ``{display_name: format_instance}`` per config.

    Format instances are stateless with respect to their configuration, so
    this is safe to cache for a full experiment run.
    """
    out: Dict[str, object] = {}
    for spec in config.formats:
        fmt = make_format(spec.factory, **spec.kwargs)
        fmt.name = spec.display_name
        out[spec.display_name] = fmt
    return out


def build_transforms(config: FourBitConfig) -> Dict[str, Transform]:
    """Return ``{display_name: transform_instance}`` per config.

    NOTE: Transforms can be *stateful* (SmoothQuant stores fitted scales).
    Callers that reuse the same transform across different (X, W) pairs must
    call ``Pipeline.fit(X, W)`` every time, or request a fresh transform
    instance via ``build_transforms`` once per call site.
    """
    out: Dict[str, Transform] = {}
    for spec in config.transforms:
        t = make_transform(spec.factory, **spec.kwargs)
        t.name = spec.display_name
        out[spec.display_name] = t
    return out


def make_fresh_transform(config: FourBitConfig, display_name: str) -> Transform:
    """Instantiate a clean transform by display name (no shared state)."""
    for spec in config.transforms:
        if spec.display_name == display_name:
            t = make_transform(spec.factory, **spec.kwargs)
            t.name = spec.display_name
            return t
    raise KeyError(f"Transform '{display_name}' not in config")


def build_pipelines(config: FourBitConfig) -> List[Pipeline]:
    """Cartesian product of transforms × formats.

    Each Pipeline pairs a *fresh* transform instance with a format instance
    so running multiple experiments concurrently does not leak SmoothQuant
    scales between (X, W) pairs.
    """
    formats = build_formats(config)
    pipelines: List[Pipeline] = []
    for fmt_spec in config.formats:
        for t_spec in config.transforms:
            t = make_transform(t_spec.factory, **t_spec.kwargs)
            t.name = t_spec.display_name
            pipelines.append(Pipeline(
                transform=t,
                fmt=formats[fmt_spec.display_name],
                name=f"{t_spec.display_name}/{fmt_spec.display_name}",
            ))
    return pipelines

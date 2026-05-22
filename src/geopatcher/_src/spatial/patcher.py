"""`SpatialPatcher` — composes the four spatial axes.

The Patcher is intentionally tiny — it just orchestrates
``SpatialSampler.anchors → Geometry.neighborhood → SpatialWindow.weights →
Field.select`` and hands the result to `SpatialAggregation.merge` when the
caller asks. Split returns an `Iterator[Patch]` so streaming is the
default; ``list(patcher.split(field))`` materialises eagerly when that's
what's wanted.

See ``design.md`` §1 for the four-axis framework.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from geopatcher._src.patch import Patch
from geopatcher._src.protocols import AsyncField, Field
from geopatcher._src.spatial.aggregation import (
    SpatialAggregation,
    _warn_if_unsafe_streaming,
)
from geopatcher._src.spatial.geometry import (
    SpatialGeometry,
    _is_raster_domain,
    _MaskedWindow,
)
from geopatcher._src.spatial.sampler import SpatialSampler
from geopatcher._src.spatial.window import SpatialWindow


@dataclass(eq=False)
class SpatialPatcher:
    """The four-axis spatial Patcher.

    Args:
        geometry: How a neighborhood is shaped around an anchor.
        sampler: Where anchors go.
        window: Boundary treatment / per-pixel weights.
        aggregation: Local → global merge strategy.

    Examples:
        Sliding-window inference over a raster::

            patcher = SpatialPatcher(
                geometry    = SpatialRectangular(size=(256, 256)),
                sampler     = SpatialRegularStride(step=(192, 192)),
                window      = SpatialHann(),
                aggregation = SpatialOverlapAdd(),
            )
            patches = list(patcher.split(field))
            outs    = [run_operator(p) for p in patches]
            stitched = patcher.merge(outs, field.domain)
    """

    geometry: SpatialGeometry
    sampler: SpatialSampler
    window: SpatialWindow
    aggregation: SpatialAggregation

    def split(self, field: Field) -> Iterator[Patch]:
        """Yield patches lazily — one per anchor placed by the sampler."""
        domain = field.domain
        try:
            base_weights = self.window.weights(self.geometry)
        except TypeError:
            base_weights = None
        boundary = getattr(self.geometry, "boundary", "drop")
        for anchor in self.sampler.anchors(domain, self.geometry):
            indices = self.geometry.neighborhood(domain, anchor)
            if boundary == "raise":
                _raise_if_overflows(indices, domain)
            data = field.select(_unwrap_for_select(indices))
            weights = _build_weights(indices, base_weights, boundary=boundary)
            yield Patch(data=data, anchor=anchor, indices=indices, weights=weights)

    def n_anchors(self, field: Field) -> int:
        """Number of patches `split(field)` will yield.

        Enumerates the sampler's anchors without touching the field —
        only the domain is consulted.

        Determinism contract: holds exactly for samplers that return the
        same anchor set on every call given the same ``(domain,
        geometry)``. That covers all five samplers when a seed is set;
        for unseeded `SpatialRandom` / `SpatialJitteredStride` /
        `SpatialPoissonDisk` the count is still well-defined
        (``n_samples`` for the first two; a probabilistic estimate for
        the third), but the anchors materialised here are different
        draws from the ones a subsequent `split` will see. See
        ``docs/decisions.md`` (ADR-001) for why `split` returns an
        iterator and this helper exists as the ``len`` substitute.
        """
        return sum(1 for _ in self.sampler.anchors(field.domain, self.geometry))

    def merge(self, patches: Iterable[Any], domain: Any) -> Any:
        """Hand off to the aggregation; warn on streaming-unsafe types."""
        _warn_if_unsafe_streaming(self.aggregation)
        return self.aggregation.merge(patches, domain)

    def get_config(self) -> dict[str, Any]:
        return {
            "geometry": {
                "class": type(self.geometry).__name__,
                "config": self.geometry.get_config(),
            },
            "sampler": {
                "class": type(self.sampler).__name__,
                "config": self.sampler.get_config(),
            },
            "window": {
                "class": type(self.window).__name__,
                "config": self.window.get_config(),
            },
            "aggregation": {
                "class": type(self.aggregation).__name__,
                "config": self.aggregation.get_config(),
            },
        }


@dataclass(eq=False)
class AsyncSpatialPatcher:
    """Async mirror of `SpatialPatcher` over an `AsyncField`.

    `split` is an ``async for``-able iterator. Useful with
    `AsyncGeoTIFFReader` for high-concurrency per-tile fan-out.
    """

    geometry: SpatialGeometry
    sampler: SpatialSampler
    window: SpatialWindow
    aggregation: SpatialAggregation

    async def split(self, field: AsyncField) -> AsyncIterator[Patch]:
        domain = field.domain
        try:
            base_weights = self.window.weights(self.geometry)
        except TypeError:
            base_weights = None
        boundary = getattr(self.geometry, "boundary", "drop")
        for anchor in self.sampler.anchors(domain, self.geometry):
            indices = self.geometry.neighborhood(domain, anchor)
            if boundary == "raise":
                _raise_if_overflows(indices, domain)
            data = await field.select(_unwrap_for_select(indices))
            weights = _build_weights(indices, base_weights, boundary=boundary)
            yield Patch(data=data, anchor=anchor, indices=indices, weights=weights)

    def n_anchors(self, field: AsyncField) -> int:
        """Number of patches `split(field)` will yield.

        See `SpatialPatcher.n_anchors`.
        """
        return sum(1 for _ in self.sampler.anchors(field.domain, self.geometry))

    def merge(self, patches: Iterable[Any], domain: Any) -> Any:
        _warn_if_unsafe_streaming(self.aggregation)
        return self.aggregation.merge(patches, domain)


def _unwrap_for_select(indices: Any) -> Any:
    """Unwrap a `_MaskedWindow` to the underlying rasterio `Window` for `Field.select`.

    `SpatialPolygonIntersection.neighborhood` returns a `_MaskedWindow`
    so `_build_weights` can recover the interior mask. But `Field.select`
    expects a plain `Window` (or dict / index list) — the wrapper would
    confuse downstream readers like `RasterField.read_from_window`. Strip
    it here at the call boundary; keep the wrapper on `Patch.indices` so
    aggregation still sees the mask via `_resolve_indices`.
    """
    if isinstance(indices, _MaskedWindow):
        return indices.window
    return indices


def _build_weights(
    indices: Any,
    base_weights: np.ndarray | None,
    *,
    boundary: str = "drop",
) -> Any:
    """Resolve a patch's weight array.

    If the indices is a `_MaskedWindow` (SpatialPolygonIntersection on a raster),
    return the interior mask — the window controls *which pixels count*,
    not how heavily they're tapered. Otherwise return the geometry-shaped
    base weights from `SpatialWindow.weights`, cropped to the actual window
    size when boundary == "shrink" (because the window was clipped).
    """
    if isinstance(indices, _MaskedWindow):
        return indices.mask
    if boundary == "shrink" and base_weights is not None:
        h = getattr(indices, "height", None)
        w = getattr(indices, "width", None)
        if h is not None and w is not None:
            bh, bw = base_weights.shape[-2:]
            if (h, w) != (bh, bw):
                return base_weights[..., : int(h), : int(w)]
    return base_weights


def _raise_if_overflows(indices: Any, domain: Any) -> None:
    """Raise ``ValueError`` if ``indices`` extends past ``domain``.

    Used by `SpatialPatcher.split` when the geometry's ``boundary``
    policy is ``"raise"``. Only meaningful for raster-shaped indices
    (rasterio `Window`); non-raster indices return early.
    """
    if not (hasattr(indices, "row_off") and hasattr(indices, "col_off")):
        return
    if not (hasattr(domain, "shape") and len(domain.shape) >= 2):
        return
    dh, dw = int(domain.shape[-2]), int(domain.shape[-1])
    r0, c0 = int(indices.row_off), int(indices.col_off)
    rh, cw = int(indices.height), int(indices.width)
    if r0 < 0 or c0 < 0 or r0 + rh > dh or c0 + cw > dw:
        raise ValueError(
            f"patch window {indices!r} overflows the domain shape "
            f"({dh}, {dw}); set boundary='pad' or 'shrink' to allow."
        )


# Re-export `_is_raster_domain` to discourage cross-imports from geometry.py.
__all__ = [
    "AsyncSpatialPatcher",
    "SpatialPatcher",
    "_is_raster_domain",
]

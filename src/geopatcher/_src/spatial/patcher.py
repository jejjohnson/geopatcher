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
        base_weights = _safe_base_weights(self.window, self.geometry)
        boundary = getattr(self.geometry, "boundary", "drop")
        for anchor in self.sampler.anchors(domain, self.geometry):
            yield _build_patch(
                field, domain, anchor, self.geometry, base_weights, boundary
            )

    def patch_at(self, field: Field, anchor: Any) -> Patch:
        """Read a single `Patch` at a specific anchor.

        The same geometry → ``field.select`` → window-weights pipeline
        as `split`, but driven by one explicit anchor instead of
        walking the sampler. Designed for random-access ML datasets
        (torch `Dataset.__getitem__`, Grain `RandomAccessDataSource`)
        that need lazy single-patch reads without materialising the
        whole iterator first.

        Args:
            field: The `Field` to read from.
            anchor: An anchor in the same format the sampler emits
                (e.g. ``(row, col)`` for raster, ``dict`` for grid).
                Typically obtained from
                ``patcher.anchors(field)[index]``.

        Returns:
            A single `Patch` bit-identical to the one ``split`` would
            yield for the same anchor.
        """
        domain = field.domain
        base_weights = _safe_base_weights(self.window, self.geometry)
        boundary = getattr(self.geometry, "boundary", "drop")
        return _build_patch(
            field, domain, anchor, self.geometry, base_weights, boundary
        )

    def anchors(self, field: Field) -> list[Any]:
        """Materialise the sampler's anchor sequence for ``field``.

        Returns the same sequence ``split(field)`` walks, as a list
        the caller can ``len()`` and index. Same determinism contract
        as `n_anchors` (deterministic given an int sampler seed,
        re-drawn when seed is ``None``).
        """
        return list(self.sampler.anchors(field.domain, self.geometry))

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
        base_weights = _safe_base_weights(self.window, self.geometry)
        boundary = getattr(self.geometry, "boundary", "drop")
        for anchor in self.sampler.anchors(domain, self.geometry):
            yield await _build_patch_async(
                field, domain, anchor, self.geometry, base_weights, boundary
            )

    async def patch_at(self, field: AsyncField, anchor: Any) -> Patch:
        """Read a single `Patch` at a specific anchor.

        Async mirror of `SpatialPatcher.patch_at` — the read goes
        through ``await field.select(...)``. Designed for random-access
        cloud-tile readers driving a Grain / torch `Dataset` with
        per-item HTTP fan-out.
        """
        domain = field.domain
        base_weights = _safe_base_weights(self.window, self.geometry)
        boundary = getattr(self.geometry, "boundary", "drop")
        return await _build_patch_async(
            field, domain, anchor, self.geometry, base_weights, boundary
        )

    def anchors(self, field: AsyncField) -> list[Any]:
        """Materialise the sampler's anchor sequence for ``field``.

        Anchors are placed without touching the field, so this is sync
        even on the async patcher. See `SpatialPatcher.anchors`.
        """
        return list(self.sampler.anchors(field.domain, self.geometry))

    def n_anchors(self, field: AsyncField) -> int:
        """Number of patches `split(field)` will yield.

        See `SpatialPatcher.n_anchors`.
        """
        return sum(1 for _ in self.sampler.anchors(field.domain, self.geometry))

    def merge(self, patches: Iterable[Any], domain: Any) -> Any:
        _warn_if_unsafe_streaming(self.aggregation)
        return self.aggregation.merge(patches, domain)


def _safe_base_weights(
    window: SpatialWindow, geometry: SpatialGeometry
) -> np.ndarray | None:
    """Compute the geometry-shaped base weights, or `None` for windows
    that don't expose a static weight grid (e.g. graph-based geometries
    where weights are anchor-dependent)."""
    try:
        return window.weights(geometry)
    except TypeError:
        return None


def _build_patch(
    field: Field,
    domain: Any,
    anchor: Any,
    geometry: SpatialGeometry,
    base_weights: np.ndarray | None,
    boundary: str,
) -> Patch:
    """Single-anchor read pipeline shared by `split` and `patch_at`."""
    indices = geometry.neighborhood(domain, anchor)
    if boundary == "raise":
        _raise_if_overflows(indices, domain)
    data = field.select(_unwrap_for_select(indices))
    weights = _build_weights(indices, base_weights, boundary=boundary)
    return Patch(data=data, anchor=anchor, indices=indices, weights=weights)


async def _build_patch_async(
    field: AsyncField,
    domain: Any,
    anchor: Any,
    geometry: SpatialGeometry,
    base_weights: np.ndarray | None,
    boundary: str,
) -> Patch:
    """Async mirror of `_build_patch` — awaits `field.select`."""
    indices = geometry.neighborhood(domain, anchor)
    if boundary == "raise":
        _raise_if_overflows(indices, domain)
    data = await field.select(_unwrap_for_select(indices))
    weights = _build_weights(indices, base_weights, boundary=boundary)
    return Patch(data=data, anchor=anchor, indices=indices, weights=weights)


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

"""`Patch` carrier — the unit of work that flows through a Patcher.

A `Patch` bundles four things together: the data slice that the operator
sees, the anchor that places it in the global field, the backend-specific
indices used to extract it, and the optional window weights used to taper
edges or mark interior pixels (e.g. for `PolygonIntersection`).

The fields are intentionally type-erased (`Any`) at the carrier level -
the type-narrowing happens per (Geometry x Domain) pairing, captured by
the `Patch[AnchorT, IndicesT, DataT]` generic parameters in user code.
See `examples.md` §Summary for the table.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(eq=False)
class Patch[AnchorT, IndicesT, DataT]:
    """A single patch produced by a `SpatialPatcher`.

    Args:
        data: The substrate slice the operator consumes (a `GeoTensor`,
            a sub-`DataArray`, a `GeoDataFrame` slice, …).
        anchor: Where this patch lives in the global field. Shape varies
            with the `Sampler` (pixel coords, lat/lon, polygon id, …).
        indices: Backend-specific indexer used to extract `data` from
            the parent field (a `rasterio.windows.Window`, a
            `dict[str, slice]`, a list of row indices, …).
        weights: Optional window weights — used by `OverlapAdd` and
            `WeightedSum` aggregations, and by `PolygonIntersection` to
            carry an interior mask. ``None`` is equivalent to a Boxcar.
    """

    data: DataT
    anchor: AnchorT
    indices: IndicesT
    weights: Any | None = None
    _release: Callable[[], None] | None = field(default=None, repr=False, compare=False)

    def with_data[NewDataT](self, data: NewDataT) -> Patch[AnchorT, IndicesT, NewDataT]:
        """Return a data-replaced copy that does not own this patch's release slot."""
        return replace(self, data=data, _release=None)

    def close(self) -> None:
        """Release any iterator backpressure slot held by this patch."""
        release, self._release = self._release, None
        if release is not None:
            release()

    def __enter__(self) -> Patch[AnchorT, IndicesT, DataT]:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


@dataclass(eq=False)
class TemporalPatch[AnchorT, IndicesT, DataT]:
    """A single patch produced by a `TemporalPatcher`.

    Mirrors `Patch` but indexes along the time axis only.
    """

    data: DataT
    anchor: AnchorT
    indices: IndicesT
    weights: Any | None = None
    _release: Callable[[], None] | None = field(default=None, repr=False, compare=False)

    def with_data[NewDataT](
        self, data: NewDataT
    ) -> TemporalPatch[AnchorT, IndicesT, NewDataT]:
        """Return a data-replaced copy that does not own this patch's release slot."""
        return replace(self, data=data, _release=None)

    def close(self) -> None:
        """Release any iterator backpressure slot held by this patch."""
        release, self._release = self._release, None
        if release is not None:
            release()

    def __enter__(self) -> TemporalPatch[AnchorT, IndicesT, DataT]:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


@dataclass(eq=False)
class SpatioTemporalPatch:
    """A patch carrying both a spatial and a temporal anchor.

    Produced by `SpatioTemporalPatcher`. The data field is the
    spatial-then-temporal (or coupled) slice; `space` and `time` keep the
    two anchors separately so downstream operators / aggregations can
    treat them as peers rather than packed.
    """

    data: Any
    space: Any
    time: Any
    spatial_indices: Any = None
    temporal_indices: Any = None
    weights: Any = field(default=None)
    _release: Callable[[], None] | None = field(default=None, repr=False, compare=False)

    def with_data(self, data: Any) -> SpatioTemporalPatch:
        """Return a data-replaced copy that does not own this patch's release slot."""
        return replace(self, data=data, _release=None)

    def close(self) -> None:
        """Release any iterator backpressure slot held by this patch."""
        release, self._release = self._release, None
        if release is not None:
            release()

    def __enter__(self) -> SpatioTemporalPatch:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

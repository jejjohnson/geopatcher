"""`TemporalPatcher` — composes the four time axes.

Mirror of `SpatialPatcher` over a 1-D time axis. The Patcher splits a
field along its time dimension; for each anchor it produces a
`TemporalPatch` of data sliced by `TemporalGeometry.window`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Iterator
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np

from geopatcher._src.hooks import (
    PatcherHook,
    _as_hooks,
    _dispatch,
    _len_or_unknown,
    _nbytes,
)
from geopatcher._src.patch import TemporalPatch
from geopatcher._src.prefetch import prefetch_iterable
from geopatcher._src.time.aggregation import TemporalAggregation
from geopatcher._src.time.geometry import TemporalGeometry
from geopatcher._src.time.sampler import TemporalSampler
from geopatcher._src.time.window import TemporalWindow


@dataclass(eq=False)
class TemporalPatcher:
    """Four-axis temporal Patcher.

    Args:
        geometry: How a temporal window is shaped around an anchor.
        sampler: Where time anchors are placed.
        window: Temporal boundary treatment (recency / taper / periodic).
        aggregation: Time → time merge strategy.

    Examples:
        Lookback + horizon forecasting on a ``(time, feature)`` array::

            tp = TemporalPatcher(
                geometry    = TemporalLookbackHorizon(lookback=12, horizon=6),
                sampler     = TemporalRegularStride(step=1),
                window      = TemporalCausalBoxcar(),
                aggregation = TemporalForecast(horizon=6),
            )
            patches = list(tp.split(series))
            preds   = [model(p.data) for p in patches]
            aligned = tp.merge(preds_as_patches)
    """

    geometry: TemporalGeometry
    sampler: TemporalSampler
    window: TemporalWindow
    aggregation: TemporalAggregation

    def split(
        self,
        series: Any,
        time_axis: int = 0,
        hooks: Iterable[PatcherHook] | None = None,
        *,
        prefetch: int = 0,
    ) -> Iterator[TemporalPatch]:
        """Yield temporal patches lazily.

        Args:
            series: Numpy array (or anything with ``shape`` + slicing) to
                slice along ``time_axis``.
            time_axis: Which axis is the time axis. Default 0.
            hooks: Optional observability hooks for split callbacks.
            prefetch: If positive, eagerly buffer up to ``prefetch`` patches
                in a background thread for I/O overlap.
        """
        return prefetch_iterable(self._split(series, time_axis, hooks=hooks), prefetch)

    def _split(
        self,
        series: Any,
        time_axis: int = 0,
        *,
        hooks: Iterable[PatcherHook] | None = None,
    ) -> Iterator[TemporalPatch]:
        arr = np.asarray(series)
        time_len = int(arr.shape[time_axis])
        hook_list = _as_hooks(hooks)
        if not hook_list:
            for anchor in self.sampler.anchors(time_len):
                yield from self._patches_for_anchor(
                    arr, time_len, int(anchor), time_axis
                )
            return
        anchors = [int(a) for a in self.sampler.anchors(time_len)]
        _dispatch(hook_list, "on_split_start", len(anchors))
        try:
            for anchor in anchors:
                yield from self._patches_for_anchor(
                    arr, time_len, anchor, time_axis, hook_list
                )
        finally:
            _dispatch(hook_list, "on_split_end")

    async def asplit(
        self,
        series: Any,
        time_axis: int = 0,
        *,
        hooks: Iterable[PatcherHook] | None = None,
    ) -> AsyncIterator[TemporalPatch]:
        """Async iterator mirror of `split` for async pipeline composition."""
        for patch in self._split(series, time_axis, hooks=hooks):
            yield patch

    def patches_at(
        self, series: Any, anchor: int, time_axis: int = 0
    ) -> list[TemporalPatch]:
        """Return the patches `split` would yield for a single anchor.

        Always a list — length 1 for the common single-slice
        geometries, length N for `TemporalMultiScale` and any future
        geometry that returns ``list[slice]`` (one entry per scale).
        The spatial counterpart returns a single `Patch`; the temporal
        side has to flatten the multi-scale case, so the return type
        is a list either way for callers to handle uniformly.

        Args:
            series: Same input shape as `split`.
            anchor: A single anchor value (typically from
                ``patcher.anchors(series)[index]``).
            time_axis: Which axis is the time axis. Default 0.
        """
        arr = np.asarray(series)
        time_len = int(arr.shape[time_axis])
        return list(self._patches_for_anchor(arr, time_len, int(anchor), time_axis))

    def anchors(self, series: Any, time_axis: int = 0) -> list[int]:
        """Materialise the sampler's anchor sequence for ``series``.

        Returns ``len(anchors) <= len(split(series))`` — multi-scale
        geometries emit multiple patches per anchor. Same determinism
        contract as `n_anchors`. See `SpatialPatcher.anchors`.
        """
        shape = getattr(series, "shape", None) or np.shape(series)
        time_len = int(shape[time_axis])
        return [int(a) for a in self.sampler.anchors(time_len)]

    def _patches_for_anchor(
        self,
        arr: np.ndarray,
        time_len: int,
        anchor: int,
        time_axis: int,
        hooks: Iterable[PatcherHook] = (),
    ) -> Iterator[TemporalPatch]:
        try:
            window = self.geometry.window(time_len, anchor)
        except Exception as exc:
            _dispatch(hooks, "on_error", anchor, exc)
            raise
        slices = window if isinstance(window, list) else [window]
        for s in slices:
            _dispatch(hooks, "on_patch_start", anchor)
            start = perf_counter()
            try:
                idx = [slice(None)] * arr.ndim
                idx[time_axis] = s
                data = arr[tuple(idx)]
                weights = self.window.weights(self.geometry, s.stop - s.start)
                patch = TemporalPatch(
                    data=data, anchor=anchor, indices=s, weights=weights
                )
            except Exception as exc:
                _dispatch(hooks, "on_error", anchor, exc)
                raise
            _dispatch(
                hooks,
                "on_patch_done",
                anchor,
                perf_counter() - start,
                _nbytes(patch.data),
            )
            yield patch

    def n_anchors(self, series: Any, time_axis: int = 0) -> int:
        """Number of patches `split(series)` will yield.

        Walks the sampler **and** the geometry's per-anchor window — a
        single geometry call may return a ``list[slice]`` (e.g.
        `TemporalMultiScale`), in which case `split` yields one patch
        per slice. We read ``series.shape`` rather than calling
        ``np.asarray(series)`` so generic / lazy series don't get
        materialised here. See ``docs/decisions.md`` (ADR-001).
        """
        shape = getattr(series, "shape", None) or np.shape(series)
        time_len = int(shape[time_axis])
        total = 0
        for anchor in self.sampler.anchors(time_len):
            window = self.geometry.window(time_len, int(anchor))
            total += len(window) if isinstance(window, list) else 1
        return total

    def merge(
        self, patches: Iterable[Any], hooks: Iterable[PatcherHook] | None = None
    ) -> Any:
        hook_list = _as_hooks(hooks)
        _dispatch(hook_list, "on_merge_start", _len_or_unknown(patches))
        try:
            output = self.aggregation.merge(patches)
        except Exception as exc:
            _dispatch(hook_list, "on_error", None, exc)
            raise
        _dispatch(hook_list, "on_merge_end", _nbytes(output))
        return output

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

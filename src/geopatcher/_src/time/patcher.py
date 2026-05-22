"""`TemporalPatcher` — composes the four time axes.

Mirror of `SpatialPatcher` over a 1-D time axis. The Patcher splits a
field along its time dimension; for each anchor it produces a
`TemporalPatch` of data sliced by `TemporalGeometry.window`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from geopatcher._src.patch import TemporalPatch
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

    def split(self, series: Any, time_axis: int = 0) -> Iterator[TemporalPatch]:
        """Yield temporal patches lazily.

        Args:
            series: Numpy array (or anything with ``shape`` + slicing) to
                slice along ``time_axis``.
            time_axis: Which axis is the time axis. Default 0.
        """
        arr = np.asarray(series)
        time_len = int(arr.shape[time_axis])
        for anchor in self.sampler.anchors(time_len):
            window = self.geometry.window(time_len, int(anchor))
            slices = window if isinstance(window, list) else [window]
            for s in slices:
                idx = [slice(None)] * arr.ndim
                idx[time_axis] = s
                data = arr[tuple(idx)]
                weights = self.window.weights(self.geometry, s.stop - s.start)
                yield TemporalPatch(
                    data=data, anchor=int(anchor), indices=s, weights=weights
                )

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

    def merge(self, patches: Iterable[Any]) -> Any:
        return self.aggregation.merge(patches)

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

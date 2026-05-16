"""Temporal counterparts of the spatial four-axis Patcher.

The shape of the API mirrors `geopatcher.spatial` exactly —
`TemporalGeometry`, `TemporalSampler`, `TemporalWindow`,
`TemporalAggregation` bases with concrete axes that drop the
``Temporal`` prefix (the submodule path provides the context). See
``design.md`` §5 for the time-axis framing.
"""

from __future__ import annotations

from geopatcher._src.time.aggregation import (
    TemporalAggregation,
    TemporalFold,
    TemporalForecast,
    TemporalHierarchicalCombine,
    TemporalMean,
)
from geopatcher._src.time.geometry import (
    TemporalFixedLookback,
    TemporalGeometry,
    TemporalLookbackHorizon,
    TemporalMultiScale,
    TemporalPhaseWindow,
)
from geopatcher._src.time.patcher import TemporalPatcher
from geopatcher._src.time.sampler import (
    TemporalCausalRolling,
    TemporalEventTriggered,
    TemporalExplicit,
    TemporalRandom,
    TemporalRegularStride,
    TemporalSampler,
)
from geopatcher._src.time.window import (
    TemporalCausalBoxcar,
    TemporalExponentialDecay,
    TemporalPeriodic,
    TemporalTaperedTukey,
    TemporalWindow,
)


__all__ = [
    "TemporalAggregation",
    "TemporalCausalBoxcar",
    "TemporalCausalRolling",
    "TemporalEventTriggered",
    "TemporalExplicit",
    "TemporalExponentialDecay",
    "TemporalFixedLookback",
    "TemporalFold",
    "TemporalForecast",
    "TemporalGeometry",
    "TemporalHierarchicalCombine",
    "TemporalLookbackHorizon",
    "TemporalMean",
    "TemporalMultiScale",
    "TemporalPatcher",
    "TemporalPeriodic",
    "TemporalPhaseWindow",
    "TemporalRandom",
    "TemporalRegularStride",
    "TemporalSampler",
    "TemporalTaperedTukey",
    "TemporalWindow",
]

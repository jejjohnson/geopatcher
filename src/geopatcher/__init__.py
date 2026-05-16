"""`geopatcher` — the four-axis Patcher framework.

Public surface re-exports:

- Carriers: `Patch`, `TemporalPatch`, `SpatioTemporalPatch`.
- Protocols: `Field`, `AsyncField`, `Domain`.
- Concrete domains: `RasterDomain`, `GridDomain`, `VectorDomain`, `PointDomain`.
- Field adapters: `RasterField`, `AsyncRasterField`. Non-raster adapters
  (`XarrayField`, `GeoPandasField`, `XvecField`, `RioXarrayField`) live
  under `geopatcher.fields` and lazy-import their optional extras.
- Top-level patchers: `SpatialPatcher`, `AsyncSpatialPatcher`,
  `TemporalPatcher`, `SpatioTemporalPatcher`.
- Spatial axes: re-exported from `geopatcher.spatial`.
- Temporal axes: re-exported from `geopatcher.time`.

Operator-graph wrappers (`GridSampler`, `ApplyToChips`, `Stitch`) that bridge
the patcher into a downstream composition library (e.g. `geotoolz`) live in
that library, not here — geopatcher itself has no Operator dependency.
"""

from __future__ import annotations

from geopatcher import fields, spatial, time
from geopatcher._src.domains import (
    GridDomain,
    PointDomain,
    RasterDomain,
    VectorDomain,
)
from geopatcher._src.fields import (
    AsyncRasterField,
    RasterField,
)
from geopatcher._src.patch import (
    Patch,
    SpatioTemporalPatch,
    TemporalPatch,
)
from geopatcher._src.protocols import (
    AsyncField,
    Domain,
    Field,
)
from geopatcher._src.spatial import (  # re-export of all spatial concretes + bases
    AsyncSpatialPatcher,
    SpatialAggregation,
    SpatialApproxCardinality,
    SpatialApproxMode,
    SpatialApproxQuantile,
    SpatialBoxcar,
    SpatialByIndex,
    SpatialCustom,
    SpatialExplicit,
    SpatialGaussian,
    SpatialGeometry,
    SpatialHann,
    SpatialHardVote,
    SpatialInvVarWeightedMean,
    SpatialJitteredStride,
    SpatialKNNGraph,
    SpatialLearned,
    SpatialMax,
    SpatialMean,
    SpatialMedian,
    SpatialMin,
    SpatialMode,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialPoissonDisk,
    SpatialPolygonIntersection,
    SpatialRadiusGraph,
    SpatialRandom,
    SpatialRectangular,
    SpatialRegularStride,
    SpatialReservoir,
    SpatialSampler,
    SpatialSoftVote,
    SpatialSphericalCap,
    SpatialStreamingHistogram,
    SpatialSum,
    SpatialTukey,
    SpatialVariance,
    SpatialWeightedSum,
    SpatialWindow,
)
from geopatcher._src.spatial_time import SpatioTemporalPatcher
from geopatcher._src.time import (  # re-export of all temporal concretes + bases
    TemporalAggregation,
    TemporalCausalBoxcar,
    TemporalCausalRolling,
    TemporalEventTriggered,
    TemporalExplicit,
    TemporalExponentialDecay,
    TemporalFixedLookback,
    TemporalFold,
    TemporalForecast,
    TemporalGeometry,
    TemporalHierarchicalCombine,
    TemporalLookbackHorizon,
    TemporalMean,
    TemporalMultiScale,
    TemporalPatcher,
    TemporalPeriodic,
    TemporalPhaseWindow,
    TemporalRandom,
    TemporalRegularStride,
    TemporalSampler,
    TemporalTaperedTukey,
    TemporalWindow,
)


__version__ = "0.0.1"

__all__ = [
    "AsyncField",
    "AsyncRasterField",
    "AsyncSpatialPatcher",
    "Domain",
    "Field",
    "GridDomain",
    "Patch",
    "PointDomain",
    "RasterDomain",
    "RasterField",
    "SpatialAggregation",
    "SpatialApproxCardinality",
    "SpatialApproxMode",
    "SpatialApproxQuantile",
    "SpatialBoxcar",
    "SpatialByIndex",
    "SpatialCustom",
    "SpatialExplicit",
    "SpatialGaussian",
    "SpatialGeometry",
    "SpatialHann",
    "SpatialHardVote",
    "SpatialInvVarWeightedMean",
    "SpatialJitteredStride",
    "SpatialKNNGraph",
    "SpatialLearned",
    "SpatialMax",
    "SpatialMean",
    "SpatialMedian",
    "SpatialMin",
    "SpatialMode",
    "SpatialOverlapAdd",
    "SpatialPatcher",
    "SpatialPoissonDisk",
    "SpatialPolygonIntersection",
    "SpatialRadiusGraph",
    "SpatialRandom",
    "SpatialRectangular",
    "SpatialRegularStride",
    "SpatialReservoir",
    "SpatialSampler",
    "SpatialSoftVote",
    "SpatialSphericalCap",
    "SpatialStreamingHistogram",
    "SpatialSum",
    "SpatialTukey",
    "SpatialVariance",
    "SpatialWeightedSum",
    "SpatialWindow",
    "SpatioTemporalPatch",
    "SpatioTemporalPatcher",
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
    "TemporalPatch",
    "TemporalPatcher",
    "TemporalPeriodic",
    "TemporalPhaseWindow",
    "TemporalRandom",
    "TemporalRegularStride",
    "TemporalSampler",
    "TemporalTaperedTukey",
    "TemporalWindow",
    "VectorDomain",
    "__version__",
    "fields",
    "spatial",
    "time",
]


# Lazy field adapters keyed off optional extras — defer to the public
# `geopatcher.fields` submodule's own lazy loader.
def __getattr__(name: str):
    """Lazy-load optional Field adapters from `geopatcher.fields`."""
    if name in {"XarrayField", "GeoPandasField", "XvecField", "RioXarrayField"}:
        return getattr(fields, name)
    raise AttributeError(name)

# `geopatcher` — Patcher framework API

Curated mkdocstrings reference, grouped by family. For the conceptual
walkthrough see [Patching](../patching.md).

## Carriers

::: geopatcher._src.patch.Patch
::: geopatcher._src.patch.TemporalPatch
::: geopatcher._src.patch.SpatioTemporalPatch

## Operational scale

::: geopatcher._src.journal.PatchJournal

## Protocols

::: geopatcher._src.hooks.PatcherHook
::: geopatcher._src.protocols.Field
::: geopatcher._src.protocols.AsyncField
::: geopatcher._src.protocols.Domain

## Domains

::: geopatcher._src.domains.GridDomain
::: geopatcher._src.domains.VectorDomain
::: geopatcher._src.domains.PointDomain

## Field adapters

::: geopatcher._src.fields.raster.RasterField
::: geopatcher._src.fields.raster.AsyncRasterField

The non-raster adapters are extras-gated; import via the submodule path:

```python
from geopatcher.fields import XarrayField, GeoPandasField, XvecField
```

## Top-level patchers

::: geopatcher._src.spatial.patcher.SpatialPatcher
::: geopatcher._src.spatial.patcher.AsyncSpatialPatcher
::: geopatcher._src.time.patcher.TemporalPatcher
::: geopatcher._src.spatial_time.SpatioTemporalPatcher

## Spatial axes

### Geometry

::: geopatcher._src.spatial.geometry.SpatialGeometry
::: geopatcher._src.spatial.geometry.SpatialRectangular
::: geopatcher._src.spatial.geometry.SpatialSphericalCap
::: geopatcher._src.spatial.geometry.SpatialKNNGraph
::: geopatcher._src.spatial.geometry.SpatialRadiusGraph
::: geopatcher._src.spatial.geometry.SpatialPolygonIntersection

### Sampler

::: geopatcher._src.spatial.sampler.SpatialSampler
::: geopatcher._src.spatial.sampler.SpatialRegularStride
::: geopatcher._src.spatial.sampler.SpatialJitteredStride
::: geopatcher._src.spatial.sampler.SpatialRandom
::: geopatcher._src.spatial.sampler.SpatialPoissonDisk
::: geopatcher._src.spatial.sampler.SpatialExplicit

### Window

::: geopatcher._src.spatial.window.SpatialWindow
::: geopatcher._src.spatial.window.SpatialBoxcar
::: geopatcher._src.spatial.window.SpatialHann
::: geopatcher._src.spatial.window.SpatialTukey
::: geopatcher._src.spatial.window.SpatialGaussian
::: geopatcher._src.spatial.window.SpatialCustom

### Aggregation

::: geopatcher._src.spatial.aggregation.SpatialAggregation
::: geopatcher._src.spatial.aggregation.SpatialSum
::: geopatcher._src.spatial.aggregation.SpatialMean
::: geopatcher._src.spatial.aggregation.SpatialVariance
::: geopatcher._src.spatial.aggregation.SpatialOverlapAdd
::: geopatcher._src.spatial.aggregation.SpatialWeightedSum
::: geopatcher._src.spatial.aggregation.SpatialInvVarWeightedMean
::: geopatcher._src.spatial.aggregation.SpatialMax
::: geopatcher._src.spatial.aggregation.SpatialMin
::: geopatcher._src.spatial.aggregation.SpatialHardVote
::: geopatcher._src.spatial.aggregation.SpatialSoftVote
::: geopatcher._src.spatial.aggregation.SpatialByIndex
::: geopatcher._src.spatial.aggregation.SpatialMedian
::: geopatcher._src.spatial.aggregation.SpatialMode
::: geopatcher._src.spatial.aggregation.SpatialLearned

#### Approximate (sketches)

::: geopatcher._src.spatial.aggregation.SpatialApproxQuantile
::: geopatcher._src.spatial.aggregation.SpatialApproxCardinality
::: geopatcher._src.spatial.aggregation.SpatialApproxMode
::: geopatcher._src.spatial.aggregation.SpatialStreamingHistogram
::: geopatcher._src.spatial.aggregation.SpatialReservoir

## Temporal axes

### Geometry

::: geopatcher._src.time.geometry.TemporalGeometry
::: geopatcher._src.time.geometry.TemporalFixedLookback
::: geopatcher._src.time.geometry.TemporalLookbackHorizon
::: geopatcher._src.time.geometry.TemporalMultiScale
::: geopatcher._src.time.geometry.TemporalPhaseWindow

### Sampler

::: geopatcher._src.time.sampler.TemporalSampler
::: geopatcher._src.time.sampler.TemporalRegularStride
::: geopatcher._src.time.sampler.TemporalCausalRolling
::: geopatcher._src.time.sampler.TemporalEventTriggered
::: geopatcher._src.time.sampler.TemporalRandom
::: geopatcher._src.time.sampler.TemporalExplicit

### Window

::: geopatcher._src.time.window.TemporalWindow
::: geopatcher._src.time.window.TemporalCausalBoxcar
::: geopatcher._src.time.window.TemporalExponentialDecay
::: geopatcher._src.time.window.TemporalTaperedTukey
::: geopatcher._src.time.window.TemporalPeriodic

### Aggregation

::: geopatcher._src.time.aggregation.TemporalAggregation
::: geopatcher._src.time.aggregation.TemporalFold
::: geopatcher._src.time.aggregation.TemporalMean
::: geopatcher._src.time.aggregation.TemporalHierarchicalCombine
::: geopatcher._src.time.aggregation.TemporalForecast

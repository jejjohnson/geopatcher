# Patching

`geopatcher` is the locality layer of the stack. Where an operator-graph
composition library (e.g. [`geotoolz`](https://github.com/jejjohnson/geotoolz)) settles
*what to compute*, and the [`Field` / `Domain`](#protocols-field-and-domain)
Protocols settle *what backend the data lives on*, the Patcher settles
the third orthogonal question: **what slice of the data does the
operator see at once, and how do local outputs become a global field?**

Three Patcher classes compose the four-axis framework:

- `SpatialPatcher` — neighborhoods in space (raster, grid, points, polygons).
- `TemporalPatcher` — windows along a time axis.
- `SpatioTemporalPatcher` — composition of the two with explicit coupling.

## The four spatial axes

| Axis | Controls | Examples |
|------|----------|----------|
| **Geometry** | Shape + scale of the neighborhood (and the domain topology). | `SpatialRectangular`, `SpatialSphericalCap`, `SpatialKNNGraph`, `SpatialRadiusGraph`, `SpatialPolygonIntersection` |
| **Sampler** | Where anchors are placed; overlap is emergent. | `SpatialRegularStride`, `SpatialJitteredStride`, `SpatialRandom`, `SpatialPoissonDisk`, `SpatialExplicit` |
| **Window** | Boundary treatment (spectral leakage, edge artefacts). | `SpatialBoxcar`, `SpatialHann`, `SpatialTukey`, `SpatialGaussian`, `SpatialCustom` |
| **Aggregation** | Local predictions → global field. | `SpatialOverlapAdd`, `SpatialMean`, `SpatialWeightedSum`, `SpatialInvVarWeightedMean`, `SpatialHardVote`, `SpatialByIndex`, … |

The Patcher composes them and exposes a tiny surface:

```python
patcher = SpatialPatcher(geometry=..., sampler=..., window=..., aggregation=...)
for patch in patcher.split(field):    # Iterator[Patch]
    out = operator(patch.data)
stitched = patcher.merge(outputs_as_patches, field.domain)
```

`split` is an iterator by design — streaming is the default; materialise
with `list(...)` when convenient.

## Determinism (stochastic samplers)

`SpatialRandom`, `SpatialJitteredStride`, `SpatialPoissonDisk`, and
`TemporalRandom` accept a `seed: int | None`. The contract (issue #18,
pinned by `tests/test_determinism.py`):

| `seed` value | Behavior |
|---|---|
| `int` | Two samplers with the same config return bit-identical anchors across calls *and* across instances. Use this whenever you need reproducible runs (ML evaluation, CI, journal-resume). |
| `None` (default) | The sampler re-seeds from OS entropy on every call; anchors will differ. Pick this for casual exploration when reproducibility doesn't matter. |

The Hypothesis round-trip suite (`tests/test_roundtrip.py`, issue #21)
leans on the `int` contract — given a seed, it shrinks failing
examples to the minimal `(shape, stride, seed)` triple and replays
them deterministically.

## Boundary policy

What happens when an anchor sits close enough to the edge that the
neighborhood would overflow the domain? `SpatialRectangular` exposes
this as a first-class parameter (issue #19):

```python
geom = SpatialRectangular(size=(256, 256), boundary="pad")
```

| Mode | Behavior |
|------|----------|
| `"drop"` (default) | Sampler clips so overflowing anchors are never emitted. Edge residual is silently dropped — exactly the pre-issue-19 behavior. |
| `"pad"` | Edge anchors are emitted; the raster `Field` reads with `boundless=True` so the patch is the full geometry size, padded in the overflow region with the reader's nodata. **Only `RasterField` / `AsyncRasterField` honor this contract today** — `RioXarrayField.select` clips via `isel` instead, so `"pad"` against a rioxarray field silently behaves like `"shrink"`. Wrap your data in `RasterField` if you need true padding, or use `"shrink"` explicitly. |
| `"shrink"` | Edge anchors are emitted; the geometry clips the returned Window so the patch is *smaller* at the edge. Weights crop to match. |
| `"raise"` | Edge anchors are emitted; `SpatialPatcher.split` raises a `ValueError` on the first overflow. Useful with `SpatialExplicit` when the caller wants strict edge handling. |

`"reflect"` and a fully aggregation-aware `"pad"` (zero-weight mask in
the overflow region for COLA-correct stitching) are planned follow-ups —
see issue #19.

Only `SpatialRectangular` on raster domains honors the parameter in v0.x;
graph and polygon geometries always behave as if `"drop"` (their natural
clipping is already correct), and `GridDomain` support is pending an
xarray-pad story.

## Protocols: `Field` and `Domain`

The Patcher consumes a `Field` (something with `domain`, `select(indexer)`,
`with_data(array)`). The raster path reuses
[`georeader.GeoData`](https://github.com/IPL-UV/georeader) verbatim through
the thin `RasterField` adapter; the non-raster Fields (`XarrayField`,
`GeoPandasField`, `XvecField`, `RioXarrayField`) live under
`geopatcher.fields` and lazy-import their optional extras.

| Field | Domain | Backend |
|---|---|---|
| `RasterField`, `AsyncRasterField` | `RasterDomain` (`georeader.GeoDataBase`) | `RasterioReader`, `AsyncGeoTIFFReader`, `GeoTensor` |
| `RioXarrayField` | `RasterDomain` | rioxarray `DataArray` |
| `XarrayField` | `GridDomain` | `xarray.DataArray` (non-raster) |
| `GeoPandasField` | `VectorDomain` / `PointDomain` | `geopandas.GeoDataFrame` |
| `XvecField` | `PointDomain` | `xvec.Dataset` |

Geometry × Domain dispatch is explicit `isinstance` (Protocol nominal typing
doesn't play well with `singledispatch`). Unsupported pairings raise
`NotImplementedError` at runtime.

## The four temporal axes

Mirror of the spatial side, with axes that encode time-specific properties
(causality, periodicity, multi-scale, forecasting):

| Axis | Controls | Examples |
|------|----------|----------|
| **Geometry** | Window shape (lookback, horizon, multi-scale, phase). | `TemporalFixedLookback`, `TemporalLookbackHorizon`, `TemporalMultiScale`, `TemporalPhaseWindow` |
| **Sampler** | Anchor placement in time. | `TemporalRegularStride`, `TemporalCausalRolling`, `TemporalEventTriggered`, `TemporalRandom`, `TemporalExplicit` |
| **Window** | Temporal boundary treatment. | `TemporalCausalBoxcar`, `TemporalExponentialDecay`, `TemporalTaperedTukey`, `TemporalPeriodic` |
| **Aggregation** | Time → time reconstruction. | `TemporalFold` (RNN-like state-passing), `TemporalMean`, `TemporalHierarchicalCombine`, `TemporalForecast` |

`TemporalFold` is the name for the RNN-like fold (renamed from the design's
`Sequential` to avoid clashing with operator-graph `Sequential` types in
downstream composition libraries).

## Spatiotemporal composition

`SpatioTemporalPatcher` composes a `SpatialPatcher` and a `TemporalPatcher`
with one of two coupling modes:

- `"product"` (default) — Cartesian product of every spatial anchor × every
  time anchor. The right shape for dense gridded data (climate output,
  regular satellite revisits).
- `"coupled"` — explicit `(space, time)` anchor pairs from the spatial
  sampler's `anchors_`. The right shape for event-triggered patches
  (methane plume detections, Argo profile locations, storm tracks).

## Operator-graph bridge

Operator-graph composition libraries (e.g.
[`geotoolz`](https://github.com/jejjohnson/geotoolz)) ship thin wrappers
that adapt the Patcher into their `Operator` world — typically a triple
of `GridSampler(patcher)`, `ApplyToChips(operator)`, and
`Stitch(aggregation, domain)`. Those wrappers live in the consuming
library, not here; geopatcher itself has no operator-graph dependency.

## Streaming aggregations

Every `SpatialAggregation` carries a `streaming_safe: ClassVar[bool]`. The
canonical streaming-safe member is `SpatialOverlapAdd`, which accepts
`streaming=True, target_path=...` to accumulate into an on-disk
[zarr](https://zarr.dev) store instead of RAM. The exact streaming family
(`Sum`, `Mean`, `Variance`, `OverlapAdd`, `WeightedSum`, `InvVarWeightedMean`,
`HardVote`, `SoftVote`) is fully implemented. The approximate sketch
family (`ApproxQuantile`, `ApproxCardinality`, `ApproxMode`,
`StreamingHistogram`, `Reservoir`) provides global streaming summaries for
operational-scale jobs that need bounded reducer state rather than a full
materialised field.

For resumable local jobs, create a `PatchJournal(path)` and pass it to
`patcher.split(field, journal=journal)`. Anchors with successful journal rows
are skipped on restart. Iterator backpressure is available through
`max_in_flight` or `max_in_flight_bytes`; close patches explicitly (or use them
as context managers) when you want to release a slot before the object is
garbage-collected.

## Optional extras

`geopatcher` keeps the base install slim and gates each non-raster
Field adapter behind an extra:

```bash
pip install 'geopatcher[grid]'           # XarrayField
pip install 'geopatcher[vector]'         # GeoPandasField
pip install 'geopatcher[point]'          # XvecField
pip install 'geopatcher[xarray-raster]'  # RioXarrayField
pip install 'geopatcher[streaming]'      # OverlapAdd(streaming=True)
pip install 'geopatcher[patch-full]'     # everything above
```

Each adapter raises a friendly `ImportError` pointing at the right extra if
the backend library is missing.

## Where the framework draws the line

- **Mesh / `uxarray`** (`UXarrayField`) is deferred to v0.2.
- **Hierarchical Patcher-of-Patchers** is supported as a *recipe* on top of
  the framework rather than a dedicated class. See the streaming tutorial
  notebook.
- **Two-pass / global-context operators** (global normalisation,
  attention across patches) are explicitly out of scope; users write the
  two passes themselves on top of the existing primitives.

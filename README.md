# geopatcher

[![Tests](https://github.com/jejjohnson/geopatcher/actions/workflows/ci.yml/badge.svg)](https://github.com/jejjohnson/geopatcher/actions/workflows/ci.yml)
[![Lint](https://github.com/jejjohnson/geopatcher/actions/workflows/lint.yml/badge.svg)](https://github.com/jejjohnson/geopatcher/actions/workflows/lint.yml)
[![Type Check](https://github.com/jejjohnson/geopatcher/actions/workflows/typecheck.yml/badge.svg)](https://github.com/jejjohnson/geopatcher/actions/workflows/typecheck.yml)
[![Deploy Docs](https://github.com/jejjohnson/geopatcher/actions/workflows/pages.yml/badge.svg)](https://github.com/jejjohnson/geopatcher/actions/workflows/pages.yml)
[![codecov](https://codecov.io/gh/jejjohnson/geopatcher/branch/main/graph/badge.svg)](https://codecov.io/gh/jejjohnson/geopatcher)
[![PyPI version](https://img.shields.io/pypi/v/geopatcher.svg)](https://pypi.org/project/geopatcher/)
[![Python versions](https://img.shields.io/pypi/pyversions/geopatcher.svg)](https://pypi.org/project/geopatcher/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com/)

> A four-axis Patcher framework for splitting geospatial fields into local
> patches, running an operator per patch, and stitching local outputs back
> into a global result.

`geopatcher` is the locality layer for remote-sensing and geospatial
pipelines. It answers a single question — *what slice of the data does
my operator see at once, and how do local outputs become a global field?* —
along four independently composable axes.

## What it gives you

Three `Patcher` classes:

- `SpatialPatcher` — neighborhoods in space (raster, grid, points, polygons).
- `TemporalPatcher` — windows along a time axis.
- `SpatioTemporalPatcher` — composition of the two with explicit coupling.

Each composes four orthogonal axes over a `Field` protocol:

| Axis            | Controls                                              | Examples                                                                                                  |
| --------------- | ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| **Geometry**    | Shape + scale of the neighborhood (domain topology).  | `SpatialRectangular`, `SpatialSphericalCap`, `SpatialKNNGraph`, `SpatialRadiusGraph`, `SpatialPolygonIntersection` |
| **Sampler**     | Where anchors are placed; overlap is emergent.        | `SpatialRegularStride`, `SpatialJitteredStride`, `SpatialRandom`, `SpatialPoissonDisk`, `SpatialExplicit`  |
| **Window**      | Boundary treatment (spectral leakage, edge artefacts). | `SpatialBoxcar`, `SpatialHann`, `SpatialTukey`, `SpatialGaussian`, `SpatialCustom`                          |
| **Aggregation** | Local predictions → global field.                     | `SpatialOverlapAdd`, `SpatialMean`, `SpatialWeightedSum`, `SpatialInvVarWeightedMean`, `SpatialHardVote`, … |

The temporal side mirrors the spatial side with axes that encode
time-specific properties (causality, periodicity, multi-scale, forecasting).

## Installation

```bash
pip install geopatcher
```

Optional extras gate the non-raster `Field` adapters:

```bash
pip install 'geopatcher[grid]'           # XarrayField
pip install 'geopatcher[vector]'         # GeoPandasField
pip install 'geopatcher[point]'          # XvecField
pip install 'geopatcher[xarray-raster]'  # RioXarrayField
pip install 'geopatcher[streaming]'      # SpatialOverlapAdd(streaming=True)
pip install 'geopatcher[patch-full]'     # everything above
```

## Quickstart

```python
import dataclasses
import geopatcher as gp

patcher = gp.SpatialPatcher(
    geometry    = gp.SpatialRectangular(size=(256, 256)),
    sampler     = gp.SpatialRegularStride(step=(192, 192)),
    window      = gp.SpatialHann(),
    aggregation = gp.SpatialOverlapAdd(),
)

# Streaming by default — `split` returns an Iterator[Patch]
outputs = []
for patch in patcher.split(field):
    out = my_operator(patch.data)
    outputs.append(dataclasses.replace(patch, data=out))

stitched = patcher.merge(outputs, field.domain)
```

For integration with an operator-graph composition library, see
[geotoolz](https://github.com/jejjohnson/geotoolz), which ships
`GridSampler` / `ApplyToChips` / `Stitch` wrappers that plug a
`SpatialPatcher` into a `Sequential` pipeline.

## Documentation

- [Patching concepts](https://jejjohnson.github.io/geopatcher/patching/)
- [API reference](https://jejjohnson.github.io/geopatcher/api/reference/)
- [Tutorial notebooks](https://jejjohnson.github.io/geopatcher/notebooks/patching_intro/)

## License

MIT — see [LICENSE](LICENSE).

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

Optional extras gate the non-raster `Field` adapters and the
operator-graph integration:

```bash
pip install 'geopatcher[grid]'           # XarrayField
pip install 'geopatcher[vector]'         # GeoPandasField
pip install 'geopatcher[point]'          # XvecField
pip install 'geopatcher[xarray-raster]'  # RioXarrayField
pip install 'geopatcher[streaming]'      # SpatialOverlapAdd(streaming=True)
pip install 'geopatcher[patch-full]'     # all substrate adapters
pip install 'geopatcher[pipekit]'        # geopatcher.integrations.pipekit (once pipekit lands on PyPI)
```

> **Pre-PyPI note.** `pipekit` isn't on PyPI yet, so the `[pipekit]`
> extra can't be resolved by plain `pip install` today. See [Pre-PyPI
> install](#pre-pypi-install-from-github) below for the `uv`-based
> workflow that works in the meantime.

### Pre-PyPI install (from GitHub)

`pipekit` isn't on PyPI yet, so the `[pipekit]` extra can't be resolved
by plain `pip install`. Use `uv` to clone-and-sync, which picks up the
git source declared in `pyproject.toml`:

```bash
git clone https://github.com/jejjohnson/geopatcher.git
cd geopatcher
uv sync --extra pipekit          # resolves pipekit from its GitHub repo
```

Or install in one shot directly from GitHub (uv reads `[tool.uv.sources]`
out of the requested project):

```bash
uv pip install "git+https://github.com/jejjohnson/geopatcher@main#egg=geopatcher[pipekit]"
```

Once `pipekit` ships to PyPI, plain `pip install 'geopatcher[pipekit]'`
will work and the git source can be removed.

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

For a built-in reference runner over independent local jobs, use
`geopatcher.runners.parallel_map`:

```python
from geopatcher.runners import parallel_map

outputs = parallel_map(patcher, field, my_operator, n_workers=8)
stitched = patcher.merge(outputs, field.domain)
```

Operators that need global context can use the codified two-pass pattern:

```python
stats = patcher.reduce(field, agg=gp.SpatialMeanStd())
stitched = patcher.two_pass(
    field,
    reduce_with=gp.SpatialMeanStd(),
    apply=lambda data, s: my_operator((data - s["mean"]) / s["std"]),
)
```

For integration with the [pipekit](https://github.com/jejjohnson/pipekit)
operator-graph framework, install the optional `[pipekit]` extra (`pip
install 'geopatcher[pipekit]'`) and import `GridSampler` /
`ApplyToChips` / `Stitch` from `geopatcher.integrations.pipekit` to plug
a `SpatialPatcher` into a `pipekit.Sequential` pipeline. The same
wrappers are also reachable through
[geotoolz](https://github.com/jejjohnson/geotoolz)`.patch_ops`.

## Documentation

- [Patching concepts](https://jejjohnson.github.io/geopatcher/patching/)
- [API reference](https://jejjohnson.github.io/geopatcher/api/reference/)
- [Tutorial notebooks](https://jejjohnson.github.io/geopatcher/notebooks/patching_intro/)

## License

MIT — see [LICENSE](LICENSE).

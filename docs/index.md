# geopatcher

> A four-axis Patcher framework for splitting geospatial fields into local
> patches and stitching local outputs back into a global result.

`geopatcher` is the locality layer for remote-sensing and geospatial
operator pipelines. It answers: *what slice of the data does an operator
see at once, and how do local outputs become a global field?*

Three `Patcher` classes compose the framework:

- `SpatialPatcher` — neighborhoods in space (raster, grid, points, polygons).
- `TemporalPatcher` — windows along a time axis.
- `SpatioTemporalPatcher` — composition of the two with explicit coupling.

Each patcher composes four orthogonal axes — **Geometry**, **Sampler**,
**Window**, **Aggregation** — over a `Field` protocol that adapts the
backend substrate (raster, xarray, geopandas, xvec, …).

## Installation

```bash
pip install geopatcher
```

Or with `uv`:

```bash
uv add geopatcher
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

outputs = []
for patch in patcher.split(field):
    out = my_operator(patch.data)
    outputs.append(dataclasses.replace(patch, data=out))
stitched = patcher.merge(outputs, field.domain)
```

## Links

- [Patching concepts](patching.md)
- [API Reference](api/reference.md)
- [GitHub](https://github.com/jejjohnson/geopatcher)

"""`XarrayField` — adapter for dense, labeled N-D grids.

Wraps an `xarray.DataArray` (typically reanalysis / climate-model
output) and exposes it under the `Field` Protocol with a `GridDomain`
view. The natural indexer is ``dict[str, slice]``, consumed by
`DataArray.isel`.

Optional extra: ``pip install 'geopatcher[grid]'``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from geopatcher._src.domains import GridDomain
from geopatcher._src.fields._extras import _missing_extra


try:
    import xarray as xr  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    xr = None  # type: ignore[assignment]


@dataclass(eq=False)
class XarrayField:
    """Wrap an `xarray.DataArray` as a `Field[GridDomain]`.

    Args:
        da: The underlying `xarray.DataArray`. Any CRS must be exposed
            via the rioxarray accessor (``da.rio.crs``); ``None`` is
            allowed for non-georeferenced cubes.
    """

    da: Any

    def __post_init__(self) -> None:
        if xr is None:
            raise _missing_extra("XarrayField", "grid", "xarray>=2024.1")

    @property
    def domain(self) -> GridDomain:
        coords = {d: np.asarray(self.da[d].values) for d in self.da.dims}
        crs = getattr(self.da, "rio", None)
        crs = crs.crs if crs is not None else None
        return GridDomain(coords=coords, crs=crs)

    def select(self, indexer: dict[str, slice]) -> XarrayField:
        return XarrayField(self.da.isel(**indexer))

    def with_data(self, array: Any) -> XarrayField:
        new = self.da.copy(data=np.asarray(array))
        return XarrayField(new)

    def time_coord(self, name: str = "time") -> np.ndarray:
        """Return the 1-D time coordinate as a NumPy array.

        Helper for the coordinate-aware temporal patcher path. Resolves the
        coordinate by name (defaults to ``"time"``) and materialises its
        values so callers don't repeat ``ds[name].values`` boilerplate.

        Args:
            name: Name of the time-like coordinate. Defaults to ``"time"``.

        Returns:
            ``np.ndarray`` of dtype ``datetime64[ns]`` (or whatever NumPy
            unit xarray exposes for that coord).

        Raises:
            KeyError: If ``name`` is absent from the DataArray's coords.
            TypeError: If the coordinate is `cftime`-typed — convert via
                ``xarray.coding.times.convert_calendar(...)`` or
                ``ds.indexes['time'].to_datetimeindex()`` first.
        """
        if name not in self.da.coords:
            raise KeyError(
                f"XarrayField has no coord named {name!r}; "
                f"available: {list(self.da.coords)}"
            )
        values = np.asarray(self.da.coords[name].values)
        if values.dtype == np.dtype("O"):
            raise TypeError(
                f"Coord {name!r} has object dtype (likely cftime). "
                "Convert with xarray.coding.times.convert_calendar(...) "
                "or DataArray.indexes['time'].to_datetimeindex() before "
                "passing to TemporalPatcher."
            )
        return values

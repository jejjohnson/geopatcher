"""Tests for the spatial `Sampler` family."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from georeader.geotensor import GeoTensor

from geopatcher import (
    GridDomain,
    SpatialExplicit,
    SpatialJitteredStride,
    SpatialPoissonDisk,
    SpatialRandom,
    SpatialRectangular,
    SpatialRegularStride,
)


@pytest.fixture
def raster_domain() -> GeoTensor:
    return GeoTensor(
        values=np.zeros((1, 64, 64), dtype=np.float32),
        transform=rasterio.Affine.identity(),
        crs="EPSG:32630",
    )


@pytest.fixture
def rect() -> SpatialRectangular:
    return SpatialRectangular(size=(16, 16))


class TestSpatialRegularStride:
    def test_raster_anchor_count(
        self, raster_domain: GeoTensor, rect: SpatialRectangular
    ) -> None:
        s = SpatialRegularStride(step=16)
        anchors = list(s.anchors(raster_domain, rect))
        # 64x64 raster, 16x16 patches, stride 16 -> 4x4 = 16 anchors
        assert len(anchors) == 16
        # All anchors are valid (row, col) pairs
        assert all(0 <= r <= 48 and 0 <= c <= 48 for r, c in anchors)


class TestSpatialJitteredStride:
    def test_reproducible_seed(
        self, raster_domain: GeoTensor, rect: SpatialRectangular
    ) -> None:
        s1 = SpatialJitteredStride(step=16, jitter=0.5, seed=0)
        s2 = SpatialJitteredStride(step=16, jitter=0.5, seed=0)
        assert list(s1.anchors(raster_domain, rect)) == list(
            s2.anchors(raster_domain, rect)
        )


class TestSpatialRandom:
    def test_count_matches_request(
        self, raster_domain: GeoTensor, rect: SpatialRectangular
    ) -> None:
        s = SpatialRandom(n_samples=7, seed=42)
        anchors = list(s.anchors(raster_domain, rect))
        assert len(anchors) == 7

    def test_grid_anchors_are_dicts(self, rect: SpatialRectangular) -> None:
        gd = GridDomain(
            coords={"a": np.arange(64), "b": np.arange(64)},
        )
        s = SpatialRandom(n_samples=3, seed=0)
        anchors = list(s.anchors(gd, rect))
        assert len(anchors) == 3
        assert all(isinstance(a, dict) for a in anchors)


class TestSpatialPoissonDisk:
    def test_min_distance_invariant(
        self, raster_domain: GeoTensor, rect: SpatialRectangular
    ) -> None:
        s = SpatialPoissonDisk(min_dist=8.0, seed=0)
        anchors = list(s.anchors(raster_domain, rect))
        # Bridson samples are floats internally; the integer pixel cast can
        # shorten the integer-space distance by up to sqrt(2). Use a tolerance.
        for i, a in enumerate(anchors):
            for b in anchors[i + 1 :]:
                d = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
                assert d >= 8.0 - 2.0  # generous: sqrt(2) per coord


class TestSpatialExplicit:
    def test_yields_supplied(
        self, raster_domain: GeoTensor, rect: SpatialRectangular
    ) -> None:
        anchors = [(0, 0), (10, 5), (32, 16)]
        s = SpatialExplicit(anchors_=anchors)
        assert list(s.anchors(raster_domain, rect)) == anchors


class TestCheckFullScan:
    """`SpatialRegularStride(check_full_scan=True)` raises on partial tiles."""

    def test_raster_exact_tiling_passes(
        self, raster_domain: GeoTensor, rect: SpatialRectangular
    ) -> None:
        # 64 / 16 = 4 — exact, both axes.
        s = SpatialRegularStride(step=16, check_full_scan=True)
        anchors = list(s.anchors(raster_domain, rect))
        assert len(anchors) == 16

    def test_raster_partial_tile_raises(
        self, raster_domain: GeoTensor, rect: SpatialRectangular
    ) -> None:
        from geopatcher import IncompleteScanConfiguration

        # step=20 against (64, 64): (64 - 16) % 20 = 8 ≠ 0
        s = SpatialRegularStride(step=20, check_full_scan=True)
        with pytest.raises(IncompleteScanConfiguration, match="row"):
            list(s.anchors(raster_domain, rect))

    def test_grid_exact_tiling_passes(self) -> None:
        grid = GridDomain(
            coords={
                "latitude": np.linspace(-30, 30, 24),
                "longitude": np.linspace(0, 60, 36),
            },
        )
        rect = SpatialRectangular(size=(6, 6))
        s = SpatialRegularStride(step=(6, 6), check_full_scan=True)
        anchors = list(s.anchors(grid, rect))
        # Both axes: (24 - 6) / 6 + 1 = 4; (36 - 6) / 6 + 1 = 6 → 4 * 6 = 24
        assert len(anchors) == 24

    def test_grid_partial_tile_raises(self) -> None:
        from geopatcher import IncompleteScanConfiguration

        grid = GridDomain(
            coords={
                "latitude": np.linspace(-30, 30, 25),
                "longitude": np.linspace(0, 60, 36),
            },
        )
        rect = SpatialRectangular(size=(6, 6))
        s = SpatialRegularStride(step=(6, 6), check_full_scan=True)
        # (25 - 6) % 6 = 1 ≠ 0 on latitude
        with pytest.raises(IncompleteScanConfiguration, match="latitude"):
            list(s.anchors(grid, rect))

    def test_default_off_preserves_silent_truncation(
        self, raster_domain: GeoTensor, rect: SpatialRectangular
    ) -> None:
        # No flag → same behaviour as before: partial tile silently dropped.
        s = SpatialRegularStride(step=20)
        anchors = list(s.anchors(raster_domain, rect))
        assert len(anchors) > 0  # no raise

    def test_get_config_round_trip(self) -> None:
        s = SpatialRegularStride(step=(6, 6), check_full_scan=True)
        cfg = s.get_config()
        assert cfg == {"step": [6, 6], "check_full_scan": True}

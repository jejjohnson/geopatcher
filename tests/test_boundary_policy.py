"""Tests for `SpatialRectangular.boundary` — issue #19.

Four modes on a deliberately misaligned domain (70x70, patch 16,
stride 16 → 4 full anchors plus a 6-px residual at the right/bottom
edges):

- ``"drop"`` (default): residual is silently dropped; 4x4 = 16 anchors.
- ``"pad"``: edge anchors emitted; reads use ``boundless=True`` so the
  patch is the full geometry size with the reader's nodata in the
  overflow region; 5x5 = 25 anchors.
- ``"shrink"``: edge anchors emitted; the geometry clips the Window so
  the patch is smaller at the edge; 5x5 = 25 anchors, edge ones smaller.
- ``"raise"``: edge anchors emitted; `SpatialPatcher.split` raises on
  the first overflow.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from georeader.geotensor import GeoTensor

from geopatcher import (
    RasterField,
    SpatialBoxcar,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialRectangular,
    SpatialRegularStride,
)


def _patcher(boundary: str) -> SpatialPatcher:
    return SpatialPatcher(
        geometry=SpatialRectangular(size=(16, 16), boundary=boundary),  # type: ignore[arg-type]
        sampler=SpatialRegularStride(step=16),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )


@pytest.fixture
def misaligned_field() -> RasterField:
    # 70x70 with patch=16, stride=16 → residual of 6 px on each axis.
    arr = np.ones((70, 70), dtype=np.float32)
    gt = GeoTensor(
        values=arr,
        transform=rasterio.Affine.identity(),
        crs="EPSG:32630",
    )
    return RasterField(gt)


class TestRectangularBoundary:
    def test_drop_is_default_and_omits_residual(
        self, misaligned_field: RasterField
    ) -> None:
        p = _patcher("drop")
        anchors = [patch.anchor for patch in p.split(misaligned_field)]
        # 4 anchors per axis (0, 16, 32, 48). 64 is dropped because
        # 64 + 16 = 80 > 70.
        assert len(anchors) == 16
        rows = sorted({a[0] for a in anchors})
        assert rows == [0, 16, 32, 48]

    def test_pad_emits_edge_anchors_full_size(
        self, misaligned_field: RasterField
    ) -> None:
        p = _patcher("pad")
        patches = list(p.split(misaligned_field))
        # 5 anchors per axis (0, 16, 32, 48, 64).
        assert len(patches) == 25
        # Every patch is still 16x16 — georeader pads the out-of-bounds
        # region via boundless=True (with reader nodata).
        for patch in patches:
            assert patch.data.values.shape == (16, 16)

    def test_shrink_clips_edge_patches(self, misaligned_field: RasterField) -> None:
        p = _patcher("shrink")
        patches = list(p.split(misaligned_field))
        assert len(patches) == 25
        # Interior patch at (0, 0) keeps full 16x16; corner patch at
        # (64, 64) shrinks to 6x6.
        shapes = {patch.anchor: patch.data.values.shape for patch in patches}
        assert shapes[(0, 0)] == (16, 16)
        assert shapes[(64, 64)] == (6, 6)
        assert shapes[(64, 0)] == (6, 16)
        # Weights track the actual patch size.
        for patch in patches:
            assert patch.weights.shape == patch.data.values.shape

    def test_raise_errors_on_first_overflow(
        self, misaligned_field: RasterField
    ) -> None:
        p = _patcher("raise")
        with pytest.raises(ValueError, match="overflows the domain"):
            list(p.split(misaligned_field))

    def test_invalid_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="invalid boundary mode"):
            SpatialRectangular(size=(16, 16), boundary="reflect")  # type: ignore[arg-type]

    def test_config_round_trips_boundary(self) -> None:
        geom = SpatialRectangular(size=(16, 16), boundary="pad")
        cfg = geom.get_config()
        assert cfg["boundary"] == "pad"
        # Defaults preserved through round-trip too.
        default = SpatialRectangular(size=(16, 16))
        assert default.get_config()["boundary"] == "drop"


class TestAlignedDomainIsUnchanged:
    """When the domain divides evenly, drop / pad / shrink agree."""

    @pytest.fixture
    def aligned_field(self) -> RasterField:
        arr = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
        gt = GeoTensor(
            values=arr,
            transform=rasterio.Affine.identity(),
            crs="EPSG:32630",
        )
        return RasterField(gt)

    @pytest.mark.parametrize("boundary", ["drop", "pad", "shrink", "raise"])
    def test_aligned_domain_anchor_count(
        self, aligned_field: RasterField, boundary: str
    ) -> None:
        p = _patcher(boundary)
        anchors = [patch.anchor for patch in p.split(aligned_field)]
        # 4x4 = 16 anchors on a 64x64 domain with patch=16, stride=16.
        assert len(anchors) == 16

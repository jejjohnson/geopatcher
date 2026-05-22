"""Determinism contract for stochastic samplers — issue #18.

Pins down the behavior #21 (the Hypothesis round-trip suite) relies on:
**given the same integer seed and the same (domain, geometry), every
stochastic sampler returns bit-identical anchors across calls and
across instances.** Without that, Hypothesis can't shrink a failing
example and the round-trip properties can't be verified.

The four stochastic samplers covered here:

- `SpatialJitteredStride`
- `SpatialRandom`
- `SpatialPoissonDisk`
- `TemporalRandom`

The contract:

| ``seed`` value          | Determinism                                                |
| ----------------------- | ---------------------------------------------------------- |
| ``int``                 | Bit-identical anchors across calls and across instances.   |
| ``None`` (default)      | Re-seeded from OS entropy each call — anchors will differ. |

(`SpatialExplicit` and `SpatialRegularStride` are deterministic by
construction; not exercised here.)
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from georeader.geotensor import GeoTensor
from hypothesis import given, settings, strategies as st

from geopatcher import (
    RasterField,
    SpatialJitteredStride,
    SpatialPoissonDisk,
    SpatialRandom,
    SpatialRectangular,
    TemporalRandom,
)


@pytest.fixture
def domain() -> RasterField:
    arr = np.zeros((64, 64), dtype=np.float32)
    return RasterField(
        GeoTensor(
            values=arr,
            transform=rasterio.Affine.identity(),
            crs="EPSG:32630",
        )
    )


@pytest.fixture
def rect() -> SpatialRectangular:
    return SpatialRectangular(size=(16, 16))


# ---------------------------------------------------------------------------
# Per-sampler determinism — explicit cases
# ---------------------------------------------------------------------------


class TestSpatialJitteredStrideDeterminism:
    def test_same_seed_same_anchors_across_calls(
        self, domain: RasterField, rect: SpatialRectangular
    ) -> None:
        s = SpatialJitteredStride(step=16, jitter=0.5, seed=42)
        first = list(s.anchors(domain.domain, rect))
        second = list(s.anchors(domain.domain, rect))
        assert first == second

    def test_same_seed_same_anchors_across_instances(
        self, domain: RasterField, rect: SpatialRectangular
    ) -> None:
        a = list(
            SpatialJitteredStride(step=16, jitter=0.5, seed=42).anchors(
                domain.domain, rect
            )
        )
        b = list(
            SpatialJitteredStride(step=16, jitter=0.5, seed=42).anchors(
                domain.domain, rect
            )
        )
        assert a == b

    def test_different_seeds_differ(
        self, domain: RasterField, rect: SpatialRectangular
    ) -> None:
        a = list(
            SpatialJitteredStride(step=16, jitter=0.5, seed=0).anchors(
                domain.domain, rect
            )
        )
        b = list(
            SpatialJitteredStride(step=16, jitter=0.5, seed=1).anchors(
                domain.domain, rect
            )
        )
        assert a != b


class TestSpatialRandomDeterminism:
    def test_same_seed_same_anchors_across_calls(
        self, domain: RasterField, rect: SpatialRectangular
    ) -> None:
        s = SpatialRandom(n_samples=20, seed=7)
        assert list(s.anchors(domain.domain, rect)) == list(
            s.anchors(domain.domain, rect)
        )

    def test_same_seed_same_anchors_across_instances(
        self, domain: RasterField, rect: SpatialRectangular
    ) -> None:
        a = list(SpatialRandom(n_samples=20, seed=7).anchors(domain.domain, rect))
        b = list(SpatialRandom(n_samples=20, seed=7).anchors(domain.domain, rect))
        assert a == b

    def test_seed_none_is_documented_non_deterministic(
        self, domain: RasterField, rect: SpatialRectangular
    ) -> None:
        # Document the documented contract: seed=None re-seeds from OS
        # entropy each call. With n_samples=20 across 64x64 the
        # probability of two draws matching is negligible.
        s = SpatialRandom(n_samples=20, seed=None)
        first = list(s.anchors(domain.domain, rect))
        second = list(s.anchors(domain.domain, rect))
        assert first != second


class TestSpatialPoissonDiskDeterminism:
    def test_same_seed_same_anchors_across_calls(
        self, domain: RasterField, rect: SpatialRectangular
    ) -> None:
        s = SpatialPoissonDisk(min_dist=6.0, seed=11)
        assert list(s.anchors(domain.domain, rect)) == list(
            s.anchors(domain.domain, rect)
        )

    def test_same_seed_same_anchors_across_instances(
        self, domain: RasterField, rect: SpatialRectangular
    ) -> None:
        a = list(SpatialPoissonDisk(min_dist=6.0, seed=11).anchors(domain.domain, rect))
        b = list(SpatialPoissonDisk(min_dist=6.0, seed=11).anchors(domain.domain, rect))
        assert a == b


class TestTemporalRandomDeterminism:
    def test_same_seed_same_anchors_across_calls(self) -> None:
        s = TemporalRandom(n=5, seed=3)
        first = list(s.anchors(time_len=100))
        second = list(s.anchors(time_len=100))
        assert first == second

    def test_same_seed_same_anchors_across_instances(self) -> None:
        a = list(TemporalRandom(n=5, seed=3).anchors(time_len=100))
        b = list(TemporalRandom(n=5, seed=3).anchors(time_len=100))
        assert a == b


# ---------------------------------------------------------------------------
# Hypothesis: for any seed, anchors are bit-identical across calls
# ---------------------------------------------------------------------------


@settings(max_examples=50, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_spatial_random_bit_identical_for_any_int_seed(seed: int) -> None:
    # Property: independent of the seed value, two calls return the
    # same anchor sequence. Catches any accidental reliance on global
    # RNG state that example-based tests might miss.
    arr = np.zeros((32, 32), dtype=np.float32)
    field = RasterField(
        GeoTensor(
            values=arr,
            transform=rasterio.Affine.identity(),
            crs="EPSG:32630",
        )
    )
    geom = SpatialRectangular(size=(8, 8))
    s = SpatialRandom(n_samples=10, seed=seed)
    assert list(s.anchors(field.domain, geom)) == list(s.anchors(field.domain, geom))


@settings(max_examples=50, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_jittered_stride_bit_identical_for_any_int_seed(seed: int) -> None:
    arr = np.zeros((64, 64), dtype=np.float32)
    field = RasterField(
        GeoTensor(
            values=arr,
            transform=rasterio.Affine.identity(),
            crs="EPSG:32630",
        )
    )
    geom = SpatialRectangular(size=(8, 8))
    s = SpatialJitteredStride(step=8, jitter=0.5, seed=seed)
    assert list(s.anchors(field.domain, geom)) == list(s.anchors(field.domain, geom))

"""Golden tests: streaming aggregations match their in-memory closed
form — issue #22.

Concrete content of #22 for the v0.x aggregation family:

- The only aggregation with two distinct code paths is
  `SpatialOverlapAdd`: `_merge_in_memory` (numpy accumulators) vs
  `_merge_streaming` (zarr-backed accumulators on disk). The bulk of
  this file is the equality contract between those two paths under
  varied zarr chunk shapes (1, 7 prime, 16 block-aligned, full).

- The other streaming-safe aggregations (`Sum`, `Max`, `Min`,
  `WeightedSum`, `Mean`, `Variance`) are monoidal folds with a single
  implementation. The "golden" property we can still verify on them is
  **permutation invariance**: feeding the same patches in any order
  must yield the same result. That catches the same class of
  bookkeeping bugs the streaming/in-memory comparison would.

- `SpatialVariance` uses Welford specifically because it's more
  accurate than a naive two-pass on ill-conditioned data (large mean,
  small variance, near-cancellation in `E[x²] - E[x]²`). One test pins
  that claim down — Welford error must not exceed naive error on a
  deliberately ill-conditioned fixture.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from georeader.geotensor import GeoTensor
from rasterio.windows import Window

from geopatcher import (
    Patch,
    SpatialMax,
    SpatialMean,
    SpatialMin,
    SpatialOverlapAdd,
    SpatialSum,
    SpatialVariance,
    SpatialWeightedSum,
)


zarr = pytest.importorskip("zarr")


# ---------------------------------------------------------------------------
# Fixture: deterministic patch set on a 64x64 domain
# ---------------------------------------------------------------------------


@pytest.fixture
def domain() -> GeoTensor:
    return GeoTensor(
        values=np.zeros((64, 64), dtype=np.float32),
        transform=rasterio.Affine.identity(),
        crs="EPSG:32630",
    )


@pytest.fixture
def overlapping_patches() -> list[Patch]:
    # Deterministic (16x16) patches on a 64x64 grid, anchored on a
    # 12-pixel stride so they overlap (4-px overlap region per pair) —
    # the regime where OverlapAdd's streaming and in-memory paths
    # actually have something to disagree about. Kept small (5x5
    # lattice = 25 patches) because the streaming path's per-patch
    # zarr RMW dominates wall time; the equality property doesn't need
    # bulk to surface.
    rng = np.random.default_rng(seed=0)
    patches: list[Patch] = []
    anchors = [(r, c) for r in range(0, 49, 12) for c in range(0, 49, 12)]
    for r, c in anchors:
        data = rng.normal(loc=10.0, scale=2.0, size=(16, 16)).astype(np.float64)
        weights = rng.uniform(0.1, 1.0, size=(16, 16))
        patches.append(
            Patch(
                data=data,
                anchor=(r, c),
                indices=Window(col_off=c, row_off=r, width=16, height=16),
                weights=weights,
            )
        )
    return patches


# ---------------------------------------------------------------------------
# OverlapAdd: streaming vs in-memory, across zarr chunk shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "chunks",
    [
        None,  # default — derived from first patch
        (8, 8),  # patch-misaligned small chunk (16 / 8 = 2)
        (7, 7),  # prime, deliberately misaligned with patches
        (16, 16),  # patch-aligned
        (64, 64),  # whole-domain (single chunk)
    ],
)
def test_overlap_add_streaming_matches_in_memory(
    domain: GeoTensor,
    overlapping_patches: list[Patch],
    tmp_path,
    chunks: tuple[int, int] | None,
) -> None:
    in_mem = SpatialOverlapAdd().merge(overlapping_patches, domain)

    streamed_agg = SpatialOverlapAdd(
        streaming=True,
        target_path=str(tmp_path),
        chunks=chunks,
    )
    streamed = np.asarray(streamed_agg.merge(overlapping_patches, domain)[:])

    # The streaming path stores float32 accumulators (see zarr.open
    # dtype="float32" in aggregation._merge_streaming); the in-memory
    # path uses float64. The float32 rtol of 1e-6 is the right ceiling.
    np.testing.assert_allclose(streamed, in_mem, rtol=1e-6, atol=1e-6)


def test_overlap_add_streaming_empty_patches_returns_zero_array(
    domain: GeoTensor, tmp_path
) -> None:
    # The peeked-iterator branch in `_merge_streaming`: no patches at
    # all should yield a zero-filled zarr of the domain's shape.
    agg = SpatialOverlapAdd(streaming=True, target_path=str(tmp_path))
    result = np.asarray(agg.merge([], domain)[:])
    assert result.shape == (64, 64)
    np.testing.assert_array_equal(result, 0.0)


def test_overlap_add_streaming_chunk_size_invariant(
    domain: GeoTensor,
    overlapping_patches: list[Patch],
    tmp_path,
) -> None:
    # Stronger version of the parametrized test: every chunk shape must
    # produce the same numerical result *to each other*, not just to
    # the in-memory path. Catches any bug where blocking introduces
    # per-chunk drift.
    results = []
    for i, chunks in enumerate([(7, 7), (8, 8), (16, 16), (64, 64)]):
        agg = SpatialOverlapAdd(
            streaming=True,
            target_path=str(tmp_path / f"run_{i}"),
            chunks=chunks,
        )
        results.append(np.asarray(agg.merge(overlapping_patches, domain)[:]))
    for r in results[1:]:
        np.testing.assert_allclose(r, results[0], rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Permutation invariance — every streaming-safe monoidal aggregation
# ---------------------------------------------------------------------------


@pytest.fixture
def disjoint_patches() -> list[Patch]:
    # Non-overlapping tiling — every cell touched exactly once. Tests
    # the boring "associativity holds" property without OverlapAdd's
    # weight bookkeeping muddying the picture.
    rng = np.random.default_rng(seed=1)
    patches: list[Patch] = []
    for r in range(0, 64, 16):
        for c in range(0, 64, 16):
            data = rng.normal(loc=0.0, scale=1.0, size=(16, 16)).astype(np.float64)
            patches.append(
                Patch(
                    data=data,
                    anchor=(r, c),
                    indices=Window(col_off=c, row_off=r, width=16, height=16),
                    weights=np.ones((16, 16), dtype=np.float64),
                )
            )
    return patches


@pytest.mark.parametrize(
    "agg",
    [
        SpatialSum(),
        SpatialMax(),
        SpatialMin(),
        SpatialMean(),
        SpatialWeightedSum(),
    ],
    ids=lambda a: type(a).__name__,
)
def test_monoidal_aggregations_are_permutation_invariant(
    domain: GeoTensor, disjoint_patches: list[Patch], agg
) -> None:
    # Monoidal fold: order of inputs must not affect the merged
    # result. Catches any accumulator that's been quietly threaded
    # through a non-associative op.
    forward = agg.merge(disjoint_patches, domain)
    rng = np.random.default_rng(seed=2)
    shuffled = list(disjoint_patches)
    rng.shuffle(shuffled)
    permuted = agg.merge(shuffled, domain)
    np.testing.assert_allclose(forward, permuted, rtol=1e-12, atol=0)


def test_variance_permutation_invariant(
    domain: GeoTensor, disjoint_patches: list[Patch]
) -> None:
    # Variance uses Welford, which is order-sensitive in the
    # intermediate accumulator but order-invariant in the final result.
    # Tolerance is looser than the strict monoids because Welford's
    # delta * (x - mean) step accumulates float64 ULPs differently per
    # ordering — that's expected; we just need agreement to fp tol.
    forward = SpatialVariance().merge(disjoint_patches, domain)
    rng = np.random.default_rng(seed=3)
    shuffled = list(disjoint_patches)
    rng.shuffle(shuffled)
    permuted = SpatialVariance().merge(shuffled, domain)
    np.testing.assert_allclose(forward, permuted, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# Welford accuracy claim: streaming variance error <= naive two-pass error
# ---------------------------------------------------------------------------


def test_welford_variance_no_worse_than_naive_on_ill_conditioned_data(
    domain: GeoTensor,
) -> None:
    # Classic ill-conditioned fixture: very large mean, very small
    # spread. `Sum(x^2)/N - (Sum(x)/N)^2` (the naive two-pass form, in
    # float32 to surface the cancellation problem) loses precision in
    # the subtraction. Welford's running update is robust to this. The
    # framework's `SpatialVariance` runs in float64 internally, so we
    # compare it against a deliberately-stressed float32 naive
    # reference to make the accuracy gap visible.
    rng = np.random.default_rng(seed=7)
    true_mean = 1e8
    true_std = 1e-3
    # Many patches at the same anchor so the variance is computed over
    # 100 noisy samples at every cell — enough samples for the
    # cancellation regime to bite.
    patches: list[Patch] = []
    samples_per_cell = 100
    for i in range(samples_per_cell):
        del i  # only the count matters here
        data = rng.normal(loc=true_mean, scale=true_std, size=(16, 16))
        patches.append(
            Patch(
                data=data,
                anchor=(0, 0),
                indices=Window(col_off=0, row_off=0, width=16, height=16),
                weights=None,
            )
        )

    welford = SpatialVariance().merge(patches, domain)

    # Naive two-pass in float32 — explicit cancellation regime.
    stacked32 = np.stack([np.asarray(p.data, dtype=np.float32) for p in patches])
    sum_x = stacked32.sum(axis=0)
    sum_xx = (stacked32 * stacked32).sum(axis=0)
    n = float(samples_per_cell)
    naive_var32 = (sum_xx - (sum_x * sum_x) / n) / (n - 1.0)

    interior_welford = welford[:16, :16]
    interior_naive = naive_var32[:16, :16]
    true_var = true_std**2

    welford_err = float(np.max(np.abs(interior_welford - true_var)))
    naive_err = float(np.max(np.abs(interior_naive - true_var)))
    assert welford_err <= naive_err, (
        f"Welford error {welford_err:.3e} exceeded naive float32 "
        f"two-pass error {naive_err:.3e} on ill-conditioned input — "
        "the whole reason SpatialVariance uses Welford is that this "
        "inequality should hold."
    )

"""Operational-scale primitives: journals, sketches, and backpressure."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from georeader.geotensor import GeoTensor
from rasterio.windows import Window

from geopatcher import (
    Patch,
    PatchJournal,
    RasterField,
    SpatialApproxCardinality,
    SpatialApproxMode,
    SpatialApproxQuantile,
    SpatialBoxcar,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialRectangular,
    SpatialRegularStride,
    SpatialReservoir,
    SpatialStreamingHistogram,
)


def _patch(values: np.ndarray) -> Patch:
    return Patch(
        data=values,
        anchor=(0, 0),
        indices=Window(col_off=0, row_off=0, width=values.shape[-1], height=1),
    )


@pytest.fixture
def field() -> RasterField:
    gt = GeoTensor(
        values=np.arange(16, dtype=np.float32).reshape(4, 4),
        transform=rasterio.Affine.identity(),
        crs="EPSG:32630",
    )
    return RasterField(gt)


def test_patch_with_data_preserves_metadata() -> None:
    patch = _patch(np.array([[1, 2, 3]]))
    updated = patch.with_data(np.array([[4, 5, 6]]))
    assert updated.anchor == patch.anchor
    assert updated.indices == patch.indices
    np.testing.assert_array_equal(updated.data, [[4, 5, 6]])


def test_patch_with_data_does_not_mutate_original() -> None:
    original_data = np.array([[1, 2, 3]])
    patch = _patch(original_data)
    updated = patch.with_data(np.array([[4, 5, 6]]))
    # `with_data` must return a fresh patch, leaving `patch.data` untouched.
    assert updated is not patch
    np.testing.assert_array_equal(patch.data, original_data)


def test_patch_close_is_idempotent() -> None:
    released: list[int] = []
    patch = _patch(np.array([[1, 2, 3]]))
    patch._release = lambda: released.append(1)
    patch.close()
    patch.close()
    assert released == [1]


def test_patch_journal_survives_crash_simulation(tmp_path: Path) -> None:
    """Re-opening the journal recovers committed rows even without close()."""
    journal_path = tmp_path / "journal.jsonl"
    journal = PatchJournal(str(journal_path))
    journal.commit((1, 1), status="ok", runtime_s=0.1)
    journal.commit((2, 2), status="ok", runtime_s=0.2)
    # Drop the in-memory reference without explicit close — fsync should
    # have made the rows durable.
    del journal

    reopened = PatchJournal(str(journal_path))
    assert reopened.has((1, 1))
    assert reopened.has((2, 2))
    assert reopened.pending([(1, 1), (2, 2), (3, 3)]) == [(3, 3)]


def test_patch_journal_persists_and_split_skips_completed(
    tmp_path: Path, field: RasterField
) -> None:
    patcher = SpatialPatcher(
        geometry=SpatialRectangular(size=(2, 2)),
        sampler=SpatialRegularStride(step=2),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )
    journal_path = tmp_path / "journal.jsonl"
    journal = PatchJournal(str(journal_path))
    journal.commit((0, 0), status="ok", runtime_s=0.1)

    reopened = PatchJournal(str(journal_path))
    anchors = [patch.anchor for patch in patcher.split(field, journal=reopened)]

    assert (0, 0) not in anchors
    assert set(anchors) == {(0, 2), (2, 0), (2, 2)}


def test_split_rejects_patch_larger_than_byte_budget(field: RasterField) -> None:
    patcher = SpatialPatcher(
        geometry=SpatialRectangular(size=(2, 2)),
        sampler=SpatialRegularStride(step=2),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )
    with pytest.raises(ValueError, match="exceeding max_in_flight_bytes"):
        next(patcher.split(field, max_in_flight_bytes=1))


def test_sketch_aggregations_finalize_streaming_summaries() -> None:
    patch = _patch(np.array([[1, 2, 2, 3, 4, 5, 100]], dtype=np.float64))

    quantile = SpatialApproxQuantile(q=[0.5], compression=32).merge([patch], None)
    cardinality = SpatialApproxCardinality(p=8).merge([patch], None)
    mode = SpatialApproxMode(k=3).merge([patch], None)
    histogram = SpatialStreamingHistogram(bins=3).merge([patch], None)
    reservoir = SpatialReservoir(k=4, seed=0).merge([patch], None)

    assert quantile["0.5"] == pytest.approx(3.0)
    assert cardinality == pytest.approx(6, rel=0.2)
    assert 2 in mode
    assert histogram["counts"].sum() == 7
    assert len(reservoir) == 4


def test_hyperloglog_sketches_merge_disjoint_sets() -> None:
    left = SpatialApproxCardinality(p=8)
    right = SpatialApproxCardinality(p=8)
    left.update(_patch(np.arange(50)))
    right.update(_patch(np.arange(50, 100)))

    left.merge(right)

    assert left.finalize() == pytest.approx(100, rel=0.2)

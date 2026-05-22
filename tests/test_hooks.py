"""Tests for patcher observability hooks."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from georeader.geotensor import GeoTensor

from geopatcher import (
    PatcherHook,
    RasterField,
    SpatialBoxcar,
    SpatialOverlapAdd,
    SpatialPatcher,
    SpatialRectangular,
    SpatialRegularStride,
)


@pytest.fixture
def field() -> RasterField:
    arr = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    gt = GeoTensor(
        values=arr,
        transform=rasterio.Affine.identity(),
        crs="EPSG:32630",
    )
    return RasterField(gt)


@pytest.fixture
def patcher() -> SpatialPatcher:
    return SpatialPatcher(
        geometry=SpatialRectangular(size=(16, 16)),
        sampler=SpatialRegularStride(step=16),
        window=SpatialBoxcar(),
        aggregation=SpatialOverlapAdd(),
    )


class RecordingHook:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []
        self.bytes_: list[int] = []

    def on_split_start(self, n_anchors: int) -> None:
        self.events.append(("split_start", n_anchors))

    def on_patch_start(self, anchor: object) -> None:
        self.events.append(("patch_start", anchor))

    def on_patch_done(self, anchor: object, runtime_s: float, bytes_: int) -> None:
        assert runtime_s >= 0
        self.events.append(("patch_done", anchor))
        self.bytes_.append(bytes_)

    def on_split_end(self) -> None:
        self.events.append(("split_end", None))


def test_spatial_split_dispatches_hooks_in_order(
    field: RasterField, patcher: SpatialPatcher
) -> None:
    hook = RecordingHook()

    patches = list(patcher.split(field, hooks=[hook]))

    assert len(patches) == 16
    assert hook.events[0] == ("split_start", 16)
    assert hook.events[-1] == ("split_end", None)
    assert [name for name, _ in hook.events].count("patch_start") == 16
    assert [name for name, _ in hook.events].count("patch_done") == 16
    assert all(bytes_ > 0 for bytes_ in hook.bytes_)


class MergeHook:
    def __init__(self) -> None:
        self.events: list[tuple[str, int]] = []

    def on_merge_start(self, n_patches: int) -> None:
        self.events.append(("merge_start", n_patches))

    def on_merge_end(self, output_bytes: int) -> None:
        self.events.append(("merge_end", output_bytes))


def test_spatial_merge_dispatches_hooks(
    field: RasterField, patcher: SpatialPatcher
) -> None:
    hook = MergeHook()
    patches = list(patcher.split(field))

    patcher.merge(patches, field.reader, hooks=[hook])

    assert hook.events[0] == ("merge_start", 16)
    assert hook.events[1][0] == "merge_end"
    assert hook.events[1][1] > 0


class FailingHook:
    def on_patch_start(self, anchor: object) -> None:
        raise RuntimeError(f"bad hook for {anchor!r}")


def test_hook_errors_warn_without_aborting_split(
    field: RasterField, patcher: SpatialPatcher
) -> None:
    with pytest.warns(RuntimeWarning, match="PatcherHook.on_patch_start"):
        patches = list(patcher.split(field, hooks=[FailingHook()]))

    assert len(patches) == 16


def test_protocol_is_public() -> None:
    assert PatcherHook.__name__ == "PatcherHook"

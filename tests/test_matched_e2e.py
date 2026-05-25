"""End-to-end tests for `MatchedField` + `MatchedSpatialPatcher`.

Exercises the full split / merge pipeline against a stub primary
`SpatialPatcher` driven by a `MatchedField` with one identity-coreg
secondary. These tests validate the carrier wiring (what flows
through `Patch.data` and how it unpacks into `MatchedPatch`); they
don't depend on georeader/rasterio/numpy specifics so they run fast
and remain deterministic.

For real raster / xarray / vector behaviour, see the field-adapter
tests under `tests/fields/` (those test the underlying `Field`
implementations independently).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest

from geopatcher._src.matched import (
    MatchedField,
    MatchedPatch,
    MatchedSpatialPatcher,
)
from geopatcher._src.matched.patch import PRIMARY_KEY


# ---------------------------------------------------------------------------
# Stub Field / Domain / SpatialPatcher minimal enough to drive split/merge
# without touching georeader. The real samplers/aggregations live in
# `_src/spatial/*`; we re-use the genuine `Patch` carrier but stub the
# patcher itself so we control which patches arrive at MatchedSpatialPatcher.
# ---------------------------------------------------------------------------


@dataclass
class _StubDomain:
    crs: str = "EPSG:4326"
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)


@dataclass
class _StubField:
    """A `Field` whose `select(indexer)` returns ``f"{name}@{indexer}"``."""

    name: str

    @property
    def domain(self) -> _StubDomain:
        return _StubDomain()

    def select(self, indexer: Any) -> str:
        return f"{self.name}@{indexer}"

    def with_data(self, array: Any) -> Any:
        # Aggregations reconstruct a field-shaped value via with_data.
        # Echo through so the merge tests can inspect what came out.
        return ("with_data", self.name, array)


class _RecordingPrimaryPatcher:
    """Stand-in for `SpatialPatcher` that yields a fixed set of patches.

    Yields one `Patch` per anchor; the `data` comes from
    ``mfield.select(indexer)`` so `MatchedField` is genuinely
    exercised (not stubbed). `merge` calls into a recording
    aggregator that just returns the list it received.
    """

    def __init__(self, anchors: list[Any], aggregation: _RecordingAgg) -> None:
        self.anchors_ = anchors
        self.aggregation = aggregation
        self.merge_calls: list[tuple[list[Any], Any]] = []

    def split(self, field: Any) -> Iterator[Any]:
        from geopatcher._src.patch import Patch

        for anchor in self.anchors_:
            indexer = f"idx[{anchor}]"
            data = field.select(indexer)
            yield Patch(data=data, anchor=anchor, indices=indexer, weights=None)

    def n_anchors(self, field: Any) -> int:
        return len(self.anchors_)

    def anchors(self, field: Any) -> list[Any]:
        return list(self.anchors_)

    def merge(self, patches: list[Any], domain: Any) -> Any:
        self.merge_calls.append((list(patches), domain))
        return self.aggregation.merge(patches, domain)


class _RecordingAgg:
    """Aggregation stub — returns ``("merged", name, n_patches)``."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[tuple[int, Any]] = []

    def merge(self, patches: list[Any], domain: Any) -> Any:
        materialised = list(patches)
        self.calls.append((len(materialised), domain))
        return ("merged", self.name, len(materialised))


# ---------------------------------------------------------------------------
# MatchedField.select — read primary + secondaries, apply coreg
# ---------------------------------------------------------------------------


class TestMatchedFieldSelect:
    def test_primary_only(self) -> None:
        mf = MatchedField(primary=_StubField("p"))
        result = mf.select("idx-7")
        assert result == {PRIMARY_KEY: "p@idx-7"}

    def test_primary_and_one_secondary_identity_coreg(self) -> None:
        mf = MatchedField(
            primary=_StubField("p"),
            secondaries={"s2": _StubField("s2")},
            coreg={"s2": lambda raw, primary: raw},
        )
        result = mf.select("idx-7")
        assert result == {PRIMARY_KEY: "p@idx-7", "s2": "s2@idx-7"}

    def test_coreg_sees_both_raw_and_primary(self) -> None:
        # The coreg callable receives `(secondary_raw, primary_data)`;
        # we encode both into the output string to prove the call shape.
        mf = MatchedField(
            primary=_StubField("p"),
            secondaries={"s2": _StubField("s2")},
            coreg={"s2": lambda raw, primary: f"aligned({raw}|to|{primary})"},
        )
        result = mf.select("idx-1")
        assert result["s2"] == "aligned(s2@idx-1|to|p@idx-1)"

    def test_multiple_secondaries(self) -> None:
        mf = MatchedField(
            primary=_StubField("p"),
            secondaries={
                "s2": _StubField("s2"),
                "landsat": _StubField("landsat"),
            },
            coreg={
                "s2": lambda raw, primary: f"s2_aligned({raw})",
                "landsat": lambda raw, primary: f"landsat_aligned({raw})",
            },
        )
        result = mf.select("idx")
        assert set(result) == {PRIMARY_KEY, "s2", "landsat"}
        assert result["s2"] == "s2_aligned(s2@idx)"
        assert result["landsat"] == "landsat_aligned(landsat@idx)"


# ---------------------------------------------------------------------------
# MatchedSpatialPatcher.split — drive primary, unpack into MatchedPatch
# ---------------------------------------------------------------------------


class TestMatchedSpatialPatcherSplit:
    def _build(self) -> tuple[MatchedSpatialPatcher, MatchedField]:
        mf = MatchedField(
            primary=_StubField("p"),
            secondaries={"s2": _StubField("s2")},
            coreg={"s2": lambda raw, primary: f"aligned({raw})"},
        )
        primary_patcher = _RecordingPrimaryPatcher(
            anchors=[(0, 0), (0, 1), (1, 0)],
            aggregation=_RecordingAgg("primary_agg"),
        )
        msp = MatchedSpatialPatcher(
            primary=primary_patcher,  # type: ignore[arg-type]
            secondary_aggregators={"s2": _RecordingAgg("s2_agg")},  # type: ignore[dict-item]
        )
        return msp, mf

    def test_yields_one_matched_patch_per_anchor(self) -> None:
        msp, mf = self._build()
        patches = list(msp.split(mf))
        assert len(patches) == 3
        for mp in patches:
            assert isinstance(mp, MatchedPatch)

    def test_matched_patch_members_keyed_by_source(self) -> None:
        msp, mf = self._build()
        first = next(iter(msp.split(mf)))
        assert set(first.members) == {PRIMARY_KEY, "s2"}
        assert first.members[PRIMARY_KEY].data == "p@idx[(0, 0)]"
        assert first.members["s2"].data == "aligned(s2@idx[(0, 0)])"

    def test_inner_patches_carry_outer_metadata(self) -> None:
        # Each inner Patch must mirror the outer anchor / indices /
        # weights — downstream code (aggregators, ML loaders) reads
        # these to place the patch globally.
        msp, mf = self._build()
        first = next(iter(msp.split(mf)))
        assert first.anchor == (0, 0)
        for member_patch in first.members.values():
            assert member_patch.anchor == (0, 0)
            assert member_patch.indices == "idx[(0, 0)]"

    def test_non_dict_data_raises_with_clear_message(self) -> None:
        # If a user accidentally wires up a plain Field instead of a
        # MatchedField, the primary patcher will yield Patch.data of
        # whatever the Field returns (not a dict). We catch that and
        # raise loudly.
        msp = MatchedSpatialPatcher(
            primary=_RecordingPrimaryPatcher(
                anchors=[(0, 0)], aggregation=_RecordingAgg("agg")
            ),  # type: ignore[arg-type]
        )
        # Pass a plain Field; the patcher will call field.select which
        # returns a string, not a dict.
        plain_field = _StubField("only_primary")
        with pytest.raises(TypeError, match=r"dict.*MatchedField"):
            list(msp.split(plain_field))  # type: ignore[arg-type]

    def test_missing_primary_key_in_select_raises(self) -> None:
        # A MatchedField subclass that violates the protocol — select
        # returns a dict without PRIMARY_KEY. We catch immediately.
        class _BrokenMatchedField(MatchedField):
            def select(self, indexer: Any) -> dict[str, Any]:
                return {"only_secondary": f"x@{indexer}"}

        mf = _BrokenMatchedField(primary=_StubField("p"))
        msp = MatchedSpatialPatcher(
            primary=_RecordingPrimaryPatcher(
                anchors=[(0, 0)], aggregation=_RecordingAgg("agg")
            ),  # type: ignore[arg-type]
        )
        with pytest.raises(ValueError, match="must include the primary key"):
            list(msp.split(mf))

    def test_n_anchors_forwards(self) -> None:
        msp, mf = self._build()
        assert msp.n_anchors(mf) == 3

    def test_anchors_forwards(self) -> None:
        msp, mf = self._build()
        assert msp.anchors(mf) == [(0, 0), (0, 1), (1, 0)]


# ---------------------------------------------------------------------------
# MatchedSpatialPatcher.merge — per-source aggregation
# ---------------------------------------------------------------------------


class TestMatchedSpatialPatcherMerge:
    def _build(
        self,
        *,
        with_secondary_agg: bool = True,
    ) -> tuple[
        MatchedSpatialPatcher,
        MatchedField,
        _RecordingAgg,
        _RecordingAgg | None,
    ]:
        primary_agg = _RecordingAgg("primary_agg")
        secondary_agg = _RecordingAgg("s2_agg") if with_secondary_agg else None
        mf = MatchedField(
            primary=_StubField("p"),
            secondaries={"s2": _StubField("s2")},
            coreg={"s2": lambda raw, primary: f"aligned({raw})"},
        )
        primary_patcher = _RecordingPrimaryPatcher(
            anchors=[(0, 0), (0, 1)],
            aggregation=primary_agg,
        )
        secondary_aggregators: dict[str, Any] = {}
        if secondary_agg is not None:
            secondary_aggregators["s2"] = secondary_agg
        msp = MatchedSpatialPatcher(
            primary=primary_patcher,  # type: ignore[arg-type]
            secondary_aggregators=secondary_aggregators,
        )
        return msp, mf, primary_agg, secondary_agg

    def test_merge_returns_dict_keyed_by_source(self) -> None:
        msp, mf, _, _ = self._build()
        patches = list(msp.split(mf))
        out = msp.merge(patches, mf)
        assert set(out) == {PRIMARY_KEY, "s2"}

    def test_merge_dispatches_to_per_source_aggregators(self) -> None:
        # Each aggregator receives exactly the patches for its source.
        msp, mf, primary_agg, secondary_agg = self._build()
        patches = list(msp.split(mf))
        msp.merge(patches, mf)
        assert primary_agg.calls == [(2, mf.domain)]
        assert secondary_agg is not None
        assert secondary_agg.calls == [(2, mf.domain)]

    def test_merge_skips_secondary_without_aggregator(self) -> None:
        # When a secondary has no aggregator entry, it's skipped (the
        # user opted out for that source).
        msp, mf, _, _ = self._build(with_secondary_agg=False)
        patches = list(msp.split(mf))
        out = msp.merge(patches, mf)
        assert set(out) == {PRIMARY_KEY}

    def test_merge_uses_primary_domain_for_all(self) -> None:
        # Coregistration aligned every secondary onto the primary's
        # grid, so the reconstruction domain is the primary's. The
        # aggregation receives that domain as its second arg.
        msp, mf, primary_agg, secondary_agg = self._build()
        patches = list(msp.split(mf))
        msp.merge(patches, mf)
        assert primary_agg.calls[-1][1] == mf.domain
        assert secondary_agg is not None
        assert secondary_agg.calls[-1][1] == mf.domain

    def test_merge_consumes_iterator_once(self) -> None:
        # `patches` may be a generator; the merge must not iterate it
        # multiple times (would silently consume nothing on the
        # second pass).
        msp, mf, _primary_agg, _secondary_agg = self._build()
        patches_gen = msp.split(mf)
        out = msp.merge(patches_gen, mf)
        # Both sources should have seen 2 patches each — proving the
        # generator was teed into per-source lists in one pass.
        assert out[PRIMARY_KEY] == ("merged", "primary_agg", 2)
        assert out["s2"] == ("merged", "s2_agg", 2)


# ---------------------------------------------------------------------------
# Integration: a real SpatialPatcher driving a MatchedField + identity coreg
# (closes the loop with the actual sampler/geometry machinery).
# ---------------------------------------------------------------------------


class TestRealSpatialPatcherIntegration:
    """Use the genuine `SpatialPatcher` against a stub `Field` whose
    `select` returns a small numpy array, so we exercise the real
    sampler / geometry / `_build_patch` code path — not just stubs."""

    def _build_mfield(self):
        import numpy as np
        import rasterio
        from georeader.geotensor import GeoTensor

        from geopatcher._src.fields.raster import RasterField

        # A tiny in-memory raster — shared across primary and
        # secondary so the identity coreg is well-defined.
        values = np.arange(64, dtype=np.float32).reshape(8, 8)
        tensor = GeoTensor(
            values=values,
            transform=rasterio.Affine(10.0, 0.0, 500_000.0, 0.0, -10.0, 4_000_000.0),
            crs="EPSG:32629",
            fill_value_default=np.nan,
        )
        primary = RasterField(tensor)
        secondary = RasterField(tensor)
        return MatchedField(
            primary=primary,
            secondaries={"sec": secondary},
            coreg={"sec": lambda raw, primary_data: raw},
        )

    def test_real_patcher_yields_matched_patches(self) -> None:
        import numpy as np

        from geopatcher._src.spatial.geometry import SpatialRectangular
        from geopatcher._src.spatial.patcher import SpatialPatcher
        from geopatcher._src.spatial.sampler import SpatialRegularStride
        from geopatcher._src.spatial.window import SpatialBoxcar

        # Reuse a `SpatialPatcher` (the real one) inside the matched
        # patcher; aggregator unused for this test.
        primary = SpatialPatcher(
            geometry=SpatialRectangular(size=(4, 4)),
            sampler=SpatialRegularStride(step=(4, 4)),
            window=SpatialBoxcar(),
            aggregation=None,  # type: ignore[arg-type] — split only
        )
        mf = self._build_mfield()
        msp = MatchedSpatialPatcher(primary=primary)

        patches = list(msp.split(mf))
        # 8x8 field, 4x4 patches, stride 4 → 2x2 = 4 anchors.
        assert len(patches) == 4
        for mp in patches:
            assert set(mp.members) == {PRIMARY_KEY, "sec"}
            primary_chip = mp.members[PRIMARY_KEY].data
            secondary_chip = mp.members["sec"].data
            # Identity coreg: secondary equals primary numerically.
            np.testing.assert_array_equal(
                np.asarray(primary_chip), np.asarray(secondary_chip)
            )

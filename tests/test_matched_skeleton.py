"""Smoke tests for the scaffolded `geopatcher.matched` surface.

Locks in:

* `MatchedField` satisfies the `Field` Protocol (so existing samplers
  work on it),
* `MatchedPatch` exposes the primary / secondary accessors documented
  in ADR-003,
* the constructor invariants (key parity between `secondaries` and
  `coreg`) hold.

The actual `select` body is not implemented yet (Phase 4 PR); tests
that exercise it just assert `NotImplementedError`.
"""

from __future__ import annotations

from typing import Any

import pytest

import geopatcher.matched as matched_ns
from geopatcher._src.matched import (
    MatchedField,
    MatchedPatch,
    MatchedSpatialPatcher,
)
from geopatcher._src.matched.patch import PRIMARY_KEY
from geopatcher._src.patch import Patch
from geopatcher._src.protocols import Field


class _StubDomain:
    """Minimal `Domain` for tests — bounds and CRS are never read here."""

    @property
    def crs(self) -> Any:
        return "EPSG:4326"

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (0.0, 0.0, 1.0, 1.0)


class _StubField:
    """Minimal `Field` — `select` returns its name; `with_data` echoes."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._domain = _StubDomain()

    @property
    def domain(self) -> Any:
        return self._domain

    def select(self, indexer: Any) -> Any:
        return f"{self._name}@{indexer}"

    def with_data(self, array: Any) -> Any:
        return array


class TestReexports:
    def test_public_alias(self) -> None:
        assert matched_ns.MatchedField is MatchedField
        assert matched_ns.MatchedPatch is MatchedPatch
        assert matched_ns.MatchedSpatialPatcher is MatchedSpatialPatcher


class TestMatchedField:
    def test_satisfies_field_protocol(self) -> None:
        mf = MatchedField(primary=_StubField("p"))
        assert isinstance(mf, Field)

    def test_domain_forwards_to_primary(self) -> None:
        primary = _StubField("p")
        mf = MatchedField(primary=primary)
        assert mf.domain is primary.domain

    def test_keys_must_match(self) -> None:
        # Mismatch: secondary "s2" has no coreg entry.
        with pytest.raises(ValueError, match="missing coreg"):
            MatchedField(
                primary=_StubField("p"),
                secondaries={"s2": _StubField("s2")},
                coreg={},
            )
        # Mismatch: coreg "extra" has no secondary entry.
        with pytest.raises(ValueError, match="extra coreg"):
            MatchedField(
                primary=_StubField("p"),
                secondaries={},
                coreg={"extra": lambda raw, primary: raw},
            )

    def test_select_not_implemented(self) -> None:
        mf = MatchedField(primary=_StubField("p"))
        with pytest.raises(NotImplementedError):
            mf.select(object())

    def test_with_data_forwards(self) -> None:
        # `with_data` is implemented (forwards to primary) so existing
        # aggregations that need it for the primary path keep working.
        primary = _StubField("p")
        mf = MatchedField(primary=primary)
        assert mf.with_data(42) == 42

    def test_empty_secondaries_ok(self) -> None:
        # Degenerate "matchup of one source" — useful as a default in
        # code paths that conditionally add secondaries.
        mf = MatchedField(primary=_StubField("p"))
        assert mf.secondaries == {}
        assert mf.coreg == {}

    def test_secondary_named_primary_rejected(self) -> None:
        # "primary" is reserved for the primary in `MatchedPatch.members`;
        # a secondary with that name would silently overwrite it.
        with pytest.raises(ValueError, match="reserved key 'primary'"):
            MatchedField(
                primary=_StubField("p"),
                secondaries={"primary": _StubField("dup")},
                coreg={"primary": lambda raw, prim: raw},
            )


class TestMatchedPatch:
    def test_primary_accessor(self) -> None:
        primary_patch = Patch(data="p", anchor=(0, 0), indices=None)
        s2_patch = Patch(data="s", anchor=(0, 0), indices=None)
        mp = MatchedPatch(
            anchor=(0, 0),
            members={PRIMARY_KEY: primary_patch, "s2": s2_patch},
        )
        assert mp.primary is primary_patch

    def test_secondary_names_excludes_primary(self) -> None:
        mp = MatchedPatch(
            anchor=(0, 0),
            members={
                PRIMARY_KEY: Patch(data="p", anchor=(0, 0), indices=None),
                "s2": Patch(data="s2", anchor=(0, 0), indices=None),
                "landsat": Patch(data="l", anchor=(0, 0), indices=None),
            },
        )
        assert set(mp.secondary_names) == {"s2", "landsat"}

    def test_is_not_subclass_of_patch(self) -> None:
        # ADR-003: sibling carrier, not subclass.
        assert not issubclass(MatchedPatch, Patch)

    def test_missing_primary_rejected(self) -> None:
        # The docstring promises `members["primary"]` always exists —
        # enforce it at construction time so the failure mode is
        # immediate, not a later KeyError on `.primary`.
        with pytest.raises(ValueError, match="must contain the primary key"):
            MatchedPatch(
                anchor=(0, 0),
                members={"s2": Patch(data="s", anchor=(0, 0), indices=None)},
            )

    def test_valid_mask_keys_must_subset_members(self) -> None:
        import numpy as np

        # A stale mask whose key is no longer in `members` is a silent
        # bug source — reject up front.
        with pytest.raises(ValueError, match="valid_mask has keys not present"):
            MatchedPatch(
                anchor=(0, 0),
                members={PRIMARY_KEY: Patch(data="p", anchor=(0, 0), indices=None)},
                valid_mask={"ghost": np.zeros((2, 2), dtype=bool)},
            )

    def test_weights_keys_must_subset_members(self) -> None:
        import numpy as np

        with pytest.raises(ValueError, match="weights has keys not present"):
            MatchedPatch(
                anchor=(0, 0),
                members={PRIMARY_KEY: Patch(data="p", anchor=(0, 0), indices=None)},
                weights={"ghost": np.ones((2, 2))},
            )


class TestMatchedSpatialPatcher:
    def test_construction(self) -> None:
        # We don't construct a real `SpatialPatcher` here — just
        # confirm the dataclass accepts the expected fields. The real
        # split/merge wiring lands in Phase 4.
        msp = MatchedSpatialPatcher(primary=object())  # type: ignore[arg-type]
        assert msp.secondary_aggregators == {}

    def test_split_not_implemented(self) -> None:
        msp = MatchedSpatialPatcher(primary=object())  # type: ignore[arg-type]
        mf = MatchedField(primary=_StubField("p"))
        with pytest.raises(NotImplementedError):
            list(msp.split(mf))

    def test_merge_not_implemented(self) -> None:
        msp = MatchedSpatialPatcher(primary=object())  # type: ignore[arg-type]
        mf = MatchedField(primary=_StubField("p"))
        with pytest.raises(NotImplementedError):
            msp.merge([], mf)

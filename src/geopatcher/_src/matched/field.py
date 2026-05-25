"""`MatchedField` — composite Field that fans out per-anchor reads.

A `MatchedField` wraps:

* one **primary** `Field` (defines the anchor space, CRS, and domain),
* N **secondary** `Field`s keyed by name,
* a **coregistration callable** per secondary (any `Callable`; the
  intended choice is a `pipekit.Operator` from
  ``geotoolz.geom.coregister``, but the type is the broader Callable
  so geopatcher's core stays framework-free — see ADR-003).

It satisfies the existing `Field` Protocol by exposing the primary's
``domain`` and delegating reads through ``select``. On each
``select(indexer)`` it:

1. reads the primary's data,
2. reads each secondary's raw data at the same indexer,
3. pipes the (secondary_raw, primary_data) pair through that
   secondary's coreg callable,
4. returns a ``dict[str, data]`` keyed by source name (primary first).

Because it *is* a `Field`, every existing `SpatialPatcher` /
geometry / sampler / window / aggregation works on it unchanged.
The per-source data dict travels through the outer ``Patch.data``
field; `MatchedSpatialPatcher.split` unpacks it into a
`MatchedPatch` for matched-aware consumers.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from geopatcher._src.protocols import Domain, Field


# A coregistration callable maps ``(raw_secondary_patch_data,
# primary_patch_data) -> aligned_secondary_data``. The runtime
# contract is intentionally loose so any callable — a
# `pipekit.Operator`, a partial, a plain function — works.
CoregFn = Callable[[Any, Any], Any]


@dataclass(eq=False)
class MatchedField:
    """N co-registered Fields presented as one `Field`.

    Args:
        primary: The `Field` that defines the anchor space, CRS,
            and domain. Existing samplers run against this.
        secondaries: ``{name: Field}`` for the matched secondaries.
            Names appear as keys in `MatchedPatch.members`.
        coreg: ``{name: CoregFn}`` — one coregistration callable
            per secondary. Typically a
            ``geotoolz.geom.coregister.*`` operator, but any
            ``Callable[[Any, Any], Any]`` works. The callable is
            invoked as ``coreg[name](raw_secondary, primary_patch)``
            and its return value lands in
            ``MatchedPatch.members[name].data``.
        valid_mask: When True, `MatchedField` computes a per-source
            ``valid_mask`` (True = data present) and packs it on
            the `MatchedPatch`. Useful when secondaries have
            partial coverage (LEO swath ↔ GEO grid).

    Notes:
        The set of keys in ``secondaries`` and ``coreg`` must match
        exactly; mismatched keys raise on construction.
    """

    primary: Field
    secondaries: Mapping[str, Field] = field(default_factory=dict)
    coreg: Mapping[str, CoregFn] = field(default_factory=dict)
    valid_mask: bool = True

    def __post_init__(self) -> None:
        # Avoid late-import cycle: `patch.py` imports from this module
        # under TYPE_CHECKING and vice versa.
        from geopatcher._src.matched.patch import PRIMARY_KEY

        sec_keys = set(self.secondaries.keys())
        cor_keys = set(self.coreg.keys())
        if sec_keys != cor_keys:
            missing = sec_keys - cor_keys
            extra = cor_keys - sec_keys
            raise ValueError(
                "MatchedField.secondaries and .coreg must have the same keys; "
                f"missing coreg for {sorted(missing)!r}, "
                f"extra coreg for {sorted(extra)!r}."
            )
        # `PRIMARY_KEY` is reserved for the primary in `MatchedPatch.members`;
        # a secondary named "primary" would silently overwrite it on
        # patch construction. Reject up front with a clear message.
        if PRIMARY_KEY in sec_keys:
            raise ValueError(
                f"MatchedField.secondaries cannot use the reserved key "
                f"{PRIMARY_KEY!r}; pick another name."
            )

    @property
    def domain(self) -> Domain:
        """Forward the primary's domain so existing samplers work."""
        return self.primary.domain

    def select(self, indexer: Any) -> dict[str, Any]:
        """Read primary + all secondaries at ``indexer`` and align them.

        Returns a `dict[str, data]` keyed by source name — the primary
        under ``PRIMARY_KEY`` (``"primary"``), each secondary under
        the name supplied to `MatchedField.secondaries`. The values
        are whatever the underlying Fields' `select` returns: a
        `GeoTensor` for raster, a sub-`xarray.DataArray` for grid, etc.

        The per-source aligned data flows through `Patch.data` when
        a plain `SpatialPatcher` consumes a `MatchedField`. Consumers
        that want the matched-patch carrier shape go through
        `MatchedSpatialPatcher.split`, which unpacks the dict.
        """
        from geopatcher._src.matched.patch import PRIMARY_KEY

        primary_data = self.primary.select(indexer)
        result: dict[str, Any] = {PRIMARY_KEY: primary_data}
        for name, sec in self.secondaries.items():
            raw = sec.select(indexer)
            # Coreg callable: (secondary_raw, primary_data) -> aligned.
            # The runtime contract is intentionally loose so any
            # callable — pipekit.Operator, partial, lambda — works.
            result[name] = self.coreg[name](raw, primary_data)
        return result

    def with_data(self, array: Any) -> Any:
        """Forward to the primary; ``MatchedField.merge`` is per-source.

        The single-array ``with_data`` signature is for the
        primary's reconstruction path. Per-source aggregation back
        to N global fields goes through `MatchedSpatialPatcher.merge`,
        which uses each secondary's own ``with_data``.
        """
        return self.primary.with_data(array)

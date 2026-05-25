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

1. reads the primary's patch,
2. reads each secondary's raw patch at the same anchor,
3. pipes the (secondary_raw, primary_patch) pair through that
   secondary's coreg callable,
4. packs everything into a `MatchedPatch`.

Because it *is* a `Field`, every existing `SpatialPatcher` /
geometry / sampler / window / aggregation works on it unchanged.

Scaffolding — `select()` raises `NotImplementedError`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from geopatcher._src.matched.patch import MatchedPatch
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

    def select(self, indexer: Any) -> MatchedPatch:
        """Read primary + all secondaries at ``indexer`` and align.

        Body lands in Phase 4 (see design §6.2). The contract is:

        1. ``self.primary.select(indexer)`` → primary patch.
        2. For each ``name, sec in self.secondaries.items()``:
           ``sec.select(indexer)`` → raw secondary patch.
        3. ``self.coreg[name](raw_data, primary_data)`` →
           aligned secondary data, wrapped back into a `Patch`.
        4. Build & return a `MatchedPatch`.
        """
        raise NotImplementedError("Phase 4 PR — see design §6.2 and ADR-003.")

    def with_data(self, array: Any) -> Any:
        """Forward to the primary; ``MatchedField.merge`` is per-source.

        The single-array ``with_data`` signature is for the
        primary's reconstruction path. Per-source aggregation back
        to N global fields goes through `MatchedSpatialPatcher.merge`,
        which uses each secondary's own ``with_data``.
        """
        return self.primary.with_data(array)

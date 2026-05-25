"""`MatchedSpatialPatcher` — orchestrates split / merge across sources.

A thin wrapper around an existing `SpatialPatcher` that:

* splits a `MatchedField` into `MatchedPatch`es (the underlying
  patcher's iterator already does this for free — `MatchedField`
  *is* a `Field`),
* on ``merge``, dispatches to per-source aggregators and returns
  a ``dict[str, Field]`` (one reconstructed global field per
  source) instead of a single field.

The single-source `SpatialPatcher` is reused as the primary;
secondary aggregators live in a parallel mapping. The four-axis
decomposition is untouched — this class only adds the per-source
fan-out on the merge side. See ADR-003.

Scaffolding only.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from geopatcher._src.matched.field import MatchedField
    from geopatcher._src.matched.patch import MatchedPatch
    from geopatcher._src.protocols import Field
    from geopatcher._src.spatial.aggregation import SpatialAggregation
    from geopatcher._src.spatial.patcher import SpatialPatcher


@dataclass(eq=False)
class MatchedSpatialPatcher:
    """Spatial patcher that yields `MatchedPatch`es and merges per-source.

    Args:
        primary: A regular `SpatialPatcher` configured for the
            primary `Field`. Drives anchor placement, geometry,
            window, and primary aggregation.
        secondary_aggregators: ``{name: SpatialAggregation}`` — one
            aggregator per secondary. Missing names skip the
            per-source merge (you can still consume the iterator
            without merging back).
    """

    primary: SpatialPatcher
    secondary_aggregators: Mapping[str, SpatialAggregation] = field(
        default_factory=dict
    )

    def split(self, mfield: MatchedField) -> Iterator[MatchedPatch]:
        """Yield `MatchedPatch`es by walking ``mfield`` with the primary's sampler.

        Phase 4 PR fills in: this is just
        ``yield from self.primary.split(mfield)`` once
        `MatchedField.select` is implemented (since `MatchedField`
        already satisfies the `Field` Protocol).
        """
        raise NotImplementedError("Phase 4 PR — see design §6.4 and ADR-003.")

    def n_anchors(self, mfield: MatchedField) -> int:
        """Number of `MatchedPatch`es ``split`` will yield."""
        return self.primary.n_anchors(mfield)

    def anchors(self, mfield: MatchedField) -> list[Any]:
        """Materialise the sampler's anchor sequence for ``mfield``."""
        return self.primary.anchors(mfield)

    def merge(
        self,
        patches: Iterable[MatchedPatch],
        mfield: MatchedField,
    ) -> dict[str, Field]:
        """Per-source merge: dict of ``name -> reconstructed Field``.

        Returns the primary under the same key the matched-patches
        use (``MatchedPatch.PRIMARY_KEY``); secondaries appear under
        the names supplied to ``MatchedField.secondaries``. Names
        whose `secondary_aggregators` entry is missing are skipped.
        """
        raise NotImplementedError("Phase 4 PR — see design §6.4 and ADR-003.")

"""`MatchedSpatialPatcher` — orchestrates split / merge across sources.

A thin wrapper around an existing `SpatialPatcher` that:

* splits a `MatchedField` into `MatchedPatch`es. The underlying
  patcher's iterator yields one ``Patch`` per anchor whose ``data``
  is the per-source ``dict`` returned by `MatchedField.select`;
  this class unpacks that dict into a `MatchedPatch` carrier.
* on ``merge``, dispatches to per-source aggregators and returns
  a ``dict[str, Field]`` (one reconstructed global field per
  source) instead of a single field.

The single-source `SpatialPatcher` is reused as the primary;
secondary aggregators live in a parallel mapping. The four-axis
decomposition is untouched — this class only adds the per-source
fan-out on the merge side. See ADR-003.
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

        Internally drives ``self.primary.split(mfield)`` — since
        `MatchedField` already satisfies the `Field` Protocol, the
        existing sampler / geometry / window machinery works
        unchanged. Each outer ``Patch`` carries the per-source
        ``dict`` returned by `MatchedField.select` in its ``data``
        field; this method unpacks that dict into a `MatchedPatch`
        whose ``members`` is ``{name: Patch}`` and whose
        ``anchor`` / ``indices`` / ``weights`` mirror the outer
        patch so downstream aggregations see consistent metadata.
        """
        from geopatcher._src.matched.patch import PRIMARY_KEY, MatchedPatch
        from geopatcher._src.patch import Patch

        for outer in self.primary.split(mfield):
            data_by_name = outer.data
            if not isinstance(data_by_name, dict):
                # Belt-and-braces: a SpatialPatcher fed a plain Field
                # would give us a non-dict here. MatchedSpatialPatcher
                # is documented to expect a MatchedField; surface the
                # misuse rather than producing an obscure KeyError
                # later.
                raise TypeError(
                    "MatchedSpatialPatcher.split expects each Patch.data "
                    "to be a dict[str, data] (as produced by "
                    "MatchedField.select); got "
                    f"{type(data_by_name).__name__}. "
                    "Did you pass a plain Field instead of a MatchedField?"
                )
            if PRIMARY_KEY not in data_by_name:
                raise ValueError(
                    f"MatchedField.select must include the primary key "
                    f"{PRIMARY_KEY!r}; got keys {sorted(data_by_name)!r}."
                )
            members = {
                name: Patch(
                    data=data,
                    anchor=outer.anchor,
                    indices=outer.indices,
                    weights=outer.weights,
                )
                for name, data in data_by_name.items()
            }
            yield MatchedPatch(
                anchor=outer.anchor,
                members=members,
                # Per-source `valid_mask` computation is deferred — the
                # default behaviour is "no mask" so downstream
                # operators just see the data. Phase-2 enhancement:
                # plug in mask computation per Field type.
                valid_mask=None,
            )

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

        Returns the primary under ``MatchedPatch.PRIMARY_KEY``;
        secondaries appear under the names supplied to
        ``MatchedField.secondaries``. Names whose
        ``secondary_aggregators`` entry is missing are skipped (you
        can choose to only reconstruct a subset).

        Every source is aggregated against the primary's domain
        because the coregistration callable mapped each secondary
        onto the primary's grid at split time. Reconstructing a
        secondary back into its own original grid would require
        re-inverting the coregistration, which is the user's
        problem if they need it.
        """
        from geopatcher._src.matched.patch import PRIMARY_KEY

        # Patches stream lazily — materialise the per-source lists
        # in one pass so we don't iterate `patches` N+1 times.
        per_source: dict[str, list[Any]] = {PRIMARY_KEY: []}
        for name in self.secondary_aggregators:
            per_source[name] = []
        for mp in patches:
            for name, patch in mp.members.items():
                if name in per_source:
                    per_source[name].append(patch)

        primary_domain = mfield.domain
        result: dict[str, Field] = {
            PRIMARY_KEY: self.primary.merge(per_source[PRIMARY_KEY], primary_domain),
        }
        for name, agg in self.secondary_aggregators.items():
            result[name] = agg.merge(per_source[name], primary_domain)
        return result

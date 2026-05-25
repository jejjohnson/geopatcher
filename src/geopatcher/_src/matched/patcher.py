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

import numpy as np


if TYPE_CHECKING:
    from geopatcher._src.matched.field import MatchedField
    from geopatcher._src.matched.patch import MatchedPatch
    from geopatcher._src.spatial.aggregation import SpatialAggregation
    from geopatcher._src.spatial.patcher import SpatialPatcher


def _compute_valid_mask(data: Any) -> np.ndarray | None:
    """Best-effort validity mask for ``data``.

    For array-coercible numeric data (the common raster case),
    returns ``np.isfinite(data)`` — True where the value is real
    and finite, False on NaN / +-inf (the conventional nodata
    sentinel for float rasters). For non-array data or non-numeric
    arrays, returns None so the caller can simply omit that source's
    mask entry rather than emit a meaningless all-True / all-False
    array.
    """
    try:
        arr = np.asarray(data)
    except (TypeError, ValueError):
        return None
    if not np.issubdtype(arr.dtype, np.number):
        return None
    return np.isfinite(arr)


@dataclass(eq=False)
class MatchedSpatialPatcher:
    """Spatial patcher that yields `MatchedPatch`es and merges per-source.

    Args:
        primary: A regular `SpatialPatcher` configured for the
            primary `Field`. Drives anchor placement, geometry,
            window, and primary aggregation.
        secondary_aggregators: ``{name: SpatialAggregation}`` — one
            aggregator per secondary. Names that don't match any
            entry in ``mfield.secondaries`` raise on ``split`` /
            ``merge`` rather than silently skipping (catches config
            typos like ``"s22"`` instead of ``"s2"``). Omitting a
            secondary from this mapping is fine — that source is
            simply not merged back, which is the documented opt-out.
    """

    primary: SpatialPatcher
    secondary_aggregators: Mapping[str, SpatialAggregation] = field(
        default_factory=dict
    )

    def _validate_aggregator_names(self, mfield: MatchedField) -> None:
        """Reject typoed `secondary_aggregators` keys up front.

        Without this guard, a typo like
        ``secondary_aggregators={"s22": ...}`` would silently drop
        every real ``"s2"`` patch and still call the typoed
        aggregator with an empty list — producing a bogus
        reconstructed field with no error.

        Best-effort: if ``mfield`` doesn't expose ``secondaries``
        (i.e. caller mistakenly passed a plain Field), the
        type-error path in ``split`` / the empty-merge path will
        surface that misuse — we don't double-fault here.
        """
        secondaries = getattr(mfield, "secondaries", None)
        if secondaries is None:
            return
        unknown = set(self.secondary_aggregators) - set(secondaries)
        if unknown:
            raise ValueError(
                "MatchedSpatialPatcher.secondary_aggregators has names "
                "not in mfield.secondaries: "
                f"{sorted(unknown)!r}. "
                f"Known secondaries: {sorted(secondaries)!r}."
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

        Per-source ``valid_mask`` arrays are computed when
        ``mfield.valid_mask`` is True (the default): for numeric
        array-coercible data, ``np.isfinite(data)`` marks the
        positions of NaN / inf nodata sentinels. Non-array members
        are simply omitted from the mask dict (and the dict drops
        to ``None`` if no member produced a mask).
        """
        from geopatcher._src.matched.patch import PRIMARY_KEY, MatchedPatch
        from geopatcher._src.patch import Patch

        self._validate_aggregator_names(mfield)

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
            if mfield.valid_mask:
                mask_dict = {
                    name: mask
                    for name, data in data_by_name.items()
                    if (mask := _compute_valid_mask(data)) is not None
                }
                valid_mask: dict[str, np.ndarray] | None = mask_dict or None
            else:
                valid_mask = None
            yield MatchedPatch(
                anchor=outer.anchor,
                members=members,
                valid_mask=valid_mask,
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
    ) -> dict[str, Any]:
        """Per-source merge: dict of ``name -> aggregation result``.

        Returns the primary under ``MatchedPatch.PRIMARY_KEY``;
        secondaries appear under the names supplied to
        ``MatchedField.secondaries``. Names whose
        ``secondary_aggregators`` entry is missing are skipped (you
        can choose to only reconstruct a subset). Names that *are*
        in ``secondary_aggregators`` but not in
        ``mfield.secondaries`` raise — typo guard.

        The value type is intentionally ``Any`` because the
        underlying `SpatialAggregation.merge` returns whatever the
        aggregator produces — typically a `GeoTensor` for stitched
        rasters, but for ``Sum`` / ``Mean`` / ``Max`` it may be a
        plain numpy array. Callers that need a `Field` shape can
        wrap with the source's ``Field.with_data``.

        Every source is aggregated against the primary's domain
        because the coregistration callable mapped each secondary
        onto the primary's grid at split time. Reconstructing a
        secondary back into its own original grid would require
        re-inverting the coregistration, which is the user's
        problem if they need it.

        Strict-mode streaming-safety: each secondary aggregator is
        checked via the same ``_warn_if_unsafe_streaming`` helper
        the primary ``SpatialPatcher`` uses, so a non-streaming
        secondary aggregation surfaces the same warning/error in
        strict mode as the primary path.
        """
        from geopatcher._src.matched.patch import PRIMARY_KEY
        from geopatcher._src.spatial.aggregation import _warn_if_unsafe_streaming

        self._validate_aggregator_names(mfield)

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
        result: dict[str, Any] = {
            PRIMARY_KEY: self.primary.merge(per_source[PRIMARY_KEY], primary_domain),
        }
        for name, agg in self.secondary_aggregators.items():
            # Mirror the primary path's strict-mode streaming check
            # so a non-streaming secondary aggregator doesn't slip
            # through.
            _warn_if_unsafe_streaming(agg)
            result[name] = agg.merge(per_source[name], primary_domain)
        return result

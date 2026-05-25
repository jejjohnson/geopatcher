"""`MatchedPatch` — the carrier for a co-located patch across N sources.

Sibling carrier to `Patch` rather than a subclass. Two reasons,
captured as ADR-003:

1. `Patch` is parameterized over ``[AnchorT, IndicesT, DataT]`` — a
   single concrete shape per (Geometry x Domain) pairing. A
   `MatchedPatch` cannot satisfy that contract because it holds
   ``dict[str, Patch]`` with heterogeneous data types across keys.
2. Consumers that don't care about matchups continue to type
   against plain `Patch`; consumers that do care explicitly type
   against `MatchedPatch`. No LSP surprises in either direction.

`MatchedPatch.members["primary"]` always carries the patch read from
the primary field (the anchor space); secondaries are keyed by the
names supplied to `MatchedField`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import numpy as np

    from geopatcher._src.patch import Patch


# Convention: the primary patch is always stored under this key in
# ``members``. Public so callers can write
# ``mp.members[MatchedPatch.PRIMARY_KEY]`` instead of stringly typing.
PRIMARY_KEY = "primary"


@dataclass(eq=False)
class MatchedPatch:
    """A co-located patch read from N sources at a single anchor.

    Args:
        anchor: Where the patch lives in the primary's coordinate
            system. Same shape the primary's `Sampler` emits.
        members: ``{name: Patch}``. ``members["primary"]`` is the
            primary; secondary keys are the names given to
            `MatchedField.secondaries`.
        valid_mask: Optional ``{name: ndarray}`` of per-source masks
            indicating which pixels of each member contain valid
            data (False = nodata / out-of-swath / off-edge). When a
            secondary's coregistration produces partial coverage —
            e.g. the LEO swath only crosses half the GEO patch —
            the mask is the workhorse the downstream operator uses
            to decide what to ignore.
        weights: Optional ``{name: ndarray}`` of per-source window
            weights. Most callers leave this `None` and rely on the
            primary's `Window` axis.
    """

    anchor: Any
    members: dict[str, Patch]
    valid_mask: dict[str, np.ndarray] | None = None
    weights: dict[str, np.ndarray] | None = field(default=None)

    PRIMARY_KEY = PRIMARY_KEY

    @property
    def primary(self) -> Patch:
        """Convenience accessor for ``members[PRIMARY_KEY]``."""
        return self.members[PRIMARY_KEY]

    @property
    def secondary_names(self) -> tuple[str, ...]:
        """The keys of ``members`` other than the primary."""
        return tuple(k for k in self.members if k != PRIMARY_KEY)

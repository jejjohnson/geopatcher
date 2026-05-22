"""JAX-friendly batched patch splitting utilities."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np

from geopatcher._src.patch import Patch


try:
    import jax.numpy as jnp
except ImportError:  # pragma: no cover
    jnp = np


@dataclass(eq=False)
class BatchedPatch:
    """A leading-axis batch of spatial patches."""

    data: Any
    anchors: list[Any]
    valid: Any
    indices: list[Any]
    weights: list[Any]


def batch_split(
    patcher: Any,
    field: Any,
    *,
    batch_size: int,
    pad_last: bool = True,
) -> Iterator[BatchedPatch]:
    """Yield `BatchedPatch` objects with data stacked on a leading axis."""
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    batch = []
    for patch in patcher.split(field):
        batch.append(patch)
        if len(batch) == batch_size:
            yield _batch(batch, batch_size, pad=False)
            batch = []
    if batch:
        yield _batch(batch, batch_size, pad=pad_last)


def unbatch(batch: BatchedPatch, data: Any | None = None) -> list[Patch]:
    """Convert a `BatchedPatch` back to ordinary `Patch` objects."""
    arrays = batch.data if data is None else data
    valid = np.asarray(batch.valid, dtype=bool)
    patches = []
    for i, is_valid in enumerate(valid):
        if is_valid:
            patches.append(
                Patch(
                    data=arrays[i],
                    anchor=batch.anchors[i],
                    indices=batch.indices[i],
                    weights=batch.weights[i],
                )
            )
    return patches


def _batch(patches: list[Patch], batch_size: int, *, pad: bool) -> BatchedPatch:
    arrays = [jnp.asarray(p.data) for p in patches]
    valid = [True] * len(patches)
    anchors = [p.anchor for p in patches]
    indices = [p.indices for p in patches]
    weights = [p.weights for p in patches]
    if pad:
        for _ in range(batch_size - len(patches)):
            arrays.append(jnp.zeros_like(arrays[0]))
            valid.append(False)
            anchors.append(None)
            indices.append(None)
            weights.append(None)
    return BatchedPatch(
        data=jnp.stack(arrays, axis=0),
        anchors=anchors,
        valid=jnp.asarray(valid, dtype=bool),
        indices=indices,
        weights=weights,
    )

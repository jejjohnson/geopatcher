"""Reference runners for applying operators over patch streams."""

from __future__ import annotations

import pickle
import sys
import warnings
from collections.abc import Callable
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import replace
from typing import Any, Literal

from geopatcher._src.patch import Patch
from geopatcher._src.protocols import Field
from geopatcher._src.spatial.patcher import SpatialPatcher


Backend = Literal["thread", "process"]
ErrorPolicy = Literal["raise", "skip"]


def parallel_map(
    patcher: SpatialPatcher,
    field: Field,
    operator: Callable[[Any], Any],
    *,
    n_workers: int = 8,
    backend: Backend = "thread",
    show_progress: bool = False,
    journal: Any | None = None,
    on_error: ErrorPolicy = "raise",
) -> list[Patch]:
    """Apply ``operator`` to each spatial patch with a reference executor.

    Args:
        patcher: Spatial patcher that defines the anchor schedule.
        field: Field to split into patches.
        operator: Callable applied to each patch's ``data``.
        n_workers: Number of worker threads or processes.
        backend: ``"thread"`` for `ThreadPoolExecutor` or ``"process"`` for
            `ProcessPoolExecutor`.
        show_progress: If ``True``, print a lightweight completion counter
            to stderr.
        journal: Reserved for future `PatchJournal` integration.
        on_error: ``"raise"`` to fail fast, or ``"skip"`` to omit failed
            patches from the returned list.

    Returns:
        Patches with ``data`` replaced by ``operator(patch.data)``, ordered by
        the patcher's anchor schedule.
    """
    if n_workers < 1:
        raise ValueError("n_workers must be >= 1")
    if backend not in {"thread", "process"}:
        raise ValueError("backend must be 'thread' or 'process'")
    if on_error not in {"raise", "skip"}:
        raise ValueError("on_error must be 'raise' or 'skip'")
    if journal is not None:
        raise NotImplementedError("journal integration is reserved for PatchJournal")
    if backend == "process":
        _ensure_picklable_operator(operator)

    patches = list(patcher.split(field))
    executor_cls = ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor
    results: list[tuple[int, Patch]] = []
    total = len(patches)
    done = 0

    with executor_cls(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_apply_operator, i, patch, operator): i
            for i, patch in enumerate(patches)
        }
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                if on_error == "raise":
                    raise
                warnings.warn(
                    f"parallel_map skipped patch {futures[future]} after operator "
                    f"error: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            done += 1
            if show_progress:
                print(f"\r{done}/{total}", end="", file=sys.stderr)

    if show_progress:
        print(file=sys.stderr)
    return [patch for _, patch in sorted(results, key=lambda item: item[0])]


def _apply_operator(
    index: int, patch: Patch, operator: Callable[[Any], Any]
) -> tuple[int, Patch]:
    return index, replace(patch, data=operator(patch.data))


def _ensure_picklable_operator(operator: Callable[[Any], Any]) -> None:
    try:
        pickle.dumps(operator)
    except Exception as exc:
        raise TypeError(
            "parallel_map(..., backend='process') requires a picklable operator; "
            "use a top-level function, use backend='thread', or wrap your "
            "operator with a cloudpickle-based runner."
        ) from exc

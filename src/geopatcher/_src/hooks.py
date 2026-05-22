"""Callback hook protocol and dispatch helpers for patcher observability."""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable


UNKNOWN_TOTAL = -1


@runtime_checkable
class PatcherHook(Protocol):
    """Optional callbacks emitted by patcher split and merge operations.

    Hook objects may implement any subset of these methods. The patchers
    dispatch callbacks dynamically, in order, and convert hook exceptions into
    warnings so observability code cannot interrupt patch generation.
    """

    def on_split_start(self, n_anchors: int) -> None: ...

    def on_patch_start(self, anchor: Any) -> None: ...

    def on_patch_done(self, anchor: Any, runtime_s: float, bytes_: int) -> None: ...

    def on_split_end(self) -> None: ...

    def on_merge_start(self, n_patches: int) -> None: ...

    def on_merge_end(self, output_bytes: int) -> None: ...

    def on_error(self, anchor: Any, exc: Exception) -> None: ...


def _as_hooks(hooks: Iterable[PatcherHook] | None) -> tuple[PatcherHook, ...]:
    """Materialize hooks once so one-shot iterables work across callbacks."""
    return () if hooks is None else tuple(hooks)


def _dispatch(hooks: Iterable[PatcherHook], method: str, *args: Any) -> None:
    """Call ``method`` on each hook that implements it.

    Patcher methods call this after materialising user-provided iterables with
    `_as_hooks`, so generator-backed hook lists are safe to reuse across all
    callbacks in a split or merge lifecycle. The warning `stacklevel` assumes a
    direct call from a patcher method or helper so users see the patcher call
    site rather than this internal dispatcher.

    Hook failures are intentionally downgraded to warnings: callbacks are
    observability side effects and must not change patcher correctness.
    """
    for hook in hooks:
        callback = getattr(hook, method, None)
        if callback is None:
            continue
        try:
            callback(*args)
        except Exception as exc:
            warnings.warn(
                f"PatcherHook.{method} failed on {type(hook).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )


def _len_or_unknown(values: Iterable[Any]) -> int:
    try:
        return len(values)  # type: ignore[arg-type]
    except TypeError:
        return UNKNOWN_TOTAL


def _nbytes(value: Any) -> int:
    """Best-effort byte count for patch data and aggregation outputs.

    Prefer direct ``.nbytes`` (NumPy arrays and many array-like objects), then
    ``.values.nbytes`` for xarray / GeoTensor-style wrappers, then
    ``.data.nbytes`` for backends that expose their array under ``data``.
    """
    for candidate in (
        value,
        getattr(value, "values", None),
        getattr(value, "data", None),
    ):
        if candidate is None:
            continue
        nbytes = getattr(candidate, "nbytes", None)
        if nbytes is not None:
            return int(nbytes)
    return 0

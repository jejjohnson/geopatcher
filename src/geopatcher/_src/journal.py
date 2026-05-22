"""Small local journal for resumable patch jobs."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(eq=False)
class PatchJournal:
    """Append-only local journal keyed by patch anchor.

    The journal stores one JSON record per committed patch. Re-opening the
    same path reconstructs the latest status for each anchor, allowing
    ``patcher.split(..., journal=journal)`` to skip completed work after a
    crash.
    """

    uri: str

    def __post_init__(self) -> None:
        self.path = Path(self.uri)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rows: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            self._load()

    def has(self, anchor: Any) -> bool:
        """Return ``True`` when ``anchor`` has a successful journal row."""
        row = self._rows.get(_anchor_key(anchor))
        return row is not None and row["status"] == "ok"

    def commit(
        self,
        anchor: Any,
        *,
        status: str,
        runtime_s: float,
        output_uri: str | None = None,
        error: str | None = None,
    ) -> None:
        """Append a durable status row for ``anchor``."""
        row = {
            "anchor": anchor,
            "status": status,
            "runtime_s": float(runtime_s),
            "output_uri": output_uri,
            "error": error,
        }
        key = _anchor_key(anchor)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")
        self._rows[key] = row

    def pending(self, all_anchors: list[Any]) -> list[Any]:
        """Return anchors without a successful journal row."""
        return [anchor for anchor in all_anchors if not self.has(anchor)]

    def _load(self) -> None:
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    warnings.warn(
                        f"skipping malformed journal row in {self.path}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue
                self._rows[_anchor_key(row["anchor"])] = row


def _anchor_key(anchor: Any) -> str:
    return json.dumps(anchor, sort_keys=True, default=str)

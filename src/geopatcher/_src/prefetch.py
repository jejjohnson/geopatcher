"""Thread-backed prefetching for synchronous patch iterators."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from queue import Queue
from threading import Thread


_SENTINEL = object()


@dataclass(frozen=True)
class _Raised:
    exc: BaseException


def prefetch_iterable[T](iterable: Iterable[T], prefetch: int) -> Iterator[T]:
    """Return ``iterable`` with up to ``prefetch`` items read ahead."""
    if prefetch < 0:
        raise ValueError("prefetch must be >= 0")
    if prefetch == 0:
        return iter(iterable)
    return _PrefetchIterator(iterable, prefetch)


class _PrefetchIterator[T](Iterator[T]):
    def __init__(self, iterable: Iterable[T], prefetch: int) -> None:
        self._queue: Queue[T | _Raised | object] = Queue(maxsize=prefetch)
        self._thread = Thread(target=self._run, args=(iter(iterable),), daemon=True)
        self._thread.start()

    def __iter__(self) -> Iterator[T]:
        return self

    def __next__(self) -> T:
        item = self._queue.get()
        if item is _SENTINEL:
            self._thread.join()
            raise StopIteration
        if isinstance(item, _Raised):
            self._thread.join()
            raise item.exc
        return item

    def _run(self, iterator: Iterator[T]) -> None:
        try:
            for item in iterator:
                self._queue.put(item)
        except BaseException as exc:
            self._queue.put(_Raised(exc))
        finally:
            self._queue.put(_SENTINEL)

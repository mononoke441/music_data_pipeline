"""Bounded next-track loading for the resident SongFormer GPU worker."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterator, TypeVar


T = TypeVar("T")
R = TypeVar("R")
_STOP = object()


def _read_and_load(queue_input, loader: Callable[[T], R]):
    item = queue_input.get()
    if item is None:
        return _STOP
    return loader(item)


def iter_prefetched_queue(
    queue_input,
    loader: Callable[[T], R],
    *,
    prefetch: int = 1,
) -> Iterator[R]:
    """Yield queue items after loading, with at most one track decoded ahead.

    ``prefetch=0`` is the numerical-parity/debug fallback.  ``prefetch=1``
    owns one background thread: before yielding the current loaded track it
    starts loading the next, so CPU decode overlaps current GPU inference.
    """

    if prefetch not in (0, 1):
        raise ValueError("prefetch must be 0 or 1")

    if prefetch == 0:
        while True:
            value = _read_and_load(queue_input, loader)
            if value is _STOP:
                return
            yield value

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="songformer-decode") as pool:
        pending = pool.submit(_read_and_load, queue_input, loader)
        while True:
            value = pending.result()
            if value is _STOP:
                return
            pending = pool.submit(_read_and_load, queue_input, loader)
            yield value

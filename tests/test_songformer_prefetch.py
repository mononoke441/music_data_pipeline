from __future__ import annotations

import importlib.util
import queue
import threading
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_prefetch_module():
    path = ROOT / "SongFormer" / "audio_prefetch.py"
    spec = importlib.util.spec_from_file_location("songformer_audio_prefetch", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prefetch_starts_next_decode_before_current_is_consumed():
    module = load_prefetch_module()
    items: queue.Queue = queue.Queue()
    items.put("first")
    items.put("second")
    items.put(None)

    second_started = threading.Event()
    release_second = threading.Event()

    def loader(item: str) -> str:
        if item == "second":
            second_started.set()
            assert release_second.wait(timeout=1.0)
        return item.upper()

    iterator = module.iter_prefetched_queue(items, loader, prefetch=1)
    assert next(iterator) == "FIRST"
    assert second_started.wait(timeout=1.0)
    release_second.set()
    assert next(iterator) == "SECOND"
    try:
        next(iterator)
        assert False, "iterator should stop at the queue sentinel"
    except StopIteration:
        pass


def test_prefetch_zero_is_strictly_serial():
    module = load_prefetch_module()
    items: queue.Queue = queue.Queue()
    items.put("first")
    items.put("second")
    items.put(None)
    calls = []

    iterator = module.iter_prefetched_queue(items, lambda item: calls.append(item) or item, prefetch=0)
    assert next(iterator) == "first"
    assert calls == ["first"]
    assert next(iterator) == "second"
    assert calls == ["first", "second"]

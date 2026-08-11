from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MusicToolsPipeline"))

from config import Config
import ray_inference


class _FakeQueue:
    def __init__(self) -> None:
        self.items = []

    def put(self, item, block=True, timeout=None):
        assert block is True
        assert timeout == 1.0
        self.items.append(item)


def test_coordinator_sends_exactly_one_sentinel_per_model_after_all_loaders(
    monkeypatch,
):
    loader_refs = ["loader-0", "loader-1"]
    model_refs = ["model-0", "model-1"]
    save_ref = "saver"
    queue = _FakeQueue()
    order = iter(loader_refs + model_refs + [save_ref])
    values = {
        "loader-0": 2,
        "loader-1": 3,
        "model-0": 2,
        "model-1": 3,
        "saver": {"persisted": 5, "failed": 0},
    }

    def fake_wait(pending, num_returns, timeout):
        assert num_returns == 1
        assert timeout == 1.0
        ref = next(order)
        if ref == "loader-1":
            assert queue.items == []
        assert ref in pending
        return [ref], [item for item in pending if item != ref]

    monkeypatch.setattr(ray_inference.ray, "wait", fake_wait)
    monkeypatch.setattr(ray_inference.ray, "get", lambda ref: values[ref])

    loader_results, model_results, save_result = (
        ray_inference._supervise_worker_refs(
            loader_refs=loader_refs,
            model_refs=model_refs,
            save_ref=save_ref,
            input_queue=queue,
        )
    )

    assert queue.items == [None, None]
    assert loader_results == [2, 3]
    assert model_results == [2, 3]
    assert save_result == values[save_ref]


def test_supervisor_observes_saver_failure_while_loaders_are_pending(monkeypatch):
    queue = _FakeQueue()

    def fake_wait(pending, num_returns, timeout):
        assert "saver" in pending
        return ["saver"], [item for item in pending if item != "saver"]

    def fake_get(ref):
        raise RuntimeError(f"actor failed: {ref}")

    monkeypatch.setattr(ray_inference.ray, "wait", fake_wait)
    monkeypatch.setattr(ray_inference.ray, "get", fake_get)

    with pytest.raises(RuntimeError, match="actor failed: saver"):
        ray_inference._supervise_worker_refs(
            loader_refs=["loader"],
            model_refs=["model"],
            save_ref="saver",
            input_queue=queue,
        )
    assert queue.items == []


def test_worker_graph_cancels_all_refs_after_any_actor_failure(tmp_path, monkeypatch):
    class RemoteMethod:
        def __init__(self, ref):
            self.ref = ref

        def remote(self, *args, **kwargs):
            return self.ref

    class Actor:
        def __init__(self, ref):
            self.run = RemoteMethod(ref)

    class Progress:
        n = 0

        def update(self, value):
            self.n += value

        def close(self):
            pass

    cancelled = []

    def fail_supervision(**kwargs):
        raise RuntimeError("saver died")

    monkeypatch.setattr(ray_inference, "_supervise_worker_refs", fail_supervision)
    monkeypatch.setattr(
        ray_inference.ray,
        "cancel",
        lambda ref, force=False: cancelled.append((ref, force)),
    )

    with pytest.raises(RuntimeError, match="saver died"):
        ray_inference._run_worker_graph(
            loader_workers=[Actor("loader-ref")],
            model_workers=[Actor("model-ref")],
            save_worker=Actor("save-ref"),
            queue_monitor=Actor("monitor-ref"),
            input_queue=object(),
            result_queue=object(),
            db_path=str(tmp_path / "progress.jsonl"),
            total_samples=1,
            progress=Progress(),
        )

    assert ("loader-ref", True) in cancelled
    assert ("model-ref", True) in cancelled
    assert ("save-ref", True) in cancelled
    assert ("monitor-ref", False) in cancelled


def test_lance_shards_preserve_nonzero_base_offset(monkeypatch):
    calls = []

    class FakeDataLoaderWorker:
        @staticmethod
        def remote(**kwargs):
            calls.append(kwargs)
            return kwargs

    monkeypatch.setattr(ray_inference, "DataLoaderWorker", FakeDataLoaderWorker)
    config = Config()
    config.dataloader_type = "lance"
    config.num_dataloader_workers = 3
    config.batch_size = 4

    workers = ray_inference._create_loader_workers(
        config=config,
        data_path="input.lance",
        db_path="progress.jsonl",
        total_samples=7,
        dataloader_kwargs={
            "offset": 100,
            "limit": 7,
            "prompt_key": "audio_flac",
        },
    )

    assert workers == calls
    assert [(call["offset"], call["limit"]) for call in calls] == [
        (100, 3),
        (103, 2),
        (105, 2),
    ]

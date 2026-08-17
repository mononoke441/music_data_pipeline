from __future__ import annotations

import asyncio
import os
import threading
import time
from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import service_api  # noqa: E402
from service_api import (  # noqa: E402
    DynamicBatchService,
    ServiceError,
    ServiceRequest,
    create_service_app,
)


def _request(request_id: str, *, audio_id: str = "audio-a") -> ServiceRequest:
    return ServiceRequest(
        job_id="job-a",
        request_id=request_id,
        audio_id=audio_id,
        audio_path=f"/{audio_id}.wav",
        input_fingerprint=f"fp-{audio_id}",
        record={"duration": 3.0},
    )


def test_lifespan_loads_once_and_dynamic_batch_preserves_envelopes():
    async def scenario():
        loads = []
        batches = []

        def loader():
            loads.append("loaded")
            return object()

        def process(model, requests):
            assert model is not None
            batches.append([request.request_id for request in requests])
            return [
                {**request.record, "audio_id": request.audio_id, "done": True}
                for request in requests
            ]

        service = DynamicBatchService(
            loader,
            process,
            "fake_stage",
            "cpu",
            "model-v1",
            4,
            25,
            8,
        )
        await service.start()
        responses = await asyncio.gather(
            service.submit(_request("r1", audio_id="a")),
            service.submit(_request("r2", audio_id="b")),
        )
        snapshot = service.health_snapshot()
        await service.stop()

        assert loads == ["loaded"]
        assert batches == [["r1", "r2"]]
        assert [value.status for value in responses] == ["ok", "ok"]
        assert [value.audio_id for value in responses] == ["a", "b"]
        assert responses[0].record["done"] is True
        assert responses[0].stage == "fake_stage"
        assert responses[0].model_fingerprint == "model-v1"
        assert responses[0].job_id == "job-a"
        assert responses[0].input_fingerprint == "fp-a"
        assert snapshot["status"] == "ok"
        assert snapshot["process_started_at"] <= snapshot["model_loaded_at"]
        assert snapshot["model_loaded_at"] <= time.time()
        assert snapshot["load_duration"] >= 0.0
        assert snapshot["pid"] == os.getpid()
        assert snapshot["cpu_seconds"] >= 0.0
        assert snapshot["rss_bytes"] is None or snapshot["rss_bytes"] > 0

    asyncio.run(scenario())


def test_request_id_is_idempotent_inflight_and_after_completion():
    async def scenario():
        calls = []

        def process(_, requests):
            calls.append(len(requests))
            return [{"audio_id": request.audio_id} for request in requests]

        service = DynamicBatchService(
            lambda: object(), process, "stage", "cpu", "v1", 4, 20, 8
        )
        await service.start()
        request = _request("same")
        first, second = await asyncio.gather(
            service.submit(request), service.submit(request)
        )
        cached = await service.submit(request)
        with pytest.raises(ServiceError) as collision:
            await service.submit(_request("same", audio_id="different"))
        await service.stop()

        assert calls == [1]
        assert first == second == cached
        assert collision.value.status_code == 409

    asyncio.run(scenario())


def test_queue_full_is_429_while_duplicate_still_joins():
    async def scenario():
        entered = threading.Event()
        release = threading.Event()

        def process(_, requests):
            entered.set()
            assert release.wait(timeout=5)
            return [{"audio_id": request.audio_id} for request in requests]

        service = DynamicBatchService(
            lambda: object(), process, "stage", "cpu", "v1", 1, 0, 1
        )
        await service.start()
        first = asyncio.create_task(service.submit(_request("r1", audio_id="a")))
        assert await asyncio.to_thread(entered.wait, 2)
        second_request = _request("r2", audio_id="b")
        second = asyncio.create_task(service.submit(second_request))
        await asyncio.sleep(0)
        duplicate = asyncio.create_task(service.submit(second_request))
        with pytest.raises(ServiceError) as full:
            await service.submit(_request("r3", audio_id="c"))
        release.set()
        responses = await asyncio.gather(first, second, duplicate)
        await service.stop()

        assert full.value.status_code == 429
        assert responses[1] == responses[2]

    asyncio.run(scenario())


def test_fastapi_contract_exposes_health_and_infer():
    load_count = 0

    def loader():
        nonlocal load_count
        load_count += 1
        return object()

    service = DynamicBatchService(
        loader,
        lambda _, requests: [dict(request.record) for request in requests],
        "stage",
        "cpu",
        "v1",
        2,
        0,
        4,
    )
    with TestClient(create_service_app(service)) as client:
        health = client.get("/healthz")
        response = client.post("/v1/infer", json=_request("r1").model_dump())

    assert load_count == 1
    assert health.status_code == 200
    assert health.json()["stage"] == "stage"
    assert response.status_code == 200
    assert set(response.json()) == {
        "job_id",
        "request_id",
        "audio_id",
        "input_fingerprint",
        "status",
        "record",
        "stage",
        "model_fingerprint",
        "elapsed_seconds",
        "error",
    }


def test_health_telemetry_failure_does_not_change_ready_status(monkeypatch):
    async def scenario():
        service = DynamicBatchService(
            lambda: object(),
            lambda _, requests: [dict(request.record) for request in requests],
            "gpu-stage",
            "cuda:0",
            "v1",
            1,
            0,
            2,
        )
        assert service.health_snapshot()["status"] == "created"
        assert service.health_snapshot()["model_loaded_at"] is None
        await service.start()
        monkeypatch.setattr(
            service_api,
            "_gpu_memory_snapshot",
            lambda _: (_ for _ in ()).throw(RuntimeError("telemetry failed")),
        )
        snapshot = service.health_snapshot()
        await service.stop()
        assert snapshot["status"] == "ok"
        assert snapshot["gpu_memory"] is None

    asyncio.run(scenario())


def test_health_is_not_ok_until_lifespan_loader_finishes():
    async def scenario():
        entered = threading.Event()
        release = threading.Event()

        def loader():
            entered.set()
            assert release.wait(timeout=3)
            return object()

        service = DynamicBatchService(
            loader,
            lambda _, requests: [dict(request.record) for request in requests],
            "stage",
            "cpu",
            "v1",
            1,
            0,
            2,
        )
        startup = asyncio.create_task(service.start())
        assert await asyncio.to_thread(entered.wait, 2)
        loading = service.health_snapshot()
        assert loading["status"] == "starting"
        assert loading["model_loaded_at"] is None
        release.set()
        await startup
        ready = service.health_snapshot()
        await service.stop()
        assert ready["status"] == "ok"
        assert ready["model_loaded_at"] is not None

    asyncio.run(scenario())

#!/usr/bin/env python3
"""Shared bounded dynamic-batching API for persistent inference services."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import os
import resource
import sys
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Sequence

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator


_MODULE_STARTED_AT = time.time()


def _process_started_at() -> float:
    """Best-effort absolute process creation time in Unix seconds."""

    try:
        import psutil

        return float(psutil.Process(os.getpid()).create_time())
    except Exception:
        return _MODULE_STARTED_AT


def _process_resource_snapshot() -> Dict[str, Any]:
    """Collect process resource metrics without making health dependent on psutil."""

    snapshot: Dict[str, Any] = {
        "cpu_percent": None,
        "cpu_seconds": round(time.process_time(), 6),
        "rss_bytes": None,
    }
    try:
        import psutil

        process = psutil.Process(os.getpid())
        snapshot["cpu_percent"] = float(process.cpu_percent(interval=None))
        snapshot["rss_bytes"] = int(process.memory_info().rss)
        return snapshot
    except Exception:
        pass
    try:
        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB and macOS reports bytes.
        snapshot["rss_bytes"] = rss * 1024 if sys.platform.startswith("linux") else rss
    except Exception:
        pass
    return snapshot


def _gpu_memory_snapshot(device: str) -> Optional[Dict[str, Any]]:
    """Return device memory when torch can observe the selected CUDA device."""

    if not str(device).lower().startswith("cuda"):
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        raw_index = str(device).split(":", 1)[1] if ":" in str(device) else "0"
        index = int(raw_index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        return {
            "device_index": index,
            "free_bytes": int(free_bytes),
            "used_bytes": int(total_bytes - free_bytes),
            "total_bytes": int(total_bytes),
            "torch_allocated_bytes": int(torch.cuda.memory_allocated(index)),
            "torch_reserved_bytes": int(torch.cuda.memory_reserved(index)),
        }
    except Exception:
        return None


class ServiceRequest(BaseModel):
    """Canonical request shared by every local model service."""

    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(min_length=1)
    request_id: str = Field(min_length=1)
    audio_id: str = Field(min_length=1)
    audio_path: str = Field(min_length=1)
    input_fingerprint: str = Field(min_length=1)
    record: Dict[str, Any]

    @field_validator(
        "job_id", "request_id", "audio_id", "audio_path", "input_fingerprint"
    )
    @classmethod
    def _non_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class ServiceResponse(BaseModel):
    """Stable response envelope used by pipeline service clients."""

    job_id: str
    request_id: str
    audio_id: str
    input_fingerprint: str
    status: str
    record: Dict[str, Any]
    stage: str
    model_fingerprint: str
    elapsed_seconds: float
    error: Optional[str] = None


class ServiceError(RuntimeError):
    """An expected service error with an HTTP status for API translation."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.status_code = int(status_code)


@dataclass(frozen=True)
class BatchItemResult:
    """Optional per-item failure returned by a batch processor."""

    record: Mapping[str, Any]
    error: Optional[BaseException] = None


@dataclass
class _WorkItem:
    request: ServiceRequest
    request_digest: str
    future: asyncio.Future[ServiceResponse]
    submitted_at: float


Loader = Callable[[], Any]
ProcessBatch = Callable[
    [Any, Sequence[ServiceRequest]],
    Sequence[Mapping[str, Any] | BatchItemResult | BaseException]
    | Awaitable[Sequence[Mapping[str, Any] | BatchItemResult | BaseException]],
]


class DynamicBatchService:
    """Load one model and serve it through one bounded dynamic batch worker."""

    def __init__(
        self,
        loader: Loader,
        process_batch: ProcessBatch,
        stage: str,
        device: str,
        model_fingerprint: str,
        max_batch_size: int,
        max_wait_ms: int,
        queue_size: int,
    ) -> None:
        if not stage.strip() or not device.strip() or not model_fingerprint.strip():
            raise ValueError("stage, device and model_fingerprint must not be blank")
        if max_batch_size <= 0 or max_wait_ms < 0 or queue_size <= 0:
            raise ValueError(
                "max_batch_size/queue_size must be positive and max_wait_ms non-negative"
            )
        self.loader = loader
        self.process_batch = process_batch
        self.stage = stage
        self.device = device
        self.model_fingerprint = model_fingerprint
        self.max_batch_size = int(max_batch_size)
        self.max_wait_ms = int(max_wait_ms)
        self.queue_size = int(queue_size)

        self._queue: asyncio.Queue[_WorkItem] = asyncio.Queue(maxsize=self.queue_size)
        self._model: Any = None
        self._worker: Optional[asyncio.Task[None]] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._lock = asyncio.Lock()
        self._state = "created"
        self._process_started_at = _process_started_at()
        self._ready_monotonic: Optional[float] = None
        self._model_loaded_at: Optional[float] = None
        self._load_duration: Optional[float] = None
        self._inflight: Dict[str, tuple[str, asyncio.Future[ServiceResponse]]] = {}
        self._completed: OrderedDict[str, tuple[str, ServiceResponse]] = OrderedDict()
        self._completed_limit = max(64, self.queue_size * 4)

    async def _run_sync(self, function: Callable[..., Any], *args: Any) -> Any:
        loop = asyncio.get_running_loop()
        if self._executor is None:
            raise RuntimeError("service executor is not running")
        return await loop.run_in_executor(self._executor, function, *args)

    async def start(self) -> None:
        """Load the model exactly once, then start the queue consumer."""

        async with self._lock:
            if self._state == "running":
                return
            if self._state != "created":
                raise RuntimeError(f"cannot start service from state={self._state}")
            self._state = "starting"
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"{self.stage}-model",
            )
        load_started = time.perf_counter()
        try:
            if inspect.iscoroutinefunction(self.loader):
                model = await self.loader()
            else:
                model = await self._run_sync(self.loader)
                if inspect.isawaitable(model):
                    model = await model
        except BaseException:
            async with self._lock:
                self._state = "failed"
            if self._executor is not None:
                self._executor.shutdown(wait=False, cancel_futures=True)
                self._executor = None
            raise
        async with self._lock:
            self._model = model
            self._model_loaded_at = time.time()
            self._load_duration = round(
                max(0.0, time.perf_counter() - load_started), 6
            )
            self._ready_monotonic = time.monotonic()
            self._state = "running"
            self._worker = asyncio.create_task(
                self._batch_worker(), name=f"{self.stage}-dynamic-batcher"
            )

    async def stop(self) -> None:
        """Stop accepting work, fail pending requests and release the model."""

        async with self._lock:
            if self._state in {"created", "stopped"}:
                self._state = "stopped"
                return
            self._state = "stopping"
            worker = self._worker
            self._worker = None
        if worker is not None:
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass

        shutdown_error = ServiceError("service is shutting down", 503)
        async with self._lock:
            while not self._queue.empty():
                item = self._queue.get_nowait()
                if not item.future.done():
                    item.future.set_exception(shutdown_error)
                self._queue.task_done()
            for _, future in self._inflight.values():
                if not future.done():
                    future.set_exception(shutdown_error)
            self._inflight.clear()

        model = self._model
        self._model = None
        cleanup = getattr(model, "cleanup", None)
        if callable(cleanup) and self._executor is not None:
            try:
                await self._run_sync(cleanup)
            except Exception:
                pass
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        async with self._lock:
            self._state = "stopped"

    @staticmethod
    def _request_digest(request: ServiceRequest) -> str:
        payload = request.model_dump(mode="json")
        encoded = json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    async def submit(self, request: ServiceRequest) -> ServiceResponse:
        """Submit once, or join the same request_id already in flight/cached."""

        digest = self._request_digest(request)
        future: asyncio.Future[ServiceResponse]
        async with self._lock:
            if self._state != "running":
                raise ServiceError(f"service is not ready (state={self._state})", 503)
            cached = self._completed.get(request.request_id)
            if cached is not None:
                cached_digest, response = cached
                if cached_digest != digest:
                    raise ServiceError(
                        "request_id was already used with a different payload", 409
                    )
                self._completed.move_to_end(request.request_id)
                return copy.deepcopy(response)
            current = self._inflight.get(request.request_id)
            if current is not None:
                current_digest, future = current
                if current_digest != digest:
                    raise ServiceError(
                        "request_id is in flight with a different payload", 409
                    )
            else:
                if self._queue.full():
                    raise ServiceError("inference queue is full", 429)
                future = asyncio.get_running_loop().create_future()
                self._inflight[request.request_id] = (digest, future)
                self._queue.put_nowait(
                    _WorkItem(
                        request=request,
                        request_digest=digest,
                        future=future,
                        submitted_at=time.perf_counter(),
                    )
                )
        return await asyncio.shield(future)

    async def _call_process_batch(
        self, requests: Sequence[ServiceRequest]
    ) -> Sequence[Mapping[str, Any] | BatchItemResult | BaseException]:
        if inspect.iscoroutinefunction(self.process_batch):
            result = await self.process_batch(self._model, requests)
        else:
            result = await self._run_sync(self.process_batch, self._model, requests)
            if inspect.isawaitable(result):
                result = await result
        if isinstance(result, (str, bytes, Mapping)) or not isinstance(result, Sequence):
            raise TypeError("process_batch must return a sequence")
        if len(result) != len(requests):
            raise RuntimeError(
                "process_batch changed batch cardinality: "
                f"expected={len(requests)} actual={len(result)}"
            )
        return result

    @staticmethod
    def _error_text(error: BaseException) -> str:
        return f"{type(error).__name__}: {error}"

    async def _complete_item(
        self,
        item: _WorkItem,
        value: Mapping[str, Any] | BatchItemResult | BaseException,
    ) -> None:
        error: Optional[BaseException] = None
        if isinstance(value, BatchItemResult):
            record = dict(value.record)
            error = value.error
        elif isinstance(value, BaseException):
            record = dict(item.request.record)
            error = value
        elif isinstance(value, Mapping):
            record = dict(value)
        else:
            record = dict(item.request.record)
            error = TypeError(
                f"unsupported process_batch result type: {type(value).__name__}"
            )
        response = ServiceResponse(
            job_id=item.request.job_id,
            request_id=item.request.request_id,
            audio_id=item.request.audio_id,
            input_fingerprint=item.request.input_fingerprint,
            status="error" if error is not None else "ok",
            record=record,
            stage=self.stage,
            model_fingerprint=self.model_fingerprint,
            elapsed_seconds=round(
                max(0.0, time.perf_counter() - item.submitted_at), 6
            ),
            error=self._error_text(error) if error is not None else None,
        )
        async with self._lock:
            self._inflight.pop(item.request.request_id, None)
            self._completed[item.request.request_id] = (
                item.request_digest,
                response,
            )
            self._completed.move_to_end(item.request.request_id)
            while len(self._completed) > self._completed_limit:
                self._completed.popitem(last=False)
        if not item.future.done():
            item.future.set_result(response)

    async def _process_items(self, items: Sequence[_WorkItem]) -> None:
        try:
            results = await self._call_process_batch(
                [item.request for item in items]
            )
        except Exception as error:
            results = [error] * len(items)
        for item, result in zip(items, results):
            await self._complete_item(item, result)
            self._queue.task_done()

    async def _batch_worker(self) -> None:
        while True:
            first = await self._queue.get()
            items = [first]
            deadline = asyncio.get_running_loop().time() + self.max_wait_ms / 1000.0
            while len(items) < self.max_batch_size:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0.0:
                    break
                try:
                    items.append(
                        await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    )
                except asyncio.TimeoutError:
                    break
            await self._process_items(items)

    def health_snapshot(self) -> Dict[str, Any]:
        ready_monotonic = self._ready_monotonic
        snapshot: Dict[str, Any] = {
            "status": "ok" if self._state == "running" else self._state,
            "stage": self.stage,
            "device": self.device,
            "model_fingerprint": self.model_fingerprint,
            "process_started_at": self._process_started_at,
            "model_loaded_at": self._model_loaded_at,
            "load_duration": self._load_duration,
            "pid": os.getpid(),
            "queue_depth": self._queue.qsize(),
            "queue_capacity": self.queue_size,
            "in_flight": len(self._inflight),
            "max_batch_size": self.max_batch_size,
            "max_wait_ms": self.max_wait_ms,
            "uptime_seconds": (
                0.0
                if ready_monotonic is None
                else round(max(0.0, time.monotonic() - ready_monotonic), 3)
            ),
        }
        try:
            snapshot.update(_process_resource_snapshot())
        except Exception:
            snapshot.update(
                {"cpu_percent": None, "cpu_seconds": None, "rss_bytes": None}
            )
        try:
            snapshot["gpu_memory"] = _gpu_memory_snapshot(self.device)
        except Exception:
            snapshot["gpu_memory"] = None
        model_health = getattr(self._model, "health_metadata", None)
        if callable(model_health):
            try:
                metadata = model_health()
                if isinstance(metadata, Mapping):
                    for key, value in metadata.items():
                        snapshot.setdefault(str(key), value)
            except Exception:
                pass
        return snapshot


def create_service_app(service: DynamicBatchService) -> FastAPI:
    """Create the common health/inference FastAPI application."""

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await service.start()
        try:
            yield
        finally:
            await service.stop()

    app = FastAPI(title=f"{service.stage} inference service", lifespan=lifespan)

    @app.get("/healthz")
    async def healthz() -> Dict[str, Any]:
        snapshot = service.health_snapshot()
        if snapshot["status"] != "ok":
            raise HTTPException(status_code=503, detail=snapshot)
        return snapshot

    @app.post("/v1/infer", response_model=ServiceResponse)
    async def infer(request: ServiceRequest) -> ServiceResponse:
        try:
            return await service.submit(request)
        except ServiceError as error:
            raise HTTPException(
                status_code=error.status_code, detail=str(error)
            ) from error

    return app


def run_service(app: FastAPI, host: str, port: int) -> None:
    """Run one local inference service with uvicorn."""

    import uvicorn

    uvicorn.run(app, host=host, port=int(port), log_level="info")

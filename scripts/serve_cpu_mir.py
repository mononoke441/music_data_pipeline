#!/usr/bin/env python3
"""Persistent CPU MIR service for Chordino, BeatNet and global key."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from service_api import (
    BatchItemResult,
    DynamicBatchService,
    ServiceError,
    ServiceRequest,
    create_service_app,
    run_service,
)


ROOT = Path(__file__).resolve().parents[1]
MUSIC_TOOLS = ROOT / "MusicToolsPipeline"
if str(MUSIC_TOOLS) not in sys.path:
    sys.path.insert(0, str(MUSIC_TOOLS))

STAGE = "music_cpu"
MAX_REQUEST_BATCH = 4
MAX_WAIT_MS = 100
MAX_WORKERS = 4


class _CpuMirWorker:
    """One persistent model instance pinned to one dedicated worker thread."""

    def __init__(self, index: int, model_factory: Callable[[], Any]) -> None:
        self.index = int(index)
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"cpu-mir-worker-{self.index}",
        )
        self._closed = False
        try:
            self._model = self._executor.submit(model_factory).result()
        except BaseException:
            self._executor.shutdown(wait=True, cancel_futures=True)
            raise

    @property
    def ready(self) -> bool:
        return not self._closed and self._model is not None

    def submit(self, records: Sequence[Mapping[str, Any]]) -> Future[Any]:
        if not self.ready:
            raise RuntimeError(f"CPU MIR worker {self.index} is not ready")
        return self._executor.submit(self._model.generate_batch, list(records))

    def cleanup(self) -> None:
        if self._closed:
            return
        self._closed = True
        model = self._model
        self._model = None

        def cleanup_model() -> None:
            cleanup = getattr(model, "cleanup", None)
            if callable(cleanup):
                cleanup()

        try:
            self._executor.submit(cleanup_model).result()
        finally:
            self._executor.shutdown(wait=True, cancel_futures=True)


@dataclass
class CpuMirRuntime:
    workers: List[_CpuMirWorker]
    model_fingerprint: str

    def generate_batch(
        self, records: Sequence[Mapping[str, Any]]
    ) -> List[Any | BaseException]:
        """Distribute one request batch over persistent workers in parallel."""

        if not records:
            return []
        if not self.workers:
            raise RuntimeError("CPU MIR runtime has no workers")
        lane_count = min(len(self.workers), len(records))
        lane_indices: List[List[int]] = [[] for _ in range(lane_count)]
        for index in range(len(records)):
            lane_indices[index % lane_count].append(index)
        submitted = [
            (
                indices,
                self.workers[lane].submit([records[index] for index in indices]),
            )
            for lane, indices in enumerate(lane_indices)
        ]
        output: List[Any | BaseException | None] = [None] * len(records)
        for indices, future in submitted:
            try:
                values = future.result()
                if isinstance(values, (str, bytes, Mapping)) or not isinstance(
                    values, Sequence
                ):
                    raise TypeError("CPU MIR worker must return a sequence")
                if len(values) != len(indices):
                    raise RuntimeError(
                        "CPU MIR worker changed batch cardinality: "
                        f"expected={len(indices)} actual={len(values)}"
                    )
            except BaseException as error:
                for index in indices:
                    output[index] = error
            else:
                for index, value in zip(indices, values):
                    output[index] = value
        if any(value is None for value in output):
            raise RuntimeError("CPU MIR worker pool lost batch result alignment")
        return [value for value in output if value is not None]

    def health_metadata(self) -> Dict[str, int]:
        return {
            "worker_count": len(self.workers),
            "workers_ready": sum(worker.ready for worker in self.workers),
        }

    def cleanup(self) -> None:
        first_error: Optional[BaseException] = None
        for worker in self.workers:
            try:
                worker.cleanup()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


def cpu_mir_model_fingerprint() -> str:
    """Reuse the existing CPU MIR semantic provenance contract."""

    from config import Config
    from runtime_integrity import build_stage_fingerprint_payload

    config = Config()
    config.model_type = "music_cpu_pipeline"
    config.dataloader_type = "jsonl"
    payload = build_stage_fingerprint_payload(config, model_path=None)
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_cpu_mir(
    *,
    model_fingerprint: str,
    worker_count: int,
    model_factory: Optional[Callable[[], Any]] = None,
) -> CpuMirRuntime:
    """Construct each persistent MusicCpuPipelineModel exactly once."""

    if not 1 <= int(worker_count) <= MAX_WORKERS:
        raise ValueError(f"worker_count must be between 1 and {MAX_WORKERS}")
    if model_factory is None:
        from sub_models.pipeline_model import MusicCpuPipelineModel

        model_factory = lambda: MusicCpuPipelineModel(
            model_name="MusicCpuPipeline"
        )
    workers: List[_CpuMirWorker] = []
    try:
        for index in range(int(worker_count)):
            workers.append(_CpuMirWorker(index, model_factory))
    except BaseException:
        for worker in workers:
            worker.cleanup()
        raise
    return CpuMirRuntime(workers=workers, model_fingerprint=model_fingerprint)


def _source(request: ServiceRequest) -> Dict[str, Any]:
    record = dict(request.record)
    record["audio_id"] = request.audio_id
    record["audio_path"] = request.audio_path
    return record


def process_cpu_mir_batch(
    runtime: CpuMirRuntime,
    requests: Sequence[ServiceRequest],
) -> List[Mapping[str, Any] | BatchItemResult]:
    """Run one native MusicCpuPipelineModel batch without Ray or a child CLI."""

    sources = [_source(request) for request in requests]
    results = runtime.generate_batch(sources)
    output: List[Mapping[str, Any] | BatchItemResult] = []
    for request, value in zip(requests, results):
        if isinstance(value, BaseException):
            output.append(BatchItemResult(_source(request), value))
            continue
        if hasattr(value, "to_dict"):
            record = value.to_dict()
        elif isinstance(value, Mapping):
            record = dict(value)
        else:
            error = TypeError(f"unsupported CPU MIR result: {type(value).__name__}")
            output.append(BatchItemResult(_source(request), error))
            continue
        record["audio_id"] = request.audio_id
        record["audio_path"] = request.audio_path
        record["music_cpu_input_fingerprint"] = request.input_fingerprint
        model_versions = dict(record.get("model_versions") or {})
        model_versions[STAGE] = runtime.model_fingerprint
        record["model_versions"] = model_versions
        stage_status = str((record.get("stage_status") or {}).get(STAGE) or "")
        if stage_status != "ok":
            detail = (record.get("stage_errors") or {}).get(
                STAGE, "CPU MIR did not publish an ok stage status"
            )
            output.append(
                BatchItemResult(
                    record,
                    ServiceError(f"CPU MIR partial/error result: {detail}"),
                )
            )
        else:
            output.append(record)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("CPU_MIR_SERVICE_WORKERS", str(MAX_WORKERS))),
    )
    parser.add_argument("--queue-size", type=int, default=128)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18103)
    return parser


def build_service(args: argparse.Namespace) -> DynamicBatchService:
    if not 1 <= args.workers <= MAX_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_WORKERS}")
    fingerprint = cpu_mir_model_fingerprint()

    def loader() -> CpuMirRuntime:
        return load_cpu_mir(
            model_fingerprint=fingerprint,
            worker_count=args.workers,
        )

    return DynamicBatchService(
        loader=loader,
        process_batch=process_cpu_mir_batch,
        stage=STAGE,
        device="cpu",
        model_fingerprint=fingerprint,
        max_batch_size=MAX_REQUEST_BATCH,
        max_wait_ms=MAX_WAIT_MS,
        queue_size=args.queue_size,
    )


def main() -> None:
    args = build_parser().parse_args()
    app = create_service_app(build_service(args))
    run_service(app, args.host, args.port)


if __name__ == "__main__":
    main()

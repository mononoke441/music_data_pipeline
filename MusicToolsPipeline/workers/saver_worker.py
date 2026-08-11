# -*- coding: utf-8 -*-
"""Result saver with explicit worker completion and durable error records."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import ray
from ray.util.queue import Queue as RayQueue

from runtime_integrity import merge_results_atomically
from task_tracker import TaskTracker

logger = logging.getLogger(__name__)


def _make_serializable_record(
    result: Dict[str, Any],
    status: str,
    *,
    stage_name: str,
    stage_fingerprint: str,
) -> Tuple[Dict[str, Any], str]:
    try:
        json.dumps(result, ensure_ascii=False, allow_nan=False)
        return result, status
    except Exception as error:
        task_key = str(
            result.get("runtime_task_key") or result.get("audio_id") or "unknown"
        )
        fallback = {
            "runtime_task_key": task_key,
            "audio_id": str(result.get("audio_id") or task_key),
            "audio_path": str(result.get("audio_path") or ""),
            "error": (
                "result_serialization_failed:"
                f"{type(error).__name__}:{error}"
            ),
            "stage_status": {stage_name: "error"},
            "stage_errors": {stage_name: str(error)},
            "model_versions": {stage_name: stage_fingerprint},
        }
        return fallback, "error"


@ray.remote(num_cpus=1)
class SaveWorker:
    def __init__(
        self,
        output_path: str,
        db_path: str,
        worker_id: int = 0,
        buffer_size: int = 64,
        progress_interval: float = 2.0,
        log_path: Optional[str] = None,
        stage_name: str = "inference",
        stage_fingerprint: Optional[str] = None,
    ) -> None:
        if log_path:
            self._configure_logger(log_path)
        self.output_path = output_path
        self.db_path = db_path
        self.worker_id = int(worker_id)
        self.buffer_size = max(1, int(buffer_size))
        self.progress_interval = float(progress_interval)
        self.stage_name = str(stage_name)
        self.stage_fingerprint = str(stage_fingerprint or "unknown")
        self.jsonl_path = os.path.join(output_path, "results.jsonl")
        os.makedirs(os.path.dirname(self.jsonl_path) or ".", exist_ok=True)
        self.task_tracker = TaskTracker(db_path)
        self.write_buffer: List[Dict[str, Any]] = []
        self.task_ids_buffer: List[int] = []
        self.status_buffer: List[str] = []
        self.total_saved = 0
        self.total_failed = 0
        self.start_time: Optional[float] = None
        self.last_progress_time: Optional[float] = None

    def _configure_logger(self, log_path: str) -> None:
        absolute = os.path.abspath(log_path)
        if any(
            isinstance(handler, logging.FileHandler)
            and getattr(handler, "baseFilename", None) == absolute
            for handler in logger.handlers
        ):
            return
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    def _remove_bytes(self, value: Any) -> Any:
        if isinstance(value, bytes):
            return None
        if isinstance(value, dict):
            return {
                key: self._remove_bytes(item)
                for key, item in value.items()
                if not isinstance(item, bytes)
            }
        if isinstance(value, (list, tuple)):
            return [
                self._remove_bytes(item)
                for item in value
                if not isinstance(item, bytes)
            ]
        return value

    def _normalize_result(self, value: Any, task_id: int) -> Tuple[Dict[str, Any], str]:
        if hasattr(value, "to_dict"):
            result = value.to_dict()
        elif isinstance(value, dict):
            result = dict(value)
        else:
            result = {
                "error": f"unsupported_result_type:{type(value).__name__}",
            }
        result = self._remove_bytes(result)
        result.pop("url", None)
        result.pop("audio_type", None)
        extra = result.get("_extra")
        if isinstance(extra, dict):
            extra.pop("audio_type", None)

        stage_errors = dict(result.get("stage_errors") or {})
        top_error = result.get("error")
        if top_error:
            stage_errors[self.stage_name] = str(top_error)

        # CPU MIR models intentionally continue after an individual sub-model
        # error. Surface those nested errors at the canonical top level.
        music_cpu = result.get("music_cpu")
        if isinstance(music_cpu, dict):
            nested = {
                key: str(error)
                for key, error in music_cpu.items()
                if key.endswith("_error") and error
            }
            if nested:
                stage_errors[self.stage_name] = nested

        status = "error" if stage_errors.get(self.stage_name) else "ok"
        stage_status = dict(result.get("stage_status") or {})
        stage_status[self.stage_name] = (
            "partial_error" if status == "error" and not top_error else status
        )
        model_versions = dict(result.get("model_versions") or {})
        model_versions[self.stage_name] = self.stage_fingerprint
        result["stage_status"] = stage_status
        result["stage_errors"] = stage_errors
        result["model_versions"] = model_versions
        result["runtime_task_key"] = self.task_tracker._task_id_to_key.get(
            int(task_id), f"unknown:{task_id}"
        )
        return result, status

    def _write_to_file(
        self,
        batch_results: List[Dict[str, Any]],
        statuses: List[str],
    ) -> List[str]:
        serializable: List[Dict[str, Any]] = []
        effective_statuses: List[str] = []
        for result, status in zip(batch_results, statuses):
            record, effective_status = _make_serializable_record(
                result,
                status,
                stage_name=self.stage_name,
                stage_fingerprint=self.stage_fingerprint,
            )
            serializable.append(record)
            effective_statuses.append(effective_status)
        merge_results_atomically(
            self.jsonl_path,
            serializable,
            task_order=self.task_tracker._task_key_to_id,
        )
        return effective_statuses

    def _flush_buffer(self) -> int:
        if not self.write_buffer:
            return 0
        if not (
            len(self.write_buffer)
            == len(self.task_ids_buffer)
            == len(self.status_buffer)
        ):
            raise RuntimeError("Saver buffers lost result/task/status alignment")
        effective_statuses = self._write_to_file(
            self.write_buffer, self.status_buffer
        )
        self.status_buffer[:] = effective_statuses
        self.task_tracker.mark_tasks_finished(self.task_ids_buffer, self.status_buffer)
        count = len(self.write_buffer)
        self.total_failed += sum(status != "ok" for status in self.status_buffer)
        self.write_buffer.clear()
        self.task_ids_buffer.clear()
        self.status_buffer.clear()
        return count

    def _log_progress(self) -> None:
        elapsed = max(time.time() - (self.start_time or time.time()), 1e-6)
        stats = self.task_tracker.get_progress_stats()
        logger.info(
            "[SaveWorker %s] persisted=%s failed=%s tracker=%s speed=%.2f samples/s",
            self.worker_id,
            self.total_saved,
            self.total_failed,
            stats,
            self.total_saved / elapsed,
        )

    def run(
        self,
        result_queue: RayQueue,
        db_queue: RayQueue = None,
        total_tasks: Optional[int] = None,
        num_model_workers: int = 1,
    ) -> Dict[str, int]:
        del db_queue, total_tasks
        self.start_time = time.time()
        self.last_progress_time = self.start_time
        done_workers: set[str] = set()

        while len(done_workers) < max(1, int(num_model_workers)):
            # Completion is defined only by one explicit marker per model
            # worker. A queue timeout is neither progress nor failure and, on
            # Ray 2.56, timed async gets can disagree with qsize under this
            # multi-actor workload. The driver supervises every producer ref.
            item = result_queue.get()

            if isinstance(item, dict) and item.get("type") == "worker_done":
                done_workers.add(str(item.get("worker_id")))
                continue
            if item is None:
                # Backward compatibility for an older ModelWorker. Count each
                # explicit None once; queue timeouts never enter this branch.
                done_workers.add(f"legacy:{len(done_workers)}")
                continue

            batch_results, task_ids = item
            if len(batch_results) != len(task_ids):
                raise RuntimeError(
                    f"Model returned {len(batch_results)} results for {len(task_ids)} tasks"
                )
            for result, task_id in zip(batch_results, task_ids):
                normalized, status = self._normalize_result(result, int(task_id))
                self.write_buffer.append(normalized)
                self.task_ids_buffer.append(int(task_id))
                self.status_buffer.append(status)
            if len(self.write_buffer) >= self.buffer_size:
                self.total_saved += self._flush_buffer()
            if time.time() - (self.last_progress_time or 0.0) >= self.progress_interval:
                self.total_saved += self._flush_buffer()
                self._log_progress()
                self.last_progress_time = time.time()

        self.total_saved += self._flush_buffer()
        self._log_progress()
        return {
            "persisted": self.total_saved,
            "failed": self.total_failed,
            "done_workers": len(done_workers),
        }

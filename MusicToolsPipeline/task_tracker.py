"""Crash-safe task tracking keyed by stable sample identity.

The original tracker persisted only JSONL row numbers.  Row numbers are useful
for scheduling, but they are not identities: reordering an input file could
silently attach an old result to a different audio track.  Version 2 stores
completion records by a stable task key and keeps a sidecar mapping from the
current row/index to that key.  The sidecar can be rebuilt for a reordered
manifest without invalidating successful work.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Mapping, Optional, Set

logger = logging.getLogger(__name__)

TRACKER_SCHEMA_VERSION = 2


class TaskTracker:
    def __init__(self, record_path: str):
        self.record_path = record_path
        self.manifest_path = record_path + ".manifest.jsonl"
        os.makedirs(os.path.dirname(self.record_path) or ".", exist_ok=True)

        self._completed_keys: Set[str] = set()
        self._failed_keys: Set[str] = set()
        self._legacy_completed_ids: Set[int] = set()
        self._allocated_pending: Set[int] = set()
        self._task_id_to_key: Dict[int, str] = {}
        self._task_key_to_id: Dict[str, int] = {}
        self._next_candidate = 0
        self.total_tasks: Optional[int] = None
        self.input_fingerprint: Optional[str] = None
        self.input_fingerprint_schema: Optional[str] = None
        self.stage_fingerprint: Optional[str] = None
        self.run_fingerprint: Optional[str] = None

        self._load_records()
        self._load_manifest()
        self._rebuild_cursor()

    @staticmethod
    def make_run_fingerprint(input_fingerprint: str) -> str:
        return hashlib.sha256(input_fingerprint.encode("utf-8")).hexdigest()

    def _load_records(self) -> None:
        if not os.path.exists(self.record_path):
            return
        with open(self.record_path, encoding="utf-8") as handle:
            for raw_line in handle:
                try:
                    entry = json.loads(raw_line)
                except Exception:
                    continue
                if not isinstance(entry, dict):
                    continue
                if entry.get("type") == "run_meta":
                    self.total_tasks = int(entry.get("total_tasks") or 0)
                    self.input_fingerprint = (
                        str(entry.get("input_fingerprint") or "") or None
                    )
                    self.input_fingerprint_schema = (
                        str(entry.get("input_fingerprint_schema") or "") or None
                    )
                    self.stage_fingerprint = (
                        str(entry.get("stage_fingerprint") or "") or None
                    )
                    self.run_fingerprint = (
                        str(entry.get("run_fingerprint") or "") or None
                    )
                    continue
                if entry.get("type") == "meta":
                    # Legacy metadata is intentionally not enough to authorize a
                    # resume. configure_run() rejects legacy completion records.
                    if entry.get("total_tasks") is not None:
                        self.total_tasks = int(entry["total_tasks"])
                    continue
                task_key = entry.get("task_key")
                if task_key:
                    status = str(entry.get("status") or "ok")
                    key = str(task_key)
                    if status == "ok":
                        self._completed_keys.add(key)
                        self._failed_keys.discard(key)
                    else:
                        self._failed_keys.add(key)
                        self._completed_keys.discard(key)
                    continue
                if entry.get("task_id") is not None:
                    self._legacy_completed_ids.add(int(entry["task_id"]))

    def _load_manifest(self) -> None:
        if not os.path.exists(self.manifest_path):
            return
        manifest_run: Optional[str] = None
        mapping: Dict[int, str] = {}
        with open(self.manifest_path, encoding="utf-8") as handle:
            for raw_line in handle:
                try:
                    entry = json.loads(raw_line)
                except Exception:
                    continue
                if entry.get("type") == "manifest_meta":
                    manifest_run = str(entry.get("run_fingerprint") or "") or None
                elif entry.get("task_id") is not None and entry.get("task_key"):
                    mapping[int(entry["task_id"])] = str(entry["task_key"])
        if (
            manifest_run
            and self.run_fingerprint
            and manifest_run != self.run_fingerprint
        ):
            raise RuntimeError(
                f"Task manifest version mismatch for {self.record_path}; "
                "refusing an unsafe row-number resume"
            )
        self._set_task_map(mapping)

    def _append_records(self, records: List[Dict[str, Any]]) -> None:
        if not records:
            return
        with open(self.record_path, "a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _set_task_map(self, task_map: Mapping[int, str]) -> None:
        normalized = {
            int(task_id): str(task_key) for task_id, task_key in task_map.items()
        }
        if len(set(normalized.values())) != len(normalized):
            raise ValueError("Task keys must be unique within one input manifest")
        self._task_id_to_key = normalized
        self._task_key_to_id = {key: task_id for task_id, key in normalized.items()}
        if normalized:
            self.total_tasks = len(normalized)

    def _write_manifest(self) -> None:
        tmp_path = self.manifest_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "type": "manifest_meta",
                        "schema_version": TRACKER_SCHEMA_VERSION,
                        "run_fingerprint": self.run_fingerprint,
                        "total_tasks": self.total_tasks,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            for task_id, task_key in sorted(self._task_id_to_key.items()):
                handle.write(
                    json.dumps(
                        {"task_id": task_id, "task_key": task_key},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, self.manifest_path)

    def configure_run(
        self,
        *,
        input_fingerprint: str,
        stage_fingerprint: str,
        task_map: Mapping[int, str],
        input_fingerprint_schema: Optional[str] = None,
    ) -> str:
        """Bind this output directory to one stable input set.

        Reordering is safe because ``input_fingerprint`` is order-independent
        and completed keys are remapped to the current indices. Stage/model
        fingerprints are recorded as provenance but never invalidate completed
        task keys.
        """
        expected = self.make_run_fingerprint(input_fingerprint)
        if self.input_fingerprint and self.input_fingerprint != input_fingerprint:
            raise RuntimeError(
                "Existing progress belongs to a different input fingerprint. "
                "Use a new output directory (or explicitly remove the old output) instead "
                "of attaching cached results to different inputs."
            )
        if self._legacy_completed_ids and not self.run_fingerprint:
            raise RuntimeError(
                "Legacy row-number progress cannot be resumed safely. Use a new output "
                "directory so audio_id-based tracking can be initialized."
            )

        metadata_changed = (
            self.run_fingerprint != expected
            or self.stage_fingerprint != str(stage_fingerprint)
            or self.input_fingerprint_schema
            != (str(input_fingerprint_schema) if input_fingerprint_schema else None)
        )
        self.input_fingerprint = str(input_fingerprint)
        self.input_fingerprint_schema = (
            str(input_fingerprint_schema) if input_fingerprint_schema else None
        )
        self.stage_fingerprint = str(stage_fingerprint)
        self.run_fingerprint = expected
        self._set_task_map(task_map)
        if (
            not os.path.exists(self.record_path)
            or os.path.getsize(self.record_path) == 0
            or metadata_changed
        ):
            self._append_records(
                [
                    {
                        "type": "run_meta",
                        "schema_version": TRACKER_SCHEMA_VERSION,
                        "total_tasks": self.total_tasks,
                        "input_fingerprint": self.input_fingerprint,
                        "input_fingerprint_schema": self.input_fingerprint_schema,
                        "stage_fingerprint": self.stage_fingerprint,
                        "run_fingerprint": self.run_fingerprint,
                        "timestamp": time.time(),
                    }
                ]
            )

        self._write_manifest()
        self._rebuild_cursor()
        return expected

    def migrate_semantic_input_fingerprint(
        self,
        *,
        input_fingerprint: str,
        task_map: Mapping[int, str],
    ) -> str:
        """Rebind an unchanged stable task set to a narrower semantic fingerprint."""

        if not self.run_fingerprint or not self._task_id_to_key:
            raise RuntimeError("Cannot migrate an unversioned task tracker")
        current_keys = set(self._task_id_to_key.values())
        new_keys = {str(task_key) for task_key in task_map.values()}
        if current_keys != new_keys:
            raise RuntimeError(
                "Cannot migrate input fingerprint because the stable task set changed"
            )
        expected = self.make_run_fingerprint(str(input_fingerprint))
        self.input_fingerprint = str(input_fingerprint)
        self.input_fingerprint_schema = "cpu_mir_semantic_v1"
        self.run_fingerprint = expected
        self._set_task_map(task_map)
        self._append_records(
            [
                {
                    "type": "run_meta",
                    "schema_version": TRACKER_SCHEMA_VERSION,
                    "total_tasks": self.total_tasks,
                    "input_fingerprint": self.input_fingerprint,
                    "input_fingerprint_schema": self.input_fingerprint_schema,
                    "stage_fingerprint": self.stage_fingerprint,
                    "run_fingerprint": self.run_fingerprint,
                    "migration": "semantic_input_fields_v1",
                    "timestamp": time.time(),
                }
            ]
        )
        self._write_manifest()
        self._rebuild_cursor()
        return expected

    # Compatibility with the old API. New callers should configure_run().
    def init_tasks(self, total_tasks: int) -> None:
        self.total_tasks = int(total_tasks)
        if self.run_fingerprint:
            return
        self._append_records(
            [
                {
                    "type": "meta",
                    "total_tasks": self.total_tasks,
                    "timestamp": time.time(),
                }
            ]
        )

    def reset_incomplete_allocations(self) -> None:
        self._allocated_pending.clear()
        self._rebuild_cursor()

    def _rebuild_cursor(self) -> None:
        self._next_candidate = 0
        if not self._task_id_to_key:
            return
        while self._next_candidate in self._task_id_to_key:
            key = self._task_id_to_key[self._next_candidate]
            if key not in self._completed_keys:
                break
            self._next_candidate += 1

    def mark_tasks_finished(self, task_ids: List[int], statuses: List[str]) -> None:
        if len(task_ids) != len(statuses):
            raise ValueError("task_ids and statuses must have identical lengths")
        timestamp = time.time()
        records: List[Dict[str, Any]] = []
        for raw_id, raw_status in zip(task_ids, statuses):
            task_id = int(raw_id)
            task_key = self._task_id_to_key.get(task_id)
            if task_key is None:
                if self.run_fingerprint:
                    raise KeyError(
                        f"Task id {task_id} is absent from the current manifest"
                    )
                task_key = f"legacy-index:{task_id}"
            status = "ok" if str(raw_status) == "ok" else "error"
            records.append(
                {
                    "task_id": task_id,
                    "task_key": task_key,
                    "status": status,
                    "run_fingerprint": self.run_fingerprint,
                    "finished_at": timestamp,
                }
            )
            self._allocated_pending.discard(task_id)
            if status == "ok":
                self._completed_keys.add(task_key)
                self._failed_keys.discard(task_key)
            else:
                self._failed_keys.add(task_key)
                self._completed_keys.discard(task_key)
        self._append_records(records)
        self._rebuild_cursor()

    def mark_tasks_completed(self, task_ids: List[int]) -> None:
        self.mark_tasks_finished(task_ids, ["ok"] * len(task_ids))

    def get_completed_tasks(self) -> Set[int]:
        if self._task_id_to_key:
            return {
                task_id
                for task_id, task_key in self._task_id_to_key.items()
                if task_key in self._completed_keys
            }
        return set(self._legacy_completed_ids)

    def get_failed_tasks(self) -> Set[int]:
        return {
            task_id
            for task_id, task_key in self._task_id_to_key.items()
            if task_key in self._failed_keys
        }

    def get_terminal_statuses(self) -> Dict[str, str]:
        statuses: Dict[str, str] = {}
        for task_key in self._task_id_to_key.values():
            if task_key in self._completed_keys:
                statuses[task_key] = "ok"
            elif task_key in self._failed_keys:
                statuses[task_key] = "error"
        return statuses

    def mark_tasks_allocated(self, task_ids: List[int], worker_id: str = "") -> None:
        for raw_id in task_ids or []:
            task_id = int(raw_id)
            if task_id not in self.get_completed_tasks():
                self._allocated_pending.add(task_id)

    def get_unallocated_tasks(self, batch_size: int) -> List[int]:
        candidates = (
            sorted(self._task_id_to_key)
            if self._task_id_to_key
            else list(range(self.total_tasks or 0))
        )
        completed = self.get_completed_tasks()
        return [
            task_id
            for task_id in candidates
            if task_id not in completed and task_id not in self._allocated_pending
        ][: int(batch_size)]

    def get_progress_stats(self) -> Dict[str, int]:
        completed = len(self.get_completed_tasks())
        failed = len(self.get_failed_tasks())
        allocated = len(self._allocated_pending)
        total = int(self.total_tasks or 0)
        unallocated = max(total - completed - failed - allocated, 0)
        return {
            "completed": completed,
            "failed": failed,
            "allocated": allocated,
            "unallocated": unallocated,
            "total": total,
        }

    def is_complete(self) -> bool:
        stats = self.get_progress_stats()
        return stats["total"] > 0 and stats["completed"] == stats["total"]

    def cleanup(self) -> None:
        pass


def init_task_tracking(db_path: str, total_tasks: int) -> None:
    TaskTracker(db_path).init_tasks(total_tasks)


def mark_tasks_completed(db_path: str, task_ids: List[int]) -> None:
    TaskTracker(db_path).mark_tasks_completed(task_ids)


def get_completed_tasks(db_path: str) -> Set[int]:
    return TaskTracker(db_path).get_completed_tasks()


def get_progress_stats(db_path: str) -> Dict[str, int]:
    return TaskTracker(db_path).get_progress_stats()

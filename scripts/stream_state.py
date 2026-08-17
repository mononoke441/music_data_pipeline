#!/usr/bin/env python3
"""Durable per-item, per-stage state for the streaming pipeline."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


SCHEMA_VERSION = 2
TERMINAL_STAGE_STATUSES = {"succeeded", "failed", "skipped"}


def canonical_fingerprint(stage: str, record: Mapping[str, Any]) -> str:
    payload = {"stage": str(stage), "record": dict(record)}
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def deterministic_request_id(
    job_id: str, audio_id: str, stage: str, input_fingerprint: str
) -> str:
    encoded = "\0".join((job_id, audio_id, stage, input_fingerprint)).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class StageRow:
    job_id: str
    audio_id: str
    stage: str
    status: str
    input_fingerprint: str
    request_id: str
    attempt: int
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    model_fingerprint: Optional[str]
    elapsed_seconds: Optional[float]
    schema_version: int


class StreamState:
    """SQLite WAL-backed state with compare-and-set stage transitions."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            str(self.path), timeout=30.0, isolation_level=None, check_same_thread=False
        )
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._create_schema()

    def _create_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS stream_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS stream_items (
                job_id TEXT NOT NULL,
                audio_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                input_fingerprint TEXT NOT NULL,
                record_json TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                error TEXT,
                schema_version INTEGER NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (job_id, audio_id)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS stream_items_job_ordinal
                ON stream_items(job_id, ordinal);
            CREATE TABLE IF NOT EXISTS stream_stages (
                job_id TEXT NOT NULL,
                audio_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                status TEXT NOT NULL,
                input_fingerprint TEXT NOT NULL,
                request_id TEXT NOT NULL,
                attempt INTEGER NOT NULL DEFAULT 0,
                result_json TEXT,
                error TEXT,
                model_fingerprint TEXT,
                elapsed_seconds REAL,
                schema_version INTEGER NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (job_id, audio_id, stage),
                UNIQUE (request_id),
                FOREIGN KEY (job_id, audio_id)
                    REFERENCES stream_items(job_id, audio_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS stream_stages_status
                ON stream_stages(job_id, status, stage);
            """
        )
        columns = {
            str(row[1])
            for row in self._connection.execute("PRAGMA table_info(stream_stages)")
        }
        if "model_fingerprint" not in columns:
            self._connection.execute(
                "ALTER TABLE stream_stages ADD COLUMN model_fingerprint TEXT"
            )
        if "elapsed_seconds" not in columns:
            self._connection.execute(
                "ALTER TABLE stream_stages ADD COLUMN elapsed_seconds REAL"
            )
        row = self._connection.execute(
            "SELECT value FROM stream_meta WHERE key='schema_version'"
        ).fetchone()
        if row is None:
            self._connection.execute(
                "INSERT INTO stream_meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
        elif int(row["value"]) > SCHEMA_VERSION:
            raise RuntimeError(
                f"unsupported stream state schema={row['value']}; expected={SCHEMA_VERSION}"
            )
        elif int(row["value"]) < SCHEMA_VERSION:
            self._connection.execute(
                "UPDATE stream_meta SET value=? WHERE key='schema_version'",
                (str(SCHEMA_VERSION),),
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "StreamState":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def journal_mode(self) -> str:
        with self._lock:
            row = self._connection.execute("PRAGMA journal_mode").fetchone()
        return str(row[0]).lower()

    def register_items(
        self, job_id: str, records: Iterable[Mapping[str, Any]]
    ) -> None:
        now = time.time()
        values = [dict(record) for record in records]
        ids = [str(value.get("audio_id") or "").strip() for value in values]
        if any(not audio_id for audio_id in ids):
            raise ValueError("inventory record is missing audio_id")
        if len(set(ids)) != len(ids):
            duplicates = sorted({value for value in ids if ids.count(value) > 1})
            raise ValueError(f"duplicate SHA256 audio_id in input: {duplicates[:10]}")
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                next_ordinal = int(
                    self._connection.execute(
                        "SELECT COALESCE(MAX(ordinal), -1) + 1 FROM stream_items WHERE job_id=?",
                        (job_id,),
                    ).fetchone()[0]
                )
                for ordinal, (audio_id, record) in enumerate(zip(ids, values)):
                    fingerprint = canonical_fingerprint("inventory", record)
                    encoded = json.dumps(record, ensure_ascii=False, sort_keys=True)
                    existing = self._connection.execute(
                        """SELECT input_fingerprint FROM stream_items
                           WHERE job_id=? AND audio_id=?""",
                        (job_id, audio_id),
                    ).fetchone()
                    if existing is not None and existing["input_fingerprint"] != fingerprint:
                        raise ValueError(
                            f"inventory changed within job_id={job_id!r} for audio_id={audio_id}"
                        )
                    stored_ordinal = (
                        int(
                            self._connection.execute(
                                "SELECT ordinal FROM stream_items WHERE job_id=? AND audio_id=?",
                                (job_id, audio_id),
                            ).fetchone()[0]
                        )
                        if existing is not None
                        else next_ordinal + ordinal
                    )
                    self._connection.execute(
                        """INSERT INTO stream_items(
                               job_id, audio_id, ordinal, input_fingerprint, record_json,
                               status, error, schema_version, updated_at
                           ) VALUES(?, ?, ?, ?, ?, 'pending', NULL, ?, ?)
                           ON CONFLICT(job_id, audio_id) DO UPDATE SET
                               ordinal=excluded.ordinal,
                               record_json=excluded.record_json,
                               schema_version=excluded.schema_version,
                               updated_at=excluded.updated_at""",
                        (
                            job_id,
                            audio_id,
                            stored_ordinal,
                            fingerprint,
                            encoded,
                            SCHEMA_VERSION,
                            now,
                        ),
                    )
                self._connection.execute("COMMIT")
            except Exception:
                self._connection.execute("ROLLBACK")
                raise

    def recover_running(self, job_id: Optional[str] = None) -> int:
        parameters: tuple[Any, ...] = ()
        predicate = "status='running'"
        if job_id is not None:
            predicate += " AND job_id=?"
            parameters = (job_id,)
        with self._lock:
            cursor = self._connection.execute(
                f"""UPDATE stream_stages
                    SET status='pending', error=NULL, updated_at=?
                    WHERE {predicate}""",
                (time.time(), *parameters),
            )
            return int(cursor.rowcount)

    def items(self, job_id: str) -> list[Dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                """SELECT audio_id, record_json, status, error
                   FROM stream_items WHERE job_id=? ORDER BY ordinal""",
                (job_id,),
            ).fetchall()
        return [
            {
                "audio_id": row["audio_id"],
                "record": json.loads(row["record_json"]),
                "status": row["status"],
                "error": row["error"],
            }
            for row in rows
        ]

    def set_item_status(
        self, job_id: str, audio_id: str, status: str, error: Optional[str] = None
    ) -> None:
        with self._lock:
            cursor = self._connection.execute(
                """UPDATE stream_items SET status=?, error=?, updated_at=?
                   WHERE job_id=? AND audio_id=?""",
                (status, error, time.time(), job_id, audio_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown stream item job={job_id} audio_id={audio_id}")

    def prepare_stage(
        self,
        job_id: str,
        audio_id: str,
        stage: str,
        input_fingerprint: str,
    ) -> StageRow:
        request_id = deterministic_request_id(
            job_id, audio_id, stage, input_fingerprint
        )
        with self._lock:
            self._connection.execute(
                """INSERT INTO stream_stages(
                               job_id, audio_id, stage, status, input_fingerprint,
                               request_id, attempt, result_json, error,
                               model_fingerprint, elapsed_seconds, schema_version, updated_at
                   ) VALUES(?, ?, ?, 'pending', ?, ?, 0, NULL, NULL, NULL, NULL, ?, ?)
                   ON CONFLICT(job_id, audio_id, stage) DO NOTHING""",
                (
                    job_id,
                    audio_id,
                    stage,
                    input_fingerprint,
                    request_id,
                    SCHEMA_VERSION,
                    time.time(),
                ),
            )
            row = self._connection.execute(
                """SELECT * FROM stream_stages
                   WHERE job_id=? AND audio_id=? AND stage=?""",
                (job_id, audio_id, stage),
            ).fetchone()
            if row is None:
                raise RuntimeError("failed to create stream stage row")
            if row["input_fingerprint"] != input_fingerprint:
                raise ValueError(
                    f"stage input changed within job_id={job_id!r}: "
                    f"audio_id={audio_id} stage={stage}"
                )
            return self._stage_row(row)

    def claim_stage(
        self, job_id: str, audio_id: str, stage: str, input_fingerprint: str
    ) -> Optional[StageRow]:
        with self._lock:
            cursor = self._connection.execute(
                """UPDATE stream_stages
                   SET status='running', attempt=attempt+1, error=NULL, updated_at=?
                   WHERE job_id=? AND audio_id=? AND stage=?
                     AND status='pending' AND input_fingerprint=?""",
                (time.time(), job_id, audio_id, stage, input_fingerprint),
            )
            if cursor.rowcount != 1:
                return None
            row = self._connection.execute(
                """SELECT * FROM stream_stages
                   WHERE job_id=? AND audio_id=? AND stage=?""",
                (job_id, audio_id, stage),
            ).fetchone()
            return self._stage_row(row)

    def finish_stage(
        self,
        job_id: str,
        audio_id: str,
        stage: str,
        request_id: str,
        result: Mapping[str, Any],
        model_fingerprint: Optional[str] = None,
        elapsed_seconds: Optional[float] = None,
    ) -> None:
        encoded = json.dumps(dict(result), ensure_ascii=False, sort_keys=True)
        with self._lock:
            cursor = self._connection.execute(
                """UPDATE stream_stages
                   SET status='succeeded', result_json=?, error=NULL,
                       model_fingerprint=?, elapsed_seconds=?, updated_at=?
                   WHERE job_id=? AND audio_id=? AND stage=?
                     AND request_id=? AND status='running'""",
                (
                    encoded,
                    model_fingerprint,
                    elapsed_seconds,
                    time.time(),
                    job_id,
                    audio_id,
                    stage,
                    request_id,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    f"invalid successful transition: audio_id={audio_id} stage={stage}"
                )

    def fail_stage(
        self,
        job_id: str,
        audio_id: str,
        stage: str,
        request_id: str,
        error: str,
        model_fingerprint: Optional[str] = None,
        elapsed_seconds: Optional[float] = None,
    ) -> None:
        with self._lock:
            cursor = self._connection.execute(
                """UPDATE stream_stages
                   SET status='failed', result_json=NULL, error=?,
                       model_fingerprint=?, elapsed_seconds=?, updated_at=?
                   WHERE job_id=? AND audio_id=? AND stage=?
                     AND request_id=? AND status='running'""",
                (
                    error,
                    model_fingerprint,
                    elapsed_seconds,
                    time.time(),
                    job_id,
                    audio_id,
                    stage,
                    request_id,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    f"invalid failed transition: audio_id={audio_id} stage={stage}"
                )

    def stage(self, job_id: str, audio_id: str, stage: str) -> Optional[StageRow]:
        with self._lock:
            row = self._connection.execute(
                """SELECT * FROM stream_stages
                   WHERE job_id=? AND audio_id=? AND stage=?""",
                (job_id, audio_id, stage),
            ).fetchone()
        return self._stage_row(row) if row is not None else None

    @staticmethod
    def _stage_row(row: sqlite3.Row) -> StageRow:
        return StageRow(
            job_id=str(row["job_id"]),
            audio_id=str(row["audio_id"]),
            stage=str(row["stage"]),
            status=str(row["status"]),
            input_fingerprint=str(row["input_fingerprint"]),
            request_id=str(row["request_id"]),
            attempt=int(row["attempt"]),
            result=json.loads(row["result_json"]) if row["result_json"] else None,
            error=str(row["error"]) if row["error"] is not None else None,
            model_fingerprint=(
                str(row["model_fingerprint"])
                if row["model_fingerprint"] is not None
                else None
            ),
            elapsed_seconds=(
                float(row["elapsed_seconds"])
                if row["elapsed_seconds"] is not None
                else None
            ),
            schema_version=int(row["schema_version"]),
        )

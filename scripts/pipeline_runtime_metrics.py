#!/usr/bin/env python3
"""Write deterministic per-stage and whole-pipeline runtime reports."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Dict, Iterable, List


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _safe_rate(numerator: int, elapsed: float) -> float | None:
    if numerator <= 0 or elapsed <= 0:
        return None
    return round(float(numerator) / elapsed, 6)


def _safe_seconds_per_item(elapsed: float, count: int) -> float | None:
    if count <= 0:
        return None
    return round(elapsed / float(count), 6)


def format_elapsed_seconds(value: float) -> str:
    total_milliseconds = max(0, int(round(float(value) * 1000.0)))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def format_stage_summary(record: Dict[str, Any]) -> str:
    elapsed = float(record.get("elapsed_seconds") or 0.0)
    parts = [
        f"[TIME] {record.get('stage')}",
        f"elapsed={format_elapsed_seconds(elapsed)} ({elapsed:.3f}s)",
        f"processed={int(record.get('processed') or 0)}",
    ]
    seconds_per_item = record.get("seconds_per_item")
    if seconds_per_item is not None:
        parts.append(f"seconds_per_item={float(seconds_per_item):.6f}")
    items_per_second = record.get("items_per_second")
    if items_per_second is not None:
        parts.append(f"items_per_second={float(items_per_second):.6f}")
    return " ".join(parts)


def format_pipeline_summary(report: Dict[str, Any]) -> str:
    elapsed = float(report.get("elapsed_seconds") or 0.0)
    return (
        "[TIME] pipeline_total "
        f"elapsed={format_elapsed_seconds(elapsed)} ({elapsed:.3f}s) "
        f"status={report.get('status')} "
        f"completed_stages={len(report.get('completed_stages') or [])}"
    )


def stage_record(stage: str, started_at: float, processed: int) -> Dict[str, Any]:
    finished_at = time.time()
    elapsed = max(0.0, finished_at - float(started_at))
    return {
        "stage": str(stage),
        "started_at_epoch": round(float(started_at), 6),
        "finished_at_epoch": round(finished_at, 6),
        "elapsed_seconds": round(elapsed, 6),
        "processed": max(0, int(processed)),
        "seconds_per_item": _safe_seconds_per_item(elapsed, int(processed)),
        "items_per_second": _safe_rate(int(processed), elapsed),
    }


def append_stage(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    append_stage(path, record)


def load_stages(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict) or not value.get("stage"):
                raise ValueError(f"invalid stage record at {path}:{line_number}")
            records.append(value)
    return records


def load_json_objects(path: Path | None) -> List[Dict[str, Any]]:
    if path is None or not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"invalid object at {path}:{line_number}")
            records.append(value)
    return records


def final_report(
    *,
    started_at: float,
    input_count: int,
    accepted_count: int,
    review_count: int,
    rejected_count: int,
    annotation_count: int = 0,
    retry_count: int = 0,
    status: str | None = None,
    failure_stage: str | None = None,
    exit_code: int = 0,
    stage_failure_counts: Dict[str, int] | None = None,
    gpu_budget_decisions: Iterable[Dict[str, Any]] = (),
    stages: Iterable[Dict[str, Any]] = (),
) -> Dict[str, Any]:
    finished_at = time.time()
    elapsed = max(0.0, finished_at - float(started_at))
    inputs = max(0, int(input_count))
    accepted = max(0, int(accepted_count))
    retries = max(0, int(retry_count))
    resolved_status = status or ("partial_success" if retries else "success")
    if resolved_status not in {"success", "partial_success", "failed"}:
        raise ValueError(f"invalid pipeline status={resolved_status!r}")
    stage_values = list(stages)
    return {
        "status": resolved_status,
        "failure_stage": failure_stage or None,
        "exit_code": int(exit_code),
        "started_at_epoch": round(float(started_at), 6),
        "finished_at_epoch": round(finished_at, 6),
        "elapsed_seconds": round(elapsed, 6),
        "input_count": inputs,
        "accepted_count": accepted,
        "annotation_count": max(0, int(annotation_count)),
        "review_count": max(0, int(review_count)),
        "rejected_count": max(0, int(rejected_count)),
        "retry_count": retries,
        "stage_failure_counts": dict(sorted((stage_failure_counts or {}).items())),
        "gpu_budget_decisions": list(gpu_budget_decisions),
        "seconds_per_input_track": _safe_seconds_per_item(elapsed, inputs),
        "input_tracks_per_second": _safe_rate(inputs, elapsed),
        "seconds_per_accepted_track": _safe_seconds_per_item(elapsed, accepted),
        "accepted_tracks_per_second": _safe_rate(accepted, elapsed),
        "completed_stages": [str(value.get("stage")) for value in stage_values],
        "stages": stage_values,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    record_parser = subparsers.add_parser("record")
    record_parser.add_argument("--output", required=True)
    record_parser.add_argument("--stage", required=True)
    record_parser.add_argument("--started-at", type=float, required=True)
    record_parser.add_argument("--processed", type=int, default=0)
    record_parser.add_argument("--human-readable", action="store_true")

    gpu_parser = subparsers.add_parser("gpu")
    gpu_parser.add_argument("--output", required=True)
    gpu_parser.add_argument("--label", required=True)
    gpu_parser.add_argument("--decision", required=True)
    gpu_parser.add_argument("--free-memory-mib", type=float)
    gpu_parser.add_argument("--total-memory-mib", type=float)
    gpu_parser.add_argument("--selected-memory-gib", type=float)
    gpu_parser.add_argument("--details")

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--stages", required=True)
    finalize_parser.add_argument("--output", required=True)
    finalize_parser.add_argument("--started-at", type=float, required=True)
    finalize_parser.add_argument("--input-count", type=int, required=True)
    finalize_parser.add_argument("--accepted-count", type=int, required=True)
    finalize_parser.add_argument("--review-count", type=int, required=True)
    finalize_parser.add_argument("--rejected-count", type=int, required=True)
    finalize_parser.add_argument("--annotation-count", type=int, default=0)
    finalize_parser.add_argument("--retry-count", type=int, default=0)
    finalize_parser.add_argument("--status", choices=("success", "partial_success", "failed"))
    finalize_parser.add_argument("--failure-stage")
    finalize_parser.add_argument("--exit-code", type=int, default=0)
    finalize_parser.add_argument("--retry")
    finalize_parser.add_argument("--gpu-decisions")
    finalize_parser.add_argument("--human-readable", action="store_true")
    args = parser.parse_args()

    if args.command == "record":
        record = stage_record(args.stage, args.started_at, args.processed)
        append_stage(Path(args.output), record)
        if args.human_readable:
            print(format_stage_summary(record))
        else:
            print(json.dumps(record, ensure_ascii=False, sort_keys=True))
        return

    if args.command == "gpu":
        record = {
            "timestamp_epoch": round(time.time(), 6),
            "label": args.label,
            "decision": args.decision,
            "free_memory_mib": args.free_memory_mib,
            "total_memory_mib": args.total_memory_mib,
            "selected_memory_gib": args.selected_memory_gib,
            "details": args.details,
        }
        append_jsonl(Path(args.output), record)
        print(json.dumps(record, ensure_ascii=False, sort_keys=True))
        return

    retry_records = load_json_objects(Path(args.retry) if args.retry else None)
    failure_counts: Dict[str, int] = {}
    for value in retry_records:
        stage = str(value.get("failure_stage") or "unknown")
        failure_counts[stage] = failure_counts.get(stage, 0) + 1

    report = final_report(
        started_at=args.started_at,
        input_count=args.input_count,
        accepted_count=args.accepted_count,
        review_count=args.review_count,
        rejected_count=args.rejected_count,
        annotation_count=args.annotation_count,
        retry_count=max(args.retry_count, len(retry_records)),
        status=args.status,
        failure_stage=args.failure_stage,
        exit_code=args.exit_code,
        stage_failure_counts=failure_counts,
        gpu_budget_decisions=load_json_objects(
            Path(args.gpu_decisions) if args.gpu_decisions else None
        ),
        stages=load_stages(Path(args.stages)),
    )
    _atomic_json(Path(args.output), report)
    if args.human_readable:
        print(format_pipeline_summary(report))
    else:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

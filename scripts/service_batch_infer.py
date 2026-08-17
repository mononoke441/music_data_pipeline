#!/usr/bin/env python3
"""Run one stage-barrier batch through a resident model service."""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from pipeline_core import PIPELINE_VERSION, iter_jsonl, write_jsonl
from service_client import (
    InferEnvelope,
    ServiceClient,
    ServiceClientError,
    ServiceInferenceError,
)
from stream_state import canonical_fingerprint, deterministic_request_id


PARTITION_FILES = {
    "fast_gate": (
        "accepted.music.jsonl",
        "review.jsonl",
        "rejected.jsonl",
        "failures.jsonl",
    ),
    "discogs_mir": (
        "data.song.jsonl",
        "data.instrumental.jsonl",
        "review.jsonl",
        "failures.jsonl",
    ),
}
STAGE_STATUS_NAMES = {
    "fast_gate": "music_gate",
    "discogs_mir": "discogs_mir",
    "music_cpu": "music_cpu",
    "structure_raw": "structure_raw",
    "alm": "alm",
    "section_asr": "section_asr",
}


def _load_unique(paths: Iterable[Path]) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            continue
        for record in iter_jsonl(path):
            audio_id = str(record.get("audio_id") or "").strip()
            if not audio_id or audio_id in output:
                raise ValueError(f"{path}: missing or duplicate audio_id={audio_id!r}")
            output[audio_id] = dict(record)
    return output


def _same_identity(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(
        left.get(field) == right.get(field)
        for field in ("audio_id", "audio_path", "source_relpath", "duration")
    )


def _reusable(
    cached: Mapping[str, Any], source: Mapping[str, Any], stage: str, fingerprint: str
) -> bool:
    status_name = STAGE_STATUS_NAMES[stage]
    status = str((cached.get("stage_status") or {}).get(status_name) or "")
    if stage == "section_asr" and source.get("content_type") == "instrumental":
        return status == "not_run" and _same_identity(cached, source)
    if status != "ok":
        return False
    cached_fingerprint = str(cached.get("service_input_fingerprint") or "")
    if cached_fingerprint:
        return cached_fingerprint == fingerprint
    # One-time migration of pre-service caches. Stable input identity and a
    # complete successful stage are sufficient; model versions are provenance.
    return _same_identity(cached, source)


def _instrumental_asr(source: Mapping[str, Any], fingerprint: str) -> Dict[str, Any]:
    record = dict(source)
    sections = []
    for value in source.get("sections") or []:
        section = {
            "section_id": value.get("section_id"),
            "start": value.get("start"),
            "end": value.get("end"),
            "lyrics": None,
            "asr_tokens": [],
            "asr_status": "not_applicable",
            "asr_error": None,
            "alignment_error": None,
        }
        sections.append(section)
    record.update(
        {
            "sections": sections,
            "stage_status": {"section_asr": "not_run"},
            "stage_errors": {},
            "model_versions": {"section_asr": None, "forced_aligner": None},
            "service_input_fingerprint": fingerprint,
        }
    )
    return record


def _failure_record(
    source: Mapping[str, Any], stage: str, fingerprint: str, error: BaseException
) -> Dict[str, Any]:
    status_name = STAGE_STATUS_NAMES[stage]
    message = f"{type(error).__name__}: {error}"
    record = dict(source)
    statuses = dict(record.get("stage_status") or {})
    errors = dict(record.get("stage_errors") or {})
    statuses[status_name] = "error"
    errors[status_name] = message
    record.update(
        {
            "stage_status": statuses,
            "stage_errors": errors,
            "failure_stage": status_name,
            "retryable": True,
            "service_input_fingerprint": fingerprint,
            "semantic_input_fingerprint": fingerprint,
            "pipeline_version": PIPELINE_VERSION,
        }
    )
    return record


def _infer_one(
    client: ServiceClient,
    source: Mapping[str, Any],
    *,
    job_id: str,
    stage: str,
    retries: int,
) -> Dict[str, Any]:
    audio_id = str(source.get("audio_id") or "").strip()
    audio_path = str(source.get("audio_path") or "").strip()
    if not audio_id or not audio_path:
        raise ValueError("service input is missing audio_id/audio_path")
    fingerprint = canonical_fingerprint(stage, source)
    if stage == "section_asr" and source.get("content_type") == "instrumental":
        return _instrumental_asr(source, fingerprint)
    request_id = deterministic_request_id(job_id, audio_id, stage, fingerprint)
    started = time.perf_counter()
    for attempt in range(retries + 1):
        try:
            response = client.infer(
                job_id=job_id,
                request_id=request_id,
                audio_id=audio_id,
                audio_path=audio_path,
                input_fingerprint=fingerprint,
                record=source,
            )
            if isinstance(response, InferEnvelope):
                record = dict(response.record)
                server_runtime = {
                    "stage": response.stage,
                    "model_fingerprint": response.model_fingerprint,
                    "elapsed_seconds": response.elapsed_seconds,
                }
            elif isinstance(response, Mapping):
                record = dict(response)
                server_runtime = {}
            else:
                raise TypeError("service client returned an invalid response")
            record["service_input_fingerprint"] = fingerprint
            metrics = dict(record.get("service_runtime") or {})
            metrics[stage] = {
                **server_runtime,
                "client_elapsed_seconds": time.perf_counter() - started,
            }
            record["service_runtime"] = metrics
            return record
        except ServiceInferenceError:
            raise
        except ServiceClientError:
            if attempt >= retries:
                raise
            time.sleep(min(2.0**attempt * 0.25, 5.0))
    raise AssertionError("unreachable")


def _partition(stage: str, record: Mapping[str, Any]) -> str:
    status_name = STAGE_STATUS_NAMES[stage]
    if str((record.get("stage_status") or {}).get(status_name) or "") in {
        "error",
        "partial_error",
    }:
        return "failures.jsonl"
    if stage == "fast_gate":
        decision = str((record.get("music_gate") or {}).get("decision") or "")
        return {
            "music": "accepted.music.jsonl",
            "review": "review.jsonl",
            "non_music": "rejected.jsonl",
            "invalid_asset": "rejected.jsonl",
        }.get(decision, "failures.jsonl")
    if stage == "discogs_mir":
        if record.get("status") == "review":
            return "review.jsonl"
        return {
            "song": "data.song.jsonl",
            "instrumental": "data.instrumental.jsonl",
        }.get(str(record.get("content_type") or ""), "failures.jsonl")
    raise ValueError(f"stage {stage!r} does not have partitioned outputs")


def _output_paths(args: argparse.Namespace) -> list[Path]:
    if args.stage in PARTITION_FILES:
        root = Path(args.output_dir)
        return [root / name for name in PARTITION_FILES[args.stage]]
    return [Path(args.output)]


def _write_outputs(
    args: argparse.Namespace,
    sources: list[Mapping[str, Any]],
    values: Mapping[str, Mapping[str, Any]],
) -> None:
    if args.stage in PARTITION_FILES:
        root = Path(args.output_dir)
        grouped = {name: [] for name in PARTITION_FILES[args.stage]}
        for source in sources:
            record = dict(values[str(source["audio_id"])])
            grouped[_partition(args.stage, record)].append(record)
        for name, records in grouped.items():
            write_jsonl(root / name, records)
        return
    write_jsonl(
        args.output,
        (values[str(source["audio_id"])] for source in sources),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGE_STATUS_NAMES), required=True)
    parser.add_argument("--service-url", required=True)
    parser.add_argument("--input", required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output")
    destination.add_argument("--output-dir")
    parser.add_argument("--job-id")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage in PARTITION_FILES and not args.output_dir:
        parser.error(f"--stage {args.stage} requires --output-dir")
    if args.stage not in PARTITION_FILES and not args.output:
        parser.error(f"--stage {args.stage} requires --output")
    if args.concurrency <= 0 or args.timeout <= 0 or args.retries < 0:
        parser.error("concurrency/timeout must be positive and retries non-negative")

    sources = [dict(value) for value in iter_jsonl(args.input)]
    ids = [str(value.get("audio_id") or "") for value in sources]
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("service batch input has missing/duplicate audio_id")
    job_id = args.job_id or hashlib.sha256(
        f"{os.path.abspath(args.input)}\0{args.stage}".encode()
    ).hexdigest()[:24]
    cached = _load_unique(_output_paths(args)) if args.resume else {}
    values: Dict[str, Mapping[str, Any]] = {}
    pending: list[Mapping[str, Any]] = []
    for source in sources:
        audio_id = str(source["audio_id"])
        fingerprint = canonical_fingerprint(args.stage, source)
        old = cached.get(audio_id)
        if old is not None and _reusable(old, source, args.stage, fingerprint):
            values[audio_id] = old
        else:
            pending.append(source)

    # This CLI owns the retry/backoff loop so one configured retry count maps
    # to one visible attempt budget instead of multiplying client retries.
    client = ServiceClient(args.service_url, timeout=args.timeout, retries=0)
    health = client.healthz()
    failures = 0
    with ThreadPoolExecutor(max_workers=min(args.concurrency, len(pending) or 1)) as pool:
        futures: Dict[Future[Dict[str, Any]], Mapping[str, Any]] = {
            pool.submit(
                _infer_one,
                client,
                source,
                job_id=job_id,
                stage=args.stage,
                retries=args.retries,
            ): source
            for source in pending
        }
        for future in as_completed(futures):
            source = futures[future]
            audio_id = str(source["audio_id"])
            fingerprint = canonical_fingerprint(args.stage, source)
            try:
                record = future.result()
                runtime = dict(record.get("service_runtime") or {})
                runtime.setdefault(args.stage, {})["model_fingerprint"] = health.get(
                    "model_fingerprint"
                )
                record["service_runtime"] = runtime
                values[audio_id] = record
            except Exception as error:
                failures += 1
                values[audio_id] = _failure_record(
                    source, args.stage, fingerprint, error
                )
    _write_outputs(args, sources, values)
    print(
        f"[service-batch] stage={args.stage} total={len(sources)} "
        f"cached={len(sources) - len(pending)} inferred={len(pending)} "
        f"failures={failures}"
    )


if __name__ == "__main__":
    main()

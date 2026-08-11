#!/usr/bin/env python3
"""Build active manifests and the final retry partition deterministically."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from annotation_storage import (
    annotation_relative_path,
    iter_annotation_records,
)
from pipeline_core import PIPELINE_VERSION, iter_jsonl, write_jsonl


STAGE_ORDER = (
    "inventory",
    "music_gate",
    "discogs_mir",
    "music_cpu",
    "structure_raw",
    "alm",
    "structure_postprocess",
    "section_key",
    "section_caption",
    "section_asr",
    "metadata_merge",
    "annotation_path",
)
PROVENANCE_KEYS = {
    "pipeline_version",
    "stage_version",
    "stage_versions",
    "model_version",
    "model_versions",
    "runtime_metrics",
}


def load_unique(path: str | Path) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for record in iter_jsonl(path):
        audio_id = str(record.get("audio_id") or "").strip()
        if not audio_id or audio_id in output:
            raise ValueError(f"{path}: missing or duplicate audio_id={audio_id!r}")
        output[audio_id] = record
    return output


def semantic_input_fingerprint(record: Mapping[str, Any], stage: str) -> str:
    payload = {
        "stage": str(stage),
        "audio_id": str(record.get("audio_id") or ""),
        "audio_path": record.get("audio_path"),
        "source_relpath": record.get("source_relpath"),
        "duration": record.get("duration"),
        "content_type": record.get("content_type"),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _stage_status(record: Mapping[str, Any], stage: str) -> str:
    statuses = record.get("stage_status") or {}
    if stage == "structure_raw":
        return str(statuses.get(stage) or statuses.get("structure") or "error")
    return str(statuses.get(stage) or "error")


def _nonempty_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and bool(value)


def stage_error(record: Mapping[str, Any], stage: str) -> str | None:
    status = _stage_status(record, stage)
    if status != "ok":
        errors = record.get("stage_errors") or {}
        detail = errors.get(stage)
        if detail is None and stage == "structure_raw":
            detail = errors.get("structure")
        detail = detail or record.get("error") or record.get("_error") or status
        return str(detail)

    if stage == "music_cpu":
        payload = record.get("music_cpu") or record
        if not isinstance(payload, Mapping):
            return "missing_music_cpu_payload"
        missing = [name for name in ("chords", "beatnet", "key") if not payload.get(name)]
        nested = [name for name in ("chords_error", "beatnet_error", "key_error") if payload.get(name)]
        if missing or nested:
            return f"incomplete_music_cpu:missing={missing}:errors={nested}"
    elif stage == "structure_raw":
        if not isinstance(record.get("structure_raw"), list) or not record.get("structure_raw"):
            return "missing_or_empty_structure_raw"
    elif stage == "alm":
        if not str(record.get("ALM_Caption") or record.get("global_caption") or "").strip():
            return "empty_alm_caption"
    elif stage == "structure_postprocess":
        if not isinstance(record.get("sections"), list) or not record.get("sections"):
            return "missing_or_empty_sections"
    elif stage == "section_key":
        sections = record.get("sections") or []
        if not sections or any(
            section.get("status") != "ok" or not _nonempty_mapping(section.get("key"))
            for section in sections
        ):
            return "incomplete_section_key"
    elif stage == "section_caption":
        sections = record.get("sections") or []
        if not sections or any(
            section.get("status") != "ok" or not str(section.get("short_caption") or "").strip()
            for section in sections
        ):
            return "incomplete_section_caption"
    elif stage == "section_asr":
        bad = {None, "not_run", "error", "decode_error", "asr_error", "alignment_error"}
        sections = record.get("sections") or []
        if not sections or any(section.get("asr_status") in bad for section in sections):
            return "incomplete_section_asr"
    return None


def retry_record(
    source: Mapping[str, Any],
    stage: str,
    error: Any,
    stage_record: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    value = stage_record or {}
    statuses = dict(value.get("stage_status") or {})
    statuses.setdefault(stage, "error")
    errors = dict(value.get("stage_errors") or {})
    errors[stage] = errors.get(stage) or str(error)
    return {
        "audio_id": str(source.get("audio_id") or value.get("audio_id") or ""),
        "audio_path": source.get("audio_path") or value.get("audio_path"),
        "source_relpath": source.get("source_relpath") or value.get("source_relpath"),
        "failure_stage": stage,
        "retryable": True,
        "stage_status": statuses,
        "stage_errors": errors,
        "stage_versions": dict(value.get("stage_versions") or {}),
        "model_versions": dict(value.get("model_versions") or {}),
        "pipeline_version": value.get("pipeline_version") or PIPELINE_VERSION,
        "semantic_input_fingerprint": value.get("semantic_input_fingerprint")
        or semantic_input_fingerprint(source, stage),
    }


def parse_stage_spec(value: str) -> tuple[str, str]:
    stage, separator, path = value.partition("=")
    if not separator or not stage or not path:
        raise ValueError(f"invalid --stage {value!r}; expected NAME=PATH")
    if stage not in STAGE_ORDER:
        raise ValueError(f"unknown stage {stage!r}")
    return stage, path


def command_active(args: argparse.Namespace) -> None:
    base = load_unique(args.base)
    stage_values: list[tuple[str, Dict[str, Dict[str, Any]]]] = []
    expected = set(base)
    for raw in args.stage:
        stage, path = parse_stage_spec(raw)
        records = load_unique(path)
        if set(records) != expected:
            raise ValueError(
                f"{stage} terminal coverage mismatch: "
                f"missing={sorted(expected - set(records))[:10]} "
                f"extra={sorted(set(records) - expected)[:10]}"
            )
        stage_values.append((stage, records))

    active = []
    retry = []
    for audio_id, source in base.items():
        for stage, records in stage_values:
            value = records[audio_id]
            error = stage_error(value, stage)
            if error is not None:
                retry.append(retry_record(source, stage, error, value))
                break
        else:
            active.append(source)
    write_jsonl(args.output, active)
    write_jsonl(args.retry_output, retry)
    print(f"[active] base={len(base)} active={len(active)} retry={len(retry)}")


def command_filter(args: argparse.Namespace) -> None:
    manifest = load_unique(args.manifest)
    active = set(manifest)
    records = load_unique(args.input)
    missing = active - set(records)
    if missing:
        raise ValueError(f"filter input is missing active ids={sorted(missing)[:10]}")
    write_jsonl(args.output, (records[audio_id] for audio_id in manifest))


def _path_conflict_ids(records: Mapping[str, Mapping[str, Any]]) -> set[str]:
    by_path: Dict[tuple[str, ...], list[str]] = {}
    for audio_id, record in records.items():
        relative = annotation_relative_path(str(record.get("source_relpath") or ""))
        by_path.setdefault(tuple(relative.parts), []).append(audio_id)
    conflicts = {
        audio_id
        for audio_ids in by_path.values()
        if len(audio_ids) > 1
        for audio_id in audio_ids
    }
    paths = sorted(by_path)
    for index, left in enumerate(paths):
        for right in paths[index + 1 :]:
            if right[: len(left)] == left:
                conflicts.update(by_path[left])
                conflicts.update(by_path[right])
            elif right[:1] != left[:1]:
                break
    return conflicts


def command_path_conflicts(args: argparse.Namespace) -> None:
    base = load_unique(args.base)
    conflicts = _path_conflict_ids(base)
    clean = [record for audio_id, record in base.items() if audio_id not in conflicts]
    retry = [
        retry_record(base[audio_id], "annotation_path", "annotation_file_directory_path_conflict")
        for audio_id in sorted(conflicts)
    ]
    write_jsonl(args.output, clean)
    write_jsonl(args.retry_output, retry)
    print(f"[annotation-paths] clean={len(clean)} retry={len(retry)}")


def _failure_stage(record: Mapping[str, Any]) -> str:
    explicit = str(record.get("failure_stage") or "").strip()
    if explicit:
        return explicit
    statuses = record.get("stage_status") or {}
    for stage in STAGE_ORDER:
        status = statuses.get(stage)
        if status is None and stage == "structure_raw":
            status = statuses.get("structure")
        if status not in (None, "ok", "accepted", "review", "rejected"):
            return stage
    if record.get("discogs_mir_error"):
        return "discogs_mir"
    if record.get("music_gate_error") or record.get("stage_error"):
        return "music_gate"
    return "metadata_merge"


def _annotation_ids(path: str) -> set[str]:
    return {
        str(record.get("audio_id") or "")
        for _, record in iter_annotation_records(path)
    }


def command_combine_retry(args: argparse.Namespace) -> None:
    inventory = load_unique(args.inventory)
    review = load_unique(args.review)
    rejected = load_unique(args.rejected)
    annotated = _annotation_ids(args.annotations_dir)
    occupied = annotated | set(review) | set(rejected)
    retry_by_id: Dict[str, Dict[str, Any]] = {}
    for path in args.inputs:
        for value in iter_jsonl(path):
            audio_id = str(value.get("audio_id") or "").strip()
            if not audio_id or audio_id not in inventory or audio_id in occupied:
                continue
            stage = _failure_stage(value)
            error = (value.get("stage_errors") or {}).get(stage)
            error = error or value.get("stage_error") or value.get("error") or value.get("_error") or stage
            normalized = retry_record(inventory[audio_id], stage, error, value)
            previous = retry_by_id.get(audio_id)
            priority = STAGE_ORDER.index(stage) if stage in STAGE_ORDER else len(STAGE_ORDER)
            previous_stage = str(previous.get("failure_stage")) if previous else ""
            previous_priority = (
                STAGE_ORDER.index(previous_stage)
                if previous_stage in STAGE_ORDER
                else len(STAGE_ORDER)
            )
            if previous is None or priority < previous_priority:
                retry_by_id[audio_id] = normalized

    retry = set(retry_by_id)
    partitions = {
        "annotation": annotated,
        "review": set(review),
        "rejected": set(rejected),
        "retry": retry,
    }
    all_ids = set(inventory)
    for name, values in partitions.items():
        extra = values - all_ids
        if extra:
            raise ValueError(f"{name} contains ids absent from inventory={sorted(extra)[:10]}")
    memberships: Dict[str, list[str]] = {audio_id: [] for audio_id in all_ids}
    for name, values in partitions.items():
        for audio_id in values:
            memberships[audio_id].append(name)
    invalid = {audio_id: names for audio_id, names in memberships.items() if len(names) != 1}
    if invalid:
        raise ValueError(f"final partition coverage is incomplete/non-exclusive: {list(invalid.items())[:10]}")
    write_jsonl(args.output, (retry_by_id[audio_id] for audio_id in sorted(retry_by_id)))
    print(
        f"[partitions] annotation={len(annotated)} review={len(review)} "
        f"rejected={len(rejected)} retry={len(retry)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    active = commands.add_parser("active")
    active.add_argument("--base", required=True)
    active.add_argument("--stage", action="append", required=True)
    active.add_argument("--output", required=True)
    active.add_argument("--retry-output", required=True)
    active.set_defaults(function=command_active)

    filter_parser = commands.add_parser("filter")
    filter_parser.add_argument("--manifest", required=True)
    filter_parser.add_argument("--input", required=True)
    filter_parser.add_argument("--output", required=True)
    filter_parser.set_defaults(function=command_filter)

    conflicts = commands.add_parser("path-conflicts")
    conflicts.add_argument("--base", required=True)
    conflicts.add_argument("--output", required=True)
    conflicts.add_argument("--retry-output", required=True)
    conflicts.set_defaults(function=command_path_conflicts)

    combine = commands.add_parser("combine-retry")
    combine.add_argument("--inventory", required=True)
    combine.add_argument("--review", required=True)
    combine.add_argument("--rejected", required=True)
    combine.add_argument("--annotations-dir", required=True)
    combine.add_argument("--inputs", nargs="+", required=True)
    combine.add_argument("--output", required=True)
    combine.set_defaults(function=command_combine_retry)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fail closed when final pipeline metadata is incomplete or internally stale."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Set

from annotation_storage import (
    annotation_relative_path,
    iter_annotation_records,
    normalize_source_relpath,
)
from pipeline_core import ANNOTATION_SCHEMA_VERSION, PIPELINE_VERSION, iter_jsonl
from pipeline_progress import pipeline_tqdm


REQUIRED_TOP_LEVEL = {
    "audio_id", "audio_path", "source_relpath", "duration", "status", "content_type",
    "global_caption", "global_mir", "raw_structure", "full_transcript",
    "sections", "stage_status", "stage_errors", "model_versions",
    "pipeline_version", "annotation_schema_version",
}
STAGES = (
    "music_gate", "discogs_mir", "alm", "music_cpu", "structure_raw", "structure_postprocess",
    "section_asr",
)
REMOVED_SECTION_FIELDS = {
    "key", "key_status", "key_error", "short_caption", "caption_status", "caption_error",
}


def load_unique(paths: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        for record in iter_jsonl(path):
            audio_id = str(record.get("audio_id", "")).strip()
            if not audio_id or audio_id in output:
                raise ValueError(f"{path}: missing or duplicate audio_id={audio_id!r}")
            output[audio_id] = record
    return output


def load_annotation_directory(path: str) -> Dict[str, Dict[str, Any]]:
    root = Path(path).expanduser().resolve()
    output: Dict[str, Dict[str, Any]] = {}
    seen_source_paths: Set[str] = set()
    for annotation_path, record in iter_annotation_records(root):
        audio_id = str(record.get("audio_id", "")).strip()
        if not audio_id or audio_id in output:
            raise ValueError(f"{annotation_path}: missing or duplicate audio_id={audio_id!r}")
        source_relpath = normalize_source_relpath(str(record.get("source_relpath") or ""))
        if source_relpath in seen_source_paths:
            raise ValueError(f"duplicate source_relpath={source_relpath!r}")
        expected = annotation_relative_path(source_relpath).as_posix()
        actual = annotation_path.relative_to(root).as_posix()
        if actual != expected:
            raise ValueError(
                f"annotation path mismatch for source_relpath={source_relpath!r}: "
                f"actual={actual!r} expected={expected!r}"
            )
        seen_source_paths.add(source_relpath)
        output[audio_id] = record
    return output


def validate_sections(record: Mapping[str, Any], asr: bool) -> None:
    audio_id = record["audio_id"]
    duration = float(record["duration"])
    sections = sorted(record.get("sections") or [], key=lambda value: float(value["start"]))
    if not sections:
        raise ValueError(f"audio_id={audio_id} has no sections")
    previous = 0.0
    seen: Set[str] = set()
    for section in sections:
        section_id = str(section.get("section_id", "")).strip()
        if not section_id or section_id in seen:
            raise ValueError(f"audio_id={audio_id} invalid/duplicate section_id={section_id!r}")
        seen.add(section_id)
        start, end = float(section["start"]), float(section["end"])
        if abs(start - previous) > 0.01 or end <= start:
            raise ValueError(f"audio_id={audio_id} gap/overlap/invalid bounds at {section_id}")
        previous = end
        removed = REMOVED_SECTION_FIELDS & set(section)
        if removed:
            raise ValueError(
                f"audio_id={audio_id} section={section_id} contains removed fields={sorted(removed)}"
            )
        if record.get("content_type") == "instrumental":
            if section.get("asr_status") != "not_applicable" or section.get("lyrics") is not None:
                raise ValueError(f"audio_id={audio_id} instrumental section has ASR payload")
        elif not asr and section.get("asr_status") != "not_run":
            raise ValueError(f"audio_id={audio_id} disabled ASR is not marked not_run")
        elif asr and section.get("asr_status") in {None, "not_run", "error", "decode_error", "asr_error", "alignment_error"}:
            raise ValueError(f"audio_id={audio_id} section={section_id} ASR is incomplete")
        if not isinstance(section.get("asr_tokens"), list):
            raise ValueError(f"audio_id={audio_id} section={section_id} asr_tokens is not a list")
    if abs(previous - duration) > 0.01:
        raise ValueError(f"audio_id={audio_id} sections end={previous}, duration={duration}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", nargs="+", required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--annotations-dir", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--rejected", required=True)
    parser.add_argument("--retry", required=True)
    parser.add_argument("--alm-enabled", action="store_true")
    parser.add_argument("--section-asr-enabled", action="store_true")
    args = parser.parse_args()

    inventory = load_unique([args.inventory])
    base = load_unique(args.base)
    annotated = load_annotation_directory(args.annotations_dir)
    review = load_unique([args.review])
    rejected = load_unique([args.rejected])
    retry = load_unique([args.retry])

    partitions = {
        "annotation": set(annotated),
        "review": set(review),
        "rejected": set(rejected),
        "retry": set(retry),
    }
    inventory_ids = set(inventory)
    membership = {audio_id: [] for audio_id in inventory_ids}
    for name, values in partitions.items():
        extra = values - inventory_ids
        if extra:
            raise ValueError(f"{name} contains ids absent from inventory={sorted(extra)[:10]}")
        for audio_id in values:
            membership[audio_id].append(name)
    invalid_partitions = {
        audio_id: names for audio_id, names in membership.items() if len(names) != 1
    }
    if invalid_partitions:
        raise ValueError(
            "inventory partition coverage is incomplete/non-exclusive: "
            f"{list(invalid_partitions.items())[:10]}"
        )

    accepted_terminal = set(annotated) | (set(retry) & set(base))
    if set(base) != accepted_terminal:
        raise ValueError(
            f"accepted coverage mismatch: missing={sorted(set(base) - accepted_terminal)[:10]} "
            f"extra={sorted(accepted_terminal - set(base))[:10]}"
        )
    for audio_id, record in pipeline_tqdm(
        annotated.items(),
        total=len(annotated),
        desc="7/7 strict validation",
        unit="track",
    ):
        source = base[audio_id]
        missing = REQUIRED_TOP_LEVEL - set(record)
        if missing:
            raise ValueError(f"audio_id={audio_id} missing final fields={sorted(missing)}")
        if record.get("pipeline_version") != PIPELINE_VERSION or record.get("status") != "accepted":
            raise ValueError(f"audio_id={audio_id} invalid pipeline/status")
        if record.get("annotation_schema_version") != ANNOTATION_SCHEMA_VERSION:
            raise ValueError(f"audio_id={audio_id} invalid annotation schema")
        if record.get("content_type") not in {"song", "instrumental"}:
            raise ValueError(f"audio_id={audio_id} invalid content_type")
        if record.get("content_type") != source.get("content_type"):
            raise ValueError(f"audio_id={audio_id} final content_type differs from base")
        if abs(float(record.get("duration") or 0.0) - float(source.get("duration") or 0.0)) > 1e-6:
            raise ValueError(f"audio_id={audio_id} final duration differs from base")
        if (record.get("music_gate") or {}).get("decision") != "music":
            raise ValueError(f"audio_id={audio_id} final record was not accepted by music gate")
        if record.get("audio_path") != source.get("audio_path"):
            raise ValueError(f"audio_id={audio_id} final audio_path differs from base")
        if normalize_source_relpath(str(record.get("source_relpath") or "")) != normalize_source_relpath(
            str(source.get("source_relpath") or "")
        ):
            raise ValueError(f"audio_id={audio_id} final source_relpath differs from base")
        expected_status = {
            "music_gate": "ok",
            "discogs_mir": "ok",
            "alm": "ok" if args.alm_enabled else "not_run",
            "music_cpu": "ok",
            "structure_raw": "ok",
            "structure_postprocess": "ok",
            "section_asr": (
                "ok"
                if args.section_asr_enabled and record.get("content_type") == "song"
                else "not_run"
            ),
        }
        if (record.get("stage_status") or {}) != expected_status:
            raise ValueError(f"audio_id={audio_id} unexpected stage_status={record.get('stage_status')}")
        errors = record.get("stage_errors") or {}
        if set(errors) != set(STAGES) or any(value is not None for value in errors.values()):
            raise ValueError(f"audio_id={audio_id} has missing/non-null stage_errors={errors}")
        versions = record.get("model_versions") or {}
        for key in ("alm", "section_asr", "forced_aligner"):
            if key not in versions:
                raise ValueError(f"audio_id={audio_id} model_versions missing {key}")
        if "section_key" in versions or "section_caption" in versions:
            raise ValueError(f"audio_id={audio_id} model_versions contains removed stages")
        if args.alm_enabled and not str(record.get("global_caption", "")).strip():
            raise ValueError(f"audio_id={audio_id} missing global caption")
        if not args.alm_enabled and record.get("global_caption") is not None:
            raise ValueError(f"audio_id={audio_id} disabled ALM produced a caption")
        global_mir = record.get("global_mir") or {}
        if not isinstance(global_mir, Mapping):
            raise ValueError(f"audio_id={audio_id} global_mir is not an object")
        for key, value in (source.get("global_mir") or {}).items():
            if global_mir.get(key) != value:
                raise ValueError(f"audio_id={audio_id} Discogs MIR field {key!r} differs from base")
        for key in ("chords", "beatnet", "key"):
            if not global_mir.get(key):
                raise ValueError(f"audio_id={audio_id} final CPU MIR field {key!r} is empty")
        if not isinstance(record.get("raw_structure"), list) or not record.get("raw_structure"):
            raise ValueError(f"audio_id={audio_id} raw_structure is missing or empty")
        validate_sections(
            record,
            args.section_asr_enabled and record.get("content_type") == "song",
        )

    required_retry_fields = {
        "audio_id", "audio_path", "failure_stage", "retryable", "stage_status",
        "stage_errors", "pipeline_version", "semantic_input_fingerprint",
    }
    for audio_id, record in retry.items():
        missing = required_retry_fields - set(record)
        if missing:
            raise ValueError(f"audio_id={audio_id} retry record missing fields={sorted(missing)}")
        if record.get("retryable") is not True:
            raise ValueError(f"audio_id={audio_id} retryable must be true")
        stage = str(record.get("failure_stage") or "")
        if not stage or (record.get("stage_status") or {}).get(stage) in (None, "ok"):
            raise ValueError(f"audio_id={audio_id} retry stage/status is invalid")
        if not (record.get("stage_errors") or {}).get(stage):
            raise ValueError(f"audio_id={audio_id} retry stage error is missing")
        source = inventory[audio_id]
        if record.get("audio_path") != source.get("audio_path"):
            raise ValueError(f"audio_id={audio_id} retry audio_path differs from inventory")

    song_count = sum(record.get("content_type") == "song" for record in annotated.values())
    instrumental_count = sum(
        record.get("content_type") == "instrumental" for record in annotated.values()
    )
    print(
        f"[validate] ok records={len(annotated)} song={song_count} "
        f"instrumental={instrumental_count} review={len(review)} "
        f"rejected={len(rejected)} retry={len(retry)}"
    )


if __name__ == "__main__":
    main()

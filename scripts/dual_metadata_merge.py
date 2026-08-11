#!/usr/bin/env python3
"""Merge all dirty-audio stages by audio_id + section_id."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from annotation_storage import normalize_source_relpath, publish_annotation_records
from pipeline_core import PIPELINE_VERSION, iter_jsonl
from pipeline_progress import pipeline_tqdm


def load_many(paths: Optional[Iterable[str]]) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for path in paths or []:
        for record in iter_jsonl(path):
            audio_id = str(record.get("audio_id", "")).strip()
            if not audio_id:
                raise ValueError(f"{path}: record is missing audio_id")
            if audio_id in output:
                raise ValueError(f"duplicate audio_id={audio_id} across {list(paths or [])}")
            output[audio_id] = record
    return output


STAGES = (
    "music_gate", "discogs_mir", "alm", "music_cpu", "structure_raw",
    "structure_postprocess", "section_key", "section_caption", "section_asr",
)


def sections_by_id(record: Mapping[str, Any], stage: str) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for section in record.get("sections") or []:
        section_id = str(section.get("section_id", "")).strip()
        if not section_id:
            raise ValueError(f"{stage}: section is missing section_id")
        if section_id in output:
            raise ValueError(f"{stage}: duplicate section_id={section_id}")
        output[section_id] = dict(section)
    return output


def merge_section_fields(target: Dict[str, Any], source: Mapping[str, Any], stage: str) -> None:
    for key, value in source.items():
        if key in {"section_id", "start", "end", "status", "error"}:
            continue
        target[key] = value
    if stage == "section_key":
        target["key_status"] = source.get("status", "ok" if source.get("key") else "error")
        target["key_error"] = source.get("error")
    elif stage == "section_caption":
        target["caption_status"] = source.get("status", "ok" if source.get("short_caption") else "error")
        target["caption_error"] = source.get("error")


def assert_exact_coverage(stage: str, values: Mapping[str, Any], expected: Set[str]) -> None:
    actual = set(values)
    if actual != expected:
        missing = sorted(expected - actual)[:10]
        extra = sorted(actual - expected)[:10]
        raise ValueError(f"{stage} coverage mismatch: missing={missing} extra={extra}")


def require_stage_ok(record: Mapping[str, Any], stage: str) -> None:
    status = (record.get("stage_status") or {}).get(stage)
    if status != "ok":
        raise ValueError(f"audio_id={record.get('audio_id')} stage {stage} status={status!r}")


def validate_sections(record: Mapping[str, Any], duration: float, stage: str) -> List[Dict[str, Any]]:
    sections = list(record.get("sections") or [])
    if not sections:
        raise ValueError(f"audio_id={record.get('audio_id')} {stage} has no sections")
    sections = sorted((dict(value) for value in sections), key=lambda value: float(value["start"]))
    seen: Set[str] = set()
    previous = 0.0
    for index, section in enumerate(sections):
        section_id = str(section.get("section_id", "")).strip()
        if not section_id or section_id in seen:
            raise ValueError(f"audio_id={record.get('audio_id')} invalid/duplicate section_id={section_id!r}")
        seen.add(section_id)
        start, end = float(section["start"]), float(section["end"])
        if end <= start:
            raise ValueError(f"audio_id={record.get('audio_id')} section {section_id} has end <= start")
        expected_start = 0.0 if index == 0 else previous
        if abs(start - expected_start) > 0.01:
            raise ValueError(
                f"audio_id={record.get('audio_id')} section coverage gap/overlap at {section_id}: "
                f"start={start} expected={expected_start}"
            )
        previous = end
    if abs(previous - duration) > 0.01:
        raise ValueError(
            f"audio_id={record.get('audio_id')} sections end at {previous}, duration={duration}"
        )
    return sections


def validate_matching_sections(
    audio_id: str,
    reference: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
    stage: str,
) -> None:
    values = sections_by_id(candidate, stage)
    expected = {str(section["section_id"]): section for section in reference}
    if set(values) != set(expected):
        raise ValueError(f"audio_id={audio_id} {stage} section ids do not match structure")
    for section_id, section in expected.items():
        other = values[section_id]
        for boundary in ("start", "end"):
            if abs(float(section[boundary]) - float(other.get(boundary, -1))) > 0.001:
                raise ValueError(
                    f"audio_id={audio_id} {stage} {section_id} {boundary} does not match structure"
                )


def model_versions(*records: Mapping[str, Any]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for record in records:
        output.update(record.get("model_versions") or {})
        output.update(record.get("stage_versions") or {})
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", nargs="+", required=True, help="Accepted song/instrumental manifests")
    parser.add_argument("--alm", nargs="*")
    parser.add_argument("--music-cpu", nargs="*")
    parser.add_argument("--structure-raw", nargs="*")
    parser.add_argument("--sections", nargs="+", required=True)
    parser.add_argument("--section-key", nargs="*")
    parser.add_argument("--section-caption", nargs="*")
    parser.add_argument("--section-asr", nargs="*")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alm-enabled", action="store_true")
    parser.add_argument("--section-caption-enabled", action="store_true")
    parser.add_argument("--section-asr-enabled", action="store_true")
    args = parser.parse_args()

    base = load_many(args.base)
    alm = load_many(args.alm)
    cpu = load_many(args.music_cpu)
    raw_structure = load_many(args.structure_raw)
    processed = load_many(args.sections)
    section_key = load_many(args.section_key)
    section_caption = load_many(args.section_caption)
    section_asr = load_many(args.section_asr)

    expected_ids = set(base)
    for name, values in (
        ("music_cpu", cpu),
        ("structure_raw", raw_structure),
        ("structure_postprocess", processed),
        ("section_key", section_key),
    ):
        assert_exact_coverage(name, values, expected_ids)
    optional = (
        ("alm", args.alm_enabled, alm, args.alm),
        ("section_caption", args.section_caption_enabled, section_caption, args.section_caption),
        ("section_asr", args.section_asr_enabled, section_asr, args.section_asr),
    )
    for name, enabled, values, paths in optional:
        if enabled:
            if not paths:
                raise ValueError(f"{name} is enabled but no input file was provided")
            assert_exact_coverage(name, values, expected_ids)
        elif paths:
            raise ValueError(f"{name} is disabled but input files were provided")

    annotated: List[Dict[str, Any]] = []
    for audio_id, source in pipeline_tqdm(
        base.items(), total=len(base), desc="7/7 metadata merge", unit="track"
    ):
        alm_value = alm.get(audio_id, {})
        cpu_value = cpu.get(audio_id, {})
        raw_value = raw_structure.get(audio_id, {})
        processed_value = processed.get(audio_id, {})
        key_value = section_key.get(audio_id, {})
        caption_value = section_caption.get(audio_id, {})
        asr_value = section_asr.get(audio_id, {})

        duration = float(source.get("duration") or 0.0)
        if duration <= 0:
            raise ValueError(f"audio_id={audio_id} has invalid duration={duration}")
        content_type = str(source.get("content_type", "unknown")).lower()
        if content_type not in {"song", "instrumental"}:
            raise ValueError(f"audio_id={audio_id} invalid content_type={content_type!r}")
        source_relpath = normalize_source_relpath(str(source.get("source_relpath") or ""))

        require_stage_ok(source, "music_gate")
        require_stage_ok(source, "discogs_mir")
        if (source.get("music_gate") or {}).get("decision") != "music":
            raise ValueError(f"audio_id={audio_id} was not accepted by the fast music gate")
        if not isinstance(source.get("global_mir"), Mapping):
            raise ValueError(f"audio_id={audio_id} is missing Discogs global_mir")

        cpu_features = cpu_value.get("music_cpu") or cpu_value
        missing_cpu = [name for name in ("chords", "beatnet", "key") if name not in cpu_features]
        cpu_errors = [name for name in ("chords_error", "beatnet_error", "key_error") if cpu_features.get(name)]
        if missing_cpu or cpu_errors:
            raise ValueError(
                f"audio_id={audio_id} incomplete music_cpu: missing={missing_cpu} errors={cpu_errors}"
            )
        if not isinstance(raw_value.get("structure_raw"), list) or not raw_value.get("structure_raw"):
            raise ValueError(f"audio_id={audio_id} structure_raw is missing or empty")
        require_stage_ok(processed_value, "structure_postprocess")
        require_stage_ok(key_value, "section_key")
        sections = validate_sections(processed_value, duration, "structure_postprocess")
        validate_matching_sections(audio_id, sections, key_value, "section_key")
        for section in sections_by_id(key_value, "section_key").values():
            if section.get("status") != "ok" or not isinstance(section.get("key"), Mapping):
                raise ValueError(f"audio_id={audio_id} section_key contains an incomplete section")
        if args.alm_enabled:
            require_stage_ok(alm_value, "alm")
            if not str(alm_value.get("ALM_Caption", "")).strip() or alm_value.get("_error"):
                raise ValueError(f"audio_id={audio_id} ALM caption is empty or failed")
        if args.section_caption_enabled:
            require_stage_ok(caption_value, "section_caption")
            validate_matching_sections(audio_id, sections, caption_value, "section_caption")
            for section in sections_by_id(caption_value, "section_caption").values():
                if section.get("status") != "ok" or not str(section.get("short_caption", "")).strip():
                    raise ValueError(f"audio_id={audio_id} section_caption contains an incomplete section")
        if args.section_asr_enabled:
            require_stage_ok(asr_value, "section_asr")
            validate_matching_sections(audio_id, sections, asr_value, "section_asr")
            bad_asr = {None, "not_run", "error", "decode_error", "asr_error", "alignment_error"}
            for section in sections_by_id(asr_value, "section_asr").values():
                if section.get("asr_status") in bad_asr:
                    raise ValueError(f"audio_id={audio_id} section_asr contains an incomplete section")

        global_mir = dict(source.get("global_mir") or {})
        for name in ("chords", "beatnet", "key"):
            if name in cpu_features:
                global_mir[name] = cpu_features[name]

        section_map = sections_by_id(processed_value, "structure_postprocess")
        for stage, extra in (
            ("section_key", key_value),
            ("section_caption", caption_value if args.section_caption_enabled else {}),
            ("section_asr", asr_value if args.section_asr_enabled else {}),
        ):
            for section_id, values in sections_by_id(extra, stage).items():
                if section_id in section_map:
                    merge_section_fields(section_map[section_id], values, stage)

        sections = sorted(section_map.values(), key=lambda item: float(item.get("start", 0.0)))
        for section in sections:
            section.setdefault("key", None)
            section.setdefault("key_status", "error")
            section.setdefault("key_error", None)
            section.setdefault("short_caption", None)
            section.setdefault("caption_status", "not_run" if not args.section_caption_enabled else "error")
            section.setdefault("caption_error", None)
            section.setdefault("lyrics", None)
            section.setdefault("asr_tokens", [])
            if content_type == "instrumental":
                section["lyrics"] = None
                section["asr_tokens"] = []
                section["asr_status"] = "not_applicable"
            else:
                section.setdefault("asr_status", "not_run" if not args.section_asr_enabled else "error")
            section.setdefault("asr_error", None)
            section.setdefault("alignment_error", None)

        statuses = {
            "music_gate": "ok",
            "discogs_mir": "ok",
            "alm": "ok" if args.alm_enabled else "not_run",
            "music_cpu": "ok",
            "structure_raw": "ok",
            "structure_postprocess": "ok",
            "section_key": "ok",
            "section_caption": "ok" if args.section_caption_enabled else "not_run",
            "section_asr": "ok" if args.section_asr_enabled else "not_run",
        }
        versions = model_versions(
            source, alm_value, cpu_value, raw_value, processed_value,
            key_value, caption_value, asr_value,
        )
        for name in ("alm", "section_caption", "section_asr", "forced_aligner"):
            versions.setdefault(name, None)

        value = {
            "audio_id": audio_id,
            "audio_path": source.get("audio_path"),
            "source_relpath": source_relpath,
            "duration": duration,
            "status": "accepted",
            "content_type": content_type,
            "content_confidence": source.get("content_confidence"),
            "music_gate": source.get("music_gate") or {},
            "voice_analysis": source.get("voice_analysis") or {},
            "global_caption": alm_value.get("ALM_Caption") if args.alm_enabled else None,
            "global_mir": global_mir,
            "raw_structure": raw_value.get("structure_raw") or source.get("structure_raw"),
            "full_transcript": asr_value.get("full_transcript") if args.section_asr_enabled else None,
            "sections": sections,
            "stage_status": statuses,
            "stage_errors": {name: None for name in STAGES},
            "model_versions": versions,
            "pipeline_version": PIPELINE_VERSION,
        }
        annotated.append(value)

    annotations_dir = Path(args.output_dir) / "annotations"
    counts = publish_annotation_records(annotated, annotations_dir)
    print(
        f"[annotations] total={counts['total']} song={counts['song']} "
        f"instrumental={counts['instrumental']} dir={annotations_dir}"
    )


if __name__ == "__main__":
    main()

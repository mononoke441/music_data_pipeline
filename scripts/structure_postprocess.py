#!/usr/bin/env python3
"""Snap either structure decoder to downbeats and emit one row per track."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from pipeline_core import (
    PIPELINE_VERSION,
    extract_downbeats,
    iter_jsonl,
    postprocess_sections,
    write_jsonl,
)
from pipeline_progress import count_jsonl, pipeline_tqdm
from pipeline_state import semantic_input_fingerprint


def load_by_audio_id(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    output: Dict[str, Dict[str, Any]] = {}
    for record in iter_jsonl(path):
        audio_id = str(record.get("audio_id", "")).strip()
        if not audio_id:
            raise ValueError(f"{path}: record is missing audio_id")
        if audio_id in output:
            raise ValueError(f"{path}: duplicate audio_id={audio_id}")
        output[audio_id] = record
    return output


def voice_metrics(
    section: Mapping[str, Any],
    voice_analysis: Mapping[str, Any],
    threshold: float = 0.5,
) -> Dict[str, float]:
    probabilities = voice_analysis.get("probabilities") or []
    hop = float(voice_analysis.get("frame_hop_sec") or 0.0)
    if not probabilities or hop <= 0:
        return {"voice_coverage": 0.0, "voice_score": 0.0}
    begin = max(0, int(math.floor(float(section["start"]) / hop)))
    finish = min(len(probabilities), int(math.ceil(float(section["end"]) / hop)))
    values = [float(value) for value in probabilities[begin:finish]]
    if not values:
        return {"voice_coverage": 0.0, "voice_score": 0.0}
    return {
        "voice_coverage": round(
            sum(value >= threshold for value in values) / len(values), 6
        ),
        "voice_score": round(sum(values) / len(values), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL containing structure_raw")
    parser.add_argument(
        "--music-cpu", help="Optional whole-track Chordino/BeatNet/key JSONL"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--snap-tolerance", type=float, default=1.5)
    parser.add_argument("--duplicate-tolerance", type=float, default=2.0)
    parser.add_argument("--minimum-duration", type=float, default=8.0)
    parser.add_argument("--extremely-short-duration", type=float, default=2.0)
    parser.add_argument("--short-boundary-confidence", type=float, default=0.65)
    args = parser.parse_args()

    cpu_by_id = load_by_audio_id(args.music_cpu)
    output = []
    failures = 0
    for source in pipeline_tqdm(
        iter_jsonl(args.input),
        total=count_jsonl(args.input),
        desc="4/7 structure postprocess",
        unit="track",
    ):
        audio_id = str(source.get("audio_id", "")).strip()
        if not audio_id:
            raise ValueError("structure input record is missing audio_id")
        cpu = cpu_by_id.get(audio_id, {})
        joined = {**source, **cpu}
        record = {
            "audio_id": audio_id,
            "audio_path": source.get("audio_path"),
            "duration": source.get("duration"),
            "content_type": source.get("content_type"),
            "pipeline_version": PIPELINE_VERSION,
            "stage_versions": {
                "structure_postprocess": "confidence-aware-downbeat-snap-v2"
            },
            "semantic_input_fingerprint": semantic_input_fingerprint(
                source, "structure_postprocess"
            ),
        }
        try:
            raw = source.get("structure_raw") or source.get("songformer_result") or []
            sections = postprocess_sections(
                raw,
                float(source.get("duration") or cpu.get("duration") or 0.0),
                extract_downbeats(joined),
                snap_tolerance=args.snap_tolerance,
                duplicate_tolerance=args.duplicate_tolerance,
                minimum_duration=args.minimum_duration,
                extremely_short_duration=args.extremely_short_duration,
                short_boundary_confidence=args.short_boundary_confidence,
            )
            voice = source.get("voice_analysis") or {}
            for section in sections:
                section.update(voice_metrics(section, voice))
            record.update(
                {
                    "sections": sections,
                    "stage_status": {"structure_postprocess": "ok"},
                }
            )
        except Exception as error:
            failures += 1
            record.update(
                {
                    "sections": [],
                    "stage_status": {"structure_postprocess": "error"},
                    "error": f"{type(error).__name__}: {error}",
                    "stage_errors": {
                        "structure_postprocess": f"{type(error).__name__}: {error}"
                    },
                }
            )
        output.append(record)
    write_jsonl(Path(args.output), output)
    print(
        f"[structure-postprocess] tracks={len(output)} errors={failures}",
        flush=True,
    )


if __name__ == "__main__":
    main()

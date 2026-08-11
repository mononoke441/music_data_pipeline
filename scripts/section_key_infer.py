#!/usr/bin/env python3
"""Compute section key/mode by decoding only requested time ranges to memory."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import numpy as np

from pipeline_core import PIPELINE_VERSION, decode_audio_range, iter_jsonl, write_jsonl
from pipeline_progress import count_jsonl, pipeline_tqdm
from pipeline_state import semantic_input_fingerprint


SECTION_KEY_STAGE_VERSION = "essentia-keyextractor-diatonic-duration-v2"


def load_by_audio_id(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    output: Dict[str, Dict[str, Any]] = {}
    for record in iter_jsonl(path) if path else []:
        audio_id = str(record.get("audio_id") or "").strip()
        if not audio_id:
            raise ValueError(f"{path}: record is missing audio_id")
        if audio_id in output:
            raise ValueError(f"{path}: duplicate audio_id={audio_id}")
        output[audio_id] = record
    return output


def sections_hash(
    record: Mapping[str, Any], cpu_mir: Mapping[str, Any] | None = None
) -> str:
    """Bind the cache to exact IDs/boundaries/labels and chord context."""

    mir = cpu_mir or {}
    chords = (
        mir.get("chords")
        or mir.get("music_cpu", {}).get("chords")
        or mir.get("global_mir", {}).get("chords")
        or record.get("music_cpu", {}).get("chords")
        or record.get("global_mir", {}).get("chords")
        or {}
    )
    payload = {
        "audio_id": str(record.get("audio_id") or ""),
        "sections": [
            {
                "section_id": str(section.get("section_id") or ""),
                "start": round(float(section.get("start", 0.0)), 6),
                "end": round(float(section.get("end", 0.0)), 6),
                "label": section.get("label"),
            }
            for section in record.get("sections") or []
        ],
        "chords": chords,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def section_key_input_fingerprint(record: Mapping[str, Any]) -> str:
    return semantic_input_fingerprint(record, "section_key")


_NOTE_TO_PC = {
    "C": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "FB": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
    "CB": 11,
    "B#": 0,
}


def _parse_chord(chord: Any) -> Optional[tuple[int, str]]:
    """Return pitch class and triad family for Chordino/common chord labels."""

    text = str(chord or "").strip().replace("♯", "#").replace("♭", "b")
    if not text or text.upper() in {"N", "X", "NO_CHORD", "NO CHORD"}:
        return None
    text = text.split("/", 1)[0]
    match = re.match(r"^([A-Ga-g])([#b]?)(.*)$", text)
    if not match:
        return None
    note = (match.group(1).upper() + match.group(2)).upper()
    pitch_class = _NOTE_TO_PC.get(note)
    if pitch_class is None:
        return None
    suffix = match.group(3).lower().lstrip(":")
    if "dim" in suffix or "°" in suffix or suffix.startswith("o"):
        family = "diminished"
    elif "aug" in suffix or "+" in suffix:
        family = "augmented"
    elif suffix.startswith("min") or (
        suffix.startswith("m") and not suffix.startswith("maj")
    ):
        family = "minor"
    elif suffix.startswith(("sus", "5")):
        family = "neutral"
    else:
        family = "major"
    return pitch_class, family


def _diatonic_families(key: str, mode: str) -> Dict[int, set[str]]:
    tonic_name = str(key).strip().replace("♯", "#").replace("♭", "b").upper()
    tonic = _NOTE_TO_PC.get(tonic_name)
    if tonic is None:
        return {}
    if str(mode).strip().lower().startswith("min"):
        # Natural minor plus the common harmonic-minor V and vii°.
        degrees = [0, 2, 3, 5, 7, 8, 10, 11]
        families = [
            {"minor", "neutral"},
            {"diminished", "neutral"},
            {"major", "neutral"},
            {"minor", "neutral"},
            {"minor", "major", "neutral"},
            {"major", "neutral"},
            {"major", "neutral"},
            {"diminished", "neutral"},
        ]
    else:
        degrees = [0, 2, 4, 5, 7, 9, 11]
        families = [
            {"major", "neutral"},
            {"minor", "neutral"},
            {"minor", "neutral"},
            {"major", "neutral"},
            {"major", "neutral"},
            {"minor", "neutral"},
            {"diminished", "neutral"},
        ]
    return {(tonic + degree) % 12: family for degree, family in zip(degrees, families)}


def chord_key_duration_metrics(
    section: Mapping[str, Any],
    mir: Mapping[str, Any],
    key: str,
    mode: str,
) -> Dict[str, Optional[float]]:
    """Return duration-weighted diatonic and tonic-chord ratios.

    Chord intervals are formed from consecutive timestamps before clipping to
    the section.  This includes a chord which begins before the section and
    sustains across its left boundary.
    """

    chords = (
        mir.get("chords")
        or mir.get("music_cpu", {}).get("chords")
        or mir.get("global_mir", {}).get("chords")
        or {}
    )
    values = chords.get("values") or []
    events = []
    for value in values:
        try:
            timestamp = float(value.get("timestamp", value.get("time")))
        except (AttributeError, TypeError, ValueError):
            continue
        events.append((timestamp, value.get("chord", "")))
    events.sort(key=lambda item: item[0])
    allowed = _diatonic_families(key, mode)
    if not events or not allowed:
        return {
            "diatonic_chord_duration_ratio": None,
            "tonic_chord_duration_ratio": None,
        }

    tonic_name = str(key).strip().replace("♯", "#").replace("♭", "b").upper()
    tonic = _NOTE_TO_PC.get(tonic_name)

    section_start = float(section["start"])
    section_end = float(section["end"])
    mir_duration = float(
        mir.get("duration") or mir.get("global_mir", {}).get("duration") or section_end
    )
    total = 0.0
    diatonic = 0.0
    tonic_duration = 0.0
    for index, (event_start, chord) in enumerate(events):
        event_end = (
            events[index + 1][0]
            if index + 1 < len(events)
            else max(section_end, mir_duration)
        )
        overlap = max(
            0.0, min(section_end, event_end) - max(section_start, event_start)
        )
        parsed = _parse_chord(chord)
        if overlap <= 0.0 or parsed is None:
            continue
        pitch_class, family = parsed
        total += overlap
        if family in allowed.get(pitch_class, set()):
            diatonic += overlap
            if pitch_class == tonic:
                tonic_duration += overlap
    if total <= 0.0:
        return {
            "diatonic_chord_duration_ratio": None,
            "tonic_chord_duration_ratio": None,
        }
    return {
        "diatonic_chord_duration_ratio": round(diatonic / total, 6),
        "tonic_chord_duration_ratio": round(tonic_duration / total, 6),
    }


def analyze_track_sections(
    source: Mapping[str, Any],
    originals: Sequence[Mapping[str, Any]],
    cpu_mir: Mapping[str, Any],
    key_extractor: Callable[[np.ndarray], tuple[Any, Any, Any]],
    executor: concurrent.futures.ThreadPoolExecutor,
    *,
    sample_rate: int,
    decode_function: Callable[..., bytes] = decode_audio_range,
) -> tuple[list[Dict[str, Any]], int]:
    """Decode ranges concurrently while serializing the shared KeyExtractor."""

    outputs: list[Optional[Dict[str, Any]]] = [None] * len(originals)
    futures = {
        executor.submit(
            decode_function,
            str(source["audio_path"]),
            float(original["start"]),
            float(original["end"]),
            sample_rate=sample_rate,
            output_format="f32le",
        ): (index, original)
        for index, original in enumerate(originals)
    }
    ok = 0
    for future in concurrent.futures.as_completed(futures):
        index, original = futures[future]
        section: Dict[str, Any] = {
            "section_id": original["section_id"],
            "start": original["start"],
            "end": original["end"],
        }
        try:
            raw = future.result()
            waveform = np.frombuffer(raw, dtype="<f4")
            # One extractor instance stays on the main thread. Essentia is not
            # assumed thread-safe; only independent ffmpeg decodes overlap.
            key, mode, strength = key_extractor(waveform)
            chord_metrics = chord_key_duration_metrics(
                original, cpu_mir, str(key), str(mode)
            )
            section["key"] = {
                "key": str(key),
                "mode": str(mode),
                "strength": round(float(strength), 6),
                **chord_metrics,
            }
            section["status"] = "ok"
            ok += 1
        except Exception as error:
            section.update(
                {
                    "key": None,
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
        outputs[index] = section
    if any(section is None for section in outputs):  # pragma: no cover
        raise RuntimeError("section key analysis lost an output slot")
    return [section for section in outputs if section is not None], ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="data.sections.jsonl")
    parser.add_argument(
        "--music-cpu", help="Optional whole-track MIR JSONL for chord check"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--decode-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.sample_rate <= 0 or args.decode_workers <= 0:
        parser.error("sample rate and decode workers must be positive")

    cpu_by_id = load_by_audio_id(args.music_cpu)
    stage_version = SECTION_KEY_STAGE_VERSION
    existing = (
        load_by_audio_id(args.output)
        if args.resume and Path(args.output).exists()
        else {}
    )
    records = []
    key_extractor = None
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.decode_workers
    ) as executor:
        for source in pipeline_tqdm(
            iter_jsonl(args.input),
            total=count_jsonl(args.input),
            desc="5/7 section key",
            unit="track",
        ):
            audio_id = str(source.get("audio_id") or "").strip()
            if not audio_id:
                raise ValueError(f"{args.input}: input record is missing audio_id")
            cpu_mir = cpu_by_id.get(audio_id, {})
            semantic_fingerprint = section_key_input_fingerprint(source)
            try:
                originals = list(source.get("sections") or [])
                for original in originals:
                    if not isinstance(original, Mapping):
                        raise ValueError("sections must contain JSON objects")
                    if not str(original.get("section_id") or "").strip():
                        raise ValueError("section is missing section_id")
                    start = float(original["start"])
                    end = float(original["end"])
                    if end <= start:
                        raise ValueError("section end must be greater than start")
                plan_hash = sections_hash(source, cpu_mir)
            except (AttributeError, KeyError, TypeError, ValueError) as error:
                records.append(
                    {
                        "audio_id": audio_id,
                        "audio_path": source.get("audio_path"),
                        "sections": [],
                        "section_key_input_fingerprint": semantic_fingerprint,
                        "semantic_input_fingerprint": semantic_fingerprint,
                        "stage_status": {"section_key": "error"},
                        "stage_errors": {
                            "section_key": f"{type(error).__name__}: {error}"
                        },
                        "pipeline_version": PIPELINE_VERSION,
                        "model_versions": {"section_key": stage_version},
                    }
                )
                continue
            cached = existing.get(audio_id)
            if cached is not None:
                if (
                    (cached.get("stage_status") or {}).get("section_key") == "ok"
                    and cached.get("sections_hash") == plan_hash
                    and str(
                        cached.get("section_key_input_fingerprint")
                        or cached.get("semantic_input_fingerprint")
                        or ""
                    )
                    == semantic_fingerprint
                ):
                    records.append(cached)
                    continue
            if key_extractor is None and originals:
                # Essentia can transitively initialize TensorFlow and emit
                # substantial logs. Do not import it for a fully cached run.
                import essentia.standard as es

                key_extractor = es.KeyExtractor()
            sections, ok = (
                analyze_track_sections(
                    source,
                    originals,
                    cpu_mir,
                    key_extractor,
                    executor,
                    sample_rate=args.sample_rate,
                )
                if originals
                else ([], 0)
            )
            value: Dict[str, Any] = {
                "audio_id": audio_id,
                "audio_path": source.get("audio_path"),
                "sections": sections,
                "sections_hash": plan_hash,
                "section_key_input_fingerprint": semantic_fingerprint,
                "semantic_input_fingerprint": semantic_fingerprint,
                "stage_status": {
                    "section_key": "ok"
                    if sections and ok == len(sections)
                    else "error",
                },
                "pipeline_version": PIPELINE_VERSION,
                "model_versions": {"section_key": stage_version},
            }
            if value["stage_status"]["section_key"] != "ok":
                errors = [
                    f"{section.get('section_id')}: {section.get('error') or section.get('status')}"
                    for section in sections
                    if section.get("status") != "ok"
                ] or ["input contains no sections"]
                value["stage_errors"] = {"section_key": "; ".join(errors)}
            records.append(value)
    write_jsonl(Path(args.output), records)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Shared, dependency-light primitives for the dirty-audio pipeline.

This module intentionally has no model imports.  The command line stages use
the same scoring, routing, section post-processing, JSONL and audio-range
contracts, while unit tests can exercise them without downloading weights.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


PIPELINE_VERSION = "music-data-pipeline-v1"


def stable_audio_id(audio_path: str, data_root: str) -> str:
    """Return a path-independent SHA-256 id for the exact source bytes.

    Hashing the encoded source (rather than decoded PCM) is deliberately
    streaming and dependency-free.  It costs one sequential read per asset,
    but is much cheaper than decoding and gives exact duplicate detection
    across paths.  ``data_root`` remains in the signature for CLI/API
    compatibility and is intentionally not part of the digest.
    """

    del data_root
    path = Path(audio_path).expanduser().resolve()
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def iter_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def load_jsonl_with_truncated_tail_recovery(path: str | Path) -> List[Dict[str, Any]]:
    """Load JSONL, recovering only one invalid non-newline-terminated tail row."""

    target = Path(path)
    if not target.exists():
        return []
    raw_lines = target.read_bytes().splitlines(keepends=True)
    records: List[Dict[str, Any]] = []
    recovered = False
    kept: List[bytes] = []
    for index, raw in enumerate(raw_lines):
        if not raw.strip():
            kept.append(raw)
            continue
        try:
            value = json.loads(raw.decode("utf-8"))
        except Exception as error:
            is_last = index == len(raw_lines) - 1
            if is_last and not raw.endswith((b"\n", b"\r")):
                recovered = True
                break
            raise RuntimeError(f"invalid JSON at {target}:{index + 1}: {error}") from error
        if not isinstance(value, dict):
            raise RuntimeError(f"{target}:{index + 1}: expected a JSON object")
        records.append(value)
        kept.append(raw if raw.endswith(b"\n") else raw + b"\n")
    if recovered:
        temporary = target.with_suffix(target.suffix + ".tmp")
        with temporary.open("wb") as handle:
            handle.writelines(kept)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    return records


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def window_ranges(
    duration_sec: float,
    window_sec: float = 10.0,
    hop_sec: float = 10.0,
    min_tail_sec: float = 2.0,
) -> List[Tuple[float, float]]:
    """Cover a track with windows and fold a tiny tail into the final window."""

    duration = max(0.0, float(duration_sec))
    if duration <= 0:
        return []
    if duration <= window_sec:
        return [(0.0, duration)]

    ranges: List[Tuple[float, float]] = []
    start = 0.0
    while start < duration:
        end = min(duration, start + window_sec)
        if end - start < min_tail_sec and ranges:
            previous_start, _ = ranges[-1]
            ranges[-1] = (previous_start, duration)
            break
        ranges.append((start, end))
        if end >= duration:
            break
        start += hop_sec
    return ranges


def trimmed_mean(values: Sequence[float], trim_fraction: float = 0.1) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    trim = int(len(ordered) * max(0.0, min(0.49, trim_fraction)))
    selected = ordered[trim : len(ordered) - trim] if trim else ordered
    return float(sum(selected) / len(selected))


def aggregate_music_gate(
    windows: Sequence[Mapping[str, float]],
    window_positive_threshold: float = 0.5,
) -> Dict[str, float]:
    music = [float(x.get("music_probability", 0.0)) for x in windows]
    singing = [float(x.get("singing_probability", 0.0)) for x in windows]
    speech = [float(x.get("speech_probability", 0.0)) for x in windows]
    weights = [
        max(0.0, float(x.get("end", 1.0)) - float(x.get("start", 0.0)))
        for x in windows
    ]
    if not any(weights):
        weights = [1.0] * len(windows)

    def weighted_trimmed(values: Sequence[float], trim_fraction: float = 0.1) -> float:
        """Weighted trimmed mean, including fractional boundary weights."""

        if not values:
            return 0.0
        ordered = sorted(zip(values, weights), key=lambda item: item[0])
        total_weight = sum(weight for _, weight in ordered)
        trim_weight = total_weight * max(0.0, min(0.49, trim_fraction))
        remaining = [[value, weight] for value, weight in ordered]
        for direction in (1, -1):
            amount = trim_weight
            indices = range(len(remaining)) if direction == 1 else range(len(remaining) - 1, -1, -1)
            for index in indices:
                take = min(amount, remaining[index][1])
                remaining[index][1] -= take
                amount -= take
                if amount <= 1e-12:
                    break
        denominator = sum(weight for _, weight in remaining)
        return (
            sum(value * weight for value, weight in remaining) / denominator
            if denominator > 0
            else 0.0
        )

    total_weight = sum(weights)
    coverage = (
        sum(weight for value, weight in zip(music, weights) if value >= window_positive_threshold)
        / total_weight
        if total_weight > 0
        else 0.0
    )
    score = 0.7 * weighted_trimmed(music) + 0.3 * coverage
    return {
        "score": round(score, 6),
        "coverage": round(coverage, 6),
        "singing_score": round(weighted_trimmed(singing), 6),
        "speech_score": round(weighted_trimmed(speech), 6),
        "window_count": len(windows),
    }


@dataclass(frozen=True)
class RouteThresholds:
    music_keep: float = 0.60
    music_reject: float = 0.30
    vocal_song: float = 0.55
    vocal_instrumental: float = 0.20
    voice_frame: float = 0.50


def fused_vocal_score(
    singing_score: float,
    voice_mean: float,
    voice_coverage: float,
    longest_voice_sec: float = 0.0,
) -> float:
    return max(
        0.0,
        min(
            1.0,
            0.50 * float(voice_mean)
            + 0.20 * float(voice_coverage)
            + 0.15 * float(singing_score)
            + 0.15 * min(1.0, max(0.0, float(longest_voice_sec)) / 30.0),
        ),
    )


def route_track(
    music_gate: Mapping[str, Any],
    voice_analysis: Optional[Mapping[str, Any]],
    thresholds: RouteThresholds,
) -> Tuple[str, str, float]:
    """Return (status, content_type, confidence)."""

    music_score = float(music_gate.get("score", 0.0))
    if music_score <= thresholds.music_reject:
        return "rejected", "unknown", 1.0 - music_score
    if music_score < thresholds.music_keep:
        return "review", "unknown", 1.0 - abs(music_score - 0.5) * 2.0
    if not voice_analysis:
        return "review", "unknown", music_score

    singing_score = float(music_gate.get("singing_score", 0.0))
    speech_score = float(music_gate.get("speech_score", 0.0))
    # Spoken-word audio over background music can otherwise look highly vocal
    # to the Discogs head.  Require a clear BEATs speech-over-singing margin so
    # rap and singing with occasional speech are not rejected automatically.
    if speech_score >= 0.65 and speech_score >= singing_score + 0.25:
        return "review", "unknown", min(1.0, speech_score)

    vocal_score = fused_vocal_score(
        singing_score,
        float(voice_analysis.get("voice_mean", 0.0)),
        float(voice_analysis.get("voice_coverage", 0.0)),
        float(voice_analysis.get("longest_voice_sec", 0.0)),
    )
    if vocal_score >= thresholds.vocal_song:
        return "accepted", "song", vocal_score
    if vocal_score <= thresholds.vocal_instrumental:
        return "accepted", "instrumental", 1.0 - vocal_score
    return "review", "unknown", 1.0 - abs(vocal_score - 0.5) * 2.0


def _nearest_downbeat(value: float, downbeats: Sequence[float], tolerance: float) -> float:
    if not downbeats:
        return value
    nearest = min((float(x) for x in downbeats), key=lambda x: abs(x - value))
    return nearest if abs(nearest - value) <= tolerance else value


def _boundary_confidence(section: Mapping[str, Any], edge: str) -> float:
    key = "start_boundary_confidence" if edge == "start" else "end_boundary_confidence"
    if key in section:
        return float(section.get(key, 0.0))
    return float(section.get("boundary_confidence", 0.0))


def postprocess_sections(
    raw_sections: Sequence[Mapping[str, Any]],
    duration: float,
    downbeats: Sequence[float],
    *,
    snap_tolerance: float = 1.5,
    duplicate_tolerance: float = 2.0,
    minimum_duration: float = 8.0,
    extremely_short_duration: float = 2.0,
    short_boundary_confidence: float = 0.65,
) -> List[Dict[str, Any]]:
    """Normalize boundaries for either SongFormer or instrumental CBM output."""

    total = max(0.0, float(duration))
    if total <= 0:
        raise ValueError("duration must be greater than zero for structure postprocess")
    source = [dict(x) for x in raw_sections if isinstance(x, Mapping)]
    if not source:
        raise ValueError("structure_raw is missing or empty")

    source.sort(key=lambda x: float(x.get("raw_start", x.get("start", 0.0))))
    normalized: List[Dict[str, Any]] = []
    for index, section in enumerate(source):
        raw_start = float(section.get("raw_start", section.get("start", 0.0)))
        raw_end = float(section.get("raw_end", section.get("end", total)))
        start = 0.0 if index == 0 else _nearest_downbeat(raw_start, downbeats, snap_tolerance)
        end = total if index == len(source) - 1 else _nearest_downbeat(raw_end, downbeats, snap_tolerance)
        item = dict(section)
        item.update({
            "raw_start": max(0.0, min(total, raw_start)),
            "raw_end": max(0.0, min(total, raw_end)),
            "start": max(0.0, min(total, start)),
            "end": max(0.0, min(total, end)),
        })
        normalized.append(item)

    # Represent internal boundaries once and resolve duplicate/over-dense ones
    # by keeping the stronger candidate.
    candidates: List[Tuple[float, float]] = [(0.0, 1.0)]
    for index in range(1, len(normalized)):
        left = normalized[index - 1]
        right = normalized[index]
        left_time = float(left["end"])
        right_time = float(right["start"])
        left_conf = _boundary_confidence(left, "end")
        right_conf = _boundary_confidence(right, "start")
        candidates.append((left_time, left_conf) if left_conf >= right_conf else (right_time, right_conf))
    candidates.append((total, 1.0))

    deduped: List[Tuple[float, float]] = [candidates[0]]
    for time_value, confidence in candidates[1:-1]:
        previous_time, previous_confidence = deduped[-1]
        if time_value - previous_time < duplicate_tolerance and len(deduped) > 1:
            if confidence > previous_confidence:
                deduped[-1] = (time_value, confidence)
        else:
            deduped.append((time_value, confidence))
    deduped.append(candidates[-1])

    protected_labels = {
        "intro", "outro", "bridge", "prechorus", "pre-chorus", "postchorus",
        "post-chorus", "interlude", "break", "breakdown", "coda", "solo",
    }

    def interval_label(begin: float, finish: float) -> str:
        midpoint = (begin + finish) / 2.0
        overlaps = [
            item for item in normalized
            if float(item["raw_start"]) <= midpoint < float(item["raw_end"])
        ]
        selected = max(
            overlaps or normalized,
            key=lambda item: max(
                0.0,
                min(finish, float(item["raw_end"])) - max(begin, float(item["raw_start"])),
            ),
        )
        return str(selected.get("label", "")).strip().lower().replace("_", "-")

    # Short functional sections are musically meaningful.  Keep them when
    # their label is protected or both surrounding boundaries are confident;
    # only sub-2-second glitches are always merged.
    changed = True
    while changed and len(deduped) > 2:
        changed = False
        for index in range(len(deduped) - 1):
            begin, left_confidence = deduped[index]
            finish, right_confidence = deduped[index + 1]
            section_duration = finish - begin
            if section_duration >= minimum_duration:
                continue
            label = interval_label(begin, finish)
            is_protected = label in protected_labels
            internal_edges: List[Tuple[int, float]] = []
            if index > 0:
                internal_edges.append((index, left_confidence))
            if index + 1 < len(deduped) - 1:
                internal_edges.append((index + 1, right_confidence))
            if not internal_edges:
                continue
            if section_duration >= extremely_short_duration:
                if is_protected:
                    continue
                if all(confidence >= short_boundary_confidence for _, confidence in internal_edges):
                    continue
            remove_index, _ = min(internal_edges, key=lambda item: (item[1], item[0]))
            del deduped[remove_index]
            changed = True
            break

    output: List[Dict[str, Any]] = []
    for index in range(len(deduped) - 1):
        start, _ = deduped[index]
        end, end_confidence = deduped[index + 1]
        midpoint = (start + end) / 2.0
        overlaps = [
            item for item in normalized
            if float(item["raw_start"]) <= midpoint < float(item["raw_end"])
        ]
        source_item = max(
            overlaps or normalized,
            key=lambda item: max(
                0.0,
                min(end, float(item["raw_end"])) - max(start, float(item["raw_start"])),
            ),
        )
        output.append({
            **source_item,
            "section_id": f"{index + 1:04d}",
            "start": round(float(start), 6),
            "end": round(float(end), 6),
            "boundary_confidence": round(float(end_confidence), 6),
        })

    # Merge adjacent identical labels after boundary cleanup.
    merged: List[Dict[str, Any]] = []
    for item in output:
        if merged and merged[-1].get("label") == item.get("label"):
            merged[-1]["end"] = item["end"]
            merged[-1]["raw_end"] = item.get("raw_end", item["end"])
            merged[-1]["boundary_confidence"] = item["boundary_confidence"]
        else:
            merged.append(item)
    for index, item in enumerate(merged, 1):
        item["section_id"] = f"{index:04d}"
    return merged


def should_run_section_asr(
    section: Mapping[str, Any],
    low_voice_threshold: float = 0.03,
) -> Tuple[bool, str]:
    label = str(section.get("label", "")).strip().lower()
    voice_coverage = float(section.get("voice_coverage", 0.0))
    if label == "silence":
        return False, "silence"
    instrumental_labels = {
        "inst", "instrumental", "solo", "gtrbreak", "guitarsolo",
        "no-vocal-intro", "no-vocal-interlude", "no-vocal-outro",
    }
    if label in instrumental_labels and voice_coverage < low_voice_threshold:
        return False, "low_voice_instrumental_section"
    return True, "eligible"


def crop_aligned_tokens(
    tokens: Sequence[Mapping[str, Any]],
    decoded_start: float,
    core_start: float,
    core_end: float,
) -> List[Dict[str, Any]]:
    """Convert segment-relative timestamps to track time and remove padding."""

    kept: List[Dict[str, Any]] = []
    for token in tokens:
        relative_start = float(token.get("start", token.get("start_time", 0.0)))
        relative_end = float(token.get("end", token.get("end_time", relative_start)))
        absolute_start = decoded_start + relative_start
        absolute_end = decoded_start + relative_end
        midpoint = (absolute_start + absolute_end) / 2.0
        if core_start <= midpoint < core_end:
            kept.append({
                "text": str(token.get("text", "")),
                "start": round(absolute_start, 6),
                "end": round(absolute_end, 6),
            })
    return kept


def decode_audio_range(
    audio_path: str,
    start: float,
    end: float,
    *,
    sample_rate: int = 16000,
    output_format: str = "f32le",
    timeout: int = 300,
) -> bytes:
    """Decode an audio range to memory using ffmpeg; never creates a segment file."""

    if end <= start:
        raise ValueError(f"invalid audio range: {start}..{end}")
    if output_format == "f32le":
        codec_args = ["-acodec", "pcm_f32le"]
        muxer = "f32le"
    elif output_format == "flac":
        codec_args = ["-acodec", "flac"]
        muxer = "flac"
    elif output_format == "wav":
        # ffmpeg cannot seek back over stdout to finalize a WAV/FLAC header.
        # Decode raw PCM first, then write it through a seekable BytesIO so
        # consumers such as libsndfile see a finite, correct frame count.
        codec_args = ["-acodec", "pcm_s16le"]
        muxer = "s16le"
    else:
        raise ValueError(f"unsupported output format: {output_format}")
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{max(0.0, start):.6f}",
        "-t", f"{end - max(0.0, start):.6f}",
        "-i", audio_path,
        "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(sample_rate),
        *codec_args, "-f", muxer, "pipe:1",
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=timeout,
    )
    if completed.returncode != 0 or not completed.stdout:
        detail = completed.stderr.decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"ffmpeg range decode failed: {detail}")
    if output_format == "wav":
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(completed.stdout)
        return output.getvalue()
    return completed.stdout


def extract_downbeats(record: Mapping[str, Any]) -> List[float]:
    beatnet = (
        record.get("beatnet")
        or record.get("music_cpu", {}).get("beatnet")
        or record.get("global_mir", {}).get("beatnet")
        or {}
    )
    values = beatnet.get("beats") or beatnet.get("values") or []
    output: List[float] = []
    for value in values:
        if isinstance(value, Mapping):
            beat_number = value.get("beat_number", value.get("beat"))
            if beat_number not in (1, "1", "downbeat", True):
                continue
            time_value = value.get("time", value.get("timestamp"))
        elif isinstance(value, (list, tuple)) and len(value) >= 2:
            time_value, beat_number = value[0], value[1]
            if beat_number not in (1, "1"):
                continue
        else:
            continue
        try:
            output.append(float(time_value))
        except (TypeError, ValueError):
            continue
    return sorted(set(output))

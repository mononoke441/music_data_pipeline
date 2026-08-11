from __future__ import annotations

import builtins
import concurrent.futures
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pipeline_core import iter_jsonl, write_jsonl
from section_key_infer import (
    analyze_track_sections,
    chord_key_duration_metrics,
    main,
    section_key_input_fingerprint,
    sections_hash,
)


def test_chord_metrics_are_duration_weighted_and_cross_section_boundary():
    section = {"start": 10.0, "end": 20.0}
    mir = {
        "duration": 30.0,
        "chords": {
            "values": [
                {"timestamp": 0.0, "chord": "Dm7"},
                {"timestamp": 12.0, "chord": "Bb:maj"},
                {"timestamp": 16.0, "chord": "E:maj"},
                {"timestamp": 20.0, "chord": "N"},
            ]
        },
    }
    assert chord_key_duration_metrics(section, mir, "D", "minor") == {
        "diatonic_chord_duration_ratio": 0.6,
        "tonic_chord_duration_ratio": 0.2,
    }


def test_chord_metrics_normalize_compact_flat_and_slash_labels():
    section = {"start": 0.0, "end": 8.0}
    mir = {
        "chords": {
            "values": [
                {"timestamp": 0.0, "chord": "Bbmaj7/D"},
                {"timestamp": 4.0, "chord": "Gm7"},
            ]
        }
    }
    assert chord_key_duration_metrics(section, mir, "Bb", "major") == {
        "diatonic_chord_duration_ratio": 1.0,
        "tonic_chord_duration_ratio": 0.5,
    }


def test_chord_metrics_return_none_without_parseable_chord_duration():
    result = chord_key_duration_metrics(
        {"start": 0.0, "end": 8.0},
        {"chords": {"values": [{"timestamp": 0.0, "chord": "N"}]}},
        "C",
        "major",
    )
    assert result == {
        "diatonic_chord_duration_ratio": None,
        "tonic_chord_duration_ratio": None,
    }


def test_successful_resume_does_not_import_or_construct_essentia(tmp_path, monkeypatch):
    input_path = tmp_path / "sections.jsonl"
    output_path = tmp_path / "section-key.jsonl"
    input_record = {
        "audio_id": "cached-audio",
        "audio_path": "/does/not/need/to/exist.wav",
        "sections": [
            {"section_id": "0001", "start": 0.0, "end": 10.0, "label": "verse"}
        ],
    }
    write_jsonl(input_path, [input_record])
    fingerprint = section_key_input_fingerprint(input_record)
    cached = {
        "audio_id": "cached-audio",
        "sections": [
            {
                "section_id": "0001",
                "start": 0.0,
                "end": 10.0,
                "key": {"key": "C", "mode": "major", "strength": 0.9},
                "status": "ok",
            }
        ],
        "sections_hash": sections_hash(input_record),
        "section_key_input_fingerprint": fingerprint,
        "semantic_input_fingerprint": fingerprint,
        "stage_status": {"section_key": "ok"},
        "pipeline_version": "older-pipeline",
        "model_versions": {"section_key": "older-key-model"},
    }
    write_jsonl(output_path, [cached])

    real_import = builtins.__import__

    def reject_essentia_import(name, *args, **kwargs):
        if name == "essentia" or name.startswith("essentia."):
            raise AssertionError("Essentia must stay unloaded on a full cache hit")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_essentia_import)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "section_key_infer.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--resume",
        ],
    )
    main()
    assert list(iter_jsonl(output_path)) == [cached]


def test_section_key_hash_changes_for_id_boundary_or_label():
    record = {
        "audio_id": "a",
        "sections": [
            {"section_id": "0001", "start": 0.0, "end": 10.0, "label": "verse"}
        ],
    }
    initial = sections_hash(record)
    record["sections"][0]["label"] = "chorus"
    assert sections_hash(record) != initial
    record["sections"][0]["label"] = "verse"
    record["sections"][0]["end"] = 11.0
    assert sections_hash(record) != initial
    record["sections"][0]["end"] = 10.0
    record["sections"][0]["section_id"] = "0002"
    assert sections_hash(record) != initial


def test_section_key_decode_failure_produces_terminal_section_error():
    def broken_decode(*args, **kwargs):
        raise RuntimeError("broken media")

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        sections, ok = analyze_track_sections(
            {"audio_path": "/broken.wav"},
            [{"section_id": "0001", "start": 0.0, "end": 10.0}],
            {},
            lambda waveform: ("C", "major", 1.0),
            executor,
            sample_rate=44100,
            decode_function=broken_decode,
        )
    assert ok == 0
    assert sections[0]["status"] == "error"
    assert "broken media" in sections[0]["error"]

from __future__ import annotations

import io
import sys
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from pipeline_core import (
    RouteThresholds,
    aggregate_music_gate,
    crop_aligned_tokens,
    decode_audio_range,
    postprocess_sections,
    route_track,
    should_run_section_asr,
    stable_audio_id,
    window_ranges,
)


def test_decode_audio_range_wav_has_finite_seekable_header(monkeypatch):
    pcm = b"\x00\x00" * 1600

    class Completed:
        returncode = 0
        stdout = pcm
        stderr = b""

    monkeypatch.setattr("pipeline_core.subprocess.run", lambda *args, **kwargs: Completed())

    encoded = decode_audio_range(
        "input.mp3", 1.0, 1.1, sample_rate=16000, output_format="wav"
    )
    with wave.open(io.BytesIO(encoded), "rb") as reader:
        assert reader.getnchannels() == 1
        assert reader.getsampwidth() == 2
        assert reader.getframerate() == 16000
        assert reader.getnframes() == 1600


@pytest.mark.parametrize(
    ("duration", "expected"),
    [
        (0.0, []),
        (1.0, [(0.0, 1.0)]),
        (10.0, [(0.0, 10.0)]),
        (21.5, [(0.0, 10.0), (10.0, 21.5)]),
        (22.0, [(0.0, 10.0), (10.0, 20.0), (20.0, 22.0)]),
    ],
)
def test_window_ranges_cover_track(duration, expected):
    assert window_ranges(duration) == expected
    if expected:
        assert expected[0][0] == 0
        assert expected[-1][1] == duration
        assert all(left[1] == right[0] for left, right in zip(expected, expected[1:]))


def test_music_gate_uses_trimmed_mean_and_coverage():
    windows = [
        {"music_probability": value, "singing_probability": 0.2, "speech_probability": 0.1}
        for value in (0.8, 0.9, 0.9, 0.95)
    ]
    result = aggregate_music_gate(windows)
    assert result["coverage"] == 1.0
    assert result["score"] > 0.9


def test_music_gate_weights_short_tail_by_duration():
    result = aggregate_music_gate([
        {
            "start": 0.0, "end": 10.0,
            "music_probability": 0.9,
            "singing_probability": 0.2,
            "speech_probability": 0.1,
        },
        {
            "start": 10.0, "end": 11.0,
            "music_probability": 0.0,
            "singing_probability": 0.0,
            "speech_probability": 0.9,
        },
    ])
    assert result["coverage"] == pytest.approx(10.0 / 11.0, abs=1e-6)
    assert result["score"] > 0.85


def test_route_track_separates_gate_and_vocal_gray_zones():
    thresholds = RouteThresholds()
    assert route_track({"score": 0.2}, None, thresholds)[0] == "rejected"
    assert route_track({"score": 0.5}, None, thresholds)[0] == "review"
    assert route_track(
        {"score": 0.9, "singing_score": 0.8},
        {"voice_mean": 0.9, "voice_coverage": 0.8},
        thresholds,
    )[1] == "song"
    assert route_track(
        {"score": 0.9, "singing_score": 0.0},
        {"voice_mean": 0.05, "voice_coverage": 0.0},
        thresholds,
    )[1] == "instrumental"


def test_route_track_reviews_spoken_word_but_keeps_rap_song():
    thresholds = RouteThresholds()
    voice = {"voice_mean": 0.9, "voice_coverage": 0.9, "longest_voice_sec": 100.0}
    assert route_track(
        {"score": 0.9, "singing_score": 0.1, "speech_score": 0.85},
        voice,
        thresholds,
    )[:2] == ("review", "unknown")
    assert route_track(
        {"score": 0.9, "singing_score": 0.65, "speech_score": 0.8},
        voice,
        thresholds,
    )[1] == "song"


def test_audio_id_is_content_based_and_path_independent(tmp_path):
    first = tmp_path / "a.bin"
    second = tmp_path / "nested" / "b.bin"
    second.parent.mkdir()
    first.write_bytes(b"same encoded audio bytes")
    second.write_bytes(first.read_bytes())
    assert stable_audio_id(str(first), str(tmp_path)) == stable_audio_id(str(second), str(tmp_path))
    second.write_bytes(b"different encoded audio bytes")
    assert stable_audio_id(str(first), str(tmp_path)) != stable_audio_id(str(second), str(tmp_path))


def test_structure_postprocess_is_contiguous_and_snaps_to_downbeat():
    raw = [
        {"label": "verse", "raw_start": 0, "raw_end": 12.2, "boundary_confidence": 0.8},
        {"label": "chorus", "raw_start": 12.2, "raw_end": 29.1, "boundary_confidence": 0.9},
        {"label": "outro", "raw_start": 29.1, "raw_end": 40, "boundary_confidence": 0.7},
    ]
    sections = postprocess_sections(raw, 40.0, [12.0, 29.0])
    assert sections[0]["start"] == 0.0
    assert sections[-1]["end"] == 40.0
    assert any(section["end"] == 12.0 for section in sections)
    assert all(left["end"] == right["start"] for left, right in zip(sections, sections[1:]))
    assert all(section["end"] - section["start"] >= 8.0 for section in sections)


def test_structure_postprocess_preserves_short_functional_and_confident_sections():
    raw = [
        {"label": "verse", "raw_start": 0, "raw_end": 12, "boundary_confidence": 0.9},
        {"label": "bridge", "raw_start": 12, "raw_end": 16, "boundary_confidence": 0.9},
        {"label": "chorus", "raw_start": 16, "raw_end": 30, "boundary_confidence": 0.9},
    ]
    sections = postprocess_sections(raw, 30.0, [])
    assert [section["label"] for section in sections] == ["verse", "bridge", "chorus"]
    assert sections[1]["end"] - sections[1]["start"] == 4.0
    assert all(left["end"] == right["start"] for left, right in zip(sections, sections[1:]))


def test_structure_postprocess_merges_extremely_short_glitch_even_if_protected():
    raw = [
        {"label": "verse", "raw_start": 0, "raw_end": 12, "boundary_confidence": 0.9},
        {"label": "bridge", "raw_start": 12, "raw_end": 13, "boundary_confidence": 0.9},
        {"label": "chorus", "raw_start": 13, "raw_end": 30, "boundary_confidence": 0.9},
    ]
    sections = postprocess_sections(raw, 30.0, [])
    assert len(sections) == 2
    assert all(left["end"] == right["start"] for left, right in zip(sections, sections[1:]))


def test_structure_postprocess_rejects_missing_structure():
    with pytest.raises(ValueError, match="structure_raw"):
        postprocess_sections([], 30.0, [])
    with pytest.raises(ValueError, match="duration"):
        postprocess_sections([{"label": "A", "start": 0.0, "end": 1.0}], 0.0, [])


def test_asr_padding_tokens_are_cropped_and_shifted():
    tokens = [
        {"text": "padding", "start": 0.0, "end": 0.4},
        {"text": "kept", "start": 1.4, "end": 2.0},
        {"text": "next", "start": 11.8, "end": 12.2},
    ]
    assert crop_aligned_tokens(tokens, 8.5, 10.0, 20.0) == [
        {"text": "kept", "start": 9.9, "end": 10.5},
    ]


def test_asr_skips_only_silence_and_low_voice_instrumental_labels():
    assert should_run_section_asr({"label": "silence", "voice_coverage": 1.0})[0] is False
    assert should_run_section_asr({"label": "solo", "voice_coverage": 0.0})[0] is False
    assert should_run_section_asr({"label": "verse", "voice_coverage": 0.0})[0] is True

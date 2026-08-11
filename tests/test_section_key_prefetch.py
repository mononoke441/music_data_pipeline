from __future__ import annotations

import concurrent.futures
import struct
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from section_key_infer import analyze_track_sections  # noqa: E402


def test_section_key_decodes_overlap_but_extractor_is_serial_and_order_is_stable():
    active_decodes = 0
    maximum_decodes = 0
    extractor_active = 0
    maximum_extractors = 0
    lock = threading.Lock()

    originals = [
        {"section_id": f"{index:04d}", "start": float(index), "end": float(index + 1)}
        for index in range(4)
    ]

    def decode(_path, start, _end, **_kwargs):
        nonlocal active_decodes, maximum_decodes
        with lock:
            active_decodes += 1
            maximum_decodes = max(maximum_decodes, active_decodes)
        time.sleep(0.04 if start == 0.0 else 0.01)
        with lock:
            active_decodes -= 1
        return struct.pack("<f", start)

    def extractor(waveform):
        nonlocal extractor_active, maximum_extractors
        with lock:
            extractor_active += 1
            maximum_extractors = max(maximum_extractors, extractor_active)
        time.sleep(0.005)
        value = int(waveform[0])
        with lock:
            extractor_active -= 1
        return f"K{value}", "major", 0.8

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        sections, ok = analyze_track_sections(
            {"audio_id": "a", "audio_path": "/unused.wav"},
            originals,
            {},
            extractor,
            executor,
            sample_rate=44100,
            decode_function=decode,
        )

    assert ok == 4
    assert maximum_decodes > 1
    assert maximum_decodes <= 3
    assert maximum_extractors == 1
    assert [section["section_id"] for section in sections] == [
        "0000", "0001", "0002", "0003"
    ]
    assert [section["key"]["key"] for section in sections] == ["K0", "K1", "K2", "K3"]


def test_section_key_decode_failure_is_isolated():
    originals = [
        {"section_id": "ok", "start": 0.0, "end": 1.0},
        {"section_id": "bad", "start": 1.0, "end": 2.0},
    ]

    def decode(_path, start, _end, **_kwargs):
        if start == 1.0:
            raise RuntimeError("broken range")
        return struct.pack("<f", start)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        sections, ok = analyze_track_sections(
            {"audio_id": "a", "audio_path": "/unused.wav"},
            originals,
            {},
            lambda _waveform: ("C", "major", 1.0),
            executor,
            sample_rate=44100,
            decode_function=decode,
        )

    assert ok == 1
    assert sections[0]["status"] == "ok"
    assert sections[1]["status"] == "error"
    assert "broken range" in sections[1]["error"]

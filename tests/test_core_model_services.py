from __future__ import annotations

import sys
import threading
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MusicToolsPipeline"))
sys.path.insert(0, str(ROOT / "scripts"))

import serve_discogs  # noqa: E402
from fast_gate_core import CascadeResult  # noqa: E402
from service_api import BatchItemResult, ServiceRequest  # noqa: E402
from serve_cpu_mir import load_cpu_mir, process_cpu_mir_batch  # noqa: E402
from serve_discogs import DiscogsRuntime, process_discogs_batch  # noqa: E402
from serve_fast_gate import (  # noqa: E402
    FastGateRuntime,
    MAX_REQUEST_BATCH as FAST_GATE_BATCH,
    MAX_WAIT_MS as FAST_GATE_WAIT,
    process_fast_gate_batch,
)


def _request(audio_id: str, record: dict) -> ServiceRequest:
    return ServiceRequest(
        job_id="job",
        request_id=f"request-{audio_id}",
        audio_id=audio_id,
        audio_path=f"/{audio_id}.wav",
        input_fingerprint=f"fingerprint-{audio_id}",
        record=record,
    )


class _FakeGate:
    def __init__(self):
        self.calls = []

    def classify_records(self, records):
        self.calls.append([record["audio_id"] for record in records])
        return [
            CascadeResult(
                backend="fake",
                scoring_version="score-v",
                stage_probabilities={"stage_a": [0.9], "stage_b": []},
                stage_scores={"stage_a": 0.9, "stage_b": None},
                offsets={"stage_a": [0.0], "stage_b": []},
                decision="accepted",
                probability=0.9,
                window_seconds=8.0,
                sample_rate=32000,
            )
            for _ in records
        ]


def test_fast_gate_adapter_reuses_cross_track_core_and_keeps_alignment():
    gate = _FakeGate()
    runtime = FastGateRuntime(gate, "stage-v", "fake->fake", "score-v")
    requests = [
        _request("a", {"duration": 10.0, "decode_status": "ok"}),
        _request(
            "bad",
            {"duration": 0.0, "decode_status": "error", "error": "bad audio"},
        ),
        _request("b", {"duration": 20.0}),
    ]

    records = process_fast_gate_batch(runtime, requests)

    assert gate.calls == [["a", "b"]]
    assert [record["audio_id"] for record in records] == ["a", "bad", "b"]
    assert records[0]["music_gate_input_fingerprint"] == "fingerprint-a"
    assert records[1]["content_type"] == "invalid_asset"
    assert FAST_GATE_BATCH == 64
    assert FAST_GATE_WAIT == 100


def _accepted(audio_id: str) -> dict:
    return {
        "audio_id": audio_id,
        "audio_path": f"/{audio_id}.wav",
        "duration": 12.0,
        "status": "accepted",
        "content_type": "music",
        "music_gate": {"probability": 0.9},
    }


def _analysis(frame_count: int) -> dict:
    return {
        "voice_analysis": {
            "voice_mean": 0.9,
            "voice_coverage": 0.8,
            "longest_voice_sec": 12.0,
        },
        "genre": [],
        "mood_theme": [],
        "danceability": [],
        "instruments": [],
        "instrument_changes": [],
        "discogs_frame_count": frame_count,
        "discogs_provider": "FakeCUDAProvider",
    }


def test_discogs_adapter_preserves_frame_batching(
    monkeypatch,
):
    captured = []

    monkeypatch.setattr(
        serve_discogs,
        "decode_track",
        lambda record: np.zeros(16000, dtype=np.float32),
    )

    def fake_prepare(engine, record, waveform, max_cached_frames=None):
        del engine, waveform, max_cached_frames
        return serve_discogs.PreparedAudio(
            record=dict(record),
            patches=np.zeros((3, 1, 1), dtype=np.float32),
            duration=12.0,
        )

    def fake_infer(engine, prepared, max_frames):
        del engine
        captured.append((len(prepared), max_frames))
        return [_analysis(int(item.patches.shape[0])) for item in prepared]

    monkeypatch.setattr(serve_discogs, "prepare_audio", fake_prepare)
    monkeypatch.setattr(serve_discogs, "infer_prepared_batch", fake_infer)
    runtime = DiscogsRuntime(
        engine=types.SimpleNamespace(),
        stage_version="stage-v",
        model_version="model-v",
        frame_batch_size=5,
        buffered_frames=8,
        decode_workers=2,
        decode_prefetch=2,
        vocal_song=0.55,
        vocal_instrumental=0.2,
    )

    records = process_discogs_batch(
        runtime,
        [_request("a", _accepted("a")), _request("b", _accepted("b"))],
    )

    assert captured == [(2, 5)]
    assert [record["content_type"] for record in records] == ["song", "song"]
    assert records[0]["stage_input_fingerprint"] == "fingerprint-a"
    assert records[0]["model_versions"]["discogs_mir"] == "model-v"


def test_cpu_mir_workers_load_once_run_parallel_and_reuse_instances():
    created = []
    barrier = threading.Barrier(4)

    class FakeCpuModel:
        def __init__(self):
            self.instance_id = len(created)
            self.calls = 0
            self.cleanup_calls = 0
            created.append(self)

        def generate_batch(self, records):
            self.calls += 1
            barrier.wait(timeout=3)
            return [
                {
                    **record,
                    "worker_instance": self.instance_id,
                    "music_cpu": {
                        "chords": {},
                        "beatnet": {},
                        "key": {},
                    },
                    "stage_status": {"music_cpu": "ok"},
                    "stage_errors": {},
                }
                for record in records
            ]

        def cleanup(self):
            self.cleanup_calls += 1

    runtime = load_cpu_mir(
        model_fingerprint="cpu-model-v",
        worker_count=4,
        model_factory=FakeCpuModel,
    )
    try:
        first = process_cpu_mir_batch(
            runtime,
            [_request(value, {"duration": 5.0}) for value in ("a", "b", "c", "d")],
        )
        second = process_cpu_mir_batch(
            runtime,
            [_request(value, {"duration": 6.0}) for value in ("e", "f", "g", "h")],
        )

        assert len(created) == 4
        assert [record["worker_instance"] for record in first] == [0, 1, 2, 3]
        assert [record["worker_instance"] for record in second] == [0, 1, 2, 3]
        assert all(model.calls == 2 for model in created)
        assert all(not isinstance(record, BatchItemResult) for record in first + second)
        assert first[0]["music_cpu_input_fingerprint"] == "fingerprint-a"
        assert first[0]["model_versions"]["music_cpu"] == "cpu-model-v"
        assert runtime.health_metadata() == {"worker_count": 4, "workers_ready": 4}
    finally:
        runtime.cleanup()

    assert all(model.cleanup_calls == 1 for model in created)

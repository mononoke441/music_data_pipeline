from __future__ import annotations

import json
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from service_client import InferEnvelope  # noqa: E402
from stream_pipeline import REMOTE_STAGES, StreamingPipeline  # noqa: E402
from stream_state import StreamState  # noqa: E402


class Tracker:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.calls: Counter[tuple[str, str]] = Counter()
        self.request_ids: dict[tuple[str, str], list[str]] = defaultdict(list)
        self.active: dict[str, set[str]] = defaultdict(set)
        self.maximum: Counter[str] = Counter()
        self.three_started: dict[str, threading.Event] = defaultdict(threading.Event)
        self.stage_started: dict[tuple[str, str], threading.Event] = defaultdict(
            threading.Event
        )

    def enter(self, stage: str, audio_id: str, request_id: str) -> None:
        with self.lock:
            self.calls[(stage, audio_id)] += 1
            self.request_ids[(stage, audio_id)].append(request_id)
            self.stage_started[(stage, audio_id)].set()
            if stage in {"music_cpu", "structure_raw", "alm"}:
                self.active[audio_id].add(stage)
                self.maximum[audio_id] = max(
                    self.maximum[audio_id], len(self.active[audio_id])
                )
                if len(self.active[audio_id]) == 3:
                    self.three_started[audio_id].set()

    def leave(self, stage: str, audio_id: str) -> None:
        with self.lock:
            self.active[audio_id].discard(stage)


class FakeService:
    def __init__(self, stage: str, tracker: Tracker) -> None:
        self.stage = stage
        self.tracker = tracker

    def healthz(self) -> dict:
        return {"status": "ok"}

    def envelope(self, request: dict, record: dict) -> InferEnvelope:
        return InferEnvelope(
            job_id=str(request["job_id"]),
            request_id=str(request["request_id"]),
            audio_id=str(request["audio_id"]),
            input_fingerprint=str(request["input_fingerprint"]),
            record=record,
            stage=self.stage,
            model_fingerprint=f"{self.stage}-model-v1",
            elapsed_seconds=0.01,
        )

    def infer(self, **request: object) -> dict:
        audio_id = str(request["audio_id"])
        request_id = str(request["request_id"])
        record = dict(request["record"])  # type: ignore[arg-type]
        self.tracker.enter(self.stage, audio_id, request_id)
        try:
            if self.stage in {"music_cpu", "structure_raw", "alm"}:
                assert self.tracker.three_started[audio_id].wait(2), (
                    "three whole-track tasks were not in flight together"
                )
                time.sleep(float(record.get("fake_delay") or 0.0))
            if self.stage == "fast_gate":
                return self.envelope(request, {
                    "audio_id": audio_id,
                    "status": "accepted",
                    "content_type": "music",
                    "music_gate": {"decision": "music", "score": 0.99},
                    "stage_status": {"music_gate": "ok"},
                })
            if self.stage == "discogs_mir":
                return self.envelope(request, {
                    "audio_id": audio_id,
                    "status": "accepted",
                    "content_type": record["desired_content_type"],
                    "content_confidence": 0.98,
                    "voice_analysis": {},
                    "global_mir": {"genre": {"rock": 0.8}},
                    "stage_status": {"discogs_mir": "ok"},
                })
            if self.stage == "music_cpu":
                return self.envelope(request, {
                    "audio_id": audio_id,
                    "music_cpu": {
                        "chords": {"values": [[0.0, "C"]]},
                        "beatnet": {"beats": [[0.0, 1], [4.0, 1], [8.0, 1]]},
                        "key": {"key": "C", "scale": "major"},
                    },
                    "stage_status": {"music_cpu": "ok"},
                })
            if self.stage == "structure_raw":
                return self.envelope(request, {
                    "audio_id": audio_id,
                    "structure_raw": [
                        {
                            "raw_start": 0.0,
                            "raw_end": float(record["duration"]),
                            "label": "verse",
                            "boundary_confidence": 0.9,
                        }
                    ],
                    "stage_status": {"structure_raw": "ok"},
                })
            if self.stage == "alm":
                return self.envelope(request, {
                    "audio_id": audio_id,
                    "ALM_Caption": f"caption for {audio_id}",
                    "stage_status": {"alm": "ok"},
                })
            if self.stage == "asr":
                assert record.get("sections"), "ASR must receive postprocessed sections"
                with self.tracker.lock:
                    assert not self.tracker.active[audio_id]
                sections = []
                for source in record["sections"]:
                    sections.append(
                        {
                            "section_id": source["section_id"],
                            "start": source["start"],
                            "end": source["end"],
                            "lyrics": "hello world",
                            "asr_tokens": [
                                {"text": "hello world", "start": 1.0, "end": 2.0}
                            ],
                            "asr_status": "ok",
                            "asr_error": None,
                            "alignment_error": None,
                        }
                    )
                return self.envelope(request, {
                    "audio_id": audio_id,
                    "full_transcript": "hello world",
                    "sections": sections,
                    "stage_status": {"section_asr": "ok"},
                    "model_versions": {
                        "section_asr": "asr-model",
                        "forced_aligner": "aligner-model",
                    },
                })
            raise AssertionError(self.stage)
        finally:
            self.tracker.leave(self.stage, audio_id)


def inventory_record(
    audio_id: str, name: str, content_type: str, delay: float
) -> dict:
    return {
        "audio_id": audio_id,
        "audio_path": f"/input/{name}.wav",
        "source_relpath": f"batch/{name}.wav",
        "duration": 12.0,
        "decode_status": "ok",
        "error": None,
        "desired_content_type": content_type,
        "fake_delay": delay,
    }


def test_streams_per_item_parallel_branches_and_resumes_without_repeat(tmp_path: Path):
    tracker = Tracker()
    clients = {stage: FakeService(stage, tracker) for stage in REMOTE_STAGES}
    records = [
        inventory_record("a" * 64, "A", "instrumental", 0.01),
        inventory_record("b" * 64, "B", "song", 0.12),
    ]
    published: list[tuple[str, float, Path]] = []
    state_path = tmp_path / "state.sqlite3"
    with StreamState(state_path) as state:
        summary = StreamingPipeline(
            job_id="job",
            result_dir=tmp_path / "result",
            state=state,
            clients=clients,
            max_inflight=64,
            on_publish=lambda record, path: published.append(
                (str(record["audio_id"]), time.monotonic(), path)
            ),
        ).run(records)

    assert summary == {
        "total": 2,
        "published": 2,
        "review": 0,
        "rejected": 0,
        "retry": 0,
    }
    assert [value[0] for value in published] == ["a" * 64, "b" * 64]
    assert published[0][2].is_file()
    assert published[0][1] < published[1][1]
    assert tracker.maximum["a" * 64] == 3
    assert tracker.maximum["b" * 64] == 3
    assert tracker.calls[("asr", "a" * 64)] == 0
    assert tracker.calls[("asr", "b" * 64)] == 1

    instrumental = json.loads(published[0][2].read_text())
    song = json.loads(published[1][2].read_text())
    assert instrumental["annotation_schema_version"] == "music-data-annotation-v2"
    assert "section_key" not in instrumental["stage_status"]
    assert "section_caption" not in instrumental["stage_status"]
    removed = {
        "key",
        "key_status",
        "key_error",
        "short_caption",
        "caption_status",
        "caption_error",
    }
    assert not (removed & set(instrumental["sections"][0]))
    assert instrumental["sections"][0]["asr_status"] == "not_applicable"
    assert song["full_transcript"] == "hello world"
    assert song["sections"][0]["asr_status"] == "ok"

    with StreamState(state_path) as state:
        row = state.stage("job", "a" * 64, "music_cpu")
        assert row is not None
        assert row.model_fingerprint == "music_cpu-model-v1"
        assert row.elapsed_seconds == 0.01

    calls_before_resume = tracker.calls.copy()
    with StreamState(state_path) as state:
        resumed = StreamingPipeline(
            job_id="job",
            result_dir=tmp_path / "result",
            state=state,
            clients=clients,
            max_inflight=64,
        ).run(records)
    assert resumed == summary
    assert tracker.calls == calls_before_resume


def test_incremental_inventory_dispatches_gate_before_next_item(tmp_path: Path):
    tracker = Tracker()
    clients = {stage: FakeService(stage, tracker) for stage in REMOTE_STAGES}
    first = inventory_record("c" * 64, "C", "instrumental", 0.0)
    second = inventory_record("d" * 64, "D", "instrumental", 0.0)

    def incremental_inventory():
        yield first
        assert tracker.stage_started[("fast_gate", "c" * 64)].wait(2)
        yield second

    with StreamState(tmp_path / "state.sqlite3") as state:
        summary = StreamingPipeline(
            job_id="incremental",
            result_dir=tmp_path / "result",
            state=state,
            clients=clients,
            max_inflight=64,
        ).run(incremental_inventory())
    assert summary["published"] == 2


def test_resume_fails_when_state_contains_item_absent_from_current_inventory(
    tmp_path: Path,
):
    tracker = Tracker()
    clients = {stage: FakeService(stage, tracker) for stage in REMOTE_STAGES}
    current = inventory_record("e" * 64, "E", "instrumental", 0.0)
    stale = inventory_record("f" * 64, "F", "instrumental", 0.0)
    state_path = tmp_path / "state.sqlite3"
    with StreamState(state_path) as state:
        state.register_items("stale-job", [current, stale])
    with StreamState(state_path) as state:
        with pytest.raises(ValueError, match="absent from the current inventory"):
            StreamingPipeline(
                job_id="stale-job",
                result_dir=tmp_path / "result",
                state=state,
                clients=clients,
                max_inflight=64,
            ).run([current])
    assert not tracker.calls

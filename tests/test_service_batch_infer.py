from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import service_batch_infer as batch  # noqa: E402
from pipeline_core import iter_jsonl, write_jsonl  # noqa: E402


class FakeClient:
    calls: list[str] = []

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def healthz(self) -> dict:
        return {"status": "ok", "model_fingerprint": "model-v1"}

    def infer(self, **request) -> dict:
        audio_id = request["audio_id"]
        self.calls.append(audio_id)
        record = dict(request["record"])
        record.update(
            {
                "status": "accepted",
                "content_type": "music",
                "music_gate": {"decision": "music"},
                "stage_status": {"music_gate": "ok"},
            }
        )
        return record


def test_partitioned_batch_resume_reuses_success_without_service_call(
    tmp_path: Path, monkeypatch
):
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "gate"
    source = {
        "audio_id": "a" * 64,
        "audio_path": "/a.wav",
        "source_relpath": "a.wav",
        "duration": 10.0,
    }
    write_jsonl(input_path, [source])
    monkeypatch.setattr(batch, "ServiceClient", FakeClient)
    FakeClient.calls.clear()
    argv = [
        "service_batch_infer.py",
        "--stage",
        "fast_gate",
        "--service-url",
        "http://fake",
        "--input",
        str(input_path),
        "--output-dir",
        str(output_dir),
        "--resume",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    batch.main()
    assert FakeClient.calls == [source["audio_id"]]
    accepted = list(iter_jsonl(output_dir / "accepted.music.jsonl"))
    assert accepted[0]["service_runtime"]["fast_gate"]["model_fingerprint"] == "model-v1"

    FakeClient.calls.clear()
    batch.main()
    assert FakeClient.calls == []
    assert json.loads((output_dir / "accepted.music.jsonl").read_text())["audio_id"] == source["audio_id"]


def test_instrumental_asr_is_model_free(tmp_path: Path, monkeypatch):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "asr.jsonl"
    source = {
        "audio_id": "b" * 64,
        "audio_path": "/b.wav",
        "source_relpath": "b.wav",
        "duration": 10.0,
        "content_type": "instrumental",
        "sections": [{"section_id": "0001", "start": 0.0, "end": 10.0}],
    }
    write_jsonl(input_path, [source])
    monkeypatch.setattr(batch, "ServiceClient", FakeClient)
    FakeClient.calls.clear()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "service_batch_infer.py",
            "--stage",
            "section_asr",
            "--service-url",
            "http://fake",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
    )
    batch.main()
    assert FakeClient.calls == []
    value = json.loads(output_path.read_text())
    assert value["stage_status"] == {"section_asr": "not_run"}
    assert value["sections"][0]["asr_status"] == "not_applicable"

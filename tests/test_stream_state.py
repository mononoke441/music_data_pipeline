from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from stream_state import StreamState, canonical_fingerprint  # noqa: E402


def source(audio_id: str = "a" * 64) -> dict:
    return {
        "audio_id": audio_id,
        "audio_path": f"/audio/{audio_id}.wav",
        "source_relpath": f"{audio_id}.wav",
        "duration": 12.0,
        "decode_status": "ok",
    }


def test_wal_running_recovery_preserves_idempotent_request_id(tmp_path: Path):
    path = tmp_path / "state.sqlite3"
    record = source()
    fingerprint = canonical_fingerprint("fast_gate", record)
    with StreamState(path) as state:
        assert state.journal_mode() == "wal"
        state.register_items("job", [record])
        prepared = state.prepare_stage("job", record["audio_id"], "fast_gate", fingerprint)
        claimed = state.claim_stage("job", record["audio_id"], "fast_gate", fingerprint)
        assert claimed is not None
        assert claimed.request_id == prepared.request_id
        assert claimed.attempt == 1

    with StreamState(path) as state:
        assert state.recover_running("job") == 1
        recovered = state.stage("job", record["audio_id"], "fast_gate")
        assert recovered is not None
        assert recovered.status == "pending"
        assert recovered.request_id == prepared.request_id
        claimed_again = state.claim_stage(
            "job", record["audio_id"], "fast_gate", fingerprint
        )
        assert claimed_again is not None
        assert claimed_again.request_id == prepared.request_id
        assert claimed_again.attempt == 2
        state.finish_stage(
            "job",
            record["audio_id"],
            "fast_gate",
            claimed_again.request_id,
            {"status": "accepted"},
            "gate-model-v1",
            0.125,
        )
        completed = state.stage("job", record["audio_id"], "fast_gate")
        assert completed is not None
        assert completed.model_fingerprint == "gate-model-v1"
        assert completed.elapsed_seconds == 0.125
        assert completed.schema_version == 2
        assert state.claim_stage(
            "job", record["audio_id"], "fast_gate", fingerprint
        ) is None


def test_duplicate_content_ids_fail_explicitly(tmp_path: Path):
    state = StreamState(tmp_path / "state.sqlite3")
    try:
        first = source()
        second = {**first, "audio_path": "/audio/copy.wav", "source_relpath": "copy.wav"}
        try:
            state.register_items("job", [first, second])
        except ValueError as error:
            assert "duplicate SHA256 audio_id" in str(error)
        else:
            raise AssertionError("duplicate content IDs were accepted")
    finally:
        state.close()

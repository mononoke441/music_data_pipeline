from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from annotation_storage import publish_annotation_records  # noqa: E402
from pipeline_core import (  # noqa: E402
    iter_jsonl,
    load_jsonl_with_truncated_tail_recovery,
    write_jsonl,
)


def record(audio_id: str, source_relpath: str | None = None) -> dict:
    return {
        "audio_id": audio_id,
        "audio_path": f"/audio/{audio_id}.wav",
        "source_relpath": source_relpath or f"{audio_id}.wav",
        "duration": 12.0,
        "content_type": "instrumental",
    }


def run_state(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "pipeline_state.py"), *args],
        text=True,
        capture_output=True,
    )


def test_active_manifest_isolates_per_item_error_and_requires_terminal_coverage(tmp_path: Path):
    base = tmp_path / "base.jsonl"
    stage = tmp_path / "stage.jsonl"
    active = tmp_path / "active.jsonl"
    retry = tmp_path / "retry.jsonl"
    write_jsonl(base, [record("ok"), record("bad")])
    write_jsonl(
        stage,
        [
            {
                **record("ok"),
                "music_cpu": {"chords": {"values": [1]}, "beatnet": {"beats": [1]}, "key": {"key": "C"}},
                "stage_status": {"music_cpu": "ok"},
            },
            {
                **record("bad"),
                "music_cpu": {"chords": {}, "beatnet": {}, "key": {}},
                "stage_status": {"music_cpu": "partial_error"},
                "stage_errors": {"music_cpu": "decoder failed"},
            },
        ],
    )
    result = run_state(
        "active", "--base", str(base), "--stage", f"music_cpu={stage}",
        "--output", str(active), "--retry-output", str(retry),
    )
    assert result.returncode == 0, result.stderr
    assert [value["audio_id"] for value in iter_jsonl(active)] == ["ok"]
    retry_value = list(iter_jsonl(retry))[0]
    assert retry_value["audio_id"] == "bad"
    assert retry_value["failure_stage"] == "music_cpu"
    assert retry_value["retryable"] is True
    assert retry_value["semantic_input_fingerprint"]

    write_jsonl(stage, list(iter_jsonl(stage))[:1])
    result = run_state(
        "active", "--base", str(base), "--stage", f"music_cpu={stage}",
        "--output", str(active), "--retry-output", str(retry),
    )
    assert result.returncode != 0
    assert "terminal coverage mismatch" in result.stderr


def test_annotation_file_directory_prefix_conflicts_are_all_isolated(tmp_path: Path):
    base = tmp_path / "base.jsonl"
    clean = tmp_path / "clean.jsonl"
    retry = tmp_path / "retry.jsonl"
    write_jsonl(
        base,
        [record("left", "foo"), record("right", "foo.json/bar.wav"), record("safe", "safe.wav")],
    )
    result = run_state(
        "path-conflicts", "--base", str(base), "--output", str(clean),
        "--retry-output", str(retry),
    )
    assert result.returncode == 0, result.stderr
    assert {value["audio_id"] for value in iter_jsonl(clean)} == {"safe"}
    assert {value["audio_id"] for value in iter_jsonl(retry)} == {"left", "right"}


def test_combine_retry_builds_four_complete_exclusive_partitions(tmp_path: Path):
    inventory = tmp_path / "inventory.jsonl"
    review = tmp_path / "review.jsonl"
    rejected = tmp_path / "rejected.jsonl"
    failure = tmp_path / "failure.jsonl"
    retry = tmp_path / "retry.jsonl"
    annotations = tmp_path / "annotations"
    values = [record(name) for name in ("annotation", "review", "rejected", "retry")]
    write_jsonl(inventory, values)
    write_jsonl(review, [{**values[1], "status": "review"}])
    write_jsonl(rejected, [{**values[2], "status": "rejected"}])
    write_jsonl(
        failure,
        [{**values[3], "stage_status": {"alm": "error"}, "stage_errors": {"alm": "empty"}}],
    )
    publish_annotation_records([{**values[0], "status": "accepted"}], annotations)
    result = run_state(
        "combine-retry", "--inventory", str(inventory), "--review", str(review),
        "--rejected", str(rejected), "--annotations-dir", str(annotations),
        "--inputs", str(failure), "--output", str(retry),
    )
    assert result.returncode == 0, result.stderr
    value = list(iter_jsonl(retry))[0]
    assert value["audio_id"] == "retry"
    assert value["failure_stage"] == "alm"

    write_jsonl(review, [])
    result = run_state(
        "combine-retry", "--inventory", str(inventory), "--review", str(review),
        "--rejected", str(rejected), "--annotations-dir", str(annotations),
        "--inputs", str(failure), "--output", str(retry),
    )
    assert result.returncode != 0
    assert "coverage is incomplete/non-exclusive" in result.stderr


def test_resume_jsonl_recovers_only_a_truncated_last_row(tmp_path: Path):
    path = tmp_path / "resume.jsonl"
    path.write_bytes(b'{"audio_id":"ok"}\n{"audio_id":')
    assert load_jsonl_with_truncated_tail_recovery(path) == [{"audio_id": "ok"}]
    assert path.read_bytes() == b'{"audio_id":"ok"}\n'

    path.write_bytes(b'{"audio_id":\n{"audio_id":"later"}\n')
    result = run_state(
        "active", "--base", str(path), "--stage", f"music_cpu={path}",
        "--output", str(tmp_path / "out.jsonl"),
        "--retry-output", str(tmp_path / "retry-invalid.jsonl"),
    )
    assert result.returncode != 0
    assert "Expecting value" in result.stderr

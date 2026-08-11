from __future__ import annotations

import json
import sys
from pathlib import Path


SAVER = (
    Path(__file__).resolve().parents[1]
    / "MusicToolsPipeline"
    / "workers"
    / "saver_worker.py"
)
sys.path.insert(0, str(SAVER.parents[1]))

from runtime_integrity import merge_results_atomically
from workers.saver_worker import _make_serializable_record


def test_result_queue_uses_explicit_completion_without_poll_timeout():
    source = SAVER.read_text(encoding="utf-8")

    assert "item = result_queue.get()" in source
    assert "result_queue.get(timeout=" not in source
    assert 'item.get("type") == "worker_done"' in source


def test_serialization_fallback_changes_effective_status_to_error():
    record, status = _make_serializable_record(
        {
            "runtime_task_key": "a",
            "audio_id": "a",
            "audio_path": "/a.wav",
            "bad_value": float("nan"),
            "stage_status": {"music_cpu": "ok"},
        },
        "ok",
        stage_name="music_cpu",
        stage_fingerprint="stage",
    )

    assert status == "error"
    assert record["stage_status"]["music_cpu"] == "error"
    assert record["stage_errors"]["music_cpu"]
    json.dumps(record, allow_nan=False)


def test_atomic_merge_keeps_current_failure_retryable(tmp_path: Path):
    results = tmp_path / "results.jsonl"
    merge_results_atomically(
        str(results),
        [
            {
                "runtime_task_key": "a",
                "stage_status": {"music_cpu": "ok"},
                "stage_errors": {},
                "music_cpu": {
                    "chords": {"items": [1]},
                    "beatnet": {"beats": [1]},
                    "key": {"key": "C"},
                },
            }
        ],
    )
    merge_results_atomically(
        str(results),
        [
            {
                "runtime_task_key": "a",
                "stage_status": {"music_cpu": "error"},
                "stage_errors": {"music_cpu": "temporary"},
            }
        ],
    )

    rows = [json.loads(line) for line in results.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["stage_status"]["music_cpu"] == "error"

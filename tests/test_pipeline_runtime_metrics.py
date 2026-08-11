from pathlib import Path

from scripts.pipeline_runtime_metrics import (
    append_stage,
    final_report,
    load_stages,
    stage_record,
)


def test_stage_record_and_jsonl_roundtrip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("scripts.pipeline_runtime_metrics.time.time", lambda: 15.0)
    record = stage_record("fast_music_gate", 10.0, 20)
    assert record["elapsed_seconds"] == 5.0
    assert record["seconds_per_item"] == 0.25
    assert record["items_per_second"] == 4.0

    path = tmp_path / "stages.jsonl"
    append_stage(path, record)
    assert load_stages(path) == [record]


def test_final_report_handles_empty_and_accepted_counts(monkeypatch) -> None:
    monkeypatch.setattr("scripts.pipeline_runtime_metrics.time.time", lambda: 25.0)
    report = final_report(
        started_at=5.0,
        input_count=10,
        accepted_count=4,
        review_count=1,
        rejected_count=5,
        stages=[{"stage": "inventory", "elapsed_seconds": 2.0}],
    )
    assert report["elapsed_seconds"] == 20.0
    assert report["seconds_per_input_track"] == 2.0
    assert report["seconds_per_accepted_track"] == 5.0
    assert report["input_tracks_per_second"] == 0.5
    assert report["accepted_tracks_per_second"] == 0.2

    empty = final_report(
        started_at=5.0,
        input_count=0,
        accepted_count=0,
        review_count=0,
        rejected_count=0,
        stages=[],
    )
    assert empty["seconds_per_input_track"] is None
    assert empty["seconds_per_accepted_track"] is None

from pathlib import Path

from scripts.pipeline_runtime_metrics import (
    append_stage,
    final_report,
    format_elapsed_seconds,
    format_pipeline_summary,
    format_stage_summary,
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


def test_human_readable_runtime_summaries() -> None:
    assert format_elapsed_seconds(3661.2344) == "01:01:01.234"
    stage = format_stage_summary(
        {
            "stage": "inventory",
            "elapsed_seconds": 5.25,
            "processed": 20,
            "seconds_per_item": 0.2625,
            "items_per_second": 3.809524,
        }
    )
    assert stage.startswith("[TIME] inventory elapsed=00:00:05.250 (5.250s)")
    assert "processed=20" in stage
    assert "seconds_per_item=0.262500" in stage
    assert "items_per_second=3.809524" in stage

    pipeline = format_pipeline_summary(
        {
            "elapsed_seconds": 65.5,
            "status": "partial_success",
            "completed_stages": ["inventory", "fast_music_gate"],
        }
    )
    assert pipeline == (
        "[TIME] pipeline_total elapsed=00:01:05.500 (65.500s) "
        "status=partial_success completed_stages=2"
    )

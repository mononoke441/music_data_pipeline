from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MusicToolsPipeline"))
sys.path.insert(0, str(ROOT / "scripts"))

from config import Config
import ray_inference
from runtime_integrity import (
    CPU_MIR_SEMANTIC_INPUT_FIELDS,
    CPU_MIR_SEMANTIC_SOURCE_NAMES,
    _semantic_python_hash,
    build_stage_fingerprint,
    build_stage_fingerprint_payload,
    build_task_manifest,
    merge_results_atomically,
    reset_incompatible_stage_state,
    validate_result_tracker_coverage,
    write_stage_fingerprint_manifest,
)
from task_tracker import TaskTracker


class _FakeLoader:
    def __init__(self, records):
        self.records = records

    def _raw_iter(self):
        yield from self.records


def test_completed_audio_id_is_remapped_after_input_reorder(tmp_path: Path):
    progress = tmp_path / "progress.jsonl"
    tracker = TaskTracker(str(progress))
    tracker.configure_run(
        input_fingerprint="same-unordered-input",
        stage_fingerprint="same-stage",
        task_map={0: "audio-a", 1: "audio-b"},
    )
    tracker.mark_tasks_completed([0])

    resumed = TaskTracker(str(progress))
    resumed.configure_run(
        input_fingerprint="same-unordered-input",
        stage_fingerprint="same-stage",
        task_map={0: "audio-b", 1: "audio-a"},
    )

    assert resumed.get_completed_tasks() == {1}


def test_failed_task_is_terminal_but_retryable_on_resume(tmp_path: Path):
    progress = tmp_path / "progress.jsonl"
    tracker = TaskTracker(str(progress))
    tracker.configure_run(
        input_fingerprint="input",
        stage_fingerprint="stage",
        task_map={0: "audio-a"},
    )
    tracker.mark_tasks_finished([0], ["error"])

    assert tracker.get_progress_stats()["failed"] == 1
    assert tracker.get_completed_tasks() == set()
    assert TaskTracker(str(progress)).get_completed_tasks() == set()


def test_changed_stage_fingerprint_keeps_completed_input_tasks(tmp_path: Path):
    progress = tmp_path / "progress.jsonl"
    tracker = TaskTracker(str(progress))
    tracker.configure_run(
        input_fingerprint="input",
        stage_fingerprint="stage-a",
        task_map={0: "audio-a"},
    )
    tracker.mark_tasks_completed([0])
    resumed = TaskTracker(str(progress))
    resumed.configure_run(
        input_fingerprint="input",
        stage_fingerprint="stage-b",
        task_map={0: "audio-a"},
    )

    assert resumed.get_completed_tasks() == {0}
    assert resumed.stage_fingerprint == "stage-b"


def test_reset_incompatible_stage_state_removes_only_versioned_artifacts(
    tmp_path: Path,
):
    output = tmp_path / "music-cpu"
    output.mkdir()
    progress = output / "progress.jsonl"
    manifest = output / "progress.jsonl.manifest.jsonl"
    results = output / "results.jsonl"
    success = output / "success.jsonl"
    fingerprint = output / "stage_fingerprint.json"
    log = output / "inference.log"
    for path in (progress, manifest, results, success, fingerprint, log):
        path.write_text("stale\n", encoding="utf-8")

    removed = reset_incompatible_stage_state(str(output), str(progress))

    assert set(removed) == {
        str(progress),
        str(manifest),
        str(results),
        str(success),
        str(fingerprint),
    }
    assert log.read_text(encoding="utf-8") == "stale\n"


def test_cpu_mir_fingerprint_excludes_operational_sources():
    config = Config()
    config.model_type = "music_cpu_pipeline"
    payload = build_stage_fingerprint_payload(config, model_path="dummy")

    assert payload["schema"] == "music-cpu-semantic-v3"
    assert payload["semantic_contract"]["features"] == ["chords", "beatnet", "key"]
    assert set(payload["sources"]) == set(CPU_MIR_SEMANTIC_SOURCE_NAMES)
    assert "ray_inference.py" not in payload["sources"]
    assert "task_tracker.py" not in payload["sources"]
    assert "workers/model_worker.py" not in payload["sources"]
    assert "workers/saver_worker.py" not in payload["sources"]


def test_cpu_mir_scheduling_changes_do_not_change_semantic_fingerprint():
    config = Config()
    config.model_type = "music_cpu_pipeline"
    config.num_workers = 1
    config.batch_size = 1
    first = build_stage_fingerprint(config, model_path="dummy")
    config.num_workers = 32
    config.batch_size = 128
    config.num_dataloader_workers = 16
    second = build_stage_fingerprint(config, model_path="dummy")
    assert first == second


def test_semantic_source_hash_ignores_docs_and_logging_but_tracks_code(tmp_path: Path):
    source = tmp_path / "model.py"
    source.write_text(
        '"""first docs"""\n'
        "def infer(value):\n"
        "    logger.info('first message')\n"
        "    return value + 1\n",
        encoding="utf-8",
    )
    first = _semantic_python_hash(source)
    source.write_text(
        '"""different docs"""\n'
        "def infer(value):\n"
        "    logger.warning('different message')\n"
        "    return value + 1\n",
        encoding="utf-8",
    )
    assert _semantic_python_hash(source) == first
    source.write_text(
        "def infer(value):\n    return value + 2\n",
        encoding="utf-8",
    )
    assert _semantic_python_hash(source) != first


def test_stage_fingerprint_manifest_is_atomic_and_auditable(tmp_path: Path):
    payload = {"schema": "semantic-test", "sources": {"model.py": "abc"}}
    write_stage_fingerprint_manifest(str(tmp_path), "fingerprint", payload)
    value = json.loads(
        (tmp_path / "stage_fingerprint.json").read_text(encoding="utf-8")
    )
    assert value == {"stage_fingerprint": "fingerprint", "payload": payload}
    assert not (tmp_path / "stage_fingerprint.json.tmp").exists()


def test_cpu_mir_reset_policy_is_explicitly_opt_in():
    assert Config().reset_incompatible_output is False


def test_manifest_fingerprint_is_order_independent():
    first_map, first_fingerprint = build_task_manifest(
        _FakeLoader(
            [
                (0, {"audio_id": "a", "audio_path": "/a.wav", "duration": 1.0}),
                (1, {"audio_id": "b", "audio_path": "/b.wav", "duration": 2.0}),
            ]
        )
    )
    second_map, second_fingerprint = build_task_manifest(
        _FakeLoader(
            [
                (0, {"audio_id": "b", "audio_path": "/b.wav", "duration": 2.0}),
                (1, {"audio_id": "a", "audio_path": "/a.wav", "duration": 1.0}),
            ]
        )
    )

    assert first_fingerprint == second_fingerprint
    assert first_map == {0: "a", 1: "b"}
    assert second_map == {0: "b", 1: "a"}


def test_ray_no_task_exit_does_not_create_model_workers(tmp_path: Path, monkeypatch):
    record = {
        "audio_id": "a",
        "audio_path": "/a.wav",
        "duration": 3.0,
        "decode_status": "ok",
    }
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    output_path = tmp_path / "output"
    output_path.mkdir()

    config = Config()
    config.data_path = str(input_path)
    config.output_path = str(output_path)
    config.dataloader_type = "jsonl"
    config.model_type = "music_cpu_pipeline"
    config.model_path = "dummy"
    config.batch_size = 1
    # Seed the pre-semantic-fingerprint layout used by the retained WebSource run.
    task_map, input_fingerprint = build_task_manifest(_FakeLoader([(0, record)]))
    stage_fingerprint = build_stage_fingerprint(config, model_path="dummy")
    tracker = TaskTracker(str(output_path / "progress.jsonl"))
    tracker.configure_run(
        input_fingerprint=input_fingerprint,
        stage_fingerprint=stage_fingerprint,
        task_map=task_map,
    )
    tracker.mark_tasks_completed([0])
    (output_path / "results.jsonl").write_text(
        json.dumps(
            {
                "audio_id": "a",
                "model_versions": {"music_cpu": stage_fingerprint},
                "stage_status": {"music_cpu": "ok"},
                "stage_errors": {},
                "music_cpu": {
                    "chords": {"items": [1]},
                    "beatnet": {"beats": [1]},
                    "key": {"key": "C"},
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def unexpected_worker_creation(*args, **kwargs):
        raise AssertionError("model workers must not be created for a completed run")

    monkeypatch.setattr(
        ray_inference, "_ensure_model_workers", unexpected_worker_creation
    )
    returned_workers = ray_inference.run_inference(
        str(input_path),
        str(output_path),
        workers=None,
        model_path="dummy",
        config=config,
    )

    assert returned_workers is None
    marker = json.loads((output_path / "success.jsonl").read_text(encoding="utf-8"))
    assert marker["resume_status"] == "already_complete"
    assert (
        TaskTracker(str(output_path / "progress.jsonl")).input_fingerprint_schema
        == "cpu_mir_semantic_v1"
    )


def test_cpu_mir_input_fingerprint_excludes_provenance_fields():
    base = {
        "audio_id": "a",
        "audio_path": "/a.wav",
        "duration": 3.0,
        "stage_status": {"discogs_mir": "ok"},
        "stage_versions": {"discogs_mir": "old"},
        "model_versions": {"discogs_mir": "old"},
        "stage_errors": {},
        "global_mir": {"genre": [{"label": "pop", "probability": 0.9}]},
    }
    changed_provenance = dict(base)
    changed_provenance.update(
        {
            "stage_status": {"discogs_mir": "error"},
            "stage_versions": {"discogs_mir": "new"},
            "model_versions": {"discogs_mir": "new"},
            "stage_errors": {"discogs_mir": "temporary"},
            "global_mir": {"genre": []},
        }
    )
    _, first = build_task_manifest(
        _FakeLoader([(0, base)]),
        semantic_fields=CPU_MIR_SEMANTIC_INPUT_FIELDS,
    )
    _, second = build_task_manifest(
        _FakeLoader([(0, changed_provenance)]),
        semantic_fields=CPU_MIR_SEMANTIC_INPUT_FIELDS,
    )
    assert first == second

    changed_audio = dict(base, audio_path="/different.wav")
    _, third = build_task_manifest(
        _FakeLoader([(0, changed_audio)]),
        semantic_fields=CPU_MIR_SEMANTIC_INPUT_FIELDS,
    )
    assert third != first


def test_semantic_fingerprint_migration_preserves_stable_completed_tasks(
    tmp_path: Path,
):
    progress = tmp_path / "progress.jsonl"
    tracker = TaskTracker(str(progress))
    tracker.configure_run(
        input_fingerprint="legacy-full-record-fingerprint",
        stage_fingerprint="stage",
        task_map={0: "a", 1: "b"},
    )
    tracker.mark_tasks_completed([0, 1])

    tracker.migrate_semantic_input_fingerprint(
        input_fingerprint="semantic-fingerprint",
        task_map={0: "b", 1: "a"},
    )

    resumed = TaskTracker(str(progress))
    assert resumed.input_fingerprint == "semantic-fingerprint"
    assert resumed.input_fingerprint_schema == "cpu_mir_semantic_v1"
    assert resumed.get_completed_tasks() == {0, 1}


def test_semantic_fingerprint_change_is_not_repeatedly_migrated(tmp_path: Path):
    tracker = TaskTracker(str(tmp_path / "progress.jsonl"))
    tracker.configure_run(
        input_fingerprint="semantic-a",
        input_fingerprint_schema="cpu_mir_semantic_v1",
        stage_fingerprint="stage",
        task_map={0: "a"},
    )

    with pytest.raises(RuntimeError, match="different input fingerprint"):
        tracker.configure_run(
            input_fingerprint="semantic-b",
            input_fingerprint_schema="cpu_mir_semantic_v1",
            stage_fingerprint="stage",
            task_map={0: "a"},
        )


def test_atomic_result_merge_replaces_error_with_success(tmp_path: Path):
    results = tmp_path / "results.jsonl"
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
    rows = [json.loads(line) for line in results.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["stage_status"]["music_cpu"] == "ok"
    assert not (tmp_path / "results.jsonl.tmp").exists()


def test_result_tracker_coverage_rejects_empty_cpu_payload(tmp_path: Path):
    progress = tmp_path / "progress.jsonl"
    results = tmp_path / "results.jsonl"
    tracker = TaskTracker(str(progress))
    tracker.configure_run(
        input_fingerprint="input",
        stage_fingerprint="stage",
        task_map={0: "a"},
    )
    tracker.mark_tasks_completed([0])
    merge_results_atomically(
        str(results),
        [
            {
                "runtime_task_key": "a",
                "stage_status": {"music_cpu": "ok"},
                "stage_errors": {},
                "music_cpu": {"chords": {}, "beatnet": {}, "key": {}},
            }
        ],
    )

    with pytest.raises(RuntimeError, match="required payload"):
        validate_result_tracker_coverage(
            results_path=str(results),
            stage_name="music_cpu",
            tracker=tracker,
            task_map={0: "a"},
            required_payload_fields=("chords", "beatnet", "key"),
        )


def test_cpu_mir_item_error_publishes_partial_success(tmp_path: Path):
    output = tmp_path / "output"
    output.mkdir()
    progress = output / "progress.jsonl"
    tracker = TaskTracker(str(progress))
    task_map = {0: "ok", 1: "retry"}
    run_fingerprint = tracker.configure_run(
        input_fingerprint="input",
        stage_fingerprint="stage",
        task_map=task_map,
    )
    tracker.mark_tasks_finished([0, 1], ["ok", "error"])
    merge_results_atomically(
        str(output / "results.jsonl"),
        [
            {
                "runtime_task_key": "ok",
                "stage_status": {"music_cpu": "ok"},
                "stage_errors": {},
                "music_cpu": {
                    "chords": {"items": [1]},
                    "beatnet": {"beats": [1]},
                    "key": {"key": "C"},
                },
            },
            {
                "runtime_task_key": "retry",
                "stage_status": {"music_cpu": "error"},
                "stage_errors": {"music_cpu": "temporary"},
            },
        ],
    )
    state = ray_inference.ResumeState(
        tracker=tracker,
        task_map=task_map,
        input_fingerprint="input",
        stage_fingerprint="stage",
        run_fingerprint=run_fingerprint,
        completed_count=1,
    )

    ray_inference._validate_and_publish_success(
        state=state,
        output_path=str(output),
        db_path=str(progress),
        total_samples=2,
        loader_results=[1],
        model_results=[1],
        save_result={"persisted": 2, "failed": 1},
        stage_name="music_cpu",
        required_payload_fields=("chords", "beatnet", "key"),
    )

    marker = json.loads((output / "success.jsonl").read_text())
    assert marker["status"] == "partial_success"
    assert marker["failed"] == 1

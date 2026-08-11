from __future__ import annotations

import importlib.util
import json
import queue
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SONGFORMER = ROOT / "SongFormer"


def load_infer_jsonl():
    sys.path.insert(0, str(SONGFORMER))
    spec = importlib.util.spec_from_file_location(
        "songformer_infer_jsonl_for_test", SONGFORMER / "infer_jsonl.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_structure_cache_reuse_ignores_provenance_version(tmp_path: Path):
    module = load_infer_jsonl()
    prediction = tmp_path / "track.json"
    prediction.write_text(json.dumps([[0.0, "verse"]]), encoding="utf-8")
    (tmp_path / "track.meta.json").write_text(
        json.dumps({"stage_version": "older-version", "content_type": "song"}),
        encoding="utf-8",
    )

    assert module._cache_is_reusable(str(prediction), "song")


def test_structure_cache_rejects_empty_corrupt_or_incompatible_entries(tmp_path: Path):
    module = load_infer_jsonl()
    prediction = tmp_path / "track.json"
    metadata = tmp_path / "track.meta.json"

    prediction.write_text("[]\n", encoding="utf-8")
    metadata.write_text(
        json.dumps({"content_type": "song", "stage_version": "v"}) + "\n",
        encoding="utf-8",
    )
    assert not module._cache_is_reusable(str(prediction), "song")

    prediction.write_text("{broken", encoding="utf-8")
    assert not module._cache_is_reusable(str(prediction), "song")

    prediction.write_text(
        json.dumps([{"label": "verse", "start": 0.0, "end": 10.0}]) + "\n",
        encoding="utf-8",
    )
    metadata.write_text("{broken", encoding="utf-8")
    assert not module._cache_is_reusable(str(prediction), "song")

    metadata.write_text(
        json.dumps({"content_type": "instrumental", "stage_version": "v"}) + "\n",
        encoding="utf-8",
    )
    assert not module._cache_is_reusable(str(prediction), "song")


def test_prediction_and_metadata_are_committed_without_temp_files(tmp_path: Path):
    module = load_infer_jsonl()
    prediction = tmp_path / "track.json"
    structure = [{"label": "verse", "start": 0.0, "end": 10.0}]

    module._write_prediction_cache(
        str(prediction),
        structure,
        {"state": "ok", "stage_version": "v", "content_type": "song"},
    )

    assert json.loads(prediction.read_text(encoding="utf-8")) == structure
    metadata = json.loads((tmp_path / "track.meta.json").read_text(encoding="utf-8"))
    assert metadata["state"] == "ok"
    assert module._cache_is_reusable(str(prediction), "song")
    assert not list(tmp_path.glob(".*.tmp.*"))


def test_merge_persists_per_item_error_and_replaces_output_atomically(tmp_path: Path):
    module = load_infer_jsonl()
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "cache"
    output_dir.mkdir()
    output_path = tmp_path / "merged.jsonl"
    source = {
        "audio_id": "failed-audio",
        "audio_path": "/missing.wav",
        "content_type": "song",
        "stage_status": {"music_gate": "ok", "structure": "stale"},
        "stage_errors": {"structure": "stale"},
    }
    input_path.write_text(json.dumps(source) + "\n", encoding="utf-8")
    uid = module.uid_from_obj(source, 0)

    module.merge_back_to_jsonl(
        str(input_path),
        str(output_dir),
        str(output_path),
        "audio_path",
        "structure-v",
        failures={uid: "RuntimeError: injected per-track failure"},
    )

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["structure_raw"] is None
    assert result["stage_status"] == {
        "music_gate": "ok",
        "structure_raw": "error",
    }
    assert result["stage_errors"] == {
        "structure_raw": "RuntimeError: injected per-track failure"
    }
    assert result["stage_versions"]["structure_raw"] == "structure-v"
    assert not list(tmp_path.glob(".merged.jsonl.tmp.*"))


def test_interrupted_merge_keeps_previous_complete_output(tmp_path: Path):
    module = load_infer_jsonl()
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "merged.jsonl"
    cache = tmp_path / "cache"
    cache.mkdir()
    output_path.write_text('{"previous": true}\n', encoding="utf-8")
    input_path.write_text(
        json.dumps(
            {
                "audio_id": "a",
                "audio_path": "/a.wav",
                "content_type": "song",
            }
        )
        + "\n{truncated",
        encoding="utf-8",
    )

    with pytest.raises(json.JSONDecodeError):
        module.merge_back_to_jsonl(
            str(input_path),
            str(cache),
            str(output_path),
            "audio_path",
            "structure-v",
        )

    assert output_path.read_text(encoding="utf-8") == '{"previous": true}\n'
    assert not list(tmp_path.glob(".merged.jsonl.tmp.*"))


class _DeadProcess:
    pid = 123
    exitcode = 1

    @staticmethod
    def is_alive():
        return False


class _NeverUsedInputQueue:
    def __init__(self):
        self.put_calls = 0

    def put(self, _item, timeout):
        self.put_calls += 1
        raise AssertionError(f"task was admitted before workers were ready: {timeout}")


class _EmptyOutputQueue:
    @staticmethod
    def get(timeout):
        raise queue.Empty(timeout)


def test_2049_tasks_are_not_enqueued_when_worker_initialization_fails():
    module = load_infer_jsonl()
    tasks = [(f"uid-{index}", f"/{index}.wav", "song") for index in range(2049)]
    input_queue = _NeverUsedInputQueue()

    with pytest.raises(RuntimeError, match="crashed during initialization"):
        module._admit_and_dispatch(
            tasks,
            [_DeadProcess()],
            input_queue,
            _EmptyOutputQueue(),
            ready_timeout=0.1,
            poll_timeout=0.01,
            max_wait=0.1,
        )

    assert input_queue.put_calls == 0


class _AlwaysFullQueue:
    @staticmethod
    def put(_item, timeout):
        raise queue.Full(timeout)


def test_queue_put_timeout_checks_worker_liveness():
    module = load_infer_jsonl()

    with pytest.raises(RuntimeError, match="crashed during queue submission"):
        module._put_with_worker_supervision(
            _AlwaysFullQueue(),
            ("uid", "/a.wav", "song"),
            [_DeadProcess()],
            poll_timeout=0.01,
            max_wait=0.1,
        )


class _LiveProcess:
    pid = 321
    exitcode = None

    @staticmethod
    def is_alive():
        return True


def test_per_item_worker_error_is_returned_without_stage_failure():
    module = load_infer_jsonl()
    output_queue: queue.Queue = queue.Queue()
    output_queue.put(
        {
            "type": "result",
            "worker_id": 0,
            "uid": "ok",
            "ok": True,
        }
    )
    output_queue.put(
        {
            "type": "result",
            "worker_id": 0,
            "uid": "bad",
            "ok": False,
            "error": "RuntimeError: bad audio",
        }
    )

    assert module._collect_worker_results(
        [("ok", "/ok.wav", "song"), ("bad", "/bad.wav", "song")],
        [_LiveProcess()],
        output_queue,
        stall_timeout=0.1,
    ) == (1, 1, 0, {"bad": "RuntimeError: bad audio"})


class _HungProcess:
    pid = 456
    exitcode = None

    def __init__(self):
        self.alive = True
        self.terminated = False

    def is_alive(self):
        return self.alive

    def join(self, timeout):
        del timeout

    def terminate(self):
        self.terminated = True
        self.alive = False
        self.exitcode = -15


def test_worker_join_timeout_terminates_hung_process():
    module = load_infer_jsonl()
    process = _HungProcess()

    with pytest.raises(TimeoutError, match="did not exit"):
        module._join_workers([process], timeout=0.01)

    assert process.terminated

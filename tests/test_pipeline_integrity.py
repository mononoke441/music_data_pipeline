from __future__ import annotations

import importlib.util
import asyncio
import json
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, records) -> None:
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in records), encoding="utf-8"
    )


def minimal_inputs(tmp_path: Path):
    audio_id = "a1"
    section = {"section_id": "0001", "label": "A", "start": 0.0, "end": 10.0}
    values = {
        "base": [
            {
                "audio_id": audio_id,
                "audio_path": "/x.wav",
                "source_relpath": "nested/x.wav",
                "duration": 10.0,
                "content_type": "instrumental",
                "content_confidence": 0.9,
                "global_mir": {"genre": {}},
                "music_gate": {"decision": "music"},
                "voice_analysis": {},
                "stage_status": {"music_gate": "ok", "discogs_mir": "ok"},
                "model_versions": {
                    "music_gate": "gate-test",
                    "discogs_mir": "discogs-test",
                },
            }
        ],
        "cpu": [
            {
                "audio_id": audio_id,
                "music_cpu": {
                    "chords": {"values": [{"timestamp": 0.0, "chord": "C"}]},
                    "beatnet": {"beats": [{"time": 0.0, "beat_number": 1}]},
                    "key": {"key": "C", "mode": "major"},
                },
            }
        ],
        "raw": [{"audio_id": audio_id, "structure_raw": [section]}],
        "processed": [
            {
                "audio_id": audio_id,
                "sections": [section],
                "stage_status": {"structure_postprocess": "ok"},
            }
        ],
        "key": [
            {
                "audio_id": audio_id,
                "sections": [
                    {**section, "key": {"key": "C", "mode": "major"}, "status": "ok"}
                ],
                "stage_status": {"section_key": "ok"},
            }
        ],
    }
    paths = {}
    for name, records in values.items():
        paths[name] = tmp_path / f"{name}.jsonl"
        write_jsonl(paths[name], records)
    return paths


def merge_command(paths, output: Path):
    return [
        sys.executable,
        str(SCRIPTS / "dual_metadata_merge.py"),
        "--base",
        str(paths["base"]),
        "--music-cpu",
        str(paths["cpu"]),
        "--structure-raw",
        str(paths["raw"]),
        "--sections",
        str(paths["processed"]),
        "--section-key",
        str(paths["key"]),
        "--output-dir",
        str(output),
    ]


def test_disabled_stage_cannot_be_merged_and_schema_is_nullable(tmp_path: Path):
    paths = minimal_inputs(tmp_path)
    output = tmp_path / "out"
    subprocess.run(merge_command(paths, output), check=True)
    annotation_path = output / "annotations" / "nested" / "x.wav.json"
    record = json.loads(annotation_path.read_text())
    assert record["global_caption"] is None
    assert record["stage_status"]["alm"] == "not_run"
    assert record["sections"][0]["caption_status"] == "not_run"
    assert record["sections"][0]["asr_status"] == "not_applicable"
    assert not (output / "data.annotated.jsonl").exists()
    assert not (output / "accepted.jsonl").exists()
    empty_partition = tmp_path / "empty.jsonl"
    write_jsonl(empty_partition, [])
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "validate_pipeline_output.py"),
            "--base",
            str(paths["base"]),
            "--inventory",
            str(paths["base"]),
            "--annotations-dir",
            str(output / "annotations"),
            "--review",
            str(empty_partition),
            "--rejected",
            str(empty_partition),
            "--retry",
            str(empty_partition),
        ],
        check=True,
    )

    stale = tmp_path / "stale_alm.jsonl"
    write_jsonl(stale, [{"audio_id": "a1", "ALM_Caption": "stale"}])
    result = subprocess.run(
        merge_command(paths, tmp_path / "bad") + ["--alm", str(stale)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "disabled" in result.stderr


@pytest.mark.parametrize("tamper", ["content_type", "duration", "mir", "bounds"])
def test_validator_rejects_tampered_annotation_payload(tmp_path: Path, tamper: str):
    paths = minimal_inputs(tmp_path)
    output = tmp_path / "out"
    subprocess.run(merge_command(paths, output), check=True)
    annotation_path = output / "annotations" / "nested" / "x.wav.json"
    value = json.loads(annotation_path.read_text())
    if tamper == "content_type":
        value["content_type"] = "song"
    elif tamper == "duration":
        value["duration"] = 9.0
    elif tamper == "mir":
        value["global_mir"]["genre"] = {"top": "tampered"}
    else:
        value["sections"][0]["end"] = 9.0
    annotation_path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    empty_partition = tmp_path / "empty.jsonl"
    write_jsonl(empty_partition, [])
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "validate_pipeline_output.py"),
            "--base",
            str(paths["base"]),
            "--inventory",
            str(paths["base"]),
            "--annotations-dir",
            str(output / "annotations"),
            "--review",
            str(empty_partition),
            "--rejected",
            str(empty_partition),
            "--retry",
            str(empty_partition),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def test_merge_rejects_section_boundary_mismatch(tmp_path: Path):
    paths = minimal_inputs(tmp_path)
    key = json.loads(paths["key"].read_text())
    key["sections"][0]["end"] = 9.0
    write_jsonl(paths["key"], [key])
    result = subprocess.run(
        merge_command(paths, tmp_path / "out"), capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "does not match structure" in result.stderr


def test_section_cache_hash_tracks_boundaries():
    caption = load_script("section_caption_infer")
    record = {
        "audio_id": "a",
        "content_type": "song",
        "sections": [{"section_id": "1", "start": 0.0, "end": 10.0, "label": "verse"}],
    }
    before = caption.sections_hash(record)
    record["sections"][0]["end"] = 11.0
    assert caption.sections_hash(record) != before


def test_section_caption_prefetches_next_decode_while_request_is_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    caption = load_script("section_caption_infer")
    input_path = tmp_path / "sections.jsonl"
    output_path = tmp_path / "captions.jsonl"
    write_jsonl(
        input_path,
        [
            {
                "audio_id": "a",
                "audio_path": "/x.wav",
                "content_type": "song",
                "sections": [
                    {"section_id": "1", "start": 0.0, "end": 8.0, "label": "verse"},
                    {"section_id": "2", "start": 8.0, "end": 16.0, "label": "chorus"},
                ],
            }
        ],
    )

    request_started = threading.Event()
    overlap_observed = threading.Event()
    decode_calls = 0
    decode_lock = threading.Lock()

    def fake_decode(*args, **kwargs):
        nonlocal decode_calls
        with decode_lock:
            decode_calls += 1
            call_number = decode_calls
        if call_number == 2:
            assert request_started.wait(timeout=1.0)
            overlap_observed.set()
        return b"RIFF-test"

    async def fake_request(*args, **kwargs):
        request_started.set()
        await asyncio.sleep(0.05)
        return "caption"

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(caption, "decode_audio_range", fake_decode)
    monkeypatch.setattr(caption, "request_caption", fake_request)
    monkeypatch.setattr(
        caption.aiohttp, "ClientSession", lambda **kwargs: FakeSession()
    )

    args = SimpleNamespace(
        input=str(input_path),
        output=str(output_path),
        servers=["http://test"],
        model="test-model",
        api_key_env="UNSET_TEST_API_KEY",
        timeout=10,
        concurrency=1,
        decode_workers=1,
        decoded_buffer=2,
        sample_rate=24000,
        max_tokens=16,
        temperature=0.0,
        retries=0,
        resume=False,
        track_buffer=1,
    )
    asyncio.run(caption.main_async(args))

    assert overlap_observed.is_set()
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["stage_status"]["section_caption"] == "ok"
    assert [section["short_caption"] for section in result["sections"]] == [
        "caption",
        "caption",
    ]


def test_alm_failed_record_is_not_resume_cache(tmp_path: Path):
    alm = load_script("alm_caption_infer")
    output = tmp_path / "alm.jsonl"
    write_jsonl(
        output,
        [
            {
                "audio_id": "a",
                "audio_path": "/x.wav",
                "ALM_Caption": "",
                "_error": "failed",
                "model_versions": {"alm": "m"},
                "stage_status": {"alm": "error"},
            }
        ],
    )
    assert (
        alm.load_cached_records(output, "audio_id", "audio_path", "ALM_Caption") == {}
    )


def test_alm_audio_url_encodes_reserved_path_characters(tmp_path: Path):
    alm = load_script("alm_caption_infer")
    audio_path = tmp_path / "#living my best life?.mp3"

    audio_url = alm.to_audio_url(str(audio_path))

    assert audio_url == audio_path.as_uri()
    assert "%23living%20my%20best%20life%3F.mp3" in audio_url


def test_alm_resume_reuses_complete_caption_across_model_versions(tmp_path: Path):
    alm = load_script("alm_caption_infer")
    output = tmp_path / "alm.jsonl"
    cached = {
        "audio_id": "a",
        "audio_path": "/x.wav",
        "ALM_Caption": "cached caption",
        "model_versions": {"alm": "older-model"},
        "stage_status": {"alm": "ok"},
    }
    write_jsonl(output, [cached])

    assert alm.load_cached_records(output, "audio_id", "audio_path", "ALM_Caption") == {
        "a": cached
    }
    migrated = alm.reusable_cached_record(cached, dict(cached))
    assert migrated is not None
    assert migrated["alm_input_fingerprint"]
    changed = {**cached, "content_type": "instrumental"}
    assert alm.reusable_cached_record(cached, changed) is None


def test_caption_responses_reject_empty_missing_text_and_length_finish():
    for module_name in ("alm_caption_infer", "section_caption_infer"):
        module = load_script(module_name)
        with pytest.raises(module.RetryableCaptionError, match="empty"):
            module.caption_from_response(
                {"choices": [{"finish_reason": "stop", "message": {"content": ""}}]}
            )
        with pytest.raises(module.RetryableCaptionError, match="text content"):
            module.caption_from_response(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": [{"type": "audio", "value": "x"}]},
                        }
                    ]
                }
            )
        with pytest.raises(module.RetryableCaptionError, match="finish_reason=length"):
            module.caption_from_response(
                {
                    "choices": [
                        {"finish_reason": "length", "message": {"content": "partial"}}
                    ]
                }
            )


def test_section_caption_item_failure_is_terminal_without_stage_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    caption = load_script("section_caption_infer")
    input_path = tmp_path / "sections.jsonl"
    output_path = tmp_path / "captions.jsonl"
    write_jsonl(
        input_path,
        [
            {
                "audio_id": "a",
                "audio_path": "/x.wav",
                "duration": 8.0,
                "content_type": "song",
                "sections": [
                    {"section_id": "1", "start": 0.0, "end": 8.0, "label": "verse"}
                ],
            }
        ],
    )

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    async def fail_request(*args, **kwargs):
        raise caption.RetryableCaptionError("response caption is empty")

    monkeypatch.setattr(caption, "decode_audio_range", lambda *args, **kwargs: b"RIFF")
    monkeypatch.setattr(caption, "request_caption", fail_request)
    monkeypatch.setattr(
        caption.aiohttp, "ClientSession", lambda **kwargs: FakeSession()
    )
    args = SimpleNamespace(
        input=str(input_path),
        output=str(output_path),
        servers=["http://test"],
        model="test-model",
        api_key_env="UNSET_TEST_API_KEY",
        timeout=10,
        concurrency=1,
        decode_workers=1,
        decoded_buffer=1,
        sample_rate=24000,
        max_tokens=16,
        temperature=0.0,
        retries=0,
        resume=False,
        track_buffer=1,
    )
    asyncio.run(caption.main_async(args))
    value = json.loads(output_path.read_text(encoding="utf-8"))
    assert value["stage_status"] == {"section_caption": "error"}
    assert "empty" in value["stage_errors"]["section_caption"]


def test_section_caption_keeps_previous_cache_when_final_write_is_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    caption = load_script("section_caption_infer")
    input_path = tmp_path / "sections.jsonl"
    output_path = tmp_path / "captions.jsonl"
    old_bytes = b'{"audio_id":"old","sentinel":true}\n'
    output_path.write_bytes(old_bytes)
    write_jsonl(
        input_path,
        [
            {
                "audio_id": "a",
                "audio_path": "/x.wav",
                "duration": 8.0,
                "content_type": "song",
                "sections": [{"section_id": "1", "start": 0.0, "end": 8.0}],
            }
        ],
    )

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    async def ok_request(*args, **kwargs):
        return "caption"

    def interrupted_write(path, records):
        list(records)
        raise RuntimeError("simulated write interruption")

    monkeypatch.setattr(caption, "decode_audio_range", lambda *args, **kwargs: b"RIFF")
    monkeypatch.setattr(caption, "request_caption", ok_request)
    monkeypatch.setattr(caption, "write_jsonl", interrupted_write)
    monkeypatch.setattr(
        caption.aiohttp, "ClientSession", lambda **kwargs: FakeSession()
    )
    args = SimpleNamespace(
        input=str(input_path),
        output=str(output_path),
        servers=["http://test"],
        model="test-model",
        api_key_env="UNSET_TEST_API_KEY",
        timeout=10,
        concurrency=1,
        decode_workers=1,
        decoded_buffer=1,
        sample_rate=24000,
        max_tokens=16,
        temperature=0.0,
        retries=0,
        resume=False,
        track_buffer=1,
    )
    with pytest.raises(RuntimeError, match="write interruption"):
        asyncio.run(caption.main_async(args))
    assert output_path.read_bytes() == old_bytes


def test_alm_awaits_completed_tasks_and_preserves_previous_cache_on_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    alm = load_script("alm_caption_infer")
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"
    write_jsonl(
        input_path,
        [
            {
                "audio_id": "a",
                "audio_path": "/x.wav",
                "duration": 8.0,
                "content_type": "song",
            }
        ],
    )
    old_bytes = b'{"audio_id":"old","ALM_Caption":"safe"}\n'
    output_path.write_bytes(old_bytes)

    async def unexpected_failure(**kwargs):
        raise RuntimeError("unhandled task failure")

    class Progress:
        def update(self, count):
            assert count == 1

    monkeypatch.setattr(alm, "infer_one", unexpected_failure)
    args = SimpleNamespace(
        in_path=str(input_path),
        out_path=str(output_path),
        resume=False,
        resume_key="audio_id",
        audio_field="audio_path",
        output_field="ALM_Caption",
        sem=asyncio.Semaphore(1),
        server_rr_lock=asyncio.Lock(),
        rr_state={"i": 0},
        skip_existing=False,
        servers=["http://test"],
        model="test-model",
        max_tokens=16,
        temperature=0.0,
        timeout=10,
        retries=0,
        retry_base_sleep=0.0,
        task_buffer=1,
    )

    async def run():
        with pytest.raises(RuntimeError, match="unhandled task failure"):
            await alm.process_one_file(args, None, Progress())

    asyncio.run(run())
    assert output_path.read_bytes() == old_bytes


def test_structure_postprocess_item_error_is_terminal_with_zero_exit(tmp_path: Path):
    input_path = tmp_path / "raw.jsonl"
    output_path = tmp_path / "sections.jsonl"
    write_jsonl(
        input_path,
        [
            {
                "audio_id": "broken",
                "audio_path": "/broken.wav",
                "duration": 0.0,
                "content_type": "song",
                "structure_raw": [],
            }
        ],
    )
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "structure_postprocess.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    value = json.loads(output_path.read_text(encoding="utf-8"))
    assert value["stage_status"] == {"structure_postprocess": "error"}
    assert "duration" in value["stage_errors"]["structure_postprocess"]


def test_asr_batch_cardinality_mismatch_is_an_error():
    asr_module = load_script("section_asr_infer")

    class FakeASR:
        def transcribe(self, **kwargs):
            return []

    decoded = [({"target": {}}, (np.zeros(8, dtype=np.float32), 16000, 0.0))]
    with pytest.raises(asr_module.BatchCardinalityError):
        asr_module.run_batch(FakeASR(), decoded)


def test_songformer_cuda_cache_policy_matches_upstream_explicitly():
    source = (ROOT / "SongFormer" / "infer_jsonl.py").read_text(encoding="utf-8")
    helper = (ROOT / "SongFormer" / "embedding_batch.py").read_text(encoding="utf-8")

    assert 'choices=("none", "upstream")' in source
    assert 'default="upstream"' in source
    assert 'args.cuda_cache_policy == "upstream"' in source
    assert "empty_cuda_cache=args.cuda_cache_policy" in source
    assert "if empty_cuda_cache:" in helper
    assert "torch.cuda.empty_cache()" in helper

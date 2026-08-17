from __future__ import annotations

import importlib.util
import asyncio
import json
import subprocess
import sys
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
    assert record["sections"][0]["asr_status"] == "not_applicable"
    assert record["annotation_schema_version"] == "music-data-annotation-v2"
    removed = {
        "key",
        "key_status",
        "key_error",
        "short_caption",
        "caption_status",
        "caption_error",
    }
    assert removed.isdisjoint(record["sections"][0])
    assert "section_key" not in record["stage_status"]
    assert "section_caption" not in record["stage_status"]
    assert "section_key" not in record["model_versions"]
    assert "section_caption" not in record["model_versions"]
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


def test_instrumental_never_requires_or_claims_section_asr(tmp_path: Path):
    paths = minimal_inputs(tmp_path)
    empty_asr = tmp_path / "asr.jsonl"
    write_jsonl(empty_asr, [])
    output = tmp_path / "out"
    subprocess.run(
        merge_command(paths, output)
        + ["--section-asr", str(empty_asr), "--section-asr-enabled"],
        check=True,
    )
    record = json.loads(
        (output / "annotations" / "nested" / "x.wav.json").read_text()
    )
    assert record["stage_status"]["section_asr"] == "not_run"
    assert record["sections"][0]["asr_status"] == "not_applicable"

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
            "--section-asr-enabled",
        ],
        check=True,
    )


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


def test_merge_strips_deprecated_section_metadata_without_model_rerun(tmp_path: Path):
    paths = minimal_inputs(tmp_path)
    processed = json.loads(paths["processed"].read_text())
    processed["sections"][0].update(
        {
            "key": {"key": "C", "mode": "major"},
            "key_status": "ok",
            "key_error": None,
            "short_caption": "obsolete",
            "caption_status": "ok",
            "caption_error": None,
        }
    )
    write_jsonl(paths["processed"], [processed])
    output = tmp_path / "out"
    subprocess.run(merge_command(paths, output), check=True)
    record = json.loads((output / "annotations" / "nested" / "x.wav.json").read_text())
    assert {
        "key",
        "key_status",
        "key_error",
        "short_caption",
        "caption_status",
        "caption_error",
    }.isdisjoint(record["sections"][0])


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
    module = load_script("alm_caption_infer")
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
    runner = (ROOT / "run_pipeline.sh").read_text(encoding="utf-8")
    service = (ROOT / "scripts" / "serve_songformer.py").read_text(encoding="utf-8")

    assert 'choices=("none", "upstream")' in source
    assert 'default="upstream"' in source
    assert 'args.cuda_cache_policy == "upstream"' in source
    assert "empty_cuda_cache=args.cuda_cache_policy" in source
    assert "if empty_cuda_cache:" in helper
    assert "torch.cuda.empty_cache()" in helper
    assert "scripts/service_batch_infer.py" in runner
    assert "SongFormer/infer_jsonl.py" not in runner
    assert "empty_cuda_cache=False" in service

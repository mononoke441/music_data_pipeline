from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import annotation_storage
import calc_duration


def record(source_relpath: str, audio_id: str = "id-1", content_type: str = "song"):
    return {
        "audio_id": audio_id,
        "audio_path": f"/input/{source_relpath}",
        "source_relpath": source_relpath,
        "content_type": content_type,
        "sections": [],
    }


def test_relative_layout_preserves_directories_unicode_and_audio_extension(tmp_path: Path):
    destination = tmp_path / "final" / "annotations"
    values = [
        record("中文 目录/歌曲(现场).mp3", "one"),
        record("中文 目录/歌曲(现场).flac", "two", "instrumental"),
        record("另一目录/歌曲(现场).mp3", "three"),
    ]
    counts = annotation_storage.publish_annotation_records(values, destination)
    assert counts == {"total": 3, "song": 2, "instrumental": 1}
    assert (destination / "中文 目录" / "歌曲(现场).mp3.json").is_file()
    assert (destination / "中文 目录" / "歌曲(现场).flac.json").is_file()
    assert (destination / "另一目录" / "歌曲(现场).mp3.json").is_file()


def test_long_filename_is_deterministic_and_within_filesystem_limit():
    source = "目录/" + ("很长的歌曲名" * 40) + ".mp3"
    first = annotation_storage.annotation_relative_path(source)
    second = annotation_storage.annotation_relative_path(source)
    assert first == second
    assert len(first.name.encode("utf-8")) <= 255
    assert first.name.endswith(".json")


def test_unsafe_relative_paths_are_rejected(tmp_path: Path):
    with pytest.raises(ValueError):
        annotation_storage.publish_annotation_records(
            [record("../outside.mp3")], tmp_path / "annotations"
        )


def test_publish_failure_keeps_previous_complete_tree(tmp_path: Path, monkeypatch):
    destination = tmp_path / "annotations"
    annotation_storage.publish_annotation_records([record("old.mp3", "old")], destination)
    original = annotation_storage.atomic_write_json

    def fail_on_new(path, value):
        if value["audio_id"] == "new":
            raise RuntimeError("injected write failure")
        original(path, value)

    monkeypatch.setattr(annotation_storage, "atomic_write_json", fail_on_new)
    with pytest.raises(RuntimeError, match="injected"):
        annotation_storage.publish_annotation_records([record("new.mp3", "new")], destination)
    assert (destination / "old.mp3.json").is_file()
    assert not (destination / "new.mp3.json").exists()
    assert not (tmp_path / ".annotations.staging").exists()


def test_empty_publish_replaces_old_tree_with_empty_directory(tmp_path: Path):
    destination = tmp_path / "annotations"
    annotation_storage.publish_annotation_records([record("old.mp3")], destination)
    annotation_storage.publish_annotation_records([], destination)
    assert destination.is_dir()
    assert list(destination.rglob("*.json")) == []


def test_path_query_derives_file_without_hashing_or_scanning(tmp_path: Path):
    input_root = tmp_path / "input"
    audio = input_root / "专辑" / "歌.mp3"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"audio bytes are never read by the query")
    result = tmp_path / "result"
    destination = result / "final" / "annotations"
    value = record("专辑/歌.mp3")
    value["audio_path"] = str(audio)
    annotation_storage.publish_annotation_records([value], destination)

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "find_annotation.py"),
            "--input-root", str(input_root),
            "--result-dir", str(result),
            "--audio", str(audio),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout)["source_relpath"] == "专辑/歌.mp3"
    query_source = (SCRIPTS / "find_annotation.py").read_text(encoding="utf-8")
    assert "stable_audio_id" not in query_source
    assert "sha256" not in query_source.lower()


def test_inventory_scan_preserves_input_symlink_path(tmp_path: Path):
    input_root = tmp_path / "input"
    input_root.mkdir()
    target = tmp_path / "outside.mp3"
    target.write_bytes(b"audio")
    link = input_root / "linked.mp3"
    link.symlink_to(target)
    assert calc_duration.scan_media(input_root) == [str(link.absolute())]
    assert annotation_storage.source_relpath_for_audio(link, input_root) == "linked.mp3"


def test_inventory_worker_reuses_precomputed_content_hash(tmp_path, monkeypatch):
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"not decoded in this unit test")
    expected_id = "a" * 64

    def unexpected_hash(*_args, **_kwargs):
        raise AssertionError("inventory worker must not hash the canonical file twice")

    monkeypatch.setattr(calc_duration, "stable_audio_id", unexpected_hash)
    monkeypatch.setattr(calc_duration, "ffprobe_duration_seconds", lambda _path: 12.5)
    monkeypatch.setattr(calc_duration, "decode_probe", lambda _path, _seconds: (True, ""))

    record = calc_duration.inventory_one(
        (str(audio), str(tmp_path), 1.0, "inventory-test", expected_id)
    )

    assert record["audio_id"] == expected_id
    assert record["source_relpath"] == "song.wav"
    assert record["decode_status"] == "ok"


def test_inventory_content_hash_cache_reuses_unchanged_file_and_invalidates_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"first-audio-bytes")
    path = str(audio)

    first_ids, cache_records, reused = calc_duration.resolve_content_ids(
        [path], str(tmp_path), {}, hash_jobs=2
    )
    assert reused == 0
    assert first_ids[path] == calc_duration.stable_audio_id(path, str(tmp_path))

    cache_path = tmp_path / "data.hash_cache.jsonl"
    calc_duration.write_hash_cache(cache_path, cache_records)
    loaded = calc_duration.load_hash_cache(cache_path)

    def unexpected_hash(*_args, **_kwargs):
        raise AssertionError("unchanged source must reuse its cached content SHA")

    original_hash = calc_duration.stable_audio_id
    monkeypatch.setattr(calc_duration, "stable_audio_id", unexpected_hash)
    second_ids, _, reused = calc_duration.resolve_content_ids(
        [path], str(tmp_path), loaded, hash_jobs=2
    )
    assert reused == 1
    assert second_ids == first_ids

    monkeypatch.setattr(calc_duration, "stable_audio_id", original_hash)
    audio.write_bytes(b"second-audio-data")
    third_ids, _, reused = calc_duration.resolve_content_ids(
        [path], str(tmp_path), loaded, hash_jobs=2
    )
    assert reused == 0
    assert third_ids[path] != first_ids[path]


def test_inventory_cache_reuse_ignores_provenance_versions():
    record = {
        "audio_id": "a" * 64,
        "decode_status": "ok",
        "error": None,
        "pipeline_version": "older-pipeline",
        "stage_versions": {
            "inventory": "inventory-v2:probe=1.000:code=oldcode12345"
        },
    }
    assert calc_duration.reusable_inventory_record(record, "a" * 64, False)
    assert not calc_duration.reusable_inventory_record(record, "b" * 64, False)


def test_unchanged_inventory_failure_is_cached_unless_retry_requested():
    record = {
        "audio_id": "a" * 64,
        "decode_status": "failed",
        "error": "duration_probe_failed",
        "pipeline_version": calc_duration.PIPELINE_VERSION,
        "stage_versions": {
            "inventory": "inventory-v2:probe=1.000:code=oldcode12345"
        },
    }
    assert calc_duration.reusable_inventory_record(record, "a" * 64, False)
    assert not calc_duration.reusable_inventory_record(record, "a" * 64, True)


def test_resumed_duplicate_uses_new_canonical_relative_path(tmp_path: Path):
    input_root = tmp_path / "input"
    input_root.mkdir()
    earlier = input_root / "a.mp3"
    previous = input_root / "b.mp3"
    cached = {
        "audio_id": "same-content",
        "audio_path": str(previous),
        "source_relpath": "b.mp3",
    }
    updated = calc_duration.attach_duplicate_metadata(
        cached, [str(earlier), str(previous)], input_root
    )
    assert updated["audio_path"] == str(earlier)
    assert updated["source_relpath"] == "a.mp3"
    assert updated["duplicate_paths"] == [str(previous)]


def test_legacy_jsonl_converter_matches_per_track_layout(tmp_path: Path):
    input_root = tmp_path / "input"
    audio = input_root / "album" / "song.mp3"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"x")
    legacy = tmp_path / "data.annotated.jsonl"
    legacy.write_text(
        json.dumps({
            "audio_id": "legacy",
            "audio_path": str(audio),
            "content_type": "song",
            "sections": [],
        }) + "\n",
        encoding="utf-8",
    )
    result = tmp_path / "result"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPTS / "split_annotations.py"),
            "--input-jsonl", str(legacy),
            "--input-root", str(input_root),
            "--result-dir", str(result),
        ],
        check=True,
    )
    converted = result / "final" / "annotations" / "album" / "song.mp3.json"
    assert json.loads(converted.read_text())["source_relpath"] == "album/song.mp3"

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import alm_caption_infer  # noqa: E402
from pipeline_core import iter_jsonl, write_jsonl  # noqa: E402


def _run_main(monkeypatch, argv: list[str], main) -> None:
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 0


def test_alm_preflight_full_hit_rewrites_in_order_and_never_opens_session(
    tmp_path, monkeypatch, capsys
):
    input_path = tmp_path / "songs.jsonl"
    out_dir = tmp_path / "out"
    output_path = out_dir / "songs.alm.jsonl"
    cached_input = {
        "audio_id": "cached",
        "audio_path": "/cached.wav",
        "content_type": "song",
    }
    missing_input = {"audio_id": "missing", "content_type": "song"}
    write_jsonl(input_path, [cached_input, missing_input])
    write_jsonl(
        output_path,
        [
            {
                **cached_input,
                "ALM_Caption": "A cached caption.",
                "model_versions": {"alm": "old-model"},
            }
        ],
    )

    class ForbiddenSession:
        def __init__(self, *args, **kwargs):
            raise AssertionError("cache preflight must not construct an HTTP session")

    monkeypatch.setattr(alm_caption_infer.aiohttp, "ClientSession", ForbiddenSession)
    _run_main(
        monkeypatch,
        [
            "alm_caption_infer.py",
            "--inputs",
            str(input_path),
            "--out_dir",
            str(out_dir),
            "--model",
            "current-model",
            "--cache-preflight",
        ],
        alm_caption_infer.main,
    )

    captured = capsys.readouterr()
    assert captured.out == "0\n"
    assert captured.err == ""
    records = list(iter_jsonl(output_path))
    assert [record["audio_id"] for record in records] == ["cached", "missing"]
    assert records[0]["stage_status"]["alm"] == "ok"
    assert records[0]["alm_input_fingerprint"] == alm_caption_infer.alm_input_fingerprint(
        cached_input
    )
    assert records[1]["stage_status"]["alm"] == "error"
    assert records[1]["stage_errors"]["alm"] == "missing_audio_path"


def test_alm_preflight_cache_miss_preserves_output_bytes(tmp_path, monkeypatch, capsys):
    input_path = tmp_path / "songs.jsonl"
    out_dir = tmp_path / "out"
    output_path = out_dir / "songs.alm.jsonl"
    write_jsonl(
        input_path,
        [{"audio_id": "new", "audio_path": "/new.wav", "content_type": "song"}],
    )
    output_path.parent.mkdir(parents=True)
    original = b'{"audio_id":"old","ALM_Caption":"keep me"}\n'
    output_path.write_bytes(original)

    _run_main(
        monkeypatch,
        [
            "alm_caption_infer.py",
            "--inputs",
            str(input_path),
            "--out_dir",
            str(out_dir),
            "--model",
            "model",
            "--cache-preflight",
        ],
        alm_caption_infer.main,
    )
    captured = capsys.readouterr()
    assert captured.out == "1\n"
    assert captured.err == ""
    assert output_path.read_bytes() == original


def test_alm_preflight_rejects_duplicate_and_malformed_cache(tmp_path, monkeypatch):
    input_path = tmp_path / "songs.jsonl"
    out_dir = tmp_path / "out"
    output_path = out_dir / "songs.alm.jsonl"
    write_jsonl(input_path, [{"audio_id": "a", "audio_path": "/a.wav"}])
    output_path.parent.mkdir(parents=True)
    duplicate = {"audio_id": "a", "audio_path": "/a.wav", "ALM_Caption": "x"}
    write_jsonl(output_path, [duplicate, duplicate])
    argv = [
        "alm_caption_infer.py",
        "--inputs",
        str(input_path),
        "--out_dir",
        str(out_dir),
        "--model",
        "model",
        "--cache-preflight",
    ]
    main = alm_caption_infer.main
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="duplicate"):
        main()

    output_path.write_text(json.dumps({"audio_id": "a"}) + "\n{broken\n")
    with pytest.raises(json.JSONDecodeError):
        main()

from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_progress_remains_visible_when_stderr_is_captured():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "scripts")
    env["PIPELINE_PROGRESS"] = "1"
    env["PIPELINE_PROGRESS_MIN_INTERVAL"] = "0.1"
    result = subprocess.run(
        [
            "python",
            "-c",
            (
                "from pipeline_progress import pipeline_tqdm; "
                "list(pipeline_tqdm(range(2), total=2, desc='test-stage', unit='row'))"
            ),
        ],
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "test-stage" in result.stderr
    assert "2/2" in result.stderr


def test_completed_cache_progress_is_printed_once():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "scripts")
    result = subprocess.run(
        [
            "python",
            "-c",
            (
                "from pipeline_progress import pipeline_tqdm; "
                "p=pipeline_tqdm(total=2, initial=2, desc='cached-stage', unit='row'); "
                "p.close()"
            ),
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr.count("cached-stage") == 1
    assert "2/2 [cached]" in result.stderr


def test_every_numbered_stage_has_a_progress_reporter():
    expected = {
        "0/7": ROOT / "scripts" / "calc_duration.py",
        "1a/7": ROOT / "scripts" / "fast_music_gate.py",
        "1b/7": ROOT / "scripts" / "discogs_mir_infer.py",
        "2/7 CPU MIR": ROOT / "MusicToolsPipeline" / "ray_inference.py",
        "2/7 structure": ROOT / "SongFormer" / "infer_jsonl.py",
        "3/7": ROOT / "scripts" / "alm_caption_infer.py",
        "4/7": ROOT / "scripts" / "structure_postprocess.py",
        "5/7 section key": ROOT / "scripts" / "section_key_infer.py",
        "5/7 section caption": ROOT / "scripts" / "section_caption_infer.py",
        "6/7": ROOT / "scripts" / "section_asr_infer.py",
        "7/7 metadata": ROOT / "scripts" / "dual_metadata_merge.py",
        "7/7 strict": ROOT / "scripts" / "validate_pipeline_output.py",
    }
    for label, path in expected.items():
        source = path.read_text(encoding="utf-8")
        assert label in source, f"missing progress label {label!r} in {path}"


def test_runner_defaults_to_quiet_throttled_progress():
    source = (ROOT / "run_pipeline.sh").read_text(encoding="utf-8")
    assert 'PIPELINE_PROGRESS="${PIPELINE_PROGRESS:-1}"' in source
    assert 'PIPELINE_QUIET_LOGS="${PIPELINE_QUIET_LOGS:-1}"' in source
    assert 'PIPELINE_PROGRESS_MIN_INTERVAL="${PIPELINE_PROGRESS_MIN_INTERVAL:-2.0}"' in source
    assert "--cfg num_dataloader_workers=1" in source
    assert '--processed "$processed"' in source
    assert source.count("--human-readable") >= 2
    assert 'echo "timings:      printed after each completed stage;' in source
    assert '--rejected-count "$rejected_count"' in source
    assert 'pipeline_runtime_metrics.py" "${runtime_args[@]}"' in source
    assert "VLLM_LOGGING_LEVEL" in source
    assert 'tee "$LOG_DIR/pipeline.log"' in source
    assert 'tee -a "$LOG_DIR/pipeline.log"' not in source


def test_songformer_keeps_only_the_inference_progress_bar():
    source = (ROOT / "SongFormer" / "infer_jsonl.py").read_text(encoding="utf-8")
    assert 'desc="scan jsonl -> tasks"' not in source
    assert 'desc="enqueue tasks"' not in source
    assert 'desc="merge back to jsonl"' not in source
    assert 'desc="2/7 structure inference"' in source

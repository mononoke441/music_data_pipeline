from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_progress_remains_visible_when_stderr_is_captured():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "scripts")
    env["PIPELINE_PROGRESS"] = "1"
    env["PIPELINE_PROGRESS_MIN_INTERVAL"] = "0.1"
    result = subprocess.run(
        [
            sys.executable,
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
            sys.executable,
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
    runner = (ROOT / "run_pipeline.sh").read_text(encoding="utf-8")
    for label in ("[0/7]", "[1a/7]", "[1b/7]", "[2-3/7]", "[4/7]", "[5-6/7]", "[7/7]"):
        assert label in runner
    assert "desc=\"7/7 metadata merge\"" in (
        ROOT / "scripts" / "dual_metadata_merge.py"
    ).read_text(encoding="utf-8")
    assert "desc=\"7/7 strict validation\"" in (
        ROOT / "scripts" / "validate_pipeline_output.py"
    ).read_text(encoding="utf-8")


def test_runner_defaults_to_quiet_throttled_progress():
    source = (ROOT / "run_pipeline.sh").read_text(encoding="utf-8")
    assert 'PIPELINE_PROGRESS="${PIPELINE_PROGRESS:-1}"' in source
    assert 'PIPELINE_QUIET_LOGS="${PIPELINE_QUIET_LOGS:-1}"' in source
    assert 'PIPELINE_PROGRESS_MIN_INTERVAL="${PIPELINE_PROGRESS_MIN_INTERVAL:-2.0}"' in source
    assert "scripts/pipeline_runtime_metrics.py" in source
    assert "--processed" in source
    assert "--human-readable" in source
    assert 'tee "$LOG_DIR/pipeline.log"' in source
    assert 'tee -a "$LOG_DIR/pipeline.log"' not in source


def test_songformer_keeps_only_the_inference_progress_bar():
    source = (ROOT / "SongFormer" / "infer_jsonl.py").read_text(encoding="utf-8")
    assert 'desc="scan jsonl -> tasks"' not in source
    assert 'desc="enqueue tasks"' not in source
    assert 'desc="merge back to jsonl"' not in source
    assert 'desc="2/7 structure inference"' in source

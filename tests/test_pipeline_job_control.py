from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOB_CONTROL = ROOT / "scripts" / "pipeline_job_control.sh"
RUNNER = ROOT / "run_pipeline.sh"


def run_bash(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script],
        text=True,
        capture_output=True,
    )


def test_named_job_can_be_awaited_without_barriering_other_job():
    result = run_bash(f'''
set -Eeuo pipefail
source "{JOB_CONTROL}"
pipeline_pids=()
pipeline_names=()
(sleep 0.02) & first=$!
(sleep 0.20) & second=$!
add_pipeline_job "$first" first
add_pipeline_job "$second" second
wait_pipeline_job "$first"
[[ "${{#pipeline_pids[@]}}" == 1 ]]
[[ "${{pipeline_pids[0]}}" == "$second" ]]
kill -0 "$second"
wait_named_jobs
[[ "${{#pipeline_pids[@]}}" == 0 ]]
''')
    assert result.returncode == 0, result.stderr


def test_failed_named_job_is_removed_and_failure_propagates():
    result = run_bash(f'''
set -Eeuo pipefail
source "{JOB_CONTROL}"
pipeline_pids=()
pipeline_names=()
(exit 7) & failed=$!
add_pipeline_job "$failed" failed-job
if wait_pipeline_job "$failed"; then
    exit 9
fi
[[ "${{#pipeline_pids[@]}}" == 0 ]]
''')
    assert result.returncode == 0, result.stderr
    assert "[FAIL] failed-job" in result.stderr


def test_guarded_job_stops_promptly_when_service_dies():
    result = run_bash(f'''
set -Eeuo pipefail
source "{JOB_CONTROL}"
terminate_tree() {{ kill -TERM "$1" 2>/dev/null || true; }}
healthcheck_calls=0
fake_healthcheck() {{
    healthcheck_calls=$((healthcheck_calls + 1))
    (( healthcheck_calls < 2 ))
}}
if run_guarded_job fake_healthcheck fake-service 0.01 3 bash -c 'sleep 5'; then
    exit 9
else
    status=$?
fi
[[ "$status" == 125 ]]
''')
    assert result.returncode == 0, result.stderr
    assert "fake-service failed 3 consecutive checks" in result.stderr


def test_guarded_job_preserves_success_status():
    result = run_bash(f'''
set -Eeuo pipefail
source "{JOB_CONTROL}"
terminate_tree() {{ kill -TERM "$1" 2>/dev/null || true; }}
always_healthy() {{ return 0; }}
run_guarded_job always_healthy fake-service 0.01 3 bash -c 'sleep 0.02'
''')
    assert result.returncode == 0, result.stderr


def test_guarded_job_tolerates_transient_healthcheck_failures():
    result = run_bash(f'''
set -Eeuo pipefail
source "{JOB_CONTROL}"
terminate_tree() {{ kill -TERM "$1" 2>/dev/null || true; }}
healthcheck_calls=0
transient_healthcheck() {{
    healthcheck_calls=$((healthcheck_calls + 1))
    (( healthcheck_calls == 1 || healthcheck_calls >= 4 ))
}}
run_guarded_job transient_healthcheck fake-service 0.01 3 bash -c 'sleep 0.08'
''')
    assert result.returncode == 0, result.stderr
    assert "health check failed (2/3); retrying" in result.stderr


def test_batch_runner_only_orchestrates_resident_services():
    source = RUNNER.read_text(encoding="utf-8")
    assert "scripts/service_healthcheck.py" in source
    assert "scripts/service_batch_infer.py" in source
    assert "model_service_manager.py" not in source
    assert "manage_model_services.sh" not in source
    assert "pipeline_job_control.sh" not in source
    assert "start_local_alm" not in source
    assert "stop_local_alm" not in source


def test_runner_shell_syntax():
    result = subprocess.run(["bash", "-n", str(RUNNER)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

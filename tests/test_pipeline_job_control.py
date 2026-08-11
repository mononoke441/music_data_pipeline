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


def test_local_runner_encodes_safe_gpu_handoffs():
    source = RUNNER.read_text(encoding="utf-8")
    assert 'ALM_CONCURRENCY="${ALM_CONCURRENCY:-2}"' in source
    assert 'OMNI_MAX_NUM_SEQS="${OMNI_MAX_NUM_SEQS:-2}"' in source
    assert 'OMNI_GPU_MEMORY_UTILIZATION="${OMNI_GPU_MEMORY_UTILIZATION:-0.90}"' in source
    assert 'ALM_GUARD_MAX_FAILURES="${ALM_GUARD_MAX_FAILURES:-3}"' in source
    assert "if (requested < selected) selected = requested" in source
    assert "run_with_alm_guard() {" in source
    assert 'run_with_alm_guard "$PY_PIPELINE" "$PIPELINE_ROOT/scripts/alm_caption_infer.py"' in source
    assert 'run_with_alm_guard "$PY_PIPELINE" "$PIPELINE_ROOT/scripts/section_caption_infer.py"' in source
    local_guard = source[
        source.index("local_alm_is_alive() {") : source.index("run_with_alm_guard() {")
    ]
    owned_service_branch = local_guard[: local_guard.index("    fi")]
    assert 'pipeline_process_is_running "$alm_pid"' in owned_service_branch
    assert "alm_is_ready" not in owned_service_branch
    step2 = source[source.index('echo "[2/7]'):source.index('echo "[4/7]')]
    assert step2.index('wait_pipeline_job "$structure_job_pid"') < step2.index(
        "start_local_alm"
    )

    step5 = source[source.index('echo "[5/7]'):source.index('echo "[6/7]')]
    caption_done = step5.index('wait_pipeline_job "$section_caption_job_pid"')
    omni_stopped = step5.index("stop_local_alm", caption_done)
    asr_started = step5.index("run_section_asr &", omni_stopped)
    assert caption_done < omni_stopped < asr_started


def test_runner_shell_syntax():
    result = subprocess.run(["bash", "-n", str(RUNNER)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

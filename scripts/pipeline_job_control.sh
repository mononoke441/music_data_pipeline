#!/usr/bin/env bash
# This file is sourced by run_pipeline.sh; it is not executed directly.

add_pipeline_job() {
    local pid="$1" name="$2"
    pipeline_pids+=("$pid")
    pipeline_names+=("$name")
}

pipeline_process_is_running() {
    local pid="$1" process_state=""
    if ! process_state="$(ps -o stat= -p "$pid" 2>&1)"; then
        return 1
    fi
    process_state="${process_state//[[:space:]]/}"
    [[ -n "$process_state" && "${process_state:0:1}" != Z ]]
}

# Run a workload while a required service stays healthy.  Exit 125 is reserved
# for guard failure so callers can distinguish a dead service from a workload
# error.  terminate_tree is supplied by the sourcing runner.
run_guarded_job() {
    local healthcheck_fn="$1" guard_name="$2" poll_seconds="$3"
    local max_consecutive_failures="$4" workload_pid status=0
    local consecutive_failures=0
    shift 4

    if [[ ! "$max_consecutive_failures" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] guard failure limit must be a positive integer" >&2
        return 2
    fi

    "$@" &
    workload_pid="$!"
    while pipeline_process_is_running "$workload_pid"; do
        if ! "$healthcheck_fn"; then
            # Do not turn one delayed HTTP response or scheduler hiccup into a
            # destructive service failure. The guarded workload may also have
            # completed while the health check was in flight.
            if ! pipeline_process_is_running "$workload_pid"; then
                break
            fi
            consecutive_failures=$((consecutive_failures + 1))
            if (( consecutive_failures >= max_consecutive_failures )); then
                echo "[ERROR] $guard_name failed ${consecutive_failures} consecutive checks; terminating guarded job pid=$workload_pid" >&2
                terminate_tree "$workload_pid"
                wait "$workload_pid" 2>/dev/null || true
                return 125
            fi
            echo "[WARN] $guard_name health check failed (${consecutive_failures}/${max_consecutive_failures}); retrying" >&2
        else
            consecutive_failures=0
        fi
        sleep "$poll_seconds"
    done

    if wait "$workload_pid"; then
        return 0
    else
        status=$?
    fi
    return "$status"
}

wait_pipeline_job() {
    local target_pid="$1" index=0 found=-1 name="" status=0
    local -a old_pids=() old_names=()
    for index in "${!pipeline_pids[@]}"; do
        if [[ "${pipeline_pids[$index]}" == "$target_pid" ]]; then
            found="$index"
            name="${pipeline_names[$index]}"
            break
        fi
    done
    if (( found < 0 )); then
        echo "[ERROR] unknown pipeline job pid=$target_pid" >&2
        return 2
    fi

    if wait "$target_pid"; then
        echo "[OK] $name"
    else
        echo "[FAIL] $name" >&2
        status=1
    fi

    old_pids=("${pipeline_pids[@]}")
    old_names=("${pipeline_names[@]}")
    pipeline_pids=()
    pipeline_names=()
    for index in "${!old_pids[@]}"; do
        if (( index != found )); then
            pipeline_pids+=("${old_pids[$index]}")
            pipeline_names+=("${old_names[$index]}")
        fi
    done
    return "$status"
}

wait_named_jobs() {
    local failed=0 target_pid=""
    while (( ${#pipeline_pids[@]} > 0 )); do
        target_pid="${pipeline_pids[0]}"
        if ! wait_pipeline_job "$target_pid"; then
            failed=1
        fi
    done
    return "$failed"
}

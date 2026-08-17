#!/usr/bin/env bash
# Stage-barrier runner backed exclusively by resident inference services.

set -Eeuo pipefail

PIPELINE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY_PIPELINE="${PY_PIPELINE:-/mnt/data/yuyin/user_workspace/liuhongjia/conda_envs/moss-music-pipeline/bin/python}"

FAST_GATE_SERVICE_URL="${FAST_GATE_SERVICE_URL:-http://127.0.0.1:18101}"
DISCOGS_MIR_SERVICE_URL="${DISCOGS_MIR_SERVICE_URL:-http://127.0.0.1:18102}"
MUSIC_CPU_SERVICE_URL="${MUSIC_CPU_SERVICE_URL:-http://127.0.0.1:18103}"
STRUCTURE_RAW_SERVICE_URL="${STRUCTURE_RAW_SERVICE_URL:-http://127.0.0.1:10101}"
SECTION_ASR_SERVICE_URL="${SECTION_ASR_SERVICE_URL:-http://127.0.0.1:10102}"
ALM_SERVICE_URL="${ALM_SERVICE_URL:-http://127.0.0.1:10103}"

RUN_ALM="${RUN_ALM:-1}"
RUN_ASR="${RUN_ASR:-1}"
INVENTORY_JOBS="${INVENTORY_JOBS:-32}"
INVENTORY_HASH_JOBS="${INVENTORY_HASH_JOBS:-8}"
SERVICE_REQUEST_CONCURRENCY="${SERVICE_REQUEST_CONCURRENCY:-64}"
SERVICE_REQUEST_TIMEOUT="${SERVICE_REQUEST_TIMEOUT:-1800}"
SERVICE_REQUEST_RETRIES="${SERVICE_REQUEST_RETRIES:-3}"
PIPELINE_PROGRESS="${PIPELINE_PROGRESS:-1}"
PIPELINE_PROGRESS_MIN_INTERVAL="${PIPELINE_PROGRESS_MIN_INTERVAL:-2.0}"
PIPELINE_QUIET_LOGS="${PIPELINE_QUIET_LOGS:-1}"

SNAP_TOLERANCE="${SNAP_TOLERANCE:-1.5}"
DUPLICATE_TOLERANCE="${DUPLICATE_TOLERANCE:-2.0}"
MINIMUM_SECTION_DURATION="${MINIMUM_SECTION_DURATION:-8.0}"
EXTREMELY_SHORT_DURATION="${EXTREMELY_SHORT_DURATION:-2.0}"
SHORT_BOUNDARY_CONFIDENCE="${SHORT_BOUNDARY_CONFIDENCE:-0.65}"

if (( $# != 2 )); then
    echo "Usage: bash run_pipeline.sh INPUT_DIR RESULT_DIR" >&2
    exit 2
fi
for name in RUN_ALM RUN_ASR PIPELINE_PROGRESS PIPELINE_QUIET_LOGS; do
    value="${!name}"
    if [[ "$value" != 0 && "$value" != 1 ]]; then
        echo "[ERROR] $name must be 0 or 1, got $value" >&2
        exit 2
    fi
done
if [[ ! "$SERVICE_REQUEST_CONCURRENCY" =~ ^[1-9][0-9]*$ \
    || ! "$SERVICE_REQUEST_RETRIES" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] invalid service concurrency/retry configuration" >&2
    exit 2
fi
if [[ ! -x "$PY_PIPELINE" ]]; then
    echo "[ERROR] PY_PIPELINE is not executable: $PY_PIPELINE" >&2
    exit 1
fi

INPUT_DIR="$(cd -- "$1" && pwd)"
mkdir -p "$2"
RESULT_DIR="$(cd -- "$2" && pwd)"
FINAL_DIR="$RESULT_DIR/final"
INTERMEDIATE_DIR="$RESULT_DIR/intermediate"
ACTIVE_DIR="$INTERMEDIATE_DIR/active"
INVENTORY_DIR="$INTERMEDIATE_DIR/inventory"
ROUTING_DIR="$INTERMEDIATE_DIR/routing"
FAST_GATE_DIR="$ROUTING_DIR/fast-gate"
DISCOGS_DIR="$ROUTING_DIR/discogs"
GLOBAL_DIR="$INTERMEDIATE_DIR/global"
ALM_DIR="$GLOBAL_DIR/alm"
MUSIC_CPU_DIR="$GLOBAL_DIR/music-cpu"
SECTIONS_DIR="$INTERMEDIATE_DIR/sections"
LOG_DIR="$INTERMEDIATE_DIR/logs"
STAGE_TIMINGS_FILE="$LOG_DIR/stage_timings.jsonl"
PIPELINE_RUNTIME_FILE="$LOG_DIR/pipeline_runtime.json"
PIPELINE_STARTED_AT="$(date +%s.%N)"
CURRENT_STAGE="initialization"
pipeline_finalized=0
ACTIVE_GLOBAL_MANIFEST="$ACTIVE_DIR/accepted.global.jsonl"
ACTIVE_SECTIONS_MANIFEST="$ACTIVE_DIR/accepted.sections.jsonl"
ACTIVE_FINAL_MANIFEST="$ACTIVE_DIR/accepted.final.jsonl"
PUBLISH_BASE_MANIFEST="$ACTIVE_DIR/accepted.publishable.jsonl"
RETRY_GLOBAL_FILE="$ACTIVE_DIR/retry.global.jsonl"
RETRY_SECTIONS_FILE="$ACTIVE_DIR/retry.sections.jsonl"
RETRY_FINAL_FILE="$ACTIVE_DIR/retry.final.jsonl"
RETRY_PATH_FILE="$ACTIVE_DIR/retry.annotation_path.jsonl"

mkdir -p "$FINAL_DIR" "$ACTIVE_DIR" "$INVENTORY_DIR" "$FAST_GATE_DIR" \
    "$DISCOGS_DIR" "$ALM_DIR" "$MUSIC_CPU_DIR" "$SECTIONS_DIR" "$LOG_DIR"
: > "$STAGE_TIMINGS_FILE"
for path in "$RETRY_GLOBAL_FILE" "$RETRY_SECTIONS_FILE" "$RETRY_FINAL_FILE" "$RETRY_PATH_FILE"; do
    : > "$path"
done
exec > >(tee "$LOG_DIR/pipeline.log") 2>&1

export PIPELINE_PROGRESS PIPELINE_PROGRESS_MIN_INTERVAL PIPELINE_QUIET_LOGS
if [[ "$PIPELINE_QUIET_LOGS" == 1 ]]; then
    export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
    export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
    export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
    export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
fi

runtime_now() { date +%s.%N; }
jsonl_count() {
    if [[ -s "$1" ]]; then awk 'NF { n += 1 } END { print n + 0 }' "$1"; else echo 0; fi
}
record_stage_runtime() {
    "$PY_PIPELINE" "$PIPELINE_ROOT/scripts/pipeline_runtime_metrics.py" record \
        --output "$STAGE_TIMINGS_FILE" --stage "$1" --started-at "$2" \
        --processed "${3:-0}" --human-readable
}
finalize_runtime() {
    local status="$1" exit_code="$2" annotation_count=0
    local -a failure_args=()
    if [[ "$status" == "failed" ]]; then
        failure_args=(--failure-stage "$CURRENT_STAGE")
    fi
    if [[ -d "$FINAL_DIR/annotations" ]]; then
        annotation_count="$(find "$FINAL_DIR/annotations" -type f -name '*.json' | wc -l | tr -d ' ')"
    fi
    "$PY_PIPELINE" "$PIPELINE_ROOT/scripts/pipeline_runtime_metrics.py" finalize \
        --stages "$STAGE_TIMINGS_FILE" --output "$PIPELINE_RUNTIME_FILE" \
        --started-at "$PIPELINE_STARTED_AT" \
        --input-count "$(jsonl_count "$INVENTORY_DIR/data.jsonl")" \
        --accepted-count "$(jsonl_count "$ROUTING_DIR/accepted.input.jsonl")" \
        --annotation-count "$annotation_count" \
        --review-count "$(jsonl_count "$FINAL_DIR/review.jsonl")" \
        --rejected-count "$(jsonl_count "$FINAL_DIR/rejected.jsonl")" \
        --retry-count "$(jsonl_count "$FINAL_DIR/retry.jsonl")" \
        --status "$status" "${failure_args[@]}" \
        --exit-code "$exit_code" --human-readable
}
pipeline_exit() {
    local code="$?"
    trap - EXIT
    if [[ "$pipeline_finalized" != 1 ]]; then
        finalize_runtime failed "$code" || true
    fi
    exit "$code"
}
trap pipeline_exit EXIT
service_batch() {
    local stage="$1" url="$2" input="$3"
    shift 3
    "$PY_PIPELINE" "$PIPELINE_ROOT/scripts/service_batch_infer.py" \
        --stage "$stage" --service-url "$url" --input "$input" \
        --concurrency "$SERVICE_REQUEST_CONCURRENCY" \
        --timeout "$SERVICE_REQUEST_TIMEOUT" --retries "$SERVICE_REQUEST_RETRIES" \
        --resume "$@"
}
filter_stage_file() {
    "$PY_PIPELINE" "$PIPELINE_ROOT/scripts/pipeline_state.py" filter \
        --manifest "$1" --input "$2" --output "$3"
}

echo "============================================================"
echo "Music-Data-Pipeline (resident-service batch runner)"
echo "input:        $INPUT_DIR"
echo "result:       $RESULT_DIR"
echo "annotations:  $FINAL_DIR/annotations"
echo "stages:       ALM=$RUN_ALM ASR=$RUN_ASR"
echo "============================================================"

health_args=(
    --service "fast_gate=$FAST_GATE_SERVICE_URL"
    --service "discogs_mir=$DISCOGS_MIR_SERVICE_URL"
    --service "music_cpu=$MUSIC_CPU_SERVICE_URL"
    --service "structure_raw=$STRUCTURE_RAW_SERVICE_URL"
)
[[ "$RUN_ALM" == 1 ]] && health_args+=(--service "alm=$ALM_SERVICE_URL")
[[ "$RUN_ASR" == 1 ]] && health_args+=(--service "section_asr=$SECTION_ASR_SERVICE_URL")
echo "[health] checking resident services before inventory"
CURRENT_STAGE="service_healthcheck"
"$PY_PIPELINE" "$PIPELINE_ROOT/scripts/service_healthcheck.py" \
    --timeout 5 "${health_args[@]}"

echo "[0/7] asset inventory"
CURRENT_STAGE="inventory"
started="$(runtime_now)"
"$PY_PIPELINE" "$PIPELINE_ROOT/scripts/calc_duration.py" \
    --root "$INPUT_DIR" --out "$INVENTORY_DIR/data.jsonl" \
    --fail-log "$INVENTORY_DIR/calc_failures.log" --jobs "$INVENTORY_JOBS" \
    --hash-jobs "$INVENTORY_HASH_JOBS" \
    --hash-cache "$INVENTORY_DIR/data.hash_cache.jsonl" --resume
record_stage_runtime inventory "$started" "$(jsonl_count "$INVENTORY_DIR/data.jsonl")"

echo "[1a/7] fast music gate service"
CURRENT_STAGE="fast_music_gate"
started="$(runtime_now)"
service_batch fast_gate "$FAST_GATE_SERVICE_URL" "$INVENTORY_DIR/data.jsonl" \
    --output-dir "$FAST_GATE_DIR"
record_stage_runtime fast_music_gate "$started" "$(jsonl_count "$INVENTORY_DIR/data.jsonl")"

echo "[1b/7] Discogs service and Song/Instrumental routing"
CURRENT_STAGE="discogs_mir"
started="$(runtime_now)"
service_batch discogs_mir "$DISCOGS_MIR_SERVICE_URL" \
    "$FAST_GATE_DIR/accepted.music.jsonl" --output-dir "$DISCOGS_DIR"
cp -f "$DISCOGS_DIR/data.song.jsonl" "$ROUTING_DIR/data.song.jsonl"
cp -f "$DISCOGS_DIR/data.instrumental.jsonl" "$ROUTING_DIR/data.instrumental.jsonl"
cp -f "$FAST_GATE_DIR/rejected.jsonl" "$ROUTING_DIR/rejected.jsonl"
"$PY_PIPELINE" "$PIPELINE_ROOT/scripts/combine_jsonl.py" \
    --inputs "$FAST_GATE_DIR/review.jsonl" "$DISCOGS_DIR/review.jsonl" \
    --output "$ROUTING_DIR/review.jsonl"
"$PY_PIPELINE" "$PIPELINE_ROOT/scripts/combine_jsonl.py" \
    --inputs "$ROUTING_DIR/data.song.jsonl" "$ROUTING_DIR/data.instrumental.jsonl" \
    --output "$ROUTING_DIR/accepted.input.jsonl"
record_stage_runtime discogs_mir_and_routing "$started" \
    "$(jsonl_count "$FAST_GATE_DIR/accepted.music.jsonl")"

ACCEPTED_COUNT="$(jsonl_count "$ROUTING_DIR/accepted.input.jsonl")"
if (( ACCEPTED_COUNT > 0 )); then
    echo "[2-3/7] CPU MIR, SongFormer, and whole-track Omni in parallel"
    CURRENT_STAGE="whole_track_services"
    started="$(runtime_now)"
    service_batch music_cpu "$MUSIC_CPU_SERVICE_URL" \
        "$ROUTING_DIR/accepted.input.jsonl" --output "$MUSIC_CPU_DIR/results.jsonl" &
    cpu_pid=$!
    service_batch structure_raw "$STRUCTURE_RAW_SERVICE_URL" \
        "$ROUTING_DIR/accepted.input.jsonl" --output "$GLOBAL_DIR/structure.raw.jsonl" &
    structure_pid=$!
    alm_pid=""
    if [[ "$RUN_ALM" == 1 ]]; then
        service_batch alm "$ALM_SERVICE_URL" "$ROUTING_DIR/accepted.input.jsonl" \
            --output "$ALM_DIR/accepted.input.alm.jsonl" &
        alm_pid=$!
    fi
    wait "$cpu_pid"
    wait "$structure_pid"
    [[ -n "$alm_pid" ]] && wait "$alm_pid"
    record_stage_runtime whole_track_parallel "$started" "$ACCEPTED_COUNT"

    active_global_args=(
        active --base "$ROUTING_DIR/accepted.input.jsonl"
        --stage "music_cpu=$MUSIC_CPU_DIR/results.jsonl"
        --stage "structure_raw=$GLOBAL_DIR/structure.raw.jsonl"
        --output "$ACTIVE_GLOBAL_MANIFEST" --retry-output "$RETRY_GLOBAL_FILE"
    )
    if [[ "$RUN_ALM" == 1 ]]; then
        active_global_args+=(--stage "alm=$ALM_DIR/accepted.input.alm.jsonl")
    fi
    "$PY_PIPELINE" "$PIPELINE_ROOT/scripts/pipeline_state.py" "${active_global_args[@]}"

    echo "[4/7] model-free structure postprocess"
    CURRENT_STAGE="structure_postprocess"
    started="$(runtime_now)"
    filter_stage_file "$ACTIVE_GLOBAL_MANIFEST" "$MUSIC_CPU_DIR/results.jsonl" \
        "$ACTIVE_DIR/music_cpu.global.jsonl"
    filter_stage_file "$ACTIVE_GLOBAL_MANIFEST" "$GLOBAL_DIR/structure.raw.jsonl" \
        "$ACTIVE_DIR/structure.raw.global.jsonl"
    "$PY_PIPELINE" "$PIPELINE_ROOT/scripts/structure_postprocess.py" \
        --input "$ACTIVE_DIR/structure.raw.global.jsonl" \
        --music-cpu "$ACTIVE_DIR/music_cpu.global.jsonl" \
        --output "$SECTIONS_DIR/data.sections.jsonl" \
        --snap-tolerance "$SNAP_TOLERANCE" \
        --duplicate-tolerance "$DUPLICATE_TOLERANCE" \
        --minimum-duration "$MINIMUM_SECTION_DURATION" \
        --extremely-short-duration "$EXTREMELY_SHORT_DURATION" \
        --short-boundary-confidence "$SHORT_BOUNDARY_CONFIDENCE"
    "$PY_PIPELINE" "$PIPELINE_ROOT/scripts/pipeline_state.py" active \
        --base "$ACTIVE_GLOBAL_MANIFEST" \
        --stage "structure_postprocess=$SECTIONS_DIR/data.sections.jsonl" \
        --output "$ACTIVE_SECTIONS_MANIFEST" --retry-output "$RETRY_SECTIONS_FILE"
    filter_stage_file "$ACTIVE_SECTIONS_MANIFEST" "$ACTIVE_DIR/music_cpu.global.jsonl" \
        "$ACTIVE_DIR/music_cpu.sections.jsonl"
    filter_stage_file "$ACTIVE_SECTIONS_MANIFEST" "$SECTIONS_DIR/data.sections.jsonl" \
        "$ACTIVE_DIR/sections.active.jsonl"
    context_inputs=("$ACTIVE_SECTIONS_MANIFEST" "$ACTIVE_DIR/music_cpu.sections.jsonl" "$ACTIVE_DIR/sections.active.jsonl")
    if [[ "$RUN_ALM" == 1 ]]; then
        filter_stage_file "$ACTIVE_SECTIONS_MANIFEST" "$ALM_DIR/accepted.input.alm.jsonl" \
            "$ACTIVE_DIR/alm.sections.jsonl"
        context_inputs+=("$ACTIVE_DIR/alm.sections.jsonl")
    fi
    "$PY_PIPELINE" "$PIPELINE_ROOT/scripts/metadata_merge.py" \
        --inputs "${context_inputs[@]}" --audio-key audio_id \
        --output "$SECTIONS_DIR/data.sections.context.jsonl"
    record_stage_runtime structure_postprocess "$started" "$ACCEPTED_COUNT"

    echo "[5-6/7] Song-only Section ASR service"
    CURRENT_STAGE="section_asr"
    if [[ "$RUN_ASR" == 1 ]]; then
        started="$(runtime_now)"
        service_batch section_asr "$SECTION_ASR_SERVICE_URL" \
            "$SECTIONS_DIR/data.sections.context.jsonl" \
            --output "$SECTIONS_DIR/data.section_asr.jsonl"
        "$PY_PIPELINE" "$PIPELINE_ROOT/scripts/pipeline_state.py" active \
            --base "$ACTIVE_SECTIONS_MANIFEST" \
            --stage "section_asr=$SECTIONS_DIR/data.section_asr.jsonl" \
            --output "$ACTIVE_FINAL_MANIFEST" --retry-output "$RETRY_FINAL_FILE"
        record_stage_runtime section_asr_and_alignment "$started" \
            "$(jsonl_count "$ROUTING_DIR/data.song.jsonl")"
    else
        cp -f "$ACTIVE_SECTIONS_MANIFEST" "$ACTIVE_FINAL_MANIFEST"
    fi
    "$PY_PIPELINE" "$PIPELINE_ROOT/scripts/pipeline_state.py" path-conflicts \
        --base "$ACTIVE_FINAL_MANIFEST" --output "$PUBLISH_BASE_MANIFEST" \
        --retry-output "$RETRY_PATH_FILE"

    echo "[7/7] per-item annotation merge"
    CURRENT_STAGE="metadata_merge"
    filter_stage_file "$PUBLISH_BASE_MANIFEST" "$ACTIVE_DIR/music_cpu.sections.jsonl" \
        "$ACTIVE_DIR/publish.music_cpu.jsonl"
    filter_stage_file "$PUBLISH_BASE_MANIFEST" "$ACTIVE_DIR/structure.raw.global.jsonl" \
        "$ACTIVE_DIR/publish.structure_raw.jsonl"
    filter_stage_file "$PUBLISH_BASE_MANIFEST" "$ACTIVE_DIR/sections.active.jsonl" \
        "$ACTIVE_DIR/publish.sections.jsonl"
    merge_args=(
        --base "$PUBLISH_BASE_MANIFEST"
        --music-cpu "$ACTIVE_DIR/publish.music_cpu.jsonl"
        --structure-raw "$ACTIVE_DIR/publish.structure_raw.jsonl"
        --sections "$ACTIVE_DIR/publish.sections.jsonl"
        --output-dir "$FINAL_DIR"
    )
    if [[ "$RUN_ALM" == 1 ]]; then
        filter_stage_file "$PUBLISH_BASE_MANIFEST" "$ACTIVE_DIR/alm.sections.jsonl" \
            "$ACTIVE_DIR/publish.alm.jsonl"
        merge_args+=(--alm "$ACTIVE_DIR/publish.alm.jsonl" --alm-enabled)
    fi
    if [[ "$RUN_ASR" == 1 ]]; then
        filter_stage_file "$PUBLISH_BASE_MANIFEST" "$SECTIONS_DIR/data.section_asr.jsonl" \
            "$ACTIVE_DIR/publish.section_asr.jsonl"
        merge_args+=(--section-asr "$ACTIVE_DIR/publish.section_asr.jsonl" --section-asr-enabled)
    fi
    "$PY_PIPELINE" "$PIPELINE_ROOT/scripts/dual_metadata_merge.py" "${merge_args[@]}"
else
    : > "$ACTIVE_GLOBAL_MANIFEST"
    : > "$ACTIVE_SECTIONS_MANIFEST"
    : > "$ACTIVE_FINAL_MANIFEST"
    : > "$PUBLISH_BASE_MANIFEST"
    "$PY_PIPELINE" "$PIPELINE_ROOT/scripts/split_annotations.py" \
        --empty --result-dir "$RESULT_DIR"
fi

cp -f "$ROUTING_DIR/review.jsonl" "$FINAL_DIR/review.jsonl"
cp -f "$ROUTING_DIR/rejected.jsonl" "$FINAL_DIR/rejected.jsonl"
"$PY_PIPELINE" "$PIPELINE_ROOT/scripts/pipeline_state.py" combine-retry \
    --inventory "$INVENTORY_DIR/data.jsonl" \
    --review "$FINAL_DIR/review.jsonl" --rejected "$FINAL_DIR/rejected.jsonl" \
    --annotations-dir "$FINAL_DIR/annotations" \
    --inputs "$FAST_GATE_DIR/failures.jsonl" "$DISCOGS_DIR/failures.jsonl" \
        "$RETRY_GLOBAL_FILE" "$RETRY_SECTIONS_FILE" "$RETRY_FINAL_FILE" "$RETRY_PATH_FILE" \
    --output "$FINAL_DIR/retry.jsonl"
validate_args=(
    --inventory "$INVENTORY_DIR/data.jsonl"
    --base "$ROUTING_DIR/data.song.jsonl" "$ROUTING_DIR/data.instrumental.jsonl"
    --annotations-dir "$FINAL_DIR/annotations"
    --review "$FINAL_DIR/review.jsonl" --rejected "$FINAL_DIR/rejected.jsonl"
    --retry "$FINAL_DIR/retry.jsonl"
)
[[ "$RUN_ALM" == 1 ]] && validate_args+=(--alm-enabled)
[[ "$RUN_ASR" == 1 ]] && validate_args+=(--section-asr-enabled)
"$PY_PIPELINE" "$PIPELINE_ROOT/scripts/validate_pipeline_output.py" "${validate_args[@]}"
CURRENT_STAGE="complete"
runtime_status="success"
if [[ -s "$FINAL_DIR/retry.jsonl" ]]; then runtime_status="partial_success"; fi
finalize_runtime "$runtime_status" 0
pipeline_finalized=1

echo "============================================================"
echo "Done"
echo "Annotations:   $FINAL_DIR/annotations"
echo "Review:        $FINAL_DIR/review.jsonl"
echo "Rejected:      $FINAL_DIR/rejected.jsonl"
echo "Retry:         $FINAL_DIR/retry.jsonl"
echo "Runtime log:   $LOG_DIR/pipeline.log"
echo "============================================================"

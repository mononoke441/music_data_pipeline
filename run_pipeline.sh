#!/usr/bin/env bash
# Music-Data-Pipeline: dirty media directory -> clean training metadata.
# Edit the configuration block below when models, devices, or thresholds change.

set -Eeuo pipefail

PIPELINE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
# Configuration / hyperparameters
# Every value can also be overridden by an environment variable of the same name.
# =============================================================================

# Runtime and device
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY_PIPELINE="${PY_PIPELINE:-/mnt/data/yuyin/user_workspace/liuhongjia/conda_envs/moss-music-pipeline/bin/python}"
PY_QWEN="${PY_QWEN:-/mnt/data/yuyin/user_workspace/liuhongjia/conda_envs/qwen3-vllm/bin/python}"
PIPELINE_GPU_MAX_MEMORY_GIB="${PIPELINE_GPU_MAX_MEMORY_GIB:-0}"  # 0 = unlimited
PIPELINE_GPU_WAIT_TIMEOUT="${PIPELINE_GPU_WAIT_TIMEOUT:-60}"
PIPELINE_GPU_WAIT_POLL="${PIPELINE_GPU_WAIT_POLL:-10}"
VLLM_GPU_HEADROOM_GIB="${VLLM_GPU_HEADROOM_GIB:-4}"
PIPELINE_PROGRESS="${PIPELINE_PROGRESS:-1}"
PIPELINE_PROGRESS_MIN_INTERVAL="${PIPELINE_PROGRESS_MIN_INTERVAL:-2.0}"
PIPELINE_QUIET_LOGS="${PIPELINE_QUIET_LOGS:-1}"

# Local music-analysis weights.
FAST_GATE_CONFIG="${FAST_GATE_CONFIG:-$PIPELINE_ROOT/MusicToolsPipeline/checkpoints/fast_gate_config.json}"
DISCOGS_VOCAL_SONG="${DISCOGS_VOCAL_SONG:-0.55}"
DISCOGS_VOCAL_INSTRUMENTAL="${DISCOGS_VOCAL_INSTRUMENTAL:-0.20}"
PANNS_REPO="${PANNS_REPO:-$PIPELINE_ROOT/PANNs}"
DISCOGS_ONNX_MODEL_ROOT="${DISCOGS_ONNX_MODEL_ROOT:-$PIPELINE_ROOT/MusicToolsPipeline/discogs_onnx}"

# Stage switches: 1 = run, 0 = leave nullable/not_run fields
RUN_ALM="${RUN_ALM:-1}"
RUN_SECTION_CAPTION="${RUN_SECTION_CAPTION:-1}"
RUN_ASR="${RUN_ASR:-1}"

# ALM API mode: local starts/stops Qwen3-Omni in this script; external only connects.
ALM_MODE="${ALM_MODE:-local}"  # local | external
ALM_SERVER="${ALM_SERVER:-http://127.0.0.1:10008}"
ALM_MODEL="${ALM_MODEL:-Qwen3-Omni-30B-A3B-Instruct}"
ALM_MAX_TOKENS="${ALM_MAX_TOKENS:-2048}"
ALM_TEMPERATURE="${ALM_TEMPERATURE:-0.3}"
ALM_CONCURRENCY="${ALM_CONCURRENCY:-2}"
ALM_TIMEOUT="${ALM_TIMEOUT:-600}"
ALM_START_TIMEOUT="${ALM_START_TIMEOUT:-300}"
ALM_GUARD_MAX_FAILURES="${ALM_GUARD_MAX_FAILURES:-3}"

# Local Qwen3-Omni server (used only when ALM_MODE=local)
OMNI_VLLM_BIN="${OMNI_VLLM_BIN:-/mnt/data/yuyin/user_workspace/sunkangle/env/omni_venv/bin/vllm}"
OMNI_MODEL_PATH="${OMNI_MODEL_PATH:-/mnt/data/yuyin/user_workspace/sunkangle/pretrained_models/Qwen3-Omni-30B-A3B-Instruct}"
OMNI_HOST="${OMNI_HOST:-127.0.0.1}"
OMNI_PORT="${OMNI_PORT:-10008}"
OMNI_DTYPE="${OMNI_DTYPE:-bfloat16}"
OMNI_VLLM_MAX_MEMORY_GIB="${OMNI_VLLM_MAX_MEMORY_GIB:-$PIPELINE_GPU_MAX_MEMORY_GIB}"
# Optional lower fraction. Values above the absolute GiB cap are clamped.
OMNI_GPU_MEMORY_UTILIZATION="${OMNI_GPU_MEMORY_UTILIZATION:-0.90}"
OMNI_MAX_MODEL_LEN="${OMNI_MAX_MODEL_LEN:-16384}"
OMNI_MAX_NUM_SEQS="${OMNI_MAX_NUM_SEQS:-2}"
OMNI_MM_ATTN_BACKEND="${OMNI_MM_ATTN_BACKEND:-TORCH_SDPA}"
OMNI_LIMIT_MM_PER_PROMPT="${OMNI_LIMIT_MM_PER_PROMPT:-{\"audio\":1,\"image\":1,\"video\":1}}"
OMNI_CPATH="${OMNI_CPATH:-/opt/conda/include/python3.10}"

# Qwen3-ASR + ForcedAligner
QWEN3_ASR_MODEL_PATH="${QWEN3_ASR_MODEL_PATH:-/mnt/data/yuyin/datasets/jinwenqing/work/JwqMusic/Evaluation/Qwen3-ASR-main/Qwen/Qwen3-ASR-1.7B}"
QWEN3_ALIGNER_MODEL_PATH="${QWEN3_ALIGNER_MODEL_PATH:-/mnt/data/yuyin/datasets/jinwenqing/work/JwqMusic/Evaluation/Qwen3-ASR-main/Qwen/Qwen3-ForcedAligner-0.6B}"
ASR_BATCH_SIZE="${ASR_BATCH_SIZE:-4}"
ASR_DECODE_WORKERS="${ASR_DECODE_WORKERS:-2}"
ASR_PADDING="${ASR_PADDING:-1.5}"
ASR_VLLM_MAX_MEMORY_GIB="${ASR_VLLM_MAX_MEMORY_GIB:-$PIPELINE_GPU_MAX_MEMORY_GIB}"
ASR_FORCED_ALIGNER_RESERVE_GIB="${ASR_FORCED_ALIGNER_RESERVE_GIB:-8}"
ASR_MIN_VLLM_MEMORY_GIB="${ASR_MIN_VLLM_MEMORY_GIB:-8}"
# Optional lower fraction. Values above the absolute GiB cap are clamped.
ASR_GPU_MEMORY_UTILIZATION="${ASR_GPU_MEMORY_UTILIZATION:-}"
ASR_MAX_NEW_TOKENS="${ASR_MAX_NEW_TOKENS:-512}"
PARALLEL_ASR_WITH_EXTERNAL_ALM="${PARALLEL_ASR_WITH_EXTERNAL_ALM:-1}"

# Throughput
INVENTORY_JOBS="${INVENTORY_JOBS:-32}"
INVENTORY_HASH_JOBS="${INVENTORY_HASH_JOBS:-8}"
FAST_GATE_DECODE_WORKERS="${FAST_GATE_DECODE_WORKERS:-16}"
# 0 means auto-size from the benchmark-selected Stage A/B batch sizes.
FAST_GATE_TRACK_BUFFER_SIZE="${FAST_GATE_TRACK_BUFFER_SIZE:-0}"
DISCOGS_BATCH_SIZE="${DISCOGS_BATCH_SIZE:-512}"
DISCOGS_DECODE_WORKERS="${DISCOGS_DECODE_WORKERS:-8}"
DISCOGS_BUFFERED_FRAMES="${DISCOGS_BUFFERED_FRAMES:-2048}"
DISCOGS_TORCH_GPU_MEMORY_GIB="${DISCOGS_TORCH_GPU_MEMORY_GIB:-0}"  # 0 = unlimited
CPU_MIR_WORKERS="${CPU_MIR_WORKERS:-8}"
SONGFORMER_GPUS="${SONGFORMER_GPUS:-1}"
SONGFORMER_DECODE_PREFETCH="${SONGFORMER_DECODE_PREFETCH:-1}"
SONGFORMER_EMBEDDING_BATCH_SIZE="${SONGFORMER_EMBEDDING_BATCH_SIZE:-1}"
SECTION_CAPTION_CONCURRENCY="${SECTION_CAPTION_CONCURRENCY:-1}"
SECTION_CAPTION_DECODE_WORKERS="${SECTION_CAPTION_DECODE_WORKERS:-2}"
SECTION_CAPTION_DECODE_BUFFER="${SECTION_CAPTION_DECODE_BUFFER:-2}"
SECTION_CAPTION_MAX_TOKENS="${SECTION_CAPTION_MAX_TOKENS:-256}"
SECTION_KEY_DECODE_WORKERS="${SECTION_KEY_DECODE_WORKERS:-4}"

# Structure post-processing thresholds
SNAP_TOLERANCE="${SNAP_TOLERANCE:-1.5}"
DUPLICATE_TOLERANCE="${DUPLICATE_TOLERANCE:-2.0}"
MINIMUM_SECTION_DURATION="${MINIMUM_SECTION_DURATION:-8.0}"
EXTREMELY_SHORT_DURATION="${EXTREMELY_SHORT_DURATION:-2.0}"
SHORT_BOUNDARY_CONFIDENCE="${SHORT_BOUNDARY_CONFIDENCE:-0.65}"

# =============================================================================
# End configuration
# =============================================================================

if (( $# != 2 )); then
    echo "Usage: bash run_pipeline.sh INPUT_DIR RESULT_DIR" >&2
    exit 2
fi

INPUT_DIR="$(cd -- "$1" && pwd)"
RESULT_DIR_INPUT="$2"
mkdir -p "$RESULT_DIR_INPUT"
RESULT_DIR="$(cd -- "$RESULT_DIR_INPUT" && pwd)"

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
CACHE_DIR="$INTERMEDIATE_DIR/cache/structure"
LOG_DIR="$INTERMEDIATE_DIR/logs"
STAGE_TIMINGS_FILE="$LOG_DIR/stage_timings.jsonl"
PIPELINE_RUNTIME_FILE="$LOG_DIR/pipeline_runtime.json"
GPU_DECISIONS_FILE="$LOG_DIR/gpu_budget_decisions.jsonl"

ACTIVE_GLOBAL_MANIFEST="$ACTIVE_DIR/accepted.global.jsonl"
ACTIVE_SECTIONS_MANIFEST="$ACTIVE_DIR/accepted.sections.jsonl"
ACTIVE_FINAL_MANIFEST="$ACTIVE_DIR/accepted.final.jsonl"
PUBLISH_BASE_MANIFEST="$ACTIVE_DIR/accepted.publishable.jsonl"
RETRY_GLOBAL_FILE="$ACTIVE_DIR/retry.global.jsonl"
RETRY_SECTIONS_FILE="$ACTIVE_DIR/retry.sections.jsonl"
RETRY_FINAL_FILE="$ACTIVE_DIR/retry.final.jsonl"
RETRY_PATH_FILE="$ACTIVE_DIR/retry.annotation_path.jsonl"

mkdir -p \
    "$FINAL_DIR" "$INVENTORY_DIR" "$ROUTING_DIR" "$FAST_GATE_DIR" "$DISCOGS_DIR" "$ALM_DIR" \
    "$MUSIC_CPU_DIR" "$SECTIONS_DIR" "$CACHE_DIR" "$LOG_DIR" "$ACTIVE_DIR"

PIPELINE_STARTED_AT="$(date +%s.%N)"
: > "$STAGE_TIMINGS_FILE"
: > "$GPU_DECISIONS_FILE"
for retry_file in "$RETRY_GLOBAL_FILE" "$RETRY_SECTIONS_FILE" "$RETRY_FINAL_FILE" "$RETRY_PATH_FILE"; do
    : > "$retry_file"
done

# Keep this file scoped to the current invocation. Stage outputs and runtime
# metrics provide resume history; appending old tracebacks here is misleading.
exec > >(tee "$LOG_DIR/pipeline.log") 2>&1

early_runtime_exit_handler() {
    local exit_code="$?"
    trap - EXIT
    if (( exit_code != 0 )) && [[ -x "$PY_PIPELINE" ]]; then
        "$PY_PIPELINE" "$PIPELINE_ROOT/scripts/pipeline_runtime_metrics.py" finalize \
            --stages "$STAGE_TIMINGS_FILE" \
            --output "$PIPELINE_RUNTIME_FILE" \
            --started-at "$PIPELINE_STARTED_AT" \
            --input-count 0 \
            --accepted-count 0 \
            --annotation-count 0 \
            --review-count 0 \
            --rejected-count 0 \
            --retry-count 0 \
            --status failed \
            --failure-stage initialization \
            --exit-code "$exit_code" \
            --gpu-decisions "$GPU_DECISIONS_FILE" \
            --human-readable || true
    fi
    exit "$exit_code"
}
trap early_runtime_exit_handler EXIT

for switch_name in \
    RUN_ALM RUN_SECTION_CAPTION RUN_ASR PARALLEL_ASR_WITH_EXTERNAL_ALM \
    PIPELINE_PROGRESS PIPELINE_QUIET_LOGS; do
    switch_value="${!switch_name}"
    if [[ "$switch_value" != 0 && "$switch_value" != 1 ]]; then
        echo "[ERROR] $switch_name must be 0 or 1, got: $switch_value" >&2
        exit 2
    fi
done
if [[ "$ALM_MODE" != local && "$ALM_MODE" != external ]]; then
    echo "[ERROR] ALM_MODE must be local or external, got: $ALM_MODE" >&2
    exit 2
fi
if [[ ! "$ALM_GUARD_MAX_FAILURES" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] ALM_GUARD_MAX_FAILURES must be a positive integer" >&2
    exit 2
fi
if ! awk -v value="$PIPELINE_PROGRESS_MIN_INTERVAL" \
    'BEGIN { exit !(value ~ /^[0-9]+([.][0-9]+)?$/ && value > 0) }'; then
    echo "[ERROR] PIPELINE_PROGRESS_MIN_INTERVAL must be positive" >&2
    exit 2
fi
if [[ ! "$PIPELINE_GPU_MAX_MEMORY_GIB" =~ ^[0-9]+([.][0-9]+)?$ \
    || ! "$OMNI_VLLM_MAX_MEMORY_GIB" =~ ^[0-9]+([.][0-9]+)?$ \
    || ! "$ASR_VLLM_MAX_MEMORY_GIB" =~ ^[0-9]+([.][0-9]+)?$ \
    || ! "$DISCOGS_TORCH_GPU_MEMORY_GIB" =~ ^[0-9]+([.][0-9]+)?$ \
    || ! "$ASR_FORCED_ALIGNER_RESERVE_GIB" =~ ^[0-9]+([.][0-9]+)?$ \
    || ! "$ASR_MIN_VLLM_MEMORY_GIB" =~ ^[0-9]+([.][0-9]+)?$ \
    || ! "$VLLM_GPU_HEADROOM_GIB" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "[ERROR] GPU memory limits must be non-negative numeric GiB values" >&2
    exit 2
fi
if ! awk -v cap="$PIPELINE_GPU_MAX_MEMORY_GIB" \
    -v omni="$OMNI_VLLM_MAX_MEMORY_GIB" \
    -v asr="$ASR_VLLM_MAX_MEMORY_GIB" \
    -v discogs_torch="$DISCOGS_TORCH_GPU_MEMORY_GIB" \
    -v reserve="$ASR_FORCED_ALIGNER_RESERVE_GIB" -v headroom="$VLLM_GPU_HEADROOM_GIB" \
    -v minimum="$ASR_MIN_VLLM_MEMORY_GIB" \
    'BEGIN {
        required = reserve + headroom + minimum
        valid = cap >= 0 && omni >= 0 && asr >= 0 && discogs_torch >= 0 && reserve >= 0 && headroom >= 0 && minimum > 0
        if (cap > 0) valid = valid && required <= cap && (discogs_torch == 0 || discogs_torch < cap)
        if (asr > 0) valid = valid && required <= asr
        exit !valid
    }'; then
    echo "[ERROR] invalid GPU memory cap or ASR reserve/headroom/minimum" >&2
    exit 2
fi
if awk -v cap="$PIPELINE_GPU_MAX_MEMORY_GIB" 'BEGIN { exit !(cap > 0) }'; then
    if awk -v torch="$DISCOGS_TORCH_GPU_MEMORY_GIB" 'BEGIN { exit !(torch == 0) }'; then
        DISCOGS_TORCH_GPU_MEMORY_GIB="$(
            awk -v cap="$PIPELINE_GPU_MAX_MEMORY_GIB" \
                'BEGIN { printf "%.6f", cap < 8 ? cap / 2 : 4 }'
        )"
    fi
    DISCOGS_ORT_GPU_MEMORY_GIB="$(
        awk -v cap="$PIPELINE_GPU_MAX_MEMORY_GIB" -v torch="$DISCOGS_TORCH_GPU_MEMORY_GIB" \
            'BEGIN { printf "%.6f", cap - torch }'
    )"
else
    DISCOGS_ORT_GPU_MEMORY_GIB=0
fi
if [[ ! "$PIPELINE_GPU_WAIT_TIMEOUT" =~ ^[0-9]+$ \
    || ! "$PIPELINE_GPU_WAIT_POLL" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] PIPELINE_GPU_WAIT_TIMEOUT must be non-negative and PIPELINE_GPU_WAIT_POLL positive" >&2
    exit 2
fi
require_file() {
    local path="$1" label="$2"
    if [[ ! -f "$path" ]]; then
        echo "[ERROR] missing $label: $path" >&2
        exit 1
    fi
}

require_dir() {
    local path="$1" label="$2"
    if [[ ! -d "$path" ]]; then
        echo "[ERROR] missing $label: $path" >&2
        exit 1
    fi
}

if [[ ! -x "$PY_PIPELINE" ]]; then
    echo "[ERROR] PY_PIPELINE is not executable: $PY_PIPELINE" >&2
    exit 1
fi
require_file "$FAST_GATE_CONFIG" "verified zero-training fast-gate config"
require_dir "$DISCOGS_ONNX_MODEL_ROOT" "Discogs ONNX directory"

NEED_ALM_API=0
if [[ "$RUN_ALM" == 1 || "$RUN_SECTION_CAPTION" == 1 ]]; then
    NEED_ALM_API=1
fi
if [[ "$NEED_ALM_API" == 1 && "$ALM_MODE" == local ]]; then
    if [[ ! -x "$OMNI_VLLM_BIN" ]]; then
        echo "[ERROR] OMNI_VLLM_BIN is not executable: $OMNI_VLLM_BIN" >&2
        exit 1
    fi
    require_dir "$OMNI_MODEL_PATH" "Qwen3-Omni model"
    ALM_SERVER="http://$OMNI_HOST:$OMNI_PORT"
fi
if [[ "$RUN_ASR" == 1 ]]; then
    if [[ ! -x "$PY_QWEN" ]]; then
        echo "[ERROR] PY_QWEN is not executable: $PY_QWEN" >&2
        exit 1
    fi
    require_dir "$QWEN3_ASR_MODEL_PATH" "Qwen3-ASR model"
    require_dir "$QWEN3_ALIGNER_MODEL_PATH" "Qwen3 ForcedAligner model"
fi

export CUDA_VISIBLE_DEVICES PANNS_REPO
export PIPELINE_PROGRESS PIPELINE_PROGRESS_MIN_INTERVAL PIPELINE_QUIET_LOGS
if [[ "$PIPELINE_QUIET_LOGS" == 1 ]]; then
    export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
    export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
    export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"
    export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
fi

pipeline_pids=()
pipeline_names=()
alm_pid=""
alm_started_by_runner=0
asr_parallel_started=0
global_alm_completed=0
pipeline_finalized=0
CURRENT_STAGE="initialization"

# Named jobs can be awaited by dependency rather than only as one barrier.
# This is what permits CPU-only work to overlap the next resident GPU model.
source "$PIPELINE_ROOT/scripts/pipeline_job_control.sh"

stop_local_alm() {
    if [[ "$alm_started_by_runner" == 1 && -n "$alm_pid" ]]; then
        if kill -0 -- "-$alm_pid" 2>/dev/null || kill -0 "$alm_pid" 2>/dev/null; then
            echo "[ALM] stopping local service pid=$alm_pid"
            kill -TERM -- "-$alm_pid" 2>/dev/null || kill -TERM "$alm_pid" 2>/dev/null || true
            local deadline=$((SECONDS + 60))
            while kill -0 -- "-$alm_pid" 2>/dev/null || kill -0 "$alm_pid" 2>/dev/null; do
                if (( SECONDS >= deadline )); then
                    echo "[ALM] owned process group did not stop after 60s; sending KILL" >&2
                    kill -KILL -- "-$alm_pid" 2>/dev/null || kill -KILL "$alm_pid" 2>/dev/null || true
                    break
                fi
                sleep 1
            done
            wait "$alm_pid" 2>/dev/null || true
        fi
        alm_pid=""
        alm_started_by_runner=0
    fi
}

cleanup() {
    local pid
    for pid in "${pipeline_pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            terminate_tree "$pid"
        fi
    done
    stop_local_alm
}

terminate_tree() {
    local parent_pid="$1" child_pid
    while IFS= read -r child_pid; do
        [[ -n "$child_pid" ]] && terminate_tree "$child_pid"
    done < <(pgrep -P "$parent_pid" 2>/dev/null || true)
    kill -TERM "$parent_pid" 2>/dev/null || true
}

local_alm_port_is_in_use() {
    (exec 3<>"/dev/tcp/$OMNI_HOST/$OMNI_PORT") 2>/dev/null
}

resolve_vllm_gpu_memory_utilization() {
    local max_gib="$1" requested="${2:-}" gpu_selector free_mib total_mib
    gpu_selector="${CUDA_VISIBLE_DEVICES%%,*}"
    read -r free_mib total_mib < <(
        nvidia-smi -i "$gpu_selector" \
            --query-gpu=memory.free,memory.total --format=csv,noheader,nounits \
            | head -n 1 | tr -d " " | tr "," " "
    )
    awk -v max_gib="$max_gib" -v requested="$requested" \
        -v free_mib="$free_mib" -v total_mib="$total_mib" \
        -v headroom_gib="$VLLM_GPU_HEADROOM_GIB" '
        BEGIN {
            if (max_gib !~ /^[0-9]+([.][0-9]+)?$/ || max_gib < 0 ||
                    free_mib <= 0 || total_mib <= 0 || headroom_gib < 0) {
                exit 2
            }
            cap = max_gib == 0 ? 0.99 : max_gib * 1024 / total_mib
            if (cap > 0.99) cap = 0.99
            available = (free_mib - headroom_gib * 1024) / total_mib
            selected = cap < available ? cap : available
            if (requested != "") {
                if (requested !~ /^[0-9]+([.][0-9]+)?$/ || requested <= 0 || requested > 1) {
                    exit 2
                }
                if (requested < selected) selected = requested
            }
            # vLLM accepts a fraction. Truncate instead of round so the
            # resulting reservation can never exceed the absolute GiB cap.
            selected = int(selected * 1000000) / 1000000
            if (selected <= 0) exit 2
            printf "%.6f", selected
        }
    '
}

clamp_to_pipeline_gpu_cap() {
    local requested="$1"
    awk -v requested="$requested" -v cap="$PIPELINE_GPU_MAX_MEMORY_GIB" '
        BEGIN {
            if (cap == 0) selected = requested
            else if (requested == 0) selected = cap
            else selected = requested < cap ? requested : cap
            printf "%.6f", selected
        }
    '
}

wait_for_gpu_capacity() {
    local label="$1" gpu_selector free_mib total_mib deadline
    if awk -v cap="$PIPELINE_GPU_MAX_MEMORY_GIB" 'BEGIN { exit !(cap == 0) }'; then
        echo "[GPU] $label admitted: no runner memory ceiling"
        return 0
    fi
    gpu_selector="${CUDA_VISIBLE_DEVICES%%,*}"
    deadline=$((SECONDS + PIPELINE_GPU_WAIT_TIMEOUT))
    while true; do
        read -r free_mib total_mib < <(
            nvidia-smi -i "$gpu_selector" \
                --query-gpu=memory.free,memory.total \
                --format=csv,noheader,nounits \
                | head -n 1 | tr -d ' ' | tr ',' ' '
        )
        if ! awk -v cap="$PIPELINE_GPU_MAX_MEMORY_GIB" -v total="$total_mib" \
            'BEGIN { exit !(cap * 1024 <= total) }'; then
            echo "[ERROR] pipeline GPU cap ${PIPELINE_GPU_MAX_MEMORY_GIB}GiB exceeds GPU total ${total_mib}MiB" >&2
            return 1
        fi
        if awk -v cap="$PIPELINE_GPU_MAX_MEMORY_GIB" -v free="$free_mib" \
            'BEGIN { exit !(free >= cap * 1024) }'; then
            echo "[GPU] $label admitted: free=${free_mib}MiB budget=${PIPELINE_GPU_MAX_MEMORY_GIB}GiB"
            return 0
        fi
        echo "[GPU] waiting for $label: free=${free_mib}MiB required=${PIPELINE_GPU_MAX_MEMORY_GIB}GiB"
        if (( SECONDS >= deadline )); then
            echo "[ERROR] insufficient free GPU memory for $label after ${PIPELINE_GPU_WAIT_TIMEOUT}s" >&2
            return 1
        fi
        sleep "$PIPELINE_GPU_WAIT_POLL"
    done
}

record_gpu_decision() {
    local label="$1" decision="$2" free_mib="$3" total_mib="$4" selected_gib="$5" details="$6"
    "$PY_PIPELINE" "$PIPELINE_ROOT/scripts/pipeline_runtime_metrics.py" gpu \
        --output "$GPU_DECISIONS_FILE" \
        --label "$label" \
        --decision "$decision" \
        --free-memory-mib "$free_mib" \
        --total-memory-mib "$total_mib" \
        --selected-memory-gib "$selected_gib" \
        --details "$details" >/dev/null
}

wait_for_asr_gpu_capacity() {
    local gpu_selector free_mib total_mib selected_gib deadline
    gpu_selector="${CUDA_VISIBLE_DEVICES%%,*}"
    deadline=$((SECONDS + PIPELINE_GPU_WAIT_TIMEOUT))
    while true; do
        read -r free_mib total_mib < <(
            nvidia-smi -i "$gpu_selector" \
                --query-gpu=memory.free,memory.total --format=csv,noheader,nounits \
                | head -n 1 | tr -d ' ' | tr ',' ' '
        )
        selected_gib="$(
            awk -v free="$free_mib" -v pipeline="$PIPELINE_GPU_MAX_MEMORY_GIB" \
                -v asr="$ASR_VLLM_MAX_MEMORY_GIB" \
                -v reserve="$ASR_FORCED_ALIGNER_RESERVE_GIB" \
                -v headroom="$VLLM_GPU_HEADROOM_GIB" 'BEGIN {
                    limit = free / 1024
                    if (pipeline > 0 && pipeline < limit) limit = pipeline
                    if (asr > 0 && asr < limit) limit = asr
                    printf "%.6f", limit - reserve - headroom
                }'
        )"
        if awk -v selected="$selected_gib" -v minimum="$ASR_MIN_VLLM_MEMORY_GIB" \
            'BEGIN { exit !(selected >= minimum) }'; then
            echo "[GPU] ASR admitted: free=${free_mib}MiB vllm_budget=${selected_gib}GiB reserve=${ASR_FORCED_ALIGNER_RESERVE_GIB}GiB headroom=${VLLM_GPU_HEADROOM_GIB}GiB"
            record_gpu_decision \
                "Qwen3-ASR + ForcedAligner" "admitted" "$free_mib" "$total_mib" "$selected_gib" \
                "minimum=${ASR_MIN_VLLM_MEMORY_GIB}GiB reserve=${ASR_FORCED_ALIGNER_RESERVE_GIB}GiB headroom=${VLLM_GPU_HEADROOM_GIB}GiB"
            return 0
        fi
        echo "[GPU] waiting for ASR: free=${free_mib}MiB resolved_vllm=${selected_gib}GiB minimum=${ASR_MIN_VLLM_MEMORY_GIB}GiB"
        record_gpu_decision \
            "Qwen3-ASR + ForcedAligner" "waiting" "$free_mib" "$total_mib" "$selected_gib" \
            "minimum=${ASR_MIN_VLLM_MEMORY_GIB}GiB reserve=${ASR_FORCED_ALIGNER_RESERVE_GIB}GiB headroom=${VLLM_GPU_HEADROOM_GIB}GiB"
        if (( SECONDS >= deadline )); then
            echo "[ERROR] insufficient free GPU memory for ASR after ${PIPELINE_GPU_WAIT_TIMEOUT}s; need at least ${ASR_MIN_VLLM_MEMORY_GIB}+${ASR_FORCED_ALIGNER_RESERVE_GIB}+${VLLM_GPU_HEADROOM_GIB}GiB" >&2
            return 1
        fi
        sleep "$PIPELINE_GPU_WAIT_POLL"
    done
}

alm_is_ready() {
    local url="${ALM_SERVER%/}/v1/models"
    local response=""
    if [[ -n "${INF_API_KEY:-}" ]]; then
        if ! response="$(curl -fsS --max-time 5 -H "Authorization: Bearer $INF_API_KEY" "$url" 2>&1)"; then
            return 1
        fi
    else
        if ! response="$(curl -fsS --max-time 5 "$url" 2>&1)"; then
            return 1
        fi
    fi
    [[ "$response" == *"$ALM_MODEL"* ]]
}

wait_for_alm() {
    local deadline=$((SECONDS + ALM_START_TIMEOUT))
    until alm_is_ready; do
        if [[ -n "$alm_pid" ]] && ! pipeline_process_is_running "$alm_pid"; then
            echo "[ERROR] local ALM service exited during startup" >&2
            tail -80 "$LOG_DIR/alm_server.log" >&2 || true
            return 1
        fi
        if (( SECONDS >= deadline )); then
            echo "[ERROR] ALM API was not ready within ${ALM_START_TIMEOUT}s: $ALM_SERVER" >&2
            return 1
        fi
        sleep 2
    done
    echo "[ALM] ready: $ALM_SERVER model=$ALM_MODEL"
}

local_alm_is_alive() {
    if [[ "$alm_started_by_runner" == 1 && -n "$alm_pid" ]]; then
        # The owned process is the authoritative liveness signal. Active
        # caption requests surface engine errors directly; probing /v1/models
        # here adds a second failure path and can misclassify a busy server.
        pipeline_process_is_running "$alm_pid"
        return
    fi
    return 1
}

run_with_alm_guard() {
    local status=0
    if [[ "$ALM_MODE" != local ]]; then
        if run_guarded_job \
            alm_is_ready "external ALM service" 1 "$ALM_GUARD_MAX_FAILURES" "$@"; then
            return 0
        else
            status=$?
        fi
        if (( status == 125 )); then
            echo "[ERROR] external ALM service failed ${ALM_GUARD_MAX_FAILURES} consecutive health checks" >&2
        fi
        return "$status"
    fi
    if run_guarded_job \
        local_alm_is_alive "local ALM service" 1 "$ALM_GUARD_MAX_FAILURES" "$@"; then
        return 0
    else
        status=$?
    fi
    if (( status == 125 )); then
        echo "[ERROR] local ALM service exited while inference was active" >&2
        tail -80 "$LOG_DIR/alm_server.log" >&2 || true
    fi
    return "$status"
}

start_local_alm() {
    local effective_vllm_max resolved_gpu_memory_utilization
    if local_alm_port_is_in_use; then
        echo "[ERROR] ALM_MODE=local requires an unused port; $OMNI_HOST:$OMNI_PORT is already occupied" >&2
        return 1
    fi
    wait_for_gpu_capacity "Qwen3-Omni"
    effective_vllm_max="$(clamp_to_pipeline_gpu_cap "$OMNI_VLLM_MAX_MEMORY_GIB")"
    resolved_gpu_memory_utilization="$(
        resolve_vllm_gpu_memory_utilization \
            "$effective_vllm_max" "$OMNI_GPU_MEMORY_UTILIZATION"
    )" || {
        echo "[ERROR] failed to reserve ${VLLM_GPU_HEADROOM_GIB}GiB headroom for Omni vLLM" >&2
        return 1
    }
    echo "[ALM] starting local Qwen3-Omni service vllm_max=${effective_vllm_max}GiB utilization=$resolved_gpu_memory_utilization"
    (
        export CPATH="$OMNI_CPATH${CPATH:+:$CPATH}"
        export VLLM_WORKER_MULTIPROC_METHOD=spawn
        export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
        exec setsid "$OMNI_VLLM_BIN" serve "$OMNI_MODEL_PATH" \
            --host "$OMNI_HOST" \
            --port "$OMNI_PORT" \
            --dtype "$OMNI_DTYPE" \
            --tensor-parallel-size 1 \
            --gpu-memory-utilization "$resolved_gpu_memory_utilization" \
            --max-model-len "$OMNI_MAX_MODEL_LEN" \
            --max-num-seqs "$OMNI_MAX_NUM_SEQS" \
            --limit-mm-per-prompt "$OMNI_LIMIT_MM_PER_PROMPT" \
            --mm-encoder-attn-backend "$OMNI_MM_ATTN_BACKEND" \
            --allowed-local-media-path / \
            --served-model-name "$ALM_MODEL"
    ) >"$LOG_DIR/alm_server.log" 2>&1 &
    alm_pid=$!
    alm_started_by_runner=1
    wait_for_alm
}

run_global_alm() {
    run_with_alm_guard "$PY_PIPELINE" "$PIPELINE_ROOT/scripts/alm_caption_infer.py" \
        --inputs "$ROUTING_DIR/accepted.input.jsonl" \
        --out_dir "$ALM_DIR" \
        --servers "$ALM_SERVER" \
        --model "$ALM_MODEL" \
        --max_tokens "$ALM_MAX_TOKENS" \
        --temperature "$ALM_TEMPERATURE" \
        --timeout "$ALM_TIMEOUT" \
        --concurrency "$ALM_CONCURRENCY" \
        --task_buffer "$ALM_CONCURRENCY" \
        --resume
}

run_section_asr() {
    local effective_vllm_max
    wait_for_asr_gpu_capacity
    effective_vllm_max="$(clamp_to_pipeline_gpu_cap "$ASR_VLLM_MAX_MEMORY_GIB")"
    local -a asr_memory_args=(
        --vllm-max-memory-gib "$effective_vllm_max"
        --gpu-max-memory-gib "$PIPELINE_GPU_MAX_MEMORY_GIB"
        --forced-aligner-reserve-gib "$ASR_FORCED_ALIGNER_RESERVE_GIB"
        --vllm-headroom-gib "$VLLM_GPU_HEADROOM_GIB"
        --minimum-vllm-memory-gib "$ASR_MIN_VLLM_MEMORY_GIB"
    )
    if [[ -n "$ASR_GPU_MEMORY_UTILIZATION" ]]; then
        asr_memory_args+=(--gpu-memory-utilization "$ASR_GPU_MEMORY_UTILIZATION")
    fi
    VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}" \
    "$PY_QWEN" "$PIPELINE_ROOT/scripts/section_asr_infer.py" \
        --input "$SECTIONS_DIR/data.sections.context.jsonl" \
        --output "$SECTIONS_DIR/data.section_asr.jsonl" \
        --model "$QWEN3_ASR_MODEL_PATH" \
        --forced-aligner "$QWEN3_ALIGNER_MODEL_PATH" \
        --batch-size "$ASR_BATCH_SIZE" \
        --decode-workers "$ASR_DECODE_WORKERS" \
        --padding "$ASR_PADDING" \
        "${asr_memory_args[@]}" \
        --max-new-tokens "$ASR_MAX_NEW_TOKENS" \
        --resume
}

publish_route_outputs() {
    "$PY_PIPELINE" "$PIPELINE_ROOT/scripts/combine_jsonl.py" \
        --inputs "$ROUTING_DIR/review.jsonl" --output "$FINAL_DIR/review.jsonl"
    "$PY_PIPELINE" "$PIPELINE_ROOT/scripts/combine_jsonl.py" \
        --inputs "$ROUTING_DIR/rejected.jsonl" --output "$FINAL_DIR/rejected.jsonl"
}

remove_legacy_final_outputs() {
    rm -f -- \
        "$FINAL_DIR/accepted.jsonl" \
        "$FINAL_DIR/data.song.annotated.jsonl" \
        "$FINAL_DIR/data.instrumental.annotated.jsonl" \
        "$FINAL_DIR/data.annotated.jsonl"
}

runtime_now() {
    date +%s.%N
}

jsonl_count() {
    local path="$1"
    if [[ -s "$path" ]]; then
        awk 'NF { count += 1 } END { print count + 0 }' "$path"
    else
        echo 0
    fi
}

record_stage_runtime() {
    local stage="$1" started_at="$2" processed="${3:-0}"
    "$PY_PIPELINE" "$PIPELINE_ROOT/scripts/pipeline_runtime_metrics.py" record \
        --output "$STAGE_TIMINGS_FILE" \
        --stage "$stage" \
        --started-at "$started_at" \
        --processed "$processed" \
        --human-readable
}

finalize_pipeline_runtime() {
    local requested_status="${1:-}" failure_stage="${2:-}" exit_code="${3:-0}"
    local input_count accepted_count annotation_count review_count rejected_count retry_count runtime_status
    input_count="$(jsonl_count "$INVENTORY_DIR/data.jsonl")"
    accepted_count="$(jsonl_count "$ROUTING_DIR/accepted.input.jsonl")"
    review_count="$(jsonl_count "$ROUTING_DIR/review.jsonl")"
    rejected_count="$(jsonl_count "$ROUTING_DIR/rejected.jsonl")"
    retry_count="$(jsonl_count "$FINAL_DIR/retry.jsonl")"
    if [[ -d "$FINAL_DIR/annotations" ]]; then
        annotation_count="$(find "$FINAL_DIR/annotations" -type f -name '*.json' | wc -l | tr -d ' ')"
    else
        annotation_count=0
    fi
    runtime_status="$requested_status"
    if [[ -z "$runtime_status" ]]; then
        if (( retry_count > 0 )); then
            runtime_status="partial_success"
        else
            runtime_status="success"
        fi
    fi
    local -a runtime_args=(
        finalize
        --stages "$STAGE_TIMINGS_FILE" \
        --output "$PIPELINE_RUNTIME_FILE" \
        --started-at "$PIPELINE_STARTED_AT" \
        --input-count "$input_count" \
        --accepted-count "$accepted_count" \
        --annotation-count "$annotation_count" \
        --review-count "$review_count" \
        --rejected-count "$rejected_count" \
        --retry-count "$retry_count" \
        --status "$runtime_status" \
        --exit-code "$exit_code" \
        --gpu-decisions "$GPU_DECISIONS_FILE" \
        --human-readable
    )
    if [[ -n "$failure_stage" ]]; then
        runtime_args+=(--failure-stage "$failure_stage")
    fi
    if [[ -f "$FINAL_DIR/retry.jsonl" ]]; then
        runtime_args+=(--retry "$FINAL_DIR/retry.jsonl")
    fi
    "$PY_PIPELINE" "$PIPELINE_ROOT/scripts/pipeline_runtime_metrics.py" "${runtime_args[@]}"
    pipeline_finalized=1
}

pipeline_exit_handler() {
    local exit_code="$1"
    trap - EXIT INT TERM
    set +e
    cleanup
    if [[ "$pipeline_finalized" != 1 ]]; then
        finalize_pipeline_runtime "failed" "$CURRENT_STAGE" "$exit_code" || true
    fi
    exit "$exit_code"
}

trap 'pipeline_exit_handler $?' EXIT
trap 'exit 130' INT TERM

filter_stage_file() {
    local manifest="$1" input="$2" output="$3"
    "$PY_PIPELINE" "$PIPELINE_ROOT/scripts/pipeline_state.py" filter \
        --manifest "$manifest" --input "$input" --output "$output"
}

finalize_four_partitions() {
    local stage_started_at="${1:-$(runtime_now)}" processed="${2:-0}"
    publish_route_outputs
    "$PY_PIPELINE" "$PIPELINE_ROOT/scripts/pipeline_state.py" combine-retry \
        --inventory "$INVENTORY_DIR/data.jsonl" \
        --review "$FINAL_DIR/review.jsonl" \
        --rejected "$FINAL_DIR/rejected.jsonl" \
        --annotations-dir "$FINAL_DIR/annotations" \
        --inputs \
            "$FAST_GATE_DIR/failures.jsonl" \
            "$DISCOGS_DIR/failures.jsonl" \
            "$RETRY_GLOBAL_FILE" \
            "$RETRY_SECTIONS_FILE" \
            "$RETRY_FINAL_FILE" \
            "$RETRY_PATH_FILE" \
        --output "$FINAL_DIR/retry.jsonl"

    local -a validate_args=(
        --inventory "$INVENTORY_DIR/data.jsonl"
        --base "$ROUTING_DIR/data.song.jsonl" "$ROUTING_DIR/data.instrumental.jsonl"
        --annotations-dir "$FINAL_DIR/annotations"
        --review "$FINAL_DIR/review.jsonl"
        --rejected "$FINAL_DIR/rejected.jsonl"
        --retry "$FINAL_DIR/retry.jsonl"
    )
    [[ "$RUN_ALM" == 1 ]] && validate_args+=(--alm-enabled)
    [[ "$RUN_SECTION_CAPTION" == 1 ]] && validate_args+=(--section-caption-enabled)
    [[ "$RUN_ASR" == 1 ]] && validate_args+=(--section-asr-enabled)
    "$PY_PIPELINE" "$PIPELINE_ROOT/scripts/validate_pipeline_output.py" "${validate_args[@]}"
    remove_legacy_final_outputs
    record_stage_runtime "metadata_publish_and_validation" "$stage_started_at" "$processed"
    finalize_pipeline_runtime
}

publish_empty_annotations_and_finish() {
    local stage_started_at
    stage_started_at="$(runtime_now)"
    "$PY_PIPELINE" "$PIPELINE_ROOT/scripts/split_annotations.py" \
        --empty --result-dir "$RESULT_DIR"
    CURRENT_STAGE="final_partition_validation"
    finalize_four_partitions "$stage_started_at" 0
}

echo "============================================================"
echo "Music-Data-Pipeline"
echo "input:        $INPUT_DIR"
echo "result:       $RESULT_DIR"
echo "final:        $FINAL_DIR"
echo "intermediate: $INTERMEDIATE_DIR"
echo "ALM mode:     $ALM_MODE ($ALM_SERVER)"
if awk -v cap="$PIPELINE_GPU_MAX_MEMORY_GIB" 'BEGIN { exit !(cap == 0) }'; then
    echo "GPU budget:   unlimited"
else
    echo "GPU budget:   ${PIPELINE_GPU_MAX_MEMORY_GIB}GiB (wait timeout ${PIPELINE_GPU_WAIT_TIMEOUT}s)"
fi
echo "stages:       ALM=$RUN_ALM section_caption=$RUN_SECTION_CAPTION ASR=$RUN_ASR"
echo "timings:      printed after each completed stage; JSONL=$STAGE_TIMINGS_FILE"
echo "============================================================"

if [[ "$NEED_ALM_API" == 1 && "$ALM_MODE" == local ]] && local_alm_port_is_in_use; then
    echo "[ERROR] ALM_MODE=local refuses occupied port $OMNI_HOST:$OMNI_PORT; stop or move the external service first" >&2
    exit 1
fi

echo "[0/7] asset inventory"
CURRENT_STAGE="inventory"
stage_started_at="$(runtime_now)"
"$PY_PIPELINE" "$PIPELINE_ROOT/scripts/calc_duration.py" \
    --root "$INPUT_DIR" \
    --out "$INVENTORY_DIR/data.jsonl" \
    --fail-log "$INVENTORY_DIR/calc_failures.log" \
    --jobs "$INVENTORY_JOBS" \
    --hash-jobs "$INVENTORY_HASH_JOBS" \
    --hash-cache "$INVENTORY_DIR/data.hash_cache.jsonl" \
    --resume
record_stage_runtime "inventory" "$stage_started_at" "$(jsonl_count "$INVENTORY_DIR/data.jsonl")"

echo "[1a/7] sparse fast music gate"
CURRENT_STAGE="fast_music_gate"
stage_started_at="$(runtime_now)"
wait_for_gpu_capacity "fast music gate"
PYTHONPATH="$PIPELINE_ROOT/scripts/gpu_runtime${PYTHONPATH:+:$PYTHONPATH}" \
PIPELINE_TORCH_GPU_MAX_MEMORY_GIB="$PIPELINE_GPU_MAX_MEMORY_GIB" \
PIPELINE_TORCH_CUDA_DEVICE=0 \
"$PY_PIPELINE" "$PIPELINE_ROOT/scripts/fast_music_gate.py" \
    --input "$INVENTORY_DIR/data.jsonl" \
    --output-dir "$FAST_GATE_DIR" \
    --config "$FAST_GATE_CONFIG" \
    --device cuda:0 \
    --decode-workers "$FAST_GATE_DECODE_WORKERS" \
    --track-buffer-size "$FAST_GATE_TRACK_BUFFER_SIZE" \
    --resume

if [[ -s "$FAST_GATE_DIR/failures.jsonl" ]]; then
    echo "[WARN] fast gate isolated retryable failures: $FAST_GATE_DIR/failures.jsonl" >&2
fi
record_stage_runtime "fast_music_gate" "$stage_started_at" "$(jsonl_count "$INVENTORY_DIR/data.jsonl")"

echo "[1b/7] Discogs MIR and Song/Instrumental routing"
CURRENT_STAGE="discogs_mir_and_routing"
stage_started_at="$(runtime_now)"
wait_for_gpu_capacity "Discogs MIR"
PYTHONPATH="$PIPELINE_ROOT/scripts/gpu_runtime${PYTHONPATH:+:$PYTHONPATH}" \
PIPELINE_TORCH_GPU_MAX_MEMORY_GIB="$DISCOGS_TORCH_GPU_MEMORY_GIB" \
PIPELINE_ORT_GPU_MAX_MEMORY_GIB="$DISCOGS_ORT_GPU_MEMORY_GIB" \
PIPELINE_TORCH_CUDA_DEVICE=0 \
"$PY_PIPELINE" "$PIPELINE_ROOT/scripts/discogs_mir_infer.py" \
    --input "$FAST_GATE_DIR/accepted.music.jsonl" \
    --output-dir "$DISCOGS_DIR" \
    --discogs-root "$DISCOGS_ONNX_MODEL_ROOT" \
    --device cuda:0 \
    --decode-workers "$DISCOGS_DECODE_WORKERS" \
    --frame-batch-size "$DISCOGS_BATCH_SIZE" \
    --buffered-frames "$DISCOGS_BUFFERED_FRAMES" \
    --vocal-song "$DISCOGS_VOCAL_SONG" \
    --vocal-instrumental "$DISCOGS_VOCAL_INSTRUMENTAL" \
    --resume

if [[ -s "$DISCOGS_DIR/failures.jsonl" ]]; then
    echo "[WARN] Discogs isolated retryable failures: $DISCOGS_DIR/failures.jsonl" >&2
fi
record_stage_runtime "discogs_mir_and_routing" "$stage_started_at" "$(jsonl_count "$FAST_GATE_DIR/accepted.music.jsonl")"

cp -f "$DISCOGS_DIR/data.song.jsonl" "$ROUTING_DIR/data.song.jsonl"
cp -f "$DISCOGS_DIR/data.instrumental.jsonl" "$ROUTING_DIR/data.instrumental.jsonl"
cp -f "$FAST_GATE_DIR/rejected.jsonl" "$ROUTING_DIR/rejected.jsonl"
"$PY_PIPELINE" "$PIPELINE_ROOT/scripts/combine_jsonl.py" \
    --inputs "$FAST_GATE_DIR/review.jsonl" "$DISCOGS_DIR/review.jsonl" \
    --output "$ROUTING_DIR/review.jsonl"

"$PY_PIPELINE" "$PIPELINE_ROOT/scripts/combine_jsonl.py" \
    --inputs "$ROUTING_DIR/data.song.jsonl" "$ROUTING_DIR/data.instrumental.jsonl" \
    --output "$ROUTING_DIR/accepted.input.jsonl"

if [[ ! -s "$ROUTING_DIR/accepted.input.jsonl" ]]; then
    echo "[DONE] no accepted music; expensive stages were skipped"
    publish_empty_annotations_and_finish
    echo "Final results: $FINAL_DIR"
    echo "Runtime:       $PIPELINE_RUNTIME_FILE"
    exit 0
fi

ACCEPTED_COUNT="$(awk 'END { print NR + 0 }' "$ROUTING_DIR/accepted.input.jsonl")"
if [[ ! "$CPU_MIR_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] CPU_MIR_WORKERS must be a positive integer, got: $CPU_MIR_WORKERS" >&2
    exit 2
fi
ACTIVE_CPU_MIR_WORKERS="$CPU_MIR_WORKERS"
if (( ACCEPTED_COUNT < ACTIVE_CPU_MIR_WORKERS )); then
    ACTIVE_CPU_MIR_WORKERS="$ACCEPTED_COUNT"
fi
echo "[plan] accepted=$ACCEPTED_COUNT CPU_MIR_workers=$ACTIVE_CPU_MIR_WORKERS (configured max=$CPU_MIR_WORKERS)"

if [[ "$NEED_ALM_API" == 1 && "$ALM_MODE" == external ]]; then
    wait_for_alm
fi

echo "[2/7] shared whole-track CPU MIR and dual structure analysis"
CURRENT_STAGE="whole_track_mir_and_structure"
stage_started_at="$(runtime_now)"
(
    cd "$PIPELINE_ROOT/MusicToolsPipeline"
    export CUDA_VISIBLE_DEVICES=""
    python_base="$($PY_PIPELINE -c 'import sys; print(sys.base_prefix)')"
    expat_lib="${PIPELINE_EXPAT_LIB:-$python_base/lib/libexpat.so.1}"
    if [[ -f "$expat_lib" ]]; then
        export LD_PRELOAD="$expat_lib${LD_PRELOAD:+:$LD_PRELOAD}"
    fi
    "$PY_PIPELINE" ray_inference.py \
        --cfg data_path="$ROUTING_DIR/accepted.input.jsonl" \
        --cfg model_path=dummy \
        --cfg output_path="$MUSIC_CPU_DIR" \
        --cfg model_type=music_cpu_pipeline \
        --cfg num_workers="$ACTIVE_CPU_MIR_WORKERS" \
        --cfg batch_size=4 \
        --cfg num_dataloader_workers=1 \
        --cfg dataloader_type=jsonl \
        --cfg reset_incompatible_output=true
) &
cpu_mir_job_pid="$!"
add_pipeline_job "$cpu_mir_job_pid" "Chordino/BeatNet/global key"

wait_for_gpu_capacity "MuQ/MusicFM + SongFormer"
(
    cd "$PIPELINE_ROOT/SongFormer"
    export PYTHONPATH="$PIPELINE_ROOT/scripts/gpu_runtime${PYTHONPATH:+:$PYTHONPATH}"
    export PIPELINE_TORCH_GPU_MAX_MEMORY_GIB="$PIPELINE_GPU_MAX_MEMORY_GIB"
    export PIPELINE_TORCH_CUDA_DEVICE=0
    python_base="$($PY_PIPELINE -c 'import sys; print(sys.base_prefix)')"
    expat_lib="${PIPELINE_EXPAT_LIB:-$python_base/lib/libexpat.so.1}"
    if [[ -f "$expat_lib" ]]; then
        export LD_PRELOAD="$expat_lib${LD_PRELOAD:+:$LD_PRELOAD}"
    fi
    "$PY_PIPELINE" infer_jsonl.py \
        --input_jsonl "$ROUTING_DIR/accepted.input.jsonl" \
        --output_jsonl "$GLOBAL_DIR/structure.raw.jsonl" \
        --audio_key audio_path \
        --output_dir "$CACHE_DIR" \
        --gpu_num "$SONGFORMER_GPUS" \
        --num_thread_per_gpu 1 \
        --decode-prefetch "$SONGFORMER_DECODE_PREFETCH" \
        --embedding-chunk-batch-size "$SONGFORMER_EMBEDDING_BATCH_SIZE" \
        --model SongFormer \
        --checkpoint SongFormer.safetensors \
        --config_path SongFormer.yaml
) &
structure_job_pid="$!"
add_pipeline_job "$structure_job_pid" "MuQ/MusicFM + Song/Instrumental structure"

if [[ "$RUN_ALM" == 1 && "$ALM_MODE" == external ]]; then
    run_global_alm &
    add_pipeline_job "$!" "whole-track ALM"
fi
if [[ "$RUN_ALM" == 1 && "$ALM_MODE" == local ]]; then
    # Omni cannot coexist with SongFormer on the same GPU, but it can run
    # while the independent CPU MIR job is still finishing.
    wait_pipeline_job "$structure_job_pid"
    start_local_alm
    global_alm_started_at="$(runtime_now)"
    run_global_alm &
    global_alm_job_pid="$!"
    add_pipeline_job "$global_alm_job_pid" "whole-track ALM"
    wait_pipeline_job "$global_alm_job_pid"
    record_stage_runtime "whole_track_caption" "$global_alm_started_at" "$ACCEPTED_COUNT"
    global_alm_completed=1
fi
wait_named_jobs
record_stage_runtime "whole_track_mir_and_structure" "$stage_started_at" "$ACCEPTED_COUNT"

active_global_args=(
    active
    --base "$ROUTING_DIR/accepted.input.jsonl"
    --stage "music_cpu=$MUSIC_CPU_DIR/results.jsonl"
    --stage "structure_raw=$GLOBAL_DIR/structure.raw.jsonl"
    --output "$ACTIVE_GLOBAL_MANIFEST"
    --retry-output "$RETRY_GLOBAL_FILE"
)
if [[ "$RUN_ALM" == 1 ]]; then
    active_global_args+=(--stage "alm=$ALM_DIR/accepted.input.alm.jsonl")
fi
"$PY_PIPELINE" "$PIPELINE_ROOT/scripts/pipeline_state.py" "${active_global_args[@]}"
if [[ ! -s "$ACTIVE_GLOBAL_MANIFEST" ]]; then
    echo "[DONE] all accepted tracks were isolated after global stages"
    publish_empty_annotations_and_finish
    exit 0
fi
filter_stage_file "$ACTIVE_GLOBAL_MANIFEST" "$MUSIC_CPU_DIR/results.jsonl" "$ACTIVE_DIR/music_cpu.global.jsonl"
filter_stage_file "$ACTIVE_GLOBAL_MANIFEST" "$GLOBAL_DIR/structure.raw.jsonl" "$ACTIVE_DIR/structure.raw.global.jsonl"
if [[ "$RUN_ALM" == 1 ]]; then
    filter_stage_file "$ACTIVE_GLOBAL_MANIFEST" "$ALM_DIR/accepted.input.alm.jsonl" "$ACTIVE_DIR/alm.global.jsonl"
fi

if [[ "$NEED_ALM_API" == 1 && "$ALM_MODE" == local && "$alm_started_by_runner" == 0 ]]; then
    start_local_alm
fi
if [[ "$RUN_ALM" == 1 && "$ALM_MODE" == local ]]; then
    if [[ "$global_alm_completed" == 1 ]]; then
        echo "[3/7] whole-track ALM completed while CPU MIR was running"
    else
        echo "[3/7] whole-track ALM captions"
        stage_started_at="$(runtime_now)"
        run_global_alm
        record_stage_runtime "whole_track_caption" "$stage_started_at" "$ACCEPTED_COUNT"
    fi
else
    echo "[3/7] whole-track ALM handled externally or disabled"
fi
if [[ "$ALM_MODE" == local && "$RUN_SECTION_CAPTION" == 0 ]]; then
    stop_local_alm
fi

echo "[4/7] shared structure post-processing and section context"
CURRENT_STAGE="structure_postprocess"
stage_started_at="$(runtime_now)"
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
    --output "$ACTIVE_SECTIONS_MANIFEST" \
    --retry-output "$RETRY_SECTIONS_FILE"
if [[ ! -s "$ACTIVE_SECTIONS_MANIFEST" ]]; then
    echo "[DONE] all globally successful tracks were isolated during structure postprocess"
    publish_empty_annotations_and_finish
    exit 0
fi
filter_stage_file "$ACTIVE_SECTIONS_MANIFEST" "$ACTIVE_DIR/music_cpu.global.jsonl" "$ACTIVE_DIR/music_cpu.sections.jsonl"
filter_stage_file "$ACTIVE_SECTIONS_MANIFEST" "$SECTIONS_DIR/data.sections.jsonl" "$ACTIVE_DIR/sections.active.jsonl"
context_inputs=(
    "$ACTIVE_SECTIONS_MANIFEST"
    "$ACTIVE_DIR/music_cpu.sections.jsonl"
    "$ACTIVE_DIR/sections.active.jsonl"
)
if [[ "$RUN_ALM" == 1 ]]; then
    filter_stage_file "$ACTIVE_SECTIONS_MANIFEST" "$ACTIVE_DIR/alm.global.jsonl" "$ACTIVE_DIR/alm.sections.jsonl"
    context_inputs+=("$ACTIVE_DIR/alm.sections.jsonl")
fi
"$PY_PIPELINE" "$PIPELINE_ROOT/scripts/metadata_merge.py" \
    --inputs "${context_inputs[@]}" \
    --audio-key audio_id \
    --output "$SECTIONS_DIR/data.sections.context.jsonl"
record_stage_runtime "structure_postprocess" "$stage_started_at" "$ACCEPTED_COUNT"

echo "[5/7] dynamic section key and caption analysis"
CURRENT_STAGE="section_key_caption"
stage_started_at="$(runtime_now)"
"$PY_PIPELINE" "$PIPELINE_ROOT/scripts/section_key_infer.py" \
    --input "$SECTIONS_DIR/data.sections.context.jsonl" \
    --music-cpu "$MUSIC_CPU_DIR/results.jsonl" \
    --output "$SECTIONS_DIR/data.section_key.jsonl" \
    --decode-workers "$SECTION_KEY_DECODE_WORKERS" \
    --resume &
section_key_job_pid="$!"
add_pipeline_job "$section_key_job_pid" "section key"

if [[ "$RUN_SECTION_CAPTION" == 1 ]]; then
    run_with_alm_guard "$PY_PIPELINE" "$PIPELINE_ROOT/scripts/section_caption_infer.py" \
        --input "$SECTIONS_DIR/data.sections.context.jsonl" \
        --output "$SECTIONS_DIR/data.section_caption.jsonl" \
        --servers "$ALM_SERVER" \
        --model "$ALM_MODEL" \
        --concurrency "$SECTION_CAPTION_CONCURRENCY" \
        --decode-workers "$SECTION_CAPTION_DECODE_WORKERS" \
        --decoded-buffer "$SECTION_CAPTION_DECODE_BUFFER" \
        --track-buffer "$SECTION_CAPTION_CONCURRENCY" \
        --max-tokens "$SECTION_CAPTION_MAX_TOKENS" \
        --timeout "$ALM_TIMEOUT" \
        --resume &
    section_caption_job_pid="$!"
    add_pipeline_job "$section_caption_job_pid" "section caption"
fi
if [[ "$RUN_ASR" == 1 && "$ALM_MODE" == external && "$PARALLEL_ASR_WITH_EXTERNAL_ALM" == 1 ]]; then
    run_section_asr &
    add_pipeline_job "$!" "section ASR and alignment"
    asr_parallel_started=1
fi
if [[ "$RUN_ASR" == 1 && "$ALM_MODE" == local ]]; then
    if [[ "$RUN_SECTION_CAPTION" == 1 ]]; then
        # Section Key is CPU-only.  Once caption releases Omni, ASR can take
        # the GPU immediately instead of waiting for all section keys.
        wait_pipeline_job "$section_caption_job_pid"
        stop_local_alm
    fi
    run_section_asr &
    add_pipeline_job "$!" "section ASR and alignment"
    asr_parallel_started=1
fi
wait_named_jobs
if [[ "$asr_parallel_started" == 1 ]]; then
    record_stage_runtime "section_key_caption_and_asr_parallel" "$stage_started_at" "$ACCEPTED_COUNT"
else
    record_stage_runtime "section_key_and_caption" "$stage_started_at" "$ACCEPTED_COUNT"
fi

if [[ "$ALM_MODE" == local ]]; then
    stop_local_alm
fi

echo "[6/7] Song section ASR and ForcedAligner"
if [[ "$asr_parallel_started" == 1 ]]; then
    echo "[OK] section ASR completed in parallel during Step 5"
elif [[ "$RUN_ASR" == 1 ]]; then
    CURRENT_STAGE="section_asr_and_alignment"
    stage_started_at="$(runtime_now)"
    run_section_asr
    record_stage_runtime "section_asr_and_alignment" "$stage_started_at" "$(jsonl_count "$ROUTING_DIR/data.song.jsonl")"
fi

active_final_args=(
    active
    --base "$ACTIVE_SECTIONS_MANIFEST"
    --stage "section_key=$SECTIONS_DIR/data.section_key.jsonl"
    --output "$ACTIVE_FINAL_MANIFEST"
    --retry-output "$RETRY_FINAL_FILE"
)
if [[ "$RUN_SECTION_CAPTION" == 1 ]]; then
    active_final_args+=(--stage "section_caption=$SECTIONS_DIR/data.section_caption.jsonl")
fi
if [[ "$RUN_ASR" == 1 ]]; then
    active_final_args+=(--stage "section_asr=$SECTIONS_DIR/data.section_asr.jsonl")
fi
"$PY_PIPELINE" "$PIPELINE_ROOT/scripts/pipeline_state.py" "${active_final_args[@]}"
"$PY_PIPELINE" "$PIPELINE_ROOT/scripts/pipeline_state.py" path-conflicts \
    --base "$ACTIVE_FINAL_MANIFEST" \
    --output "$PUBLISH_BASE_MANIFEST" \
    --retry-output "$RETRY_PATH_FILE"

if [[ ! -s "$PUBLISH_BASE_MANIFEST" ]]; then
    echo "[DONE] no publishable annotations remain after isolation and path preflight"
    publish_empty_annotations_and_finish
    exit 0
fi

filter_stage_file "$PUBLISH_BASE_MANIFEST" "$ACTIVE_DIR/music_cpu.sections.jsonl" "$ACTIVE_DIR/publish.music_cpu.jsonl"
filter_stage_file "$PUBLISH_BASE_MANIFEST" "$ACTIVE_DIR/structure.raw.global.jsonl" "$ACTIVE_DIR/publish.structure_raw.jsonl"
filter_stage_file "$PUBLISH_BASE_MANIFEST" "$ACTIVE_DIR/sections.active.jsonl" "$ACTIVE_DIR/publish.sections.jsonl"
filter_stage_file "$PUBLISH_BASE_MANIFEST" "$SECTIONS_DIR/data.section_key.jsonl" "$ACTIVE_DIR/publish.section_key.jsonl"
if [[ "$RUN_ALM" == 1 ]]; then
    filter_stage_file "$PUBLISH_BASE_MANIFEST" "$ACTIVE_DIR/alm.sections.jsonl" "$ACTIVE_DIR/publish.alm.jsonl"
fi
if [[ "$RUN_SECTION_CAPTION" == 1 ]]; then
    filter_stage_file "$PUBLISH_BASE_MANIFEST" "$SECTIONS_DIR/data.section_caption.jsonl" "$ACTIVE_DIR/publish.section_caption.jsonl"
fi
if [[ "$RUN_ASR" == 1 ]]; then
    filter_stage_file "$PUBLISH_BASE_MANIFEST" "$SECTIONS_DIR/data.section_asr.jsonl" "$ACTIVE_DIR/publish.section_asr.jsonl"
fi

echo "[7/7] unified metadata, publishing, and strict validation"
CURRENT_STAGE="metadata_publish_and_validation"
stage_started_at="$(runtime_now)"
merge_args=(
    --base "$PUBLISH_BASE_MANIFEST"
    --music-cpu "$ACTIVE_DIR/publish.music_cpu.jsonl"
    --structure-raw "$ACTIVE_DIR/publish.structure_raw.jsonl"
    --sections "$ACTIVE_DIR/publish.sections.jsonl"
    --section-key "$ACTIVE_DIR/publish.section_key.jsonl"
    --output-dir "$FINAL_DIR"
)
if [[ "$RUN_ALM" == 1 ]]; then
    merge_args+=(--alm "$ACTIVE_DIR/publish.alm.jsonl" --alm-enabled)
fi
if [[ "$RUN_SECTION_CAPTION" == 1 ]]; then
    merge_args+=(--section-caption "$ACTIVE_DIR/publish.section_caption.jsonl" --section-caption-enabled)
fi
if [[ "$RUN_ASR" == 1 ]]; then
    merge_args+=(--section-asr "$ACTIVE_DIR/publish.section_asr.jsonl" --section-asr-enabled)
fi
"$PY_PIPELINE" "$PIPELINE_ROOT/scripts/dual_metadata_merge.py" "${merge_args[@]}"
CURRENT_STAGE="final_partition_validation"
finalize_four_partitions "$stage_started_at" "$ACCEPTED_COUNT"

echo "============================================================"
echo "Done"
echo "Final results: $FINAL_DIR"
echo "Annotations:   $FINAL_DIR/annotations"
echo "Review:       $FINAL_DIR/review.jsonl"
echo "Rejected:     $FINAL_DIR/rejected.jsonl"
echo "Retry:        $FINAL_DIR/retry.jsonl"
echo "Log:          $LOG_DIR/pipeline.log"
echo "Runtime:      $PIPELINE_RUNTIME_FILE"
echo "============================================================"

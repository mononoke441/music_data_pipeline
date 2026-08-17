#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 INPUT_DIR RESULT_DIR [STREAM_OPTIONS...]" >&2
    exit 2
fi

PIPELINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_PIPELINE="${PY_PIPELINE:-/mnt/data/yuyin/user_workspace/liuhongjia/conda_envs/moss-music-pipeline/bin/python}"

export FAST_GATE_SERVICE_URL="${FAST_GATE_SERVICE_URL:-http://127.0.0.1:18101}"
export DISCOGS_MIR_SERVICE_URL="${DISCOGS_MIR_SERVICE_URL:-http://127.0.0.1:18102}"
export MUSIC_CPU_SERVICE_URL="${MUSIC_CPU_SERVICE_URL:-http://127.0.0.1:18103}"
export STRUCTURE_RAW_SERVICE_URL="${STRUCTURE_RAW_SERVICE_URL:-http://127.0.0.1:10101}"
export SECTION_ASR_SERVICE_URL="${SECTION_ASR_SERVICE_URL:-http://127.0.0.1:10102}"
export ALM_SERVICE_URL="${ALM_SERVICE_URL:-http://127.0.0.1:10103}"

exec "$PY_PIPELINE" "$PIPELINE_ROOT/scripts/stream_pipeline.py" \
    "$1" "$2" \
    --max-inflight "${STREAM_MAX_INFLIGHT:-64}" \
    --service-timeout "${STREAM_SERVICE_TIMEOUT:-1800}" \
    --service-retries "${STREAM_SERVICE_RETRIES:-3}" \
    "${@:3}"

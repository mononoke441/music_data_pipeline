#!/usr/bin/env bash
set -euo pipefail

PIPELINE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MANAGER_PYTHON="${MODEL_SERVICE_MANAGER_PYTHON:-python3}"

exec "$MANAGER_PYTHON" "$PIPELINE_ROOT/scripts/model_service_manager.py" "$@"

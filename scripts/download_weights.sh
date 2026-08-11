#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Download the model files required by Music-Data-Pipeline.

Usage:
  bash scripts/download_weights.sh [all|muq|omni|llm|asr|wav2vec|discogs]...

Targets:
  all        Download every missing dependency (default).
  muq        OpenMuQ/MuQ-large-msd-iter for SongFormer.
  omni       Qwen3-Omni used by both ALM and ASR.
  llm        Qwen3-235B used by metadata/query generation.
  wav2vec    Wav2Vec2 Conformer config required by SongFormer.
  asr        Qwen3-ASR and Qwen3-ForcedAligner snapshots.
  discogs    Dynamic-batch Discogs EffNet ONNX backbone and five heads.

Optional environment overrides:
  HF_PROXY_URL     Optional HTTP proxy URL. Empty means direct access.
  HF_BIN           Path to the Hugging Face `hf` executable.
  MODELSCOPE_BIN   Path to the ModelScope executable.
  PIPELINE_MODEL_ROOT  Directory for Qwen snapshots.
  HF_MAX_WORKERS   Parallel Hugging Face downloads (default: 4).
  MODELSCOPE_NO_PROXY
                    Hosts reached directly for ModelScope metadata
                    (default: modelscope.cn,www.modelscope.cn).
  HF_HUB_DISABLE_XET
                    Set to 1 only when legacy HTTP download is required.

This script downloads files only. It does not start inference services.
EOF
}

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

configure_hf_access() {
    export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
    if [[ -n "${HF_PROXY_URL:-}" ]]; then
        export HTTP_PROXY="$HF_PROXY_URL"
        export HTTPS_PROXY="$HF_PROXY_URL"
        export http_proxy="$HF_PROXY_URL"
        export https_proxy="$HF_PROXY_URL"
    else
        unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
    fi
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PIPELINE_MODEL_ROOT="${PIPELINE_MODEL_ROOT:-/mnt/data/yuyin/user_workspace/liuhongjia/models/Music-Data-Pipeline}"
HF_BIN="${HF_BIN:-/mnt/data/yuyin/user_workspace/liuhongjia/conda_envs/qwen3-vllm/bin/hf}"
MODELSCOPE_BIN="${MODELSCOPE_BIN:-/mnt/data/yuyin/user_workspace/liuhongjia/conda_envs/qwen3-vllm/bin/modelscope}"
HF_MAX_WORKERS="${HF_MAX_WORKERS:-4}"

MUQ_DIR="$PIPELINE_ROOT/SongFormer/ckpts/MuQ-large-msd-iter"
OMNI_DIR="$PIPELINE_MODEL_ROOT/Qwen3-Omni-30B-A3B-Instruct"
LLM_DIR="$PIPELINE_MODEL_ROOT/Qwen3-235B-A22B-Instruct-2507"
ASR_DIR="$PIPELINE_MODEL_ROOT/Qwen3-ASR-1.7B"
ALIGNER_DIR="$PIPELINE_MODEL_ROOT/Qwen3-ForcedAligner-0.6B"
WAV2VEC_DIR="$PIPELINE_ROOT/SongFormer/ckpts/wav2vec2-conformer-rope-large-960h-ft"
DISCOGS_DIR="$PIPELINE_ROOT/MusicToolsPipeline/discogs_onnx"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
    usage
    exit 0
fi

configure_hf_access
CURL_PROXY_ARGS=()
if [[ -n "${HTTPS_PROXY:-}" ]]; then
    CURL_PROXY_ARGS=(--proxy "$HTTPS_PROXY")
fi
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-0}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"

if [[ ! -x "$HF_BIN" ]]; then
    if command -v hf >/dev/null 2>&1; then
        HF_BIN="$(command -v hf)"
    else
        die "Hugging Face CLI not found. Set HF_BIN to the full path of the hf executable."
    fi
fi

command -v curl >/dev/null 2>&1 || die "curl is required."
[[ "$HF_MAX_WORKERS" =~ ^[1-9][0-9]*$ ]] || die "HF_MAX_WORKERS must be a positive integer."

download_hf_snapshot() {
    local repo_id="$1"
    local revision="$2"
    local local_dir="$3"

    mkdir -p "$local_dir"
    log "Downloading $repo_id at $revision"
    "$HF_BIN" download "$repo_id" \
        --revision "$revision" \
        --local-dir "$local_dir" \
        --max-workers "$HF_MAX_WORKERS"
}

download_hf_files() {
    local repo_id="$1"
    local revision="$2"
    local local_dir="$3"
    shift 3

    mkdir -p "$local_dir"
    log "Downloading selected files from $repo_id at $revision"
    "$HF_BIN" download "$repo_id" "$@" \
        --revision "$revision" \
        --local-dir "$local_dir" \
        --max-workers "$HF_MAX_WORKERS"
}

download_modelscope_snapshot() {
    local repo_id="$1"
    local revision="$2"
    local local_dir="$3"

    if [[ ! -x "$MODELSCOPE_BIN" ]]; then
        if command -v modelscope >/dev/null 2>&1; then
            MODELSCOPE_BIN="$(command -v modelscope)"
        else
            die "ModelScope CLI not found. Set MODELSCOPE_BIN to the full path of the modelscope executable."
        fi
    fi

    # ModelScope metadata is reachable directly from this cluster, while large
    # files are fetched from a separate CDN through the configured proxy. Keep
    # the proxy for redirects and bypass it only for the metadata hosts.
    local modelscope_no_proxy="${MODELSCOPE_NO_PROXY:-modelscope.cn,www.modelscope.cn}"
    export NO_PROXY="${NO_PROXY:+$NO_PROXY,}$modelscope_no_proxy"
    export no_proxy="${no_proxy:+$no_proxy,}$modelscope_no_proxy"

    mkdir -p "$local_dir"
    log "Downloading $repo_id from ModelScope at $revision"
    "$MODELSCOPE_BIN" download "$repo_id" \
        --revision "$revision" \
        --local-dir "$local_dir" \
        --max-workers "$HF_MAX_WORKERS"
}

require_proxy_connect() {
    local host="$1"
    local connect_code=""

    if [[ -z "${HTTPS_PROXY:-}" ]]; then
        return 0
    fi

    connect_code="$({
        curl --silent --show-error \
            --head \
            "${CURL_PROXY_ARGS[@]}" \
            --noproxy "" \
            --connect-timeout 5 \
            --max-time 10 \
            --output /dev/null \
            --write-out '%{http_connect}' \
            "https://$host/"
    } 2>/dev/null || true)"

    if [[ "$connect_code" != "200" ]]; then
        die "Proxy cannot CONNECT to $host:443 (HTTP CONNECT status: ${connect_code:-unavailable}). The Qwen weight shards are stored on this host, so no downloader can fetch them from the current server path. Ask for this hostname to be allowed, or copy the completed snapshot from another server/shared storage. No weight download was attempted."
    fi
}

require_nonempty_file() {
    local path="$1"
    [[ -s "$path" ]] || die "Expected downloaded file is missing or empty: $path"
}

download_url() {
    local url="$1"
    local output="$2"

    if [[ -s "$output" ]]; then
        log "Already present, skipping: $output"
        return
    fi

    mkdir -p "$(dirname -- "$output")"
    log "Downloading $(basename -- "$output")"
    curl "${CURL_PROXY_ARGS[@]}" \
        --fail --location \
        --connect-timeout 10 --max-time 600 \
        --retry 5 --retry-delay 3 --retry-all-errors \
        --continue-at - \
        --output "$output" \
        "$url"
    require_nonempty_file "$output"
}

download_muq() {
    local weight_path="$MUQ_DIR/model.safetensors"
    local expected_sha256="273febab2be02872c37d2c37e48a9d6c52c1c9392f3eeeabd498efa281ccb7a6"
    local actual_sha256=""

    # The repository also contains an equivalent pytorch_model.bin. The pipeline only
    # needs one weight format, so keep the safetensors copy to avoid 1.33 GB of
    # duplicate storage.
    if [[ -s "$weight_path" ]]; then
        actual_sha256="$(sha256sum "$weight_path" | awk '{print $1}')"
        if [[ "$actual_sha256" == "$expected_sha256" ]]; then
            if [[ ! -s "$MUQ_DIR/config.json" ]]; then
                download_hf_files \
                    OpenMuQ/MuQ-large-msd-iter \
                    0562a57814f6f8bbd9fdea0a25921a2fce1a841a \
                    "$MUQ_DIR" \
                    config.json
            fi
            require_nonempty_file "$MUQ_DIR/config.json"
            log "MuQ weight already present with the expected SHA256; skipping download."
            return
        fi
        log "Existing MuQ weight has an unexpected SHA256; downloading the official snapshot."
    fi

    download_hf_files \
        OpenMuQ/MuQ-large-msd-iter \
        0562a57814f6f8bbd9fdea0a25921a2fce1a841a \
        "$MUQ_DIR" \
        config.json model.safetensors
    require_nonempty_file "$MUQ_DIR/config.json"
    require_nonempty_file "$MUQ_DIR/model.safetensors"
}

omni_snapshot_complete() {
    # Exact shard sizes for Qwen/Qwen3-Omni-30B-A3B-Instruct at HF commit
    # 26291f793822fb6be9555850f06dfe95f2d7e695. ModelScope and Modelers
    # expose the same files and first-shard SHA256.
    local expected_sizes=(
        4997899632 4997754216 4997754216 4997755648 4997755792
        4997755792 4997755792 4997755792 4997755792 4997755792
        4997755792 4997755792 4999771808 4996618552 553698794
    )
    local index=""
    local shard=""
    local actual_size=""

    [[ -s "$OMNI_DIR/config.json" ]] || return 1
    [[ -s "$OMNI_DIR/model.safetensors.index.json" ]] || return 1

    for index in "${!expected_sizes[@]}"; do
        printf -v shard '%s/model-%05d-of-00015.safetensors' "$OMNI_DIR" "$((index + 1))"
        [[ -f "$shard" ]] || return 1
        actual_size="$(stat -c '%s' "$shard")"
        [[ "$actual_size" == "${expected_sizes[$index]}" ]] || return 1
    done
}

download_omni() {
    # The same snapshot is shared by the ALM and ASR services.
    if omni_snapshot_complete; then
        log "Qwen3-Omni snapshot is already complete; skipping download."
        return
    fi

    # ModelScope serves metadata from its main site and redirects large files
    # to this CDN. Test the real object path first so a blocked proxy fails in
    # seconds instead of entering the SDK's long retry loop.
    require_proxy_connect cdn-lfs-cn-1.modelscope.cn
    download_modelscope_snapshot \
        Qwen/Qwen3-Omni-30B-A3B-Instruct \
        master \
        "$OMNI_DIR"
    omni_snapshot_complete || die "Qwen3-Omni download returned without a complete 15-shard snapshot in $OMNI_DIR"
}

download_llm() {
    download_hf_snapshot \
        Qwen/Qwen3-235B-A22B-Instruct-2507 \
        ac9c66cc9b46af7306746a9250f23d47083d689e \
        "$LLM_DIR"
    require_nonempty_file "$LLM_DIR/model.safetensors.index.json"
}

download_asr() {
    download_hf_snapshot \
        Qwen/Qwen3-ASR-1.7B \
        main \
        "$ASR_DIR"
    download_hf_snapshot \
        Qwen/Qwen3-ForcedAligner-0.6B \
        main \
        "$ALIGNER_DIR"
    require_nonempty_file "$ASR_DIR/config.json"
    require_nonempty_file "$ALIGNER_DIR/config.json"
}

download_wav2vec() {
    download_hf_files \
        facebook/wav2vec2-conformer-rope-large-960h-ft \
        main \
        "$WAV2VEC_DIR" \
        config.json
    require_nonempty_file "$WAV2VEC_DIR/config.json"
}

download_discogs_pair() {
    local directory="$1"
    local filename="$2"
    download_url \
        "https://essentia.upf.edu/models/$directory/$filename.onnx" \
        "$DISCOGS_DIR/$filename.onnx"
    download_url \
        "https://essentia.upf.edu/models/$directory/$filename.json" \
        "$DISCOGS_DIR/$filename.json"
}

download_discogs() {
    download_discogs_pair \
        feature-extractors/discogs-effnet \
        discogs-effnet-bsdynamic-1
    download_discogs_pair \
        classification-heads/voice_instrumental \
        voice_instrumental-discogs-effnet-1
    download_discogs_pair \
        classification-heads/mtg_jamendo_genre \
        mtg_jamendo_genre-discogs-effnet-1
    download_discogs_pair \
        classification-heads/mtg_jamendo_moodtheme \
        mtg_jamendo_moodtheme-discogs-effnet-1
    download_discogs_pair \
        classification-heads/mtg_jamendo_instrument \
        mtg_jamendo_instrument-discogs-effnet-1
    download_discogs_pair \
        classification-heads/danceability \
        danceability-discogs-effnet-1
}

run_target() {
    case "$1" in
        muq)       download_muq ;;
        omni)      download_omni ;;
        llm)       download_llm ;;
        asr)       download_asr ;;
        wav2vec)   download_wav2vec ;;
        discogs|essentia) download_discogs ;;
        all)
            download_muq
            download_omni
            download_llm
            download_asr
            download_wav2vec
            download_discogs
            ;;
        -h|--help|help)
            usage
            ;;
        *)
            usage >&2
            die "Unknown target: $1"
            ;;
    esac
}

if (( $# == 0 )); then
    set -- all
fi

for target in "$@"; do
    run_target "$target"
done

log "Requested downloads completed."
printf '\nRuntime paths:\n'
printf '  export BEATS_CHECKPOINT=%q\n' "$BEATS_CHECKPOINT"
printf '  export BEATS_LABELS=%q\n' "$BEATS_LABELS"
printf '  export DISCOGS_ONNX_MODEL_ROOT=%q\n' "$DISCOGS_DIR"
printf '  export QWEN3_ASR_MODEL_PATH=%q\n' "$ASR_DIR"
printf '  export QWEN3_ALIGNER_MODEL_PATH=%q\n' "$ALIGNER_DIR"

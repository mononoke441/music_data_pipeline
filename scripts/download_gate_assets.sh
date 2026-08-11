#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Download or verify fast music-gate model weights on the server.

Usage:
  bash scripts/download_gate_assets.sh [--verify|--dry-run] [all|panns-mobilenet]...

Modes:
  (default)  Reuse non-empty preinstalled files or download missing files, then
             write SHA256SUMS from the bytes actually present.
  --verify   Never use the network. Compute SHA256 for every selected file and
             compare it with SHA256SUMS and/or the corresponding *_SHA256 env.
  --dry-run  Print resolved paths and sources without downloading or writing.

Environment:
  GATE_ASSET_ROOT                 Destination directory.
  HF_ENDPOINT                    Base endpoint for optional *_HF_REPO sources.
  PANNS_MOBILENET_PATH           Override destination file.
  <ASSET>_SHA256                 Optional trusted expected SHA256.
  <ASSET>_MD5                    Optional expected MD5. The PANNs default is
                                 fixed to the model author's Zenodo record.
  <ASSET>_URL                    Optional complete source URL.
  <ASSET>_HF_REPO                Optional Hugging Face repo ID. When set, the
                                 URL is built from HF_ENDPOINT, never hardcoded.
  <ASSET>_HF_REPO_TYPE           model (default) or dataset.
  <ASSET>_HF_REVISION            HF revision (default: main).
  <ASSET>_HF_FILE                HF filename (default: destination basename).

The asset prefix is PANNS_MOBILENET.
No unknown SHA256 is embedded in this script. Published PANNs MD5 values are
enforced in addition to recording SHA256. If an official source is blocked,
preinstall the file at its printed path and rerun with --verify (optionally set
the trusted *_SHA256 supplied by the fixed production config).
EOF
}

log() {
    printf '[gate-assets] %s\n' "$*"
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
GATE_ASSET_ROOT="${GATE_ASSET_ROOT:-$PIPELINE_ROOT/MusicToolsPipeline/checkpoints/fast_gate}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
SHA256_MANIFEST="${SHA256_MANIFEST:-$GATE_ASSET_ROOT/SHA256SUMS}"

PANNS_MOBILENET_PATH="${PANNS_MOBILENET_PATH:-$GATE_ASSET_ROOT/MobileNetV1_mAP=0.389.pth}"

# Author-published checksums from Zenodo record 3987831 (version v3).
PANNS_MOBILENET_MD5="${PANNS_MOBILENET_MD5:-a419303e1c88aa1b9d2ac3811563d371}"

# These are model-author-controlled sources. SHA256 values are intentionally
# not guessed from Zenodo MD5 metadata or release filenames.
PANNS_MOBILENET_OFFICIAL_URL='https://zenodo.org/records/3987831/files/MobileNetV1_mAP%3D0.389.pth?download=1'

sha256_file() {
    local path="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$path" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$path" | awk '{print $1}'
    else
        die "sha256sum or shasum is required"
    fi
}

md5_file() {
    local path="$1"
    if command -v md5sum >/dev/null 2>&1; then
        md5sum "$path" | awk '{print $1}'
    elif command -v md5 >/dev/null 2>&1; then
        md5 -q "$path"
    else
        die "md5sum or md5 is required for assets with an author-published MD5"
    fi
}

normalize_sha256() {
    local value="$1"
    value="$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]')"
    [[ "$value" =~ ^[0-9a-f]{64}$ ]] || die "Invalid SHA256 value: $1"
    printf '%s' "$value"
}

path_token() {
    local path="$1"
    if [[ "$path" == "$GATE_ASSET_ROOT/"* ]]; then
        printf '%s' "${path#"$GATE_ASSET_ROOT/"}"
    else
        printf '%s' "$path"
    fi
}

manifest_sha256() {
    local path="$1"
    local token=""
    local sha=""
    local recorded=""
    [[ -s "$SHA256_MANIFEST" ]] || return 0
    token="$(path_token "$path")"
    while read -r sha recorded; do
        recorded="${recorded#\*}"
        if [[ "$recorded" == "$token" ]]; then
            printf '%s' "$sha"
            return 0
        fi
    done < "$SHA256_MANIFEST"
}

expected_sha256() {
    local prefix="$1"
    local path="$2"
    local env_name="${prefix}_SHA256"
    local from_env="${!env_name:-}"
    local from_manifest=""
    from_manifest="$(manifest_sha256 "$path")"
    if [[ -n "$from_env" ]]; then
        from_env="$(normalize_sha256 "$from_env")"
    fi
    if [[ -n "$from_manifest" ]]; then
        from_manifest="$(normalize_sha256 "$from_manifest")"
    fi
    if [[ -n "$from_env" && -n "$from_manifest" && "$from_env" != "$from_manifest" ]]; then
        die "$env_name conflicts with $SHA256_MANIFEST for $path"
    fi
    printf '%s' "${from_env:-$from_manifest}"
}

verify_published_md5() {
    local prefix="$1" path="$2"
    local env_name="${prefix}_MD5"
    local expected="${!env_name:-}"
    local actual=""
    [[ -n "$expected" ]] || return 0
    expected="$(printf '%s' "$expected" | tr '[:upper:]' '[:lower:]')"
    if [[ ! "$expected" =~ ^[0-9a-f]{32}$ ]]; then
        printf 'ERROR: Invalid MD5 value in %s: %s\n' "$env_name" "$expected" >&2
        return 1
    fi
    actual="$(md5_file "$path")"
    if [[ "$actual" != "$expected" ]]; then
        printf 'ERROR: MD5 mismatch for %s: expected %s, got %s\n' \
            "$path" "$expected" "$actual" >&2
        return 1
    fi
    log "verified author-published MD5 for $prefix md5=$actual"
}

resolved_url() {
    local prefix="$1"
    local default_url="$2"
    local default_file="$3"
    local url_name="${prefix}_URL"
    local repo_name="${prefix}_HF_REPO"
    local revision_name="${prefix}_HF_REVISION"
    local file_name="${prefix}_HF_FILE"
    local repo_type_name="${prefix}_HF_REPO_TYPE"
    local direct_url="${!url_name:-}"
    local hf_repo="${!repo_name:-}"
    local revision="${!revision_name:-main}"
    local hf_file="${!file_name:-$default_file}"
    local repo_type="${!repo_type_name:-model}"
    if [[ -n "$direct_url" ]]; then
        printf '%s' "$direct_url"
    elif [[ -n "$hf_repo" ]]; then
        case "$repo_type" in
            model) printf '%s/%s/resolve/%s/%s' "${HF_ENDPOINT%/}" "$hf_repo" "$revision" "$hf_file" ;;
            dataset) printf '%s/datasets/%s/resolve/%s/%s' "${HF_ENDPOINT%/}" "$hf_repo" "$revision" "$hf_file" ;;
            *) die "$repo_type_name must be model or dataset, got: $repo_type" ;;
        esac
    else
        printf '%s' "$default_url"
    fi
}

verify_file() {
    local prefix="$1"
    local path="$2"
    local expected=""
    local actual=""
    [[ -s "$path" ]] || die "Missing or empty gate asset: $path"
    actual="$(sha256_file "$path")"
    expected="$(expected_sha256 "$prefix" "$path")"
    if [[ -n "$expected" && "$actual" != "$expected" ]]; then
        die "SHA256 mismatch for $path: expected $expected, got $actual"
    fi
    if [[ -n "$expected" ]]; then
        log "verified $prefix sha256=$actual path=$path"
    else
        log "present $prefix sha256=$actual path=$path (no trusted expected SHA was supplied)"
    fi
    verify_published_md5 "$prefix" "$path" || return 1
}

download_or_reuse() {
    local prefix="$1"
    local path="$2"
    local url="$3"
    local expected=""
    local actual=""
    local partial="${path}.partial"
    if [[ -s "$path" ]]; then
        log "reusing preinstalled file: $path"
        verify_file "$prefix" "$path"
        return
    fi
    command -v curl >/dev/null 2>&1 || die "curl is required to download gate assets"
    mkdir -p "$(dirname -- "$path")"
    log "downloading $prefix from $url"
    if ! curl --fail --location \
        --connect-timeout 15 --retry 3 --retry-delay 3 --retry-all-errors \
        --continue-at - --output "$partial" "$url"; then
        die "Official source is unreachable for $prefix. No completed file was installed. Preinstall it at $path, then rerun with --verify; set ${prefix}_SHA256 when a trusted checksum is available."
    fi
    [[ -s "$partial" ]] || die "Downloaded file is empty for $prefix: $partial"
    actual="$(sha256_file "$partial")"
    expected="$(expected_sha256 "$prefix" "$path")"
    if [[ -n "$expected" && "$actual" != "$expected" ]]; then
        rm -f "$partial"
        die "Downloaded $prefix failed SHA256: expected $expected, got $actual"
    fi
    if ! verify_published_md5 "$prefix" "$partial"; then
        rm -f "$partial"
        die "Downloaded $prefix failed its published MD5"
    fi
    mv "$partial" "$path"
    log "installed $prefix sha256=$actual path=$path"
}

write_manifest() {
    local temporary=""
    local path=""
    mkdir -p "$(dirname -- "$SHA256_MANIFEST")"
    temporary="$(mktemp "${SHA256_MANIFEST}.tmp.XXXXXX")"
    for path in "$PANNS_MOBILENET_PATH"; do
        if [[ -s "$path" ]]; then
            printf '%s  %s\n' "$(sha256_file "$path")" "$(path_token "$path")" >> "$temporary"
        fi
    done
    mv "$temporary" "$SHA256_MANIFEST"
    log "wrote observed SHA256 manifest: $SHA256_MANIFEST"
}

asset_fields() {
    case "$1" in
        panns-mobilenet)
            printf '%s\t%s\t%s\t%s\n' \
                PANNS_MOBILENET "$PANNS_MOBILENET_PATH" \
                "$PANNS_MOBILENET_OFFICIAL_URL" 'MobileNetV1_mAP=0.389.pth'
            ;;
        *) die "Unknown gate asset target: $1" ;;
    esac
}

MODE=download
TARGETS=()
for argument in "$@"; do
    case "$argument" in
        --verify) MODE=verify ;;
        --dry-run) MODE=dry-run ;;
        -h|--help|help) usage; exit 0 ;;
        all)
            TARGETS+=(panns-mobilenet)
            ;;
        panns-mobilenet)
            TARGETS+=("$argument")
            ;;
        *) usage >&2; die "Unknown argument: $argument" ;;
    esac
done
if (( ${#TARGETS[@]} == 0 )); then
    TARGETS=(panns-mobilenet)
fi

for target in "${TARGETS[@]}"; do
    IFS=$'\t' read -r prefix path official_url filename < <(asset_fields "$target")
    url="$(resolved_url "$prefix" "$official_url" "$filename")"
    case "$MODE" in
        dry-run)
            printf '%s\tpath=%s\turl=%s\n' "$prefix" "$path" "$url"
            ;;
        verify) verify_file "$prefix" "$path" ;;
        download) download_or_reuse "$prefix" "$path" "$url" ;;
    esac
done

if [[ "$MODE" == download ]]; then
    write_manifest
fi

printf '\nRuntime weight paths:\n'
printf '  export PANNS_MOBILENET_PATH=%q\n' "$PANNS_MOBILENET_PATH"

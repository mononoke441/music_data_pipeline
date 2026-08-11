#!/usr/bin/env python3
"""Step 0: inventory audio/video assets without creating converted files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
from pathlib import Path
from shutil import which
from typing import Any, Dict, List, Optional, Set, Tuple

from annotation_storage import source_relpath_for_audio
from pipeline_core import PIPELINE_VERSION, stable_audio_id
from pipeline_progress import pipeline_tqdm


AUDIO_EXTS = {
    ".mp3",
    ".flac",
    ".wav",
    ".m4a",
    ".aac",
    ".ogg",
    ".oga",
    ".wma",
    ".opus",
    ".ape",
    ".alac",
    ".aiff",
    ".aif",
    ".amr",
    ".mka",
    ".dts",
}
VIDEO_EXTS = {
    ".mp4",
    ".avi",
    ".mkv",
    ".mov",
    ".flv",
    ".wmv",
    ".webm",
    ".m4v",
    ".ts",
    ".mts",
    ".m2ts",
    ".vob",
    ".3gp",
    ".rm",
    ".rmvb",
}


def run_cmd(command: List[str], timeout: int) -> Tuple[int, str, str]:
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=timeout,
        )
        return completed.returncode, completed.stdout, completed.stderr
    except subprocess.TimeoutExpired as error:
        return 124, error.stdout or "", error.stderr or "timeout"


def ffprobe_duration_seconds(path: str) -> Optional[float]:
    code, output, _ = run_cmd(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        timeout=120,
    )
    if code != 0 or not output.strip():
        return None
    try:
        value = float(output.strip())
    except ValueError:
        return None
    return value if math.isfinite(value) and value > 0 else None


def decode_probe(path: str, probe_seconds: float) -> Tuple[bool, str]:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-t",
        f"{probe_seconds:.3f}",
        "-i",
        path,
        "-map",
        "0:a:0",
        "-vn",
        "-f",
        "null",
        "-",
    ]
    code, _, error = run_cmd(command, timeout=180)
    if code == 0:
        return True, ""
    return False, error.strip()[:500] or "ffmpeg_decode_failed"


def inventory_stage_version(probe_seconds: float) -> str:
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
    return f"inventory-v2:probe={probe_seconds:.3f}:code={source_hash}"


def reusable_inventory_record(
    record: Dict[str, Any],
    current_id: str,
    retry_failures: bool,
) -> bool:
    if record.get("audio_id") != current_id:
        return False
    status = record.get("decode_status")
    if status == "ok":
        return not record.get("error")
    if status == "failed":
        return not retry_failures
    return False


def inventory_one(arguments: Tuple[str, str, float, str, str]) -> Dict[str, Any]:
    audio_path, data_root, probe_seconds, stage_version, audio_id = arguments
    try:
        source_relpath = source_relpath_for_audio(audio_path, data_root)
    except Exception as error:
        audio_id = ""
        return {
            "audio_id": audio_id,
            "audio_path": audio_path,
            "source_relpath": None,
            "duration": None,
            "decode_status": "failed",
            "error": f"stat_error: {error}",
            "pipeline_version": PIPELINE_VERSION,
            "stage_versions": {"inventory": stage_version},
        }

    duration = ffprobe_duration_seconds(audio_path)
    if duration is None:
        return {
            "audio_id": audio_id,
            "audio_path": audio_path,
            "source_relpath": source_relpath,
            "duration": None,
            "decode_status": "failed",
            "error": "duration_probe_failed",
            "pipeline_version": PIPELINE_VERSION,
            "stage_versions": {"inventory": stage_version},
        }

    decode_ok, decode_error = decode_probe(audio_path, probe_seconds)
    return {
        "audio_id": audio_id,
        "audio_path": audio_path,
        "source_relpath": source_relpath,
        "duration": round(duration, 6),
        "decode_status": "ok" if decode_ok else "failed",
        "error": None if decode_ok else f"decode_probe_failed: {decode_error}",
        "pipeline_version": PIPELINE_VERSION,
        "stage_versions": {"inventory": stage_version},
    }


def scan_media(root: Path) -> List[str]:
    paths: List[str] = []
    supported = AUDIO_EXTS | VIDEO_EXTS
    for directory, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.startswith("."):
                continue
            path = Path(directory) / filename
            if path.suffix.lower() in supported:
                paths.append(str(Path(os.path.abspath(path))))
    return sorted(paths)


def load_existing(path: Path) -> Dict[str, Dict[str, Any]]:
    existing: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return existing
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except Exception:
                continue
            audio_path = record.get("audio_path")
            if isinstance(audio_path, str):
                existing[audio_path] = record
    return existing


HASH_CACHE_SCHEMA = "content-sha256-stat-cache-v1"


def file_identity(path: str) -> Dict[str, int]:
    """Return cheap fields that change whenever source bytes are replaced.

    ``ctime_ns`` prevents a same-size rewrite followed by restoring ``mtime``
    from silently reusing an old content hash. Device/inode also invalidate a
    path whose underlying file was atomically replaced.
    """

    stat = os.stat(path, follow_symlinks=True)
    return {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
    }


def load_hash_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    cached: Dict[str, Dict[str, Any]] = {}
    if not path.is_file():
        return cached
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except Exception:
                continue
            if not isinstance(value, dict) or value.get("schema") != HASH_CACHE_SCHEMA:
                continue
            audio_path = value.get("audio_path")
            audio_id = str(value.get("audio_id") or "").lower()
            if (
                not isinstance(audio_path, str)
                or len(audio_id) != 64
                or any(character not in "0123456789abcdef" for character in audio_id)
            ):
                continue
            cached[audio_path] = value
    return cached


def _cache_identity_matches(record: Dict[str, Any], identity: Dict[str, int]) -> bool:
    return all(record.get(field) == value for field, value in identity.items())


def resolve_content_ids(
    paths: List[str],
    data_root: str,
    cache: Dict[str, Dict[str, Any]],
    hash_jobs: int,
) -> Tuple[Dict[str, Optional[str]], List[Dict[str, Any]], int]:
    """Resolve exact content SHA256 IDs with a bounded parallel cold path."""

    if hash_jobs <= 0:
        raise ValueError("hash_jobs must be positive")
    identities: Dict[str, Dict[str, int]] = {}
    id_by_path: Dict[str, Optional[str]] = {}
    misses: List[str] = []
    for path in paths:
        try:
            identity = file_identity(path)
        except OSError:
            id_by_path[path] = None
            continue
        identities[path] = identity
        cached = cache.get(path)
        if cached is not None and _cache_identity_matches(cached, identity):
            id_by_path[path] = str(cached["audio_id"])
        else:
            misses.append(path)

    def hash_one(path: str) -> Tuple[str, Optional[str]]:
        try:
            return path, stable_audio_id(path, data_root)
        except Exception:
            return path, None

    hash_progress = pipeline_tqdm(
        total=len(paths),
        initial=len(paths) - len(misses),
        desc="0/7 inventory hash",
        unit="file",
    )
    if misses:
        with ThreadPoolExecutor(max_workers=min(hash_jobs, len(misses))) as executor:
            futures = {executor.submit(hash_one, path): path for path in misses}
            for future in as_completed(futures):
                path, audio_id = future.result()
                id_by_path[path] = audio_id
                hash_progress.update(1)
    hash_progress.close()

    cache_records = [
        {
            "schema": HASH_CACHE_SCHEMA,
            "audio_path": path,
            "audio_id": id_by_path[path],
            **identities[path],
        }
        for path in paths
        if path in identities and id_by_path.get(path) is not None
    ]
    return id_by_path, cache_records, len(paths) - len(misses)


def write_hash_cache(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def attach_duplicate_metadata(
    record: Dict[str, Any], paths: List[str], data_root: str | Path
) -> Dict[str, Any]:
    """Return one canonical asset record for identical-content paths."""
    value = dict(record)
    value["audio_path"] = paths[0]
    value["source_relpath"] = source_relpath_for_audio(paths[0], data_root)
    value["duplicate_paths"] = paths[1:]
    value["duplicate_count"] = max(0, len(paths) - 1)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inventory audio/video files without writing converted media files."
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default="data.jsonl")
    parser.add_argument("--fail-log", default="calc_failures.log")
    parser.add_argument("--jobs", type=int, default=0)
    parser.add_argument(
        "--hash-jobs",
        type=int,
        default=8,
        help="Bounded parallel workers for cold content SHA256 reads",
    )
    parser.add_argument(
        "--hash-cache",
        help="Internal path/stat-to-content-SHA cache (defaults beside --out)",
    )
    parser.add_argument("--decode-probe-seconds", type=float, default=1.0)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="Re-probe unchanged assets that previously failed inventory",
    )
    # Kept so an old command line fails with an actionable message instead of
    # silently writing MP3 files, which the new pipeline forbids.
    parser.add_argument("--overwrite-mp3", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--bitrate", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.overwrite_mp3 or args.bitrate:
        parser.error(
            "video-to-MP3 conversion was removed; video audio is decoded in memory"
        )
    for binary in ("ffprobe", "ffmpeg"):
        if which(binary) is None:
            parser.error(f"{binary} is required and was not found in PATH")
    if args.hash_jobs <= 0:
        parser.error("hash jobs must be positive")

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        parser.error(f"root is not a directory: {root}")
    output = Path(args.out).expanduser().resolve()
    failures = Path(args.fail_log).expanduser().resolve()
    hash_cache = (
        Path(args.hash_cache).expanduser().resolve()
        if args.hash_cache
        else output.with_name(
            (output.name[:-6] if output.name.endswith(".jsonl") else output.name)
            + ".hash_cache.jsonl"
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    failures.parent.mkdir(parents=True, exist_ok=True)

    media_all = scan_media(root)
    jobs = args.jobs if args.jobs > 0 else max(1, os.cpu_count() or 1)
    stage_version = inventory_stage_version(args.decode_probe_seconds)
    existing = load_existing(output) if args.resume else {}
    existing_by_id = {
        str(record.get("audio_id")): record
        for record in existing.values()
        if record.get("audio_id")
    }
    # stable_audio_id is content-based.  Hashing here lets us collapse exact
    # duplicates before ffprobe/decode and makes the canonical path independent
    # of multiprocessing completion order.
    id_by_path, hash_cache_records, reused_hashes = resolve_content_ids(
        media_all,
        str(root),
        load_hash_cache(hash_cache) if args.resume else {},
        args.hash_jobs,
    )
    write_hash_cache(hash_cache, hash_cache_records)
    grouped: Dict[str, List[str]] = {}
    for path in media_all:
        key = id_by_path[path] or f"path-error:{path}"
        grouped.setdefault(key, []).append(path)
    media_groups = [
        sorted(paths, key=lambda path: source_relpath_for_audio(path, root))
        for _, paths in sorted(
            grouped.items(),
            key=lambda item: source_relpath_for_audio(item[1][0], root),
        )
    ]
    canonical_paths = [paths[0] for paths in media_groups]
    groups_by_path = {paths[0]: paths for paths in media_groups}
    done: Set[str] = set()
    cached_by_path: Dict[str, Dict[str, Any]] = {}
    for path in canonical_paths:
        current_id = id_by_path[path]
        if current_id is None:
            continue
        record = existing_by_id.get(current_id) or existing.get(path)
        if not record:
            continue
        if reusable_inventory_record(
            record,
            current_id,
            args.retry_failures,
        ):
            done.add(path)
            cached_by_path[path] = attach_duplicate_metadata(
                record, groups_by_path[path], root
            )
    media = [path for path in canonical_paths if path not in done]
    total_ok = 0
    total_failed = 0
    updated: Dict[str, Dict[str, Any]] = {}
    # Reuse the content SHA calculated during duplicate grouping. Re-hashing
    # canonical files inside inventory_one would read every unique asset twice.
    work = (
        (
            path,
            str(root),
            args.decode_probe_seconds,
            stage_version,
            str(id_by_path[path]),
        )
        for path in media
    )
    probe_progress = pipeline_tqdm(
        total=len(canonical_paths),
        initial=len(done),
        desc="0/7 inventory probe",
        unit="file",
    )
    if media:
        active_jobs = min(jobs, len(media))
        with Pool(processes=active_jobs) as pool:
            for record in pool.imap_unordered(inventory_one, work, chunksize=16):
                updated[str(record["audio_path"])] = record
                if record["decode_status"] == "ok":
                    total_ok += 1
                else:
                    total_failed += 1
                probe_progress.set_postfix(
                    ok=total_ok, failed=total_failed, refresh=False
                )
                probe_progress.update(1)
    probe_progress.close()

    if args.resume:
        final_records = [
            attach_duplicate_metadata(updated[path], groups_by_path[path], root)
            if path in updated
            else cached_by_path[path]
            for path in canonical_paths
        ]
        temporary = output.with_suffix(output.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for record in final_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        temporary.replace(output)
    else:
        mode = "a" if args.append else "w"
        with output.open(mode, encoding="utf-8") as handle:
            for path in media:
                value = attach_duplicate_metadata(
                    updated[path], groups_by_path[path], root
                )
                handle.write(json.dumps(value, ensure_ascii=False) + "\n")

    with failures.open("w", encoding="utf-8") as failure_handle:
        if args.resume:
            failure_records = final_records
        else:
            failure_records = list(updated.values())
        for record in failure_records:
            if record["decode_status"] != "ok":
                failure_handle.write(
                    f"{record['audio_path']}\t{record.get('error') or 'unknown'}\n"
                )

    print(
        f"[inventory] done total={len(canonical_paths)} cached={len(done)} "
        f"processed={len(media)} failed={total_failed}"
    )


if __name__ == "__main__":
    main()

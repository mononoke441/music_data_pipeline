#!/usr/bin/env python3
"""Path-derived, one-audio-per-JSON storage for final annotations."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Iterator, Mapping, Tuple


MAX_FILENAME_BYTES = 255


def lexical_absolute_path(path: str | Path) -> Path:
    """Return an absolute path without resolving symlink targets."""

    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def normalize_source_relpath(value: str) -> str:
    text = str(value).strip()
    if not text or "\x00" in text:
        raise ValueError("source_relpath is empty or contains NUL")
    path = PurePosixPath(text)
    if path.is_absolute() or path == PurePosixPath(".") or ".." in path.parts:
        raise ValueError(f"unsafe source_relpath={value!r}")
    normalized = path.as_posix()
    if not normalized or normalized == ".":
        raise ValueError(f"invalid source_relpath={value!r}")
    return normalized


def source_relpath_for_audio(audio_path: str | Path, input_root: str | Path) -> str:
    root = lexical_absolute_path(input_root)
    audio = lexical_absolute_path(audio_path)
    try:
        relative = audio.relative_to(root)
    except ValueError as error:
        raise ValueError(f"audio path is outside input root: audio={audio} root={root}") from error
    return normalize_source_relpath(PurePosixPath(*relative.parts).as_posix())


def _truncate_utf8(text: str, maximum_bytes: int) -> str:
    if maximum_bytes <= 0:
        return ""
    encoded = text.encode("utf-8")
    if len(encoded) <= maximum_bytes:
        return text
    return encoded[:maximum_bytes].decode("utf-8", errors="ignore")


def annotation_relative_path(source_relpath: str) -> PurePosixPath:
    normalized = normalize_source_relpath(source_relpath)
    source = PurePosixPath(normalized)
    output_name = source.name + ".json"
    if len(output_name.encode("utf-8")) > MAX_FILENAME_BYTES:
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
        suffix = f"__p{digest}.json"
        prefix = _truncate_utf8(
            source.name,
            MAX_FILENAME_BYTES - len(suffix.encode("utf-8")),
        )
        output_name = (prefix or "audio") + suffix
    return source.parent / output_name


def annotation_path(annotations_dir: str | Path, source_relpath: str) -> Path:
    relative = annotation_relative_path(source_relpath)
    return Path(annotations_dir).joinpath(*relative.parts)


def atomic_write_json(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(dict(record), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def _recover_interrupted_publish(destination: Path, staging: Path, backup: Path) -> None:
    if not destination.exists() and backup.exists():
        os.replace(backup, destination)
    elif destination.exists() and backup.exists():
        _remove_path(backup)
    if staging.exists():
        _remove_path(staging)


def publish_annotation_records(
    records: Iterable[Mapping[str, Any]],
    annotations_dir: str | Path,
) -> Dict[str, int]:
    """Write and verify a complete tree before swapping it into place."""

    destination = Path(annotations_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.staging"
    backup = destination.parent / f".{destination.name}.previous"
    _recover_interrupted_publish(destination, staging, backup)
    staging.mkdir(parents=True)

    expected: Dict[str, str] = {}
    counts = {"total": 0, "song": 0, "instrumental": 0}
    try:
        for source_record in records:
            record = dict(source_record)
            source_relpath = normalize_source_relpath(str(record.get("source_relpath") or ""))
            record["source_relpath"] = source_relpath
            relative = annotation_relative_path(source_relpath).as_posix()
            if relative in expected:
                raise ValueError(
                    f"annotation path collision: {source_relpath!r} and "
                    f"{expected[relative]!r} both map to {relative!r}"
                )
            expected[relative] = source_relpath
            atomic_write_json(staging.joinpath(*PurePosixPath(relative).parts), record)
            counts["total"] += 1
            content_type = str(record.get("content_type") or "")
            if content_type in {"song", "instrumental"}:
                counts[content_type] += 1

        actual: Dict[str, str] = {}
        for path in sorted(staging.rglob("*.json")):
            relative = path.relative_to(staging).as_posix()
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            if not isinstance(value, dict):
                raise ValueError(f"annotation is not an object: {path}")
            actual[relative] = normalize_source_relpath(str(value.get("source_relpath") or ""))
        if actual != expected:
            raise ValueError("written annotation tree does not match expected source paths")
    except Exception:
        _remove_path(staging)
        raise

    if destination.exists():
        _remove_path(backup)
        os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except Exception:
        if backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    _remove_path(backup)
    return counts


def iter_annotation_records(
    annotations_dir: str | Path,
) -> Iterator[Tuple[Path, Dict[str, Any]]]:
    root = Path(annotations_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"annotations directory not found: {root}")
    for path in sorted(root.rglob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError(f"annotation is not a JSON object: {path}")
        yield path, value

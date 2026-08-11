"""Stage provenance and input-bound resume validation.

This module contains no Ray orchestration. Stage fingerprints describe the
feature-producing code, artifacts, dependencies and contracts used for a
result. They are provenance only: completed tasks resume by stable input/task
identity and result completeness, not by a matching stage version.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


# Only sources that can change successful CPU-MIR feature values belong here.
# Driver scheduling, progress reporting, logging and crash-recovery code are
# deliberately excluded so operational changes do not invalidate expensive
# Chordino/BeatNet/KeyExtractor results.
CPU_MIR_SEMANTIC_SOURCE_NAMES = (
    "audio_info.py",
    "sub_models/base_model.py",
    "sub_models/pipeline_model.py",
    "sub_models/chordino_model.py",
    "sub_models/beatnet_model.py",
    "sub_models/essentia_model.py",
)
CPU_MIR_RESULT_ENVELOPE_SCHEMA = "music-cpu-result-envelope-v1"
CPU_MIR_INPUT_ADAPTER_SCHEMA = "audio-info-jsonl-adapter-v1"
CPU_MIR_SEMANTIC_INPUT_FIELDS = (
    "audio_id",
    "audio_path",
    "url",
    "path",
    "audio_bytes",
    "bucket",
    "object_key",
    "endpoint",
    "secure",
    "start",
    "end",
    "duration",
    "target_sample_rate",
)
LEGACY_RUNTIME_SOURCE_NAMES = (
    "ray_inference.py",
    "runtime_integrity.py",
    "task_tracker.py",
    "workers/model_worker.py",
    "workers/saver_worker.py",
    "sub_models/pipeline_model.py",
    "sub_models/chordino_model.py",
    "sub_models/beatnet_model.py",
    "sub_models/essentia_model.py",
)


def _json_safe_fingerprint_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest(), "size": len(value)}
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe_fingerprint_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in {"access_key", "secret_key", "presigned_url", "url"}
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_fingerprint_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def build_task_manifest(
    data_loader: Any,
    *,
    semantic_fields: Optional[Sequence[str]] = None,
) -> Tuple[Dict[int, str], str]:
    """Build stable task identities and an order-independent input fingerprint."""
    task_map: Dict[int, str] = {}
    record_hashes: List[str] = []
    seen: Dict[str, int] = {}
    for raw_id, raw_record in data_loader._raw_iter():
        fingerprint_record = raw_record
        if semantic_fields is not None and isinstance(raw_record, Mapping):
            fingerprint_record = {
                key: raw_record[key] for key in semantic_fields if key in raw_record
            }
        safe_record = _json_safe_fingerprint_value(fingerprint_record)
        encoded = json.dumps(
            safe_record, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
        record_hash = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        record_hashes.append(record_hash)
        explicit = (
            str(raw_record.get("audio_id") or "").strip()
            if isinstance(raw_record, Mapping)
            else ""
        )
        task_key = explicit or f"record:{record_hash}"
        if task_key in seen:
            raise ValueError(
                f"Duplicate stable task key {task_key!r} at indices {seen[task_key]} and {raw_id}; "
                "audio_id must be unique before CPU MIR inference"
            )
        seen[task_key] = int(raw_id)
        task_map[int(raw_id)] = task_key
    input_fingerprint = hashlib.sha256(
        "\n".join(sorted(record_hashes)).encode("utf-8")
    ).hexdigest()
    return task_map, input_fingerprint


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _SemanticAstNormalizer(ast.NodeTransformer):
    """Ignore formatting, docstrings and standalone logger calls."""

    def visit_Expr(self, node: ast.Expr) -> Any:  # noqa: N802
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        call = node.value
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "logger"
        ):
            return None
        return self.generic_visit(node)


def _semantic_python_hash(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    normalized = _SemanticAstNormalizer().visit(tree)
    ast.fix_missing_locations(normalized)
    encoded = ast.dump(normalized, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _artifact_signatures(
    logical_name: str,
    package_name: str,
    patterns: Tuple[str, ...],
) -> Dict[str, Any]:
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        return {
            "logical_name": logical_name,
            "status": "package_not_found",
            "files": [],
        }
    roots = [Path(path) for path in (spec.submodule_search_locations or [])]
    if not roots and spec.origin:
        roots = [Path(spec.origin).resolve().parent]
    matches: Dict[str, Path] = {}
    for root in roots:
        for pattern in patterns:
            for path in root.glob(pattern):
                if path.is_file():
                    matches[_hash_file(path)] = path
    return {
        "logical_name": logical_name,
        "status": "ok" if matches else "artifact_not_found",
        "files": [
            {
                "name": path.name,
                "size": path.stat().st_size,
                "sha256": digest,
            }
            for digest, path in sorted(matches.items())
        ],
    }


def _cpu_mir_artifacts() -> Dict[str, Any]:
    artifacts = {
        "beatnet_model_1": _artifact_signatures(
            "beatnet_model_1", "BeatNet", ("models/model_1_weights.pt",)
        ),
        "chordino_nnls_chroma": _artifact_signatures(
            "chordino_nnls_chroma", "chord_extractor", ("_lib/nnls-chroma.so",)
        ),
        "essentia_native": _artifact_signatures(
            "essentia_native", "essentia", ("_essentia*.so",)
        ),
        "libsndfile_native": _artifact_signatures(
            "libsndfile_native", "_soundfile_data", ("libsndfile*.so*",)
        ),
    }
    vamp_path = os.environ.get("VAMP_PATH", "")
    if vamp_path:
        external: Dict[str, Path] = {}
        for raw_root in vamp_path.split(os.pathsep):
            root = Path(raw_root).expanduser()
            if root.is_dir():
                for path in root.glob("*nnls-chroma*.so"):
                    if path.is_file():
                        external[_hash_file(path)] = path
        if external:
            artifacts["vamp_path_nnls_chroma"] = {
                "logical_name": "vamp_path_nnls_chroma",
                "status": "ok",
                "files": [
                    {"name": path.name, "size": path.stat().st_size, "sha256": digest}
                    for digest, path in sorted(external.items())
                ],
            }
    return artifacts


def _path_signature(raw_path: Any) -> Any:
    if not raw_path:
        return None
    path = Path(str(raw_path)).expanduser()
    if not path.exists():
        return {"path": str(path), "missing": True}
    if path.is_file():
        stat = path.stat()
        return {
            "path": str(path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": _hash_file(path),
        }
    entries = []
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        stat = child.stat()
        entries.append([str(child.relative_to(path)), stat.st_size, stat.st_mtime_ns])
    return {"path": str(path.resolve()), "entries": entries}


def build_stage_fingerprint_payload(
    config: Any,
    model_path: Any = None,
    *,
    source_root: Optional[Path] = None,
) -> Dict[str, Any]:
    root = source_root or Path(__file__).resolve().parent
    model_type = str(getattr(config, "model_type", "unknown"))
    if model_type == "music_cpu_pipeline":
        schema = "music-cpu-semantic-v3"
        source_names = CPU_MIR_SEMANTIC_SOURCE_NAMES
        dependency_names = (
            "numpy",
            "scipy",
            "torch",
            "essentia-tensorflow",
            "BeatNet",
            "madmom",
            "librosa",
            "soundfile",
            "soxr",
            "numba",
            "llvmlite",
            "audioread",
            "vamp",
            "chord-extractor",
        )
        semantic_contract = {
            "features": ["chords", "beatnet", "key"],
            "result_envelope": CPU_MIR_RESULT_ENVELOPE_SCHEMA,
            "input_adapter": CPU_MIR_INPUT_ADAPTER_SCHEMA,
            "resolved_model_class": "MusicCpuPipelineModel",
        }
    else:
        # Non-CPU-MIR callers retain the conservative legacy contract.
        schema = "music-tools-runtime-v2"
        source_names = LEGACY_RUNTIME_SOURCE_NAMES
        dependency_names = ("ray", "numpy", "essentia", "BeatNet")
        semantic_contract = None
    sources = {
        name: (
            _semantic_python_hash(root / name)
            if semantic_contract is not None
            else _hash_file(root / name)
        )
        for name in source_names
        if (root / name).is_file()
    }
    # Scheduling knobs (worker count, queue/batch size) do not define model
    # semantics and must not invalidate otherwise identical results.
    semantic_config_keys = (
        ("model_type", "dataloader_type")
        if semantic_contract is not None
        else (
            "model_type",
            "dataloader_type",
            "output_sample_rate",
            "shuffle",
            "lance_prompt_key",
            "lance_offset",
            "lance_limit",
        )
    )
    config_payload = {
        key: getattr(config, key)
        for key in semantic_config_keys
        if hasattr(config, key)
    }
    dependencies = {}
    for package in dependency_names:
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = "not-installed"
    payload = {
        "schema": schema,
        "model_type": model_type,
        "model_path": None
        if semantic_contract is not None
        else _path_signature(model_path),
        "config": _json_safe_fingerprint_value(config_payload),
        "sources": sources,
        "dependencies": dependencies,
    }
    if semantic_contract is not None:
        payload["semantic_contract"] = semantic_contract
        payload["artifacts"] = _cpu_mir_artifacts()
    return payload


def build_stage_fingerprint(config: Any, model_path: Any = None) -> str:
    payload = build_stage_fingerprint_payload(config, model_path=model_path)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def write_stage_fingerprint_manifest(
    output_path: str,
    stage_fingerprint: str,
    payload: Mapping[str, Any],
) -> None:
    target = Path(output_path) / "stage_fingerprint.json"
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(
            {"stage_fingerprint": stage_fingerprint, "payload": payload},
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)


def validate_existing_results(
    results_path: str,
    stage_name: str,
    had_versioned_progress: bool,
    required_payload_fields: Sequence[str] = (),
) -> set[str]:
    if not os.path.exists(results_path) or os.path.getsize(results_path) == 0:
        return set()
    if not had_versioned_progress:
        raise RuntimeError(
            f"{results_path} predates safe versioned progress tracking; use a new output directory"
        )
    records = normalize_results_file(results_path)
    return {
        key
        for key, record in records.items()
        if _record_stage_status(record, stage_name) == "ok"
        and _has_required_payload(record, stage_name, required_payload_fields)
    }


def _result_task_key(record: Mapping[str, Any]) -> str:
    return str(record.get("runtime_task_key") or record.get("audio_id") or "")


def _record_stage_status(record: Mapping[str, Any], stage_name: str) -> str:
    return str((record.get("stage_status") or {}).get(stage_name) or "error")


def _has_required_payload(
    record: Mapping[str, Any], stage_name: str, required_fields: Sequence[str]
) -> bool:
    if not required_fields:
        return True
    payload = record.get(stage_name)
    if not isinstance(payload, Mapping):
        return False
    return all(payload.get(field) not in (None, "", [], {}) for field in required_fields)


def _atomic_write_jsonl(path: str, records: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        dict(record),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def load_latest_results(results_path: str) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(results_path):
        return latest
    with open(results_path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except Exception as error:
                raise RuntimeError(
                    f"Invalid JSON at {results_path}:{line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise RuntimeError(
                    f"{results_path}:{line_number} must contain a JSON object"
                )
            task_key = _result_task_key(record)
            if not task_key:
                raise RuntimeError(
                    f"{results_path}:{line_number} has no runtime_task_key/audio_id"
                )
            record["runtime_task_key"] = task_key
            latest[task_key] = record
    return latest


def normalize_results_file(
    results_path: str,
    *,
    task_order: Optional[Mapping[str, int]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Atomically reduce an append-era result file to one latest row per task."""

    latest = load_latest_results(results_path)
    if not latest and not os.path.exists(results_path):
        return latest
    order = task_order or {}
    ordered_keys = sorted(
        latest,
        key=lambda key: (int(order.get(key, 2**63 - 1)), key),
    )
    _atomic_write_jsonl(results_path, (latest[key] for key in ordered_keys))
    return {key: latest[key] for key in ordered_keys}


def merge_results_atomically(
    results_path: str,
    new_records: Iterable[Mapping[str, Any]],
    *,
    task_order: Optional[Mapping[str, int]] = None,
) -> Dict[str, Dict[str, Any]]:
    latest = load_latest_results(results_path)
    for raw_record in new_records:
        record = dict(raw_record)
        task_key = _result_task_key(record)
        if not task_key:
            raise RuntimeError("New result has no runtime_task_key/audio_id")
        record["runtime_task_key"] = task_key
        latest[task_key] = record
    order = task_order or {}
    ordered_keys = sorted(
        latest,
        key=lambda key: (int(order.get(key, 2**63 - 1)), key),
    )
    _atomic_write_jsonl(results_path, (latest[key] for key in ordered_keys))
    return {key: latest[key] for key in ordered_keys}


def _load_tracker_manifest(manifest_path: str) -> Tuple[Optional[str], Dict[int, str]]:
    run_fingerprint: Optional[str] = None
    mapping: Dict[int, str] = {}
    with open(manifest_path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except Exception as error:
                raise RuntimeError(
                    f"Invalid JSON at {manifest_path}:{line_number}: {error}"
                ) from error
            if record.get("type") == "manifest_meta":
                run_fingerprint = str(record.get("run_fingerprint") or "") or None
            elif record.get("task_id") is not None and record.get("task_key"):
                mapping[int(record["task_id"])] = str(record["task_key"])
    return run_fingerprint, mapping


def validate_result_tracker_coverage(
    *,
    results_path: str,
    stage_name: str,
    tracker: Any,
    task_map: Mapping[int, str],
    required_payload_fields: Sequence[str] = (),
) -> Dict[str, int]:
    """Validate the normalized result, tracker and manifest before publication."""

    task_order = {task_key: task_id for task_id, task_key in task_map.items()}
    records = normalize_results_file(results_path, task_order=task_order)
    expected_keys = set(task_map.values())
    actual_keys = set(records)
    if actual_keys != expected_keys:
        raise RuntimeError(
            "Result/input coverage mismatch: "
            f"missing={sorted(expected_keys - actual_keys)[:5]}, "
            f"unexpected={sorted(actual_keys - expected_keys)[:5]}"
        )

    manifest_run, manifest_map = _load_tracker_manifest(tracker.manifest_path)
    if manifest_run != tracker.run_fingerprint or manifest_map != dict(task_map):
        raise RuntimeError(
            "Tracker manifest does not match the active run/input mapping"
        )

    tracker_statuses = tracker.get_terminal_statuses()
    if set(tracker_statuses) != expected_keys:
        raise RuntimeError(
            "Tracker/input coverage mismatch: "
            f"missing={sorted(expected_keys - set(tracker_statuses))[:5]}, "
            f"unexpected={sorted(set(tracker_statuses) - expected_keys)[:5]}"
        )

    ok_count = 0
    error_count = 0
    for task_key, record in records.items():
        result_status = _record_stage_status(record, stage_name)
        expected_status = "ok" if result_status == "ok" else "error"
        tracker_status = tracker_statuses[task_key]
        if tracker_status != expected_status:
            raise RuntimeError(
                f"Result/tracker status mismatch for {task_key}: "
                f"result={result_status}, tracker={tracker_status}"
            )
        if expected_status == "ok":
            if not _has_required_payload(
                record, stage_name, required_payload_fields
            ):
                raise RuntimeError(
                    f"Successful {stage_name} result {task_key} is missing a required payload"
                )
            ok_count += 1
        else:
            stage_errors = record.get("stage_errors") or {}
            if not stage_errors.get(stage_name):
                raise RuntimeError(
                    f"Failed {stage_name} result {task_key} has no stage_errors entry"
                )
            error_count += 1
    return {"ok": ok_count, "error": error_count, "total": len(records)}


def reset_incompatible_stage_state(output_path: str, db_path: str) -> List[str]:
    """Remove unsafe legacy/input-mismatched resume state, preserving logs."""

    candidates = (
        Path(db_path),
        Path(f"{db_path}.manifest.jsonl"),
        Path(output_path) / "results.jsonl",
        Path(output_path) / "success.jsonl",
        Path(output_path) / "stage_fingerprint.json",
    )
    removed: List[str] = []
    for path in candidates:
        if path.exists():
            path.unlink()
            removed.append(str(path))
    return removed

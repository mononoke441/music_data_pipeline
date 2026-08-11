#!/usr/bin/env python3
"""Run the deterministic two-stage fast music gate over an inventory JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    TextIO,
    Tuple,
)

from fast_gate_core import (
    AudioSetMusicScorer,
    CascadeMusicGate,
    CascadeResult,
    DecisionThresholds,
    InvalidAudioError,
    build_backend,
    backend_source_provenance,
    load_production_gate_config,
    verify_checkpoint_sha256,
)
from pipeline_core import load_jsonl_with_truncated_tail_recovery
from pipeline_progress import pipeline_tqdm


STAGE_FINGERPRINT_SCHEMA = "fast-music-gate-semantic-v2"
DECISION_CONTRACT_VERSION = "sparse-audioset-cascade-v1"

# The original production run used a whole-file code hash. Its feature and
# decision semantics correspond exactly to the semantic digest below. Keeping
# this one audited alias preserves existing downstream caches; any model,
# config, threshold, precision, batch, or contract change receives a new ID.
CURRENT_SEMANTIC_DIGEST = (
    "a8bdce4b23e997ce7252bb43a61dac4dffd5e76d7e6367c14c96a32d8da623f7"
)
CURRENT_COMPATIBLE_STAGE_VERSION = "4be5ef43a012007a"
OUTPUT_NAMES = (
    "accepted.music.jsonl",
    "review.jsonl",
    "rejected.jsonl",
    "failures.jsonl",
)


def write_runtime_metrics(
    output_dir: Path,
    *,
    pending: int,
    total: int,
    processed: int,
    elapsed_seconds: float,
    counters: Mapping[str, int],
    stage_version: str,
) -> None:
    elapsed = max(float(elapsed_seconds), 0.0)
    payload = {
        "stage": "music_gate",
        "stage_version": stage_version,
        "input_records": int(total),
        "pending_records": int(pending),
        "processed_this_run": int(processed),
        "elapsed_seconds": round(elapsed, 6),
        "seconds_per_track": None if processed == 0 else round(elapsed / processed, 6),
        "tracks_per_second": None if elapsed <= 0 else round(processed / elapsed, 6),
        "counters": {str(key): int(value) for key, value in counters.items()},
    }
    path = output_dir / "runtime_metrics.json"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"invalid JSON at {source}:{line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise RuntimeError(f"expected a JSON object at {source}:{line_number}")
            yield value


def record_fingerprint(record: Mapping[str, Any]) -> str:
    ignored = {
        "music_gate",
        "stage_versions",
        "model_versions",
        "stage_status",
        "stage_errors",
        "music_gate_input_fingerprint",
        "status",
        "content_type",
        "content_confidence",
        "reason_codes",
    }
    payload = {
        str(key): value for key, value in record.items() if str(key) not in ignored
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_stage_fingerprint_payload(args: argparse.Namespace) -> Dict[str, Any]:
    """Describe only inputs that can change gate scores or decisions."""

    weights = Path(args.backend_weights).expanduser()
    stage_b_weights = Path(args.stage_b_backend_weights).expanduser()
    config_path = Path(args.config).expanduser()
    for path, label in (
        (weights, "backend weights"),
        (stage_b_weights, "Stage B backend weights"),
        (config_path, "pretrained gate config"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} was not found: {path}")
    return {
        "schema": STAGE_FINGERPRINT_SCHEMA,
        "decision_contract": DECISION_CONTRACT_VERSION,
        "backend": args.backend,
        "backend_repo": args.backend_repo_id,
        "backend_checkpoint_sha256": args.backend_checkpoint_sha256,
        "stage_b_backend": args.stage_b_backend,
        "stage_b_backend_repo": args.stage_b_backend_repo_id,
        "stage_b_backend_checkpoint_sha256": args.stage_b_backend_checkpoint_sha256,
        "precision": args.precision,
        "stage_b_precision": args.stage_b_precision,
        "batch_size": args.batch_size,
        "stage_b_batch_size": args.stage_b_batch_size,
        "backend_source_sha256": args.backend_source_sha256,
        "stage_b_backend_source_sha256": args.stage_b_backend_source_sha256,
        "config_sha256": _sha256_file(config_path),
        "config_version": args.config_version,
        "thresholds": [
            args.stage_a_reject,
            args.stage_a_accept,
            args.stage_b_reject,
            args.stage_b_accept,
        ],
    }


def _stage_version_for_digest(digest: str) -> str:
    if digest == CURRENT_SEMANTIC_DIGEST:
        return CURRENT_COMPATIBLE_STAGE_VERSION
    return digest[:16]


def build_stage_version(args: argparse.Namespace) -> str:
    payload = build_stage_fingerprint_payload(args)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return _stage_version_for_digest(digest)


def _artifact_path(config_path: str, value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config_path).expanduser().resolve().parent / path
    return str(path.resolve())


def resolve_production_config(
    args: argparse.Namespace,
    config: Mapping[str, Any],
) -> None:
    """Resolve CLI overrides against the immutable production config."""
    if args.config_version not in (None, config["config_version"]):
        raise RuntimeError("--config-version does not match the selected gate config")
    args.config_version = str(config["config_version"])

    expected = {
        "backend": str(config["backend"]),
        "stage_b_backend": str(config["stage_b_backend"]),
    }
    for field, selected in expected.items():
        supplied = getattr(args, field)
        if supplied is not None and supplied != selected:
            raise RuntimeError(
                f"--{field.replace('_', '-')}={supplied!r} does not match selected {selected!r}"
            )
        setattr(args, field, selected)
    if str(config["backend_architecture"]) != "mobilenet_v1":
        raise RuntimeError("only mobilenet_v1 backend architecture is supported")
    if str(config["stage_b_backend_architecture"]) != "mobilenet_v1":
        raise RuntimeError("only mobilenet_v1 stage-B backend architecture is supported")
    for field in ("precision", "stage_b_precision"):
        selected = str(config[field])
        supplied = getattr(args, field)
        if supplied is not None and supplied != selected:
            raise RuntimeError(
                f"--{field.replace('_', '-')}={supplied!r} does not match selected {selected!r}"
            )
        setattr(args, field, selected)
    for field in ("batch_size", "stage_b_batch_size"):
        selected = int(config[field])
        supplied = getattr(args, field)
        if supplied is not None and int(supplied) != selected:
            raise RuntimeError(
                f"--{field.replace('_', '-')}={supplied} differs from selected batch {selected}"
            )
        setattr(args, field, selected)

    for cli_field, artifact_field in (
        ("backend_weights", "backend_checkpoint"),
        ("stage_b_backend_weights", "stage_b_backend_checkpoint"),
    ):
        selected_path = _artifact_path(args.config, str(config[artifact_field]))
        supplied = getattr(args, cli_field)
        # Checkpoints may be relocated; the immutable SHA256 below, rather
        # than the path string, is authoritative.
        setattr(
            args,
            cli_field,
            str(Path(supplied).expanduser().resolve()) if supplied else selected_path,
        )

    for cli_field, artifact_field, path_field, identity_field in (
        ("backend_repo", "backend_repo", "backend_repo_path", "backend_repo_id"),
        (
            "stage_b_backend_repo",
            "stage_b_backend_repo",
            "stage_b_backend_repo_path",
            "stage_b_backend_repo_id",
        ),
    ):
        # Artifact repo values are stable provenance IDs, not machine-local
        # checkout paths.
        setattr(args, identity_field, str(config[artifact_field]))
        supplied = getattr(args, cli_field)
        setattr(
            args,
            cli_field,
            (
                str(Path(supplied).expanduser().resolve())
                if supplied
                else _artifact_path(args.config, str(config[path_field]))
            ),
        )

    for prefix in ("backend", "stage_b_backend"):
        source_field = f"{prefix}_source"
        actual_source = backend_source_provenance(
            getattr(args, prefix),
            getattr(args, f"{prefix}_weights"),
            getattr(args, f"{prefix}_repo"),
        )
        expected_source = config[source_field]
        if actual_source["sha256"] != expected_source["sha256"]:
            raise RuntimeError(
                f"{prefix.replace('_', ' ')} source SHA256 does not match gate config: "
                f"expected {expected_source['sha256']}, got {actual_source['sha256']}"
            )
        setattr(args, f"{prefix}_source_sha256", actual_source["sha256"])

    selected_thresholds = config["thresholds"]
    for field in (
        "stage_a_reject",
        "stage_a_accept",
        "stage_b_reject",
        "stage_b_accept",
    ):
        selected = float(selected_thresholds[field])
        supplied = getattr(args, field)
        if supplied is not None and abs(float(supplied) - selected) > 1e-12:
            raise RuntimeError(
                f"--{field.replace('_', '-')}={supplied} differs from selected threshold {selected}"
            )
        setattr(args, field, selected)

    args.backend_checkpoint_sha256 = verify_checkpoint_sha256(
        args.backend_weights,
        str(config["backend_checkpoint_sha256"]),
        "Stage A backend",
    )
    args.stage_b_backend_checkpoint_sha256 = verify_checkpoint_sha256(
        args.stage_b_backend_weights,
        str(config["stage_b_backend_checkpoint_sha256"]),
        "Stage B backend",
    )


def _atomic_rewrite(path: Path, lines: Sequence[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.writelines(lines)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_resume(output_dir: Path) -> Dict[str, str]:
    """Load complete terminal results keyed by their input fingerprints.

    Failure records are deliberately non-terminal.  They are validated and
    then cleared so the file always represents failures from the current retry
    attempt; their assets are retried when the command resumes. Stage/model
    versions remain provenance only and never invalidate a complete result.
    """
    done: Dict[str, str] = {}
    for name in OUTPUT_NAMES:
        path = output_dir / name
        if not path.exists():
            continue
        kept: List[str] = []
        for line_number, record in enumerate(
            load_jsonl_with_truncated_tail_recovery(path), 1
        ):
            if name == "failures.jsonl":
                continue
            kept.append(json.dumps(record, ensure_ascii=False) + "\n")
            audio_id = str(record.get("audio_id") or "")
            fingerprint = str(record.get("music_gate_input_fingerprint") or "")
            if not audio_id or not fingerprint:
                raise RuntimeError(
                    f"{path}:{line_number} lacks audio_id/music_gate_input_fingerprint"
                )
            previous = done.get(audio_id)
            if previous is not None and previous != fingerprint:
                raise RuntimeError(
                    f"conflicting resume fingerprints for audio_id={audio_id}"
                )
            if previous is not None:
                raise RuntimeError(
                    f"duplicate terminal output for audio_id={audio_id}"
                )
            done[audio_id] = fingerprint
        # Canonicalize partial last writes while validating the file.  Old
        # failures must not poison a later successful retry.
        _atomic_rewrite(path, kept)
    return done


def scan_inputs(input_path: str, done: Mapping[str, str]) -> Tuple[int, int]:
    seen: set[str] = set()
    unmatched = set(done)
    pending = 0
    total = 0
    for record in iter_jsonl(input_path):
        total += 1
        audio_id = str(record.get("audio_id") or "")
        if not audio_id:
            raise RuntimeError(f"fast music gate input row {total} has no audio_id")
        if audio_id in seen:
            raise RuntimeError(
                f"duplicate audio_id in fast music gate input: {audio_id}"
            )
        seen.add(audio_id)
        old_fingerprint = done.get(audio_id)
        if old_fingerprint is None:
            pending += 1
        elif old_fingerprint != record_fingerprint(record):
            raise RuntimeError(
                f"input metadata changed for completed audio_id={audio_id}; "
                "use a new output directory"
            )
        else:
            unmatched.discard(audio_id)
    if unmatched:
        preview = ", ".join(sorted(unmatched)[:5])
        raise RuntimeError(
            "music-gate outputs contain IDs absent from this input "
            f"({preview}); use a new output directory"
        )
    return pending, total


def _merge_map(
    record: MutableMapping[str, Any], key: str, values: Mapping[str, Any]
) -> None:
    merged = dict(record.get(key) or {})
    merged.update(values)
    record[key] = merged


def decorate_result(
    source: Mapping[str, Any],
    result: CascadeResult,
    stage_version: str,
) -> Tuple[Dict[str, Any], str]:
    record = dict(source)
    decision_map = {
        "accepted": ("music", "accepted", "music", "accepted.music.jsonl"),
        "review": ("review", "review", "unknown", "review.jsonl"),
        "rejected": ("non_music", "rejected", "non_music", "rejected.jsonl"),
    }
    music_decision, status, content_type, output_name = decision_map[result.decision]
    gate = result.as_music_gate()
    gate["decision"] = music_decision
    gate["stage_version"] = stage_version
    record.update(
        {
            "music_gate": gate,
            "music_gate_input_fingerprint": record_fingerprint(source),
            "status": status,
            "content_type": content_type,
            "content_confidence": round(float(result.probability), 6),
        }
    )
    _merge_map(record, "stage_versions", {"music_gate": stage_version})
    stage_backends = result.stage_backends or {
        "stage_a": result.backend,
        "stage_b": result.backend,
    }
    _merge_map(
        record,
        "model_versions",
        {
            "music_gate": (
                f"{stage_backends['stage_a']}->{stage_backends['stage_b']}:"
                f"{result.scoring_version}"
            ),
        },
    )
    _merge_map(record, "stage_status", {"music_gate": "ok"})
    stage_errors = dict(record.get("stage_errors") or {})
    stage_errors.pop("music_gate", None)
    record["stage_errors"] = stage_errors
    reason_codes = list(record.get("reason_codes") or [])
    reason = {
        "music": "fast_gate_music",
        "review": "fast_gate_gray_zone",
        "non_music": "fast_gate_non_music",
    }[music_decision]
    if reason not in reason_codes:
        reason_codes.append(reason)
    record["reason_codes"] = reason_codes
    return record, output_name


def decorate_failure(
    source: Mapping[str, Any],
    error: Exception,
    backend: str,
    scoring_version: str,
    stage_version: str,
) -> Dict[str, Any]:
    record = dict(source)
    message = f"{type(error).__name__}: {error}"
    record.update(
        {
            "music_gate": {
                "backend": backend,
                "scoring_version": scoring_version,
                "stage_probabilities": {"stage_a": [], "stage_b": []},
                "stage_scores": {"stage_a": None, "stage_b": None},
                "offsets": {"stage_a": [], "stage_b": []},
                "decision": "error",
                "probability": None,
                "stage_version": stage_version,
            },
            "music_gate_input_fingerprint": record_fingerprint(source),
            "stage_error": message,
        }
    )
    _merge_map(record, "stage_versions", {"music_gate": stage_version})
    _merge_map(
        record,
        "model_versions",
        {
            "music_gate": f"{backend}:{scoring_version}",
        },
    )
    _merge_map(record, "stage_status", {"music_gate": "error"})
    _merge_map(record, "stage_errors", {"music_gate": message})
    return record


def decorate_invalid_asset_rejection(
    source: Mapping[str, Any], stage_version: str, detail: Optional[str] = None
) -> Dict[str, Any]:
    """Reject inventory-level decode failures without invoking or poisoning the gate."""
    record = dict(source)
    inventory_failure = source.get("decode_status") not in (None, "ok")
    asset_error = str(detail or source.get("error") or "inventory_decode_failed")
    record.update(
        {
            "music_gate": {
                "backend": "not_run",
                "scoring_version": AudioSetMusicScorer.VERSION,
                "stage_probabilities": {"stage_a": [], "stage_b": []},
                "stage_scores": {"stage_a": None, "stage_b": None},
                "offsets": {"stage_a": [], "stage_b": []},
                "decision": "invalid_asset",
                "probability": None,
                "stage_version": stage_version,
            },
            "music_gate_input_fingerprint": record_fingerprint(source),
            "status": "rejected",
            "content_type": "invalid_asset",
            "content_confidence": 0.0,
            "stage_error": asset_error,
        }
    )
    _merge_map(record, "stage_versions", {"music_gate": stage_version})
    _merge_map(record, "model_versions", {"music_gate": "not_run"})
    _merge_map(record, "stage_status", {"music_gate": "skipped_invalid_asset"})
    _merge_map(
        record,
        "stage_errors",
        {"inventory" if inventory_failure else "music_gate": asset_error},
    )
    reason_codes = list(record.get("reason_codes") or [])
    reason = "inventory_decode_failed" if inventory_failure else "audio_decode_failed"
    if reason not in reason_codes:
        reason_codes.append(reason)
    record["reason_codes"] = reason_codes
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="inventory.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--backend",
        choices=("panns_mobilenet",),
        default=None,
    )
    parser.add_argument("--backend-weights", default=None)
    parser.add_argument("--backend-repo", default=None)
    parser.add_argument(
        "--stage-b-backend",
        choices=("panns_mobilenet",),
        default=None,
    )
    parser.add_argument("--stage-b-backend-weights", default=None)
    parser.add_argument("--stage-b-backend-repo", default=None)
    parser.add_argument(
        "--config",
        default=None,
        help="fixed zero-training gate config JSON",
    )
    parser.add_argument("--config-version", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default=None)
    parser.add_argument("--stage-b-precision", choices=("fp32", "bf16"), default=None)
    parser.add_argument("--decode-workers", type=int, default=4)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="must match the fixed Stage A batch",
    )
    parser.add_argument(
        "--stage-b-batch-size",
        type=int,
        default=None,
        help="must match the fixed Stage B batch",
    )
    parser.add_argument(
        "--track-buffer-size",
        type=int,
        default=0,
        help="tracks per pipeline buffer; 0 auto-sizes to the selected maximum batch",
    )
    parser.add_argument("--stage-a-reject", type=float, default=None)
    parser.add_argument("--stage-a-accept", type=float, default=None)
    parser.add_argument("--stage-b-reject", type=float, default=None)
    parser.add_argument("--stage-b-accept", type=float, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    started = time.perf_counter()
    args = build_parser().parse_args(argv)
    if not args.config:
        raise ValueError("--config is required")
    if args.decode_workers <= 0 or args.track_buffer_size < 0:
        raise ValueError(
            "decode-workers must be positive and track-buffer-size non-negative"
        )
    config = load_production_gate_config(args.config)
    resolve_production_config(args, config)
    if args.batch_size <= 0 or args.stage_b_batch_size <= 0:
        raise ValueError("fixed Stage A/B batch sizes must be positive")
    # Keep at least one selected batch plus the bounded prefetch queue.  Even
    # tracks whose three offsets collapse to one window can then overlap decode
    # with GPU inference instead of force-flushing only after every decode ends.
    minimum_track_buffer = (
        max(args.batch_size, args.stage_b_batch_size) + 2 * args.decode_workers
    )
    if args.track_buffer_size < minimum_track_buffer:
        requested_track_buffer = args.track_buffer_size
        args.track_buffer_size = minimum_track_buffer
        print(
            "[fast-music-gate] adjusted track-buffer-size "
            f"from {requested_track_buffer} to {args.track_buffer_size} "
            "so ffmpeg decode can overlap the selected GPU batch"
        )
    # Validate fixed threshold ordering before touching outputs.
    stage_a_thresholds = DecisionThresholds(args.stage_a_reject, args.stage_a_accept)
    stage_b_thresholds = DecisionThresholds(args.stage_b_reject, args.stage_b_accept)
    current_stage_version = build_stage_version(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    done = load_resume(output_dir) if args.resume else {}
    pending, total = scan_inputs(args.input, done)
    output_paths = {name: output_dir / name for name in OUTPUT_NAMES}
    if pending == 0:
        progress = pipeline_tqdm(
            total=total, initial=total, desc="1a/7 fast music gate", unit="track"
        )
        progress.close()
        for path in output_paths.values():
            path.touch(exist_ok=True)
        write_runtime_metrics(
            output_dir,
            pending=0,
            total=total,
            processed=0,
            elapsed_seconds=time.perf_counter() - started,
            counters={name: 0 for name in OUTPUT_NAMES},
            stage_version=current_stage_version,
        )
        print(
            f"[fast-music-gate] no pending assets; validated={total} "
            f"stage_version={current_stage_version}"
        )
        return

    # Load model dependencies only after resume validation identifies work.
    # Both stages use the same fixed AudioSet scoring contract.
    head = AudioSetMusicScorer(str(config["scoring"]["version"]))
    stage_b_head = AudioSetMusicScorer(str(config["scoring"]["version"]))
    backend = build_backend(
        args.backend,
        args.backend_weights,
        device=args.device,
        repo_path=args.backend_repo,
        precision=args.precision,
    )
    same_backend = (
        args.stage_b_backend == args.backend
        and args.stage_b_backend_weights == args.backend_weights
        and args.stage_b_backend_repo == args.backend_repo
        and args.stage_b_backend_repo_id == args.backend_repo_id
        and args.stage_b_precision == args.precision
    )
    stage_b_backend = (
        backend
        if same_backend
        else build_backend(
            args.stage_b_backend,
            args.stage_b_backend_weights,
            device=args.device,
            repo_path=args.stage_b_backend_repo,
            precision=args.stage_b_precision,
        )
    )
    gate = CascadeMusicGate(
        backend=backend,
        head=head,
        stage_a_thresholds=stage_a_thresholds,
        stage_b_thresholds=stage_b_thresholds,
        batch_size=args.batch_size,
        stage_b_batch_size=args.stage_b_batch_size,
        decode_workers=args.decode_workers,
        stage_b_backend=stage_b_backend,
        stage_b_head=stage_b_head,
    )
    mode = "a" if args.resume else "w"
    handles: Dict[str, TextIO] = {
        name: path.open(mode, encoding="utf-8") for name, path in output_paths.items()
    }
    counters = {name: 0 for name in OUTPUT_NAMES}
    progress = pipeline_tqdm(
        total=total,
        initial=total - pending,
        desc="1a/7 fast music gate",
        unit="track",
    )

    def save(name: str, record: Mapping[str, Any]) -> None:
        handles[name].write(json.dumps(record, ensure_ascii=False) + "\n")
        handles[name].flush()
        counters[name] += 1
        progress.update(1)
        progress.set_postfix(
            music=counters["accepted.music.jsonl"],
            review=counters["review.jsonl"],
            rejected=counters["rejected.jsonl"],
            failed=counters["failures.jsonl"],
            refresh=False,
        )

    def fail(source: Mapping[str, Any], error: Exception) -> None:
        save(
            "failures.jsonl",
            decorate_failure(
                source,
                error,
                f"{backend.name}->{stage_b_backend.name}",
                head.scoring_version,
                current_stage_version,
            ),
        )

    def process_buffer(buffer: Sequence[Mapping[str, Any]]) -> None:
        eligible: List[Mapping[str, Any]] = []
        for source in buffer:
            if source.get("decode_status") not in (None, "ok"):
                save(
                    "rejected.jsonl",
                    decorate_invalid_asset_rejection(source, current_stage_version),
                )
            else:
                eligible.append(source)
        if not eligible:
            return
        try:
            results = gate.classify_records(eligible)
        except Exception:
            # Isolate corrupt assets without giving up cross-track batching on
            # the normal path.  A backend-wide error remains explicit per row.
            for source in eligible:
                try:
                    result = gate.classify_records([source])[0]
                    record, name = decorate_result(
                        source, result, current_stage_version
                    )
                    save(name, record)
                except InvalidAudioError as error:
                    save(
                        "rejected.jsonl",
                        decorate_invalid_asset_rejection(
                            source, current_stage_version, str(error)
                        ),
                    )
                except Exception as error:
                    fail(source, error)
            return
        for source, result in zip(eligible, results):
            record, name = decorate_result(source, result, current_stage_version)
            save(name, record)

    try:
        buffer: List[Mapping[str, Any]] = []
        for source in iter_jsonl(args.input):
            audio_id = str(source.get("audio_id") or "")
            if audio_id in done:
                continue
            buffer.append(source)
            if len(buffer) >= args.track_buffer_size:
                process_buffer(buffer)
                buffer = []
        if buffer:
            process_buffer(buffer)
    finally:
        for handle in handles.values():
            handle.close()
        progress.close()
    processed = sum(counters.values())
    elapsed = time.perf_counter() - started
    write_runtime_metrics(
        output_dir,
        pending=pending,
        total=total,
        processed=processed,
        elapsed_seconds=elapsed,
        counters=counters,
        stage_version=current_stage_version,
    )
    print(
        f"[fast-music-gate] done stage_version={current_stage_version} counters={counters} "
        f"seconds_per_track={elapsed / processed if processed else 0.0:.6f}"
    )


if __name__ == "__main__":
    main()

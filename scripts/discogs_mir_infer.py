#!/usr/bin/env python3
"""Run shared Discogs MIR inference for fast-gate accepted music only.

The stage decodes tracks with bounded CPU prefetch, packs frames from multiple
tracks into bounded batches, and computes the EffNet backbone embedding exactly
once for every frame before sharing it across all five classification heads.
Inference failures are kept out of the vocal-routing review manifest and remain
retryable with ``--resume``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
MTP_DIR = ROOT / "MusicToolsPipeline"
sys.path.insert(0, str(MTP_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from pipeline_core import (  # noqa: E402
    PIPELINE_VERSION,
    decode_audio_range,
    iter_jsonl,
    load_jsonl_with_truncated_tail_recovery,
)
from pipeline_progress import pipeline_tqdm  # noqa: E402
from sub_models.discogs_onnx_model import (  # noqa: E402
    DiscogsModelPaths,
    DiscogsOnnxEngine,
)


STAGE_NAME = "discogs_mir"
MODEL_FINGERPRINT_SCHEMA = "discogs-model-semantic-v2"
STAGE_FINGERPRINT_SCHEMA = "discogs-mir-semantic-v2"
INFERENCE_CONTRACT_VERSION = "discogs-effnet-onnx-v1"
ROUTING_CONTRACT_VERSION = "discogs-vocal-fusion-v1"

# These aliases bind the existing WebSource cache to the equivalent semantic
# payload. Model artifacts, metadata, routing thresholds, or either contract
# changing will produce a new version automatically.
CURRENT_MODEL_SEMANTIC_DIGEST = (
    "e72f75450d9babdc5e20fad4175e611ed4ccec8f7ce2581cdf221eb63b7eb473"
)
CURRENT_COMPATIBLE_MODEL_VERSION = "5ab1aedcf96a460f"
CURRENT_STAGE_SEMANTIC_DIGEST = (
    "078c355776eaa56e66b7067bd0809d54b4b4be6118dbfbf6c8ada3188bebf5a5"
)
CURRENT_COMPATIBLE_STAGE_VERSION = "1c5cd958c702b7ce"
DISCOGS_VOCAL_SCORE_DECIMALS = 6
OUTPUT_NAMES = (
    "data.song.jsonl",
    "data.instrumental.jsonl",
    "review.jsonl",
    "failures.jsonl",
)
HEAD_NAMES = ("voice", "genre", "mood", "instrument", "danceability")

T = TypeVar("T")
R = TypeVar("R")


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
        "stage": STAGE_NAME,
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


@dataclass(frozen=True)
class FrameSlice:
    """A half-open frame span belonging to one audio item."""

    audio_index: int
    start: int
    stop: int

    @property
    def frame_count(self) -> int:
        return self.stop - self.start


@dataclass
class PreparedAudio:
    record: Dict[str, Any]
    patches: torch.Tensor | np.ndarray
    duration: float


@dataclass(frozen=True)
class PrefetchResult:
    item: Any
    value: Any = None
    error: Optional[Exception] = None


def plan_frame_batches(
    frame_counts: Sequence[int],
    max_frames: int,
) -> List[List[FrameSlice]]:
    """Plan bounded cross-audio batches without dropping or repeating frames."""

    limit = int(max_frames)
    if limit <= 0:
        raise ValueError("max_frames must be positive")
    batches: List[List[FrameSlice]] = []
    current: List[FrameSlice] = []
    remaining = limit
    for audio_index, raw_count in enumerate(frame_counts):
        count = int(raw_count)
        if count < 0:
            raise ValueError("frame counts must be non-negative")
        cursor = 0
        while cursor < count:
            take = min(remaining, count - cursor)
            current.append(FrameSlice(audio_index, cursor, cursor + take))
            cursor += take
            remaining -= take
            if remaining == 0:
                batches.append(current)
                current = []
                remaining = limit
    if current:
        batches.append(current)
    return batches


def bounded_prefetch_map(
    items: Iterable[T],
    function: Callable[[T], R],
    *,
    max_workers: int,
    prefetch: int,
) -> Iterator[PrefetchResult]:
    """Completion-ordered map with a strict bound on submitted futures.

    JSONL row order is not semantic anywhere in this stage; every record is
    joined by ``audio_id``.  Yielding whichever ffmpeg decode finishes first
    prevents one slow/corrupt source from holding ready tracks away from the
    GPU while preserving the configured memory/process bound.
    """

    worker_count = int(max_workers)
    pending_limit = int(prefetch)
    if worker_count <= 0:
        raise ValueError("max_workers must be positive")
    if pending_limit <= 0:
        raise ValueError("prefetch must be positive")
    worker_count = min(worker_count, pending_limit)
    iterator = iter(items)
    pending: Dict[Future[R], T] = {}

    with ThreadPoolExecutor(max_workers=worker_count) as executor:

        def submit_one() -> bool:
            try:
                item = next(iterator)
            except StopIteration:
                return False
            pending[executor.submit(function, item)] = item
            return True

        for _ in range(pending_limit):
            if not submit_one():
                break
        while pending:
            completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in completed:
                item = pending.pop(future)
                try:
                    result = PrefetchResult(item=item, value=future.result())
                except Exception as error:
                    result = PrefetchResult(item=item, error=error)
                # Refill only after removing a terminal future, so submitted
                # and running work never exceeds ``pending_limit``.
                submit_one()
                yield result


def fused_discogs_vocal_score(
    voice_mean: float,
    voice_coverage: float,
    longest_voice_sec: float,
) -> float:
    """Combine Discogs vocal evidence into one stable routing score."""

    values = (float(voice_mean), float(voice_coverage), float(longest_voice_sec))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Discogs voice analysis values must be finite")
    score = (
        0.55 * values[0]
        + 0.30 * values[1]
        + 0.15 * min(1.0, max(0.0, values[2]) / 30.0)
    )
    return round(max(0.0, min(1.0, score)), DISCOGS_VOCAL_SCORE_DECIMALS)


def route_discogs_voice(
    voice_analysis: Mapping[str, Any],
    *,
    song_threshold: float,
    instrumental_threshold: float,
) -> Tuple[str, str, float, float]:
    """Return status, content type, confidence and fused vocal score."""

    if not 0.0 <= instrumental_threshold < song_threshold <= 1.0:
        raise ValueError("vocal thresholds must satisfy 0 <= instrumental < song <= 1")
    vocal_score = fused_discogs_vocal_score(
        float(voice_analysis.get("voice_mean", 0.0)),
        float(voice_analysis.get("voice_coverage", 0.0)),
        float(voice_analysis.get("longest_voice_sec", 0.0)),
    )
    if vocal_score >= song_threshold:
        return "accepted", "song", vocal_score, vocal_score
    if vocal_score <= instrumental_threshold:
        return "accepted", "instrumental", 1.0 - vocal_score, vocal_score
    gray_confidence = 1.0 - abs(vocal_score - 0.5) * 2.0
    return "review", "unknown", max(0.0, gray_confidence), vocal_score


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_fingerprint(record: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(record),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def discogs_model_fingerprint_payload(args: argparse.Namespace) -> Dict[str, Any]:
    paths = DiscogsModelPaths.from_root(args.discogs_root)
    model_paths = [Path(value) for value in paths.values()]
    missing = [str(path) for path in model_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing Discogs ONNX model(s): " + ", ".join(missing))
    models = []
    for model_path in model_paths:
        sidecar_path = model_path.with_suffix(".json")
        sidecar_present = sidecar_path.is_file()
        models.append(
            {
                "name": model_path.name,
                "sha256": _sha256_file(model_path),
                "sidecar": {
                    "name": sidecar_path.name,
                    "present": sidecar_present,
                    "sha256": (_sha256_file(sidecar_path) if sidecar_present else None),
                },
            }
        )
    return {
        "schema": MODEL_FINGERPRINT_SCHEMA,
        "inference_contract": INFERENCE_CONTRACT_VERSION,
        "models": models,
    }


def _model_version_for_digest(digest: str) -> str:
    if digest == CURRENT_MODEL_SEMANTIC_DIGEST:
        return CURRENT_COMPATIBLE_MODEL_VERSION
    return digest[:16]


def discogs_model_version(args: argparse.Namespace) -> str:
    payload = discogs_model_fingerprint_payload(args)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return _model_version_for_digest(digest)


def stage_version(
    args: argparse.Namespace,
    *,
    model_version: Optional[str] = None,
) -> str:
    payload = {
        "schema": STAGE_FINGERPRINT_SCHEMA,
        "routing_contract": ROUTING_CONTRACT_VERSION,
        "model_version": model_version or discogs_model_version(args),
        "thresholds": [args.vocal_song, args.vocal_instrumental],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    if digest == CURRENT_STAGE_SEMANTIC_DIGEST:
        return CURRENT_COMPATIBLE_STAGE_VERSION
    return digest[:16]


def validate_accepted_music(record: Mapping[str, Any]) -> None:
    audio_id = str(record.get("audio_id") or "")
    if not audio_id:
        raise ValueError("Discogs input record lacks audio_id")
    if record.get("status") != "accepted":
        raise ValueError(
            f"Discogs input must contain accepted music only: audio_id={audio_id}"
        )
    content_type = record.get("content_type")
    if content_type not in (None, "music"):
        raise ValueError(
            f"Discogs input has non-music content_type={content_type!r}: audio_id={audio_id}"
        )
    if not isinstance(record.get("music_gate"), Mapping):
        raise ValueError(
            f"Discogs input lacks music_gate metadata: audio_id={audio_id}"
        )
    if not record.get("audio_path"):
        raise ValueError(f"Discogs input lacks audio_path: audio_id={audio_id}")


def _merged_stage_metadata(
    source: Mapping[str, Any],
    current_version: str,
    input_fingerprint: str,
    model_version: Optional[str] = None,
) -> Dict[str, Any]:
    record = dict(source)
    record["pipeline_version"] = PIPELINE_VERSION
    stage_versions = dict(record.get("stage_versions") or {})
    stage_versions[STAGE_NAME] = current_version
    record["stage_versions"] = stage_versions
    model_versions = dict(record.get("model_versions") or {})
    model_versions[STAGE_NAME] = model_version or current_version
    record["model_versions"] = model_versions
    record["stage_input_fingerprint"] = input_fingerprint
    return record


def build_success_record(
    source: Mapping[str, Any],
    analysis: Mapping[str, Any],
    *,
    current_version: str,
    song_threshold: float,
    instrumental_threshold: float,
    model_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Attach MIR, preserve music_gate and route on Discogs vocal evidence."""

    record = _merged_stage_metadata(
        source,
        current_version,
        record_fingerprint(source),
        model_version,
    )
    voice_analysis = dict(analysis["voice_analysis"])
    global_mir = dict(record.get("global_mir") or {})
    global_mir.update(
        {
            "genre": analysis["genre"],
            "mood_theme": analysis["mood_theme"],
            "danceability": analysis["danceability"],
            "instruments": analysis["instruments"],
            "instrument_changes": analysis["instrument_changes"],
            "discogs_frame_count": analysis["discogs_frame_count"],
            "discogs_provider": analysis["discogs_provider"],
        }
    )
    record["global_mir"] = global_mir
    record["voice_analysis"] = voice_analysis
    status, content_type, confidence, vocal_score = route_discogs_voice(
        voice_analysis,
        song_threshold=song_threshold,
        instrumental_threshold=instrumental_threshold,
    )
    record.update(
        {
            "status": status,
            "content_type": content_type,
            "content_confidence": round(float(confidence), 6),
            "discogs_vocal_score": vocal_score,
        }
    )
    stage_status = dict(record.get("stage_status") or {})
    stage_status[STAGE_NAME] = "ok"
    record["stage_status"] = stage_status
    stage_errors = dict(record.get("stage_errors") or {})
    stage_errors.pop(STAGE_NAME, None)
    record["stage_errors"] = stage_errors
    reason_codes = list(record.get("reason_codes") or [])
    if status == "review" and "vocal_route_gray" not in reason_codes:
        reason_codes.append("vocal_route_gray")
    record["reason_codes"] = reason_codes
    return record


def build_failure_record(
    source: Mapping[str, Any],
    error: Exception,
    *,
    current_version: str,
    model_version: Optional[str] = None,
) -> Dict[str, Any]:
    record = _merged_stage_metadata(
        source,
        current_version,
        record_fingerprint(source),
        model_version,
    )
    message = f"{type(error).__name__}: {error}"
    stage_status = dict(record.get("stage_status") or {})
    stage_status[STAGE_NAME] = "error"
    record["stage_status"] = stage_status
    stage_errors = dict(record.get("stage_errors") or {})
    stage_errors[STAGE_NAME] = message
    record["stage_errors"] = stage_errors
    record["stage_error"] = message
    reason_codes = list(record.get("reason_codes") or [])
    if "discogs_mir_error" not in reason_codes:
        reason_codes.append("discogs_mir_error")
    record["reason_codes"] = reason_codes
    return record


def _atomic_rewrite(path: Path, lines: Sequence[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.writelines(lines)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_resume_state(output_dir: Path) -> Dict[str, str]:
    """Load terminal successes and remove retryable failures.

    Stage/model versions are retained as provenance but do not participate in
    resume decisions.
    """

    done: Dict[str, str] = {}
    for name in OUTPUT_NAMES:
        path = output_dir / name
        if not path.exists():
            continue
        kept: List[str] = []
        for line_number, value in enumerate(
            load_jsonl_with_truncated_tail_recovery(path), 1
        ):
            if name == "failures.jsonl":
                continue
            audio_id = str(value.get("audio_id") or "")
            fingerprint = str(value.get("stage_input_fingerprint") or "")
            if not audio_id or not fingerprint:
                raise RuntimeError(
                    f"{path}:{line_number} lacks safe resume metadata"
                )
            if audio_id in done:
                raise RuntimeError(
                    f"duplicate completed audio_id in outputs: {audio_id}"
                )
            done[audio_id] = fingerprint
            kept.append(json.dumps(value, ensure_ascii=False) + "\n")
        if name == "failures.jsonl":
            _atomic_rewrite(path, [])
        elif kept:
            # Normalize complete lines while retaining every successful record.
            _atomic_rewrite(path, kept)
    return done


def scan_pending_inputs(
    input_path: str | Path,
    done: Mapping[str, str],
) -> Tuple[int, int]:
    pending = 0
    total = 0
    seen: set[str] = set()
    unmatched = set(done)
    for source in iter_jsonl(input_path):
        validate_accepted_music(source)
        total += 1
        audio_id = str(source["audio_id"])
        if audio_id in seen:
            raise RuntimeError(f"duplicate audio_id in Discogs input: {audio_id}")
        seen.add(audio_id)
        completed_fingerprint = done.get(audio_id)
        if completed_fingerprint is None:
            pending += 1
        elif completed_fingerprint != record_fingerprint(source):
            raise RuntimeError(
                f"input metadata changed for audio_id={audio_id}; use a new output directory"
            )
        else:
            unmatched.discard(audio_id)
    if unmatched:
        preview = ", ".join(sorted(unmatched)[:5])
        raise RuntimeError(
            "Discogs outputs contain audio_id values absent from current input "
            f"({preview}); use a new output directory"
        )
    return pending, total


def iter_pending_inputs(
    input_path: str | Path,
    done: Mapping[str, str],
) -> Iterator[Dict[str, Any]]:
    for source in iter_jsonl(input_path):
        audio_id = str(source["audio_id"])
        if audio_id not in done:
            yield dict(source)


def decode_track(record: Mapping[str, Any], sample_rate: int = 16000) -> torch.Tensor:
    duration = float(record.get("duration") or 0.0)
    if duration <= 0.0:
        raise ValueError("missing_or_invalid_duration")
    raw = decode_audio_range(
        str(record["audio_path"]),
        0.0,
        duration,
        sample_rate=sample_rate,
        output_format="f32le",
    )
    values = np.frombuffer(raw, dtype="<f4").copy()
    if values.size == 0:
        raise RuntimeError("decoded_audio_is_empty")
    return torch.from_numpy(values)


@torch.inference_mode()
def prepare_audio(
    engine: DiscogsOnnxEngine,
    record: Mapping[str, Any],
    waveform: torch.Tensor,
    *,
    max_cached_frames: Optional[int] = None,
) -> PreparedAudio:
    patches = engine.frontend(waveform).detach().contiguous()
    if patches.ndim != 3 or patches.shape[0] == 0:
        raise RuntimeError("too_short_for_discogs")
    if max_cached_frames is not None:
        cache_limit = int(max_cached_frames)
        if cache_limit <= 0:
            raise ValueError("max_cached_frames must be positive")
        if int(patches.shape[0]) > cache_limit:
            raise RuntimeError(
                "discogs_patch_cache_exceeds_buffered_frames: "
                f"frames={int(patches.shape[0])}, limit={cache_limit}; "
                "increase --buffered-frames for this track"
            )
    if bool(getattr(engine, "require_cuda", False)) and not patches.is_cuda:
        raise RuntimeError(
            "Discogs require_cuda=True frontend returned non-CUDA patches; "
            "NumPy fallback is disabled"
        )
    duration = waveform.numel() / 16000.0
    return PreparedAudio(
        record=dict(record),
        patches=patches,
        duration=float(duration),
    )


def concatenate_frame_slices(
    prepared: Sequence[PreparedAudio],
    batch: Sequence[FrameSlice],
) -> torch.Tensor | np.ndarray:
    """Pack one planned batch without moving tensor-backed patches off device."""

    parts = [
        prepared[part.audio_index].patches[part.start : part.stop] for part in batch
    ]
    if not parts:
        raise ValueError("frame batch must not be empty")
    if all(isinstance(part, torch.Tensor) for part in parts):
        tensors = [part for part in parts if isinstance(part, torch.Tensor)]
        device = tensors[0].device
        if any(tensor.device != device for tensor in tensors):
            raise ValueError("prepared CUDA patches must share one device")
        return torch.cat(tensors, dim=0).to(dtype=torch.float32).contiguous()
    if any(isinstance(part, torch.Tensor) for part in parts):
        raise TypeError("prepared patches cannot mix torch tensors and NumPy arrays")
    return np.ascontiguousarray(np.concatenate(parts, axis=0), dtype=np.float32)


def infer_prepared_batch(
    engine: DiscogsOnnxEngine,
    prepared: Sequence[PreparedAudio],
    *,
    max_frames: int,
) -> List[Dict[str, Any]]:
    """Infer multiple tracks with one shared backbone pass per planned frame."""

    if not prepared:
        return []
    counts = [int(item.patches.shape[0]) for item in prepared]
    if any(count <= 0 for count in counts):
        raise ValueError("prepared audio must contain at least one Discogs frame")
    predictions: List[Dict[str, List[np.ndarray]]] = [
        {name: [] for name in HEAD_NAMES} for _ in prepared
    ]
    embedding_counts = [0 for _ in prepared]

    for batch in plan_frame_batches(counts, max_frames):
        patch_batch = concatenate_frame_slices(prepared, batch)
        fused_infer = getattr(engine, "infer_patch_batch", None)
        if callable(fused_infer):
            embeddings, head_outputs = fused_infer(patch_batch)
        else:
            # Compatibility path for CPU-only fake engines used by tests.
            backbone_outputs = engine._run_session(
                engine.sessions["backbone"], patch_batch
            )
            embeddings = engine._select_embeddings(backbone_outputs)
            head_outputs = {
                name: np.asarray(
                    engine._run_session(engine.sessions[name], embeddings)[0],
                    dtype=np.float32,
                )
                for name in HEAD_NAMES
            }
        if embeddings.shape[0] != patch_batch.shape[0]:
            raise RuntimeError("Discogs backbone changed the frame count")
        for name in HEAD_NAMES:
            values = np.asarray(head_outputs[name], dtype=np.float32)
            if values.shape[0] != embeddings.shape[0]:
                raise RuntimeError(f"Discogs {name} head changed the frame count")
            head_outputs[name] = values

        cursor = 0
        for part in batch:
            stop = cursor + part.frame_count
            for name in HEAD_NAMES:
                predictions[part.audio_index][name].append(
                    head_outputs[name][cursor:stop]
                )
            embedding_counts[part.audio_index] += part.frame_count
            cursor = stop

    provider = engine.sessions["backbone"].get_providers()[0]
    output: List[Dict[str, Any]] = []
    for index, item in enumerate(prepared):
        joined = {
            name: np.concatenate(predictions[index][name], axis=0)
            for name in HEAD_NAMES
        }
        if embedding_counts[index] != counts[index]:
            raise RuntimeError(
                "Discogs frame planner did not cover every frame exactly once"
            )
        output.append(
            {
                "voice_analysis": engine._voice_summary(joined["voice"]),
                "genre": engine._top_labels("genre", joined["genre"]),
                "mood_theme": engine._top_labels("mood", joined["mood"]),
                "danceability": engine._top_labels(
                    "danceability", joined["danceability"]
                ),
                "instruments": engine._top_labels("instrument", joined["instrument"]),
                "instrument_changes": engine._instrument_changes(
                    joined["instrument"], item.duration
                ),
                "discogs_frame_count": counts[index],
                "discogs_provider": provider,
            }
        )
    return output


class JsonlOutputs:
    def __init__(self, output_dir: Path, *, resume: bool) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        mode = "a" if resume else "w"
        self.handles = {
            name: (output_dir / name).open(mode, encoding="utf-8")
            for name in OUTPUT_NAMES
        }

    def write(self, name: str, record: Mapping[str, Any]) -> None:
        handle = self.handles[name]
        handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
        handle.flush()

    def close(self) -> None:
        for handle in self.handles.values():
            handle.close()


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="fast gate accepted.music.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--discogs-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--decode-workers", type=int, default=4)
    parser.add_argument("--decode-prefetch", type=int, default=8)
    parser.add_argument("--frame-batch-size", type=int, default=256)
    parser.add_argument("--buffered-frames", type=int, default=1024)
    parser.add_argument("--vocal-song", type=float, default=0.55)
    parser.add_argument("--vocal-instrumental", type=float, default=0.20)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.decode_workers <= 0 or args.decode_prefetch <= 0:
        parser.error("decode workers/prefetch must be positive")
    if args.frame_batch_size <= 0 or args.buffered_frames <= 0:
        parser.error("frame batch/buffer limits must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    current_model_version = discogs_model_version(args)
    if not 0.0 <= args.vocal_instrumental < args.vocal_song <= 1.0:
        parser.error("vocal thresholds must satisfy 0 <= instrumental < song <= 1")
    current_version = stage_version(
        args,
        model_version=current_model_version,
    )
    done = load_resume_state(output_dir) if args.resume else {}
    pending_count, total_count = scan_pending_inputs(args.input, done)
    outputs = JsonlOutputs(output_dir, resume=args.resume)
    if pending_count == 0:
        progress = pipeline_tqdm(
            total=total_count,
            initial=total_count,
            desc="1b/7 Discogs MIR",
            unit="track",
        )
        progress.close()
        outputs.close()
        write_runtime_metrics(
            output_dir,
            pending=0,
            total=total_count,
            processed=0,
            elapsed_seconds=time.perf_counter() - started,
            counters={"song": 0, "instrumental": 0, "review": 0, "failure": 0},
            stage_version=current_version,
        )
        print(
            f"[discogs] no pending tasks; validated={total_count} "
            f"stage_version={current_version} output_dir={output_dir}"
        )
        return

    device_index = (
        int(str(args.device).split(":")[-1]) if ":" in str(args.device) else 0
    )
    engine = DiscogsOnnxEngine(
        DiscogsModelPaths.from_root(args.discogs_root),
        device_id=device_index,
        batch_size=args.frame_batch_size,
        require_cuda=True,
    )
    counters = {"song": 0, "instrumental": 0, "review": 0, "failure": 0}
    progress = pipeline_tqdm(
        total=total_count,
        initial=total_count - pending_count,
        desc="1b/7 Discogs MIR",
        unit="track",
    )
    buffered: List[PreparedAudio] = []
    buffered_frames = 0

    def write_failure(source: Mapping[str, Any], error: Exception) -> None:
        outputs.write(
            "failures.jsonl",
            build_failure_record(
                source,
                error,
                current_version=current_version,
                model_version=current_model_version,
            ),
        )
        counters["failure"] += 1
        progress.update(1)
        progress.set_postfix(**counters, refresh=False)

    def flush_buffer() -> None:
        nonlocal buffered, buffered_frames
        if not buffered:
            return
        items = buffered
        buffered = []
        buffered_frames = 0
        try:
            analyses = infer_prepared_batch(
                engine,
                items,
                max_frames=args.frame_batch_size,
            )
        except Exception as error:
            # A shared ORT failure affects every item in this bounded batch;
            # they remain individually retryable on the next resume.
            for item in items:
                write_failure(item.record, error)
            return
        for item, analysis in zip(items, analyses):
            try:
                record = build_success_record(
                    item.record,
                    analysis,
                    current_version=current_version,
                    song_threshold=args.vocal_song,
                    instrumental_threshold=args.vocal_instrumental,
                    model_version=current_model_version,
                )
                if record["status"] == "review":
                    name = "review.jsonl"
                    counters["review"] += 1
                elif record["content_type"] == "song":
                    name = "data.song.jsonl"
                    counters["song"] += 1
                else:
                    name = "data.instrumental.jsonl"
                    counters["instrumental"] += 1
                outputs.write(name, record)
                progress.update(1)
                progress.set_postfix(**counters, refresh=False)
            except Exception as error:
                write_failure(item.record, error)

    try:
        pending_inputs = iter_pending_inputs(args.input, done)
        for decoded in bounded_prefetch_map(
            pending_inputs,
            decode_track,
            max_workers=args.decode_workers,
            prefetch=args.decode_prefetch,
        ):
            source = decoded.item
            if decoded.error is not None:
                flush_buffer()
                write_failure(source, decoded.error)
                continue
            try:
                item = prepare_audio(
                    engine,
                    source,
                    decoded.value,
                    max_cached_frames=args.buffered_frames,
                )
            except Exception as error:
                flush_buffer()
                write_failure(source, error)
                continue
            frame_count = int(item.patches.shape[0])
            if buffered and buffered_frames + frame_count > args.buffered_frames:
                flush_buffer()
            buffered.append(item)
            buffered_frames += frame_count
            if buffered_frames >= args.buffered_frames:
                flush_buffer()
        flush_buffer()
    finally:
        outputs.close()
        progress.close()

    processed = sum(counters.values())
    elapsed = time.perf_counter() - started
    write_runtime_metrics(
        output_dir,
        pending=pending_count,
        total=total_count,
        processed=processed,
        elapsed_seconds=elapsed,
        counters=counters,
        stage_version=current_version,
    )
    print(
        f"[discogs] done pending={pending_count} counters={counters} "
        f"stage_version={current_version} output_dir={output_dir} "
        f"seconds_per_track={elapsed / processed if processed else 0.0:.6f}"
    )


if __name__ == "__main__":
    main()

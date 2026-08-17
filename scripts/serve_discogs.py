#!/usr/bin/env python3
"""Persistent Discogs MIR service with cross-request frame batching."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from discogs_mir_infer import (
    PreparedAudio,
    bounded_prefetch_map,
    build_failure_record,
    build_success_record,
    decode_track,
    discogs_model_version,
    infer_prepared_batch,
    prepare_audio,
    stage_version,
    validate_accepted_music,
)
from service_api import (
    BatchItemResult,
    DynamicBatchService,
    ServiceRequest,
    create_service_app,
    run_service,
)
from sub_models.discogs_onnx_model import DiscogsModelPaths, DiscogsOnnxEngine


ROOT = Path(__file__).resolve().parents[1]
STAGE = "discogs_mir"
MAX_WAIT_MS = 100


@dataclass
class DiscogsRuntime:
    engine: DiscogsOnnxEngine
    stage_version: str
    model_version: str
    frame_batch_size: int
    buffered_frames: int
    decode_workers: int
    decode_prefetch: int
    vocal_song: float
    vocal_instrumental: float

    def cleanup(self) -> None:
        cleanup = getattr(self.engine, "cleanup", None)
        if callable(cleanup):
            cleanup()


def _fingerprints(args: argparse.Namespace) -> tuple[str, str]:
    model_version = discogs_model_version(args)
    return model_version, stage_version(args, model_version=model_version)


def load_discogs(
    args: argparse.Namespace,
    *,
    model_version: str,
    current_stage_version: str,
) -> DiscogsRuntime:
    """Load all Discogs sessions once for the FastAPI lifespan."""

    device_index = int(str(args.device).split(":")[-1]) if ":" in args.device else 0
    engine = DiscogsOnnxEngine(
        DiscogsModelPaths.from_root(args.discogs_root),
        device_id=device_index,
        batch_size=args.frame_batch_size,
        require_cuda=True,
    )
    return DiscogsRuntime(
        engine=engine,
        stage_version=current_stage_version,
        model_version=model_version,
        frame_batch_size=args.frame_batch_size,
        buffered_frames=args.buffered_frames,
        decode_workers=args.decode_workers,
        decode_prefetch=args.decode_prefetch,
        vocal_song=args.vocal_song,
        vocal_instrumental=args.vocal_instrumental,
    )


def _source(request: ServiceRequest) -> Dict[str, Any]:
    record = dict(request.record)
    record["audio_id"] = request.audio_id
    record["audio_path"] = request.audio_path
    return record


def process_discogs_batch(
    runtime: DiscogsRuntime,
    requests: Sequence[ServiceRequest],
) -> List[Mapping[str, Any] | BatchItemResult]:
    """Decode concurrently, then preserve the existing bounded frame planner."""

    output: List[Mapping[str, Any] | BatchItemResult | None] = [None] * len(requests)
    indexed_sources: List[tuple[int, Dict[str, Any]]] = []
    for index, request in enumerate(requests):
        source = _source(request)
        try:
            validate_accepted_music(source)
        except Exception as error:
            record = build_failure_record(
                source,
                error,
                current_version=runtime.stage_version,
                model_version=runtime.model_version,
            )
            record["stage_input_fingerprint"] = request.input_fingerprint
            output[index] = BatchItemResult(record, error)
        else:
            indexed_sources.append((index, source))

    buffered: List[tuple[int, PreparedAudio]] = []
    buffered_frame_count = 0

    def fail(index: int, source: Mapping[str, Any], error: Exception) -> None:
        record = build_failure_record(
            source,
            error,
            current_version=runtime.stage_version,
            model_version=runtime.model_version,
        )
        record["stage_input_fingerprint"] = requests[index].input_fingerprint
        output[index] = BatchItemResult(record, error)

    def flush() -> None:
        nonlocal buffered, buffered_frame_count
        if not buffered:
            return
        items = buffered
        buffered = []
        buffered_frame_count = 0
        try:
            analyses = infer_prepared_batch(
                runtime.engine,
                [prepared for _, prepared in items],
                max_frames=runtime.frame_batch_size,
            )
        except Exception as error:
            for index, prepared in items:
                fail(index, prepared.record, error)
            return
        for (index, prepared), analysis in zip(items, analyses):
            try:
                record = build_success_record(
                    prepared.record,
                    analysis,
                    current_version=runtime.stage_version,
                    song_threshold=runtime.vocal_song,
                    instrumental_threshold=runtime.vocal_instrumental,
                    model_version=runtime.model_version,
                )
                record["stage_input_fingerprint"] = requests[
                    index
                ].input_fingerprint
                output[index] = record
            except Exception as error:
                fail(index, prepared.record, error)

    def decode(indexed: tuple[int, Dict[str, Any]]) -> Any:
        return decode_track(indexed[1])

    for decoded in bounded_prefetch_map(
        indexed_sources,
        decode,
        max_workers=runtime.decode_workers,
        prefetch=runtime.decode_prefetch,
    ):
        index, source = decoded.item
        if decoded.error is not None:
            flush()
            fail(index, source, decoded.error)
            continue
        try:
            prepared = prepare_audio(
                runtime.engine,
                source,
                decoded.value,
                max_cached_frames=runtime.buffered_frames,
            )
        except Exception as error:
            flush()
            fail(index, source, error)
            continue
        frame_count = int(prepared.patches.shape[0])
        if buffered and buffered_frame_count + frame_count > runtime.buffered_frames:
            flush()
        buffered.append((index, prepared))
        buffered_frame_count += frame_count
        if buffered_frame_count >= runtime.buffered_frames:
            flush()
    flush()

    if any(value is None for value in output):
        raise RuntimeError("Discogs service lost batch result alignment")
    return [value for value in output if value is not None]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--discogs-root",
        default=str(ROOT / "MusicToolsPipeline" / "discogs_onnx"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--decode-workers", type=int, default=8)
    parser.add_argument("--decode-prefetch", type=int, default=16)
    parser.add_argument("--frame-batch-size", type=int, default=512)
    parser.add_argument("--buffered-frames", type=int, default=2048)
    parser.add_argument("--request-batch-size", type=int, default=64)
    parser.add_argument("--vocal-song", type=float, default=0.55)
    parser.add_argument("--vocal-instrumental", type=float, default=0.20)
    parser.add_argument("--queue-size", type=int, default=256)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18102)
    return parser


def build_service(args: argparse.Namespace) -> DynamicBatchService:
    if args.decode_workers <= 0 or args.decode_prefetch <= 0:
        raise ValueError("decode-workers/decode-prefetch must be positive")
    if (
        args.frame_batch_size <= 0
        or args.buffered_frames <= 0
        or args.request_batch_size <= 0
    ):
        raise ValueError("frame/request batch and buffered frame limits must be positive")
    if not 0.0 <= args.vocal_instrumental < args.vocal_song <= 1.0:
        raise ValueError("vocal thresholds must satisfy 0 <= instrumental < song <= 1")
    model_version, current_stage_version = _fingerprints(args)

    def loader() -> DiscogsRuntime:
        return load_discogs(
            args,
            model_version=model_version,
            current_stage_version=current_stage_version,
        )

    return DynamicBatchService(
        loader=loader,
        process_batch=process_discogs_batch,
        stage=STAGE,
        device=args.device,
        model_fingerprint=model_version,
        max_batch_size=args.request_batch_size,
        max_wait_ms=MAX_WAIT_MS,
        queue_size=args.queue_size,
    )


def main() -> None:
    args = build_parser().parse_args()
    app = create_service_app(build_service(args))
    run_service(app, args.host, args.port)


if __name__ == "__main__":
    main()

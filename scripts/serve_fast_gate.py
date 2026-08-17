#!/usr/bin/env python3
"""Persistent dynamically batched Fast Gate inference service."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from fast_gate_core import (
    AudioSetMusicScorer,
    CascadeMusicGate,
    DecisionThresholds,
    InvalidAudioError,
    build_backend,
    load_production_gate_config,
)
from fast_music_gate import (
    build_stage_version,
    decorate_failure,
    decorate_invalid_asset_rejection,
    decorate_result,
    resolve_production_config,
)
from service_api import (
    BatchItemResult,
    DynamicBatchService,
    ServiceRequest,
    create_service_app,
    run_service,
)


ROOT = Path(__file__).resolve().parents[1]
STAGE = "music_gate"
MAX_REQUEST_BATCH = 64
MAX_WAIT_MS = 100


@dataclass
class FastGateRuntime:
    gate: CascadeMusicGate
    stage_version: str
    backend_name: str
    scoring_version: str

    def cleanup(self) -> None:
        seen: set[int] = set()
        for backend in (self.gate.backend, self.gate.stage_b_backend):
            if backend is None or id(backend) in seen:
                continue
            seen.add(id(backend))
            cleanup = getattr(backend, "cleanup", None)
            if callable(cleanup):
                cleanup()


def _resolved_gate_args(args: argparse.Namespace) -> tuple[argparse.Namespace, Mapping[str, Any]]:
    settings = argparse.Namespace(
        config=str(Path(args.config).expanduser().resolve()),
        config_version=None,
        backend=None,
        backend_weights=args.backend_weights,
        backend_repo=args.backend_repo,
        stage_b_backend=None,
        stage_b_backend_weights=args.stage_b_backend_weights,
        stage_b_backend_repo=args.stage_b_backend_repo,
        device=args.device,
        precision=None,
        stage_b_precision=None,
        batch_size=None,
        stage_b_batch_size=None,
        stage_a_reject=None,
        stage_a_accept=None,
        stage_b_reject=None,
        stage_b_accept=None,
    )
    config = load_production_gate_config(settings.config)
    resolve_production_config(settings, config)
    return settings, config


def load_fast_gate(
    settings: argparse.Namespace,
    config: Mapping[str, Any],
    *,
    decode_workers: int,
) -> FastGateRuntime:
    """Load the production Fast Gate once for the FastAPI lifespan."""

    head = AudioSetMusicScorer(str(config["scoring"]["version"]))
    stage_b_head = AudioSetMusicScorer(str(config["scoring"]["version"]))
    backend = build_backend(
        settings.backend,
        settings.backend_weights,
        device=settings.device,
        repo_path=settings.backend_repo,
        precision=settings.precision,
    )
    same_backend = (
        settings.stage_b_backend == settings.backend
        and settings.stage_b_backend_weights == settings.backend_weights
        and settings.stage_b_backend_repo == settings.backend_repo
        and settings.stage_b_backend_repo_id == settings.backend_repo_id
        and settings.stage_b_precision == settings.precision
    )
    stage_b_backend = (
        backend
        if same_backend
        else build_backend(
            settings.stage_b_backend,
            settings.stage_b_backend_weights,
            device=settings.device,
            repo_path=settings.stage_b_backend_repo,
            precision=settings.stage_b_precision,
        )
    )
    gate = CascadeMusicGate(
        backend=backend,
        head=head,
        stage_a_thresholds=DecisionThresholds(
            settings.stage_a_reject, settings.stage_a_accept
        ),
        stage_b_thresholds=DecisionThresholds(
            settings.stage_b_reject, settings.stage_b_accept
        ),
        batch_size=settings.batch_size,
        stage_b_batch_size=settings.stage_b_batch_size,
        decode_workers=decode_workers,
        stage_b_backend=stage_b_backend,
        stage_b_head=stage_b_head,
    )
    return FastGateRuntime(
        gate=gate,
        stage_version=build_stage_version(settings),
        backend_name=f"{backend.name}->{stage_b_backend.name}",
        scoring_version=head.scoring_version,
    )


def _source(request: ServiceRequest) -> Dict[str, Any]:
    record = dict(request.record)
    record["audio_id"] = request.audio_id
    record["audio_path"] = request.audio_path
    return record


def process_fast_gate_batch(
    runtime: FastGateRuntime,
    requests: Sequence[ServiceRequest],
) -> List[Mapping[str, Any] | BatchItemResult]:
    """Reuse CascadeMusicGate's cross-track batching for one service batch."""

    output: List[Mapping[str, Any] | BatchItemResult | None] = [None] * len(requests)
    eligible: List[Mapping[str, Any]] = []
    eligible_indices: List[int] = []
    for index, request in enumerate(requests):
        source = _source(request)
        if source.get("decode_status") not in (None, "ok"):
            record = decorate_invalid_asset_rejection(source, runtime.stage_version)
            record["music_gate_input_fingerprint"] = request.input_fingerprint
            output[index] = record
        else:
            eligible.append(source)
            eligible_indices.append(index)

    if eligible:
        try:
            results = runtime.gate.classify_records(eligible)
        except Exception:
            results = []
            for service_index, source in zip(eligible_indices, eligible):
                request = requests[service_index]
                try:
                    result = runtime.gate.classify_records([source])[0]
                    record, _ = decorate_result(
                        source, result, runtime.stage_version
                    )
                    record["music_gate_input_fingerprint"] = request.input_fingerprint
                    output[service_index] = record
                except InvalidAudioError as error:
                    record = decorate_invalid_asset_rejection(
                        source, runtime.stage_version, str(error)
                    )
                    record["music_gate_input_fingerprint"] = request.input_fingerprint
                    output[service_index] = record
                except Exception as error:
                    record = decorate_failure(
                        source,
                        error,
                        runtime.backend_name,
                        runtime.scoring_version,
                        runtime.stage_version,
                    )
                    record["music_gate_input_fingerprint"] = request.input_fingerprint
                    output[service_index] = BatchItemResult(record, error)
        else:
            for service_index, source, result in zip(
                eligible_indices, eligible, results
            ):
                record, _ = decorate_result(source, result, runtime.stage_version)
                record["music_gate_input_fingerprint"] = requests[
                    service_index
                ].input_fingerprint
                output[service_index] = record

    if any(value is None for value in output):
        raise RuntimeError("Fast Gate service lost batch result alignment")
    return [value for value in output if value is not None]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(
            ROOT / "MusicToolsPipeline" / "checkpoints" / "fast_gate_config.json"
        ),
    )
    parser.add_argument("--backend-weights")
    parser.add_argument("--backend-repo")
    parser.add_argument("--stage-b-backend-weights")
    parser.add_argument("--stage-b-backend-repo")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--decode-workers", type=int, default=16)
    parser.add_argument("--queue-size", type=int, default=256)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18101)
    return parser


def build_service(args: argparse.Namespace) -> DynamicBatchService:
    if args.decode_workers <= 0:
        raise ValueError("decode-workers must be positive")
    settings, config = _resolved_gate_args(args)
    fingerprint = build_stage_version(settings)

    def loader() -> FastGateRuntime:
        return load_fast_gate(
            settings,
            config,
            decode_workers=args.decode_workers,
        )

    return DynamicBatchService(
        loader=loader,
        process_batch=process_fast_gate_batch,
        stage=STAGE,
        device=args.device,
        model_fingerprint=fingerprint,
        max_batch_size=MAX_REQUEST_BATCH,
        max_wait_ms=MAX_WAIT_MS,
        queue_size=args.queue_size,
    )


def main() -> None:
    args = build_parser().parse_args()
    app = create_service_app(build_service(args))
    run_service(app, args.host, args.port)


if __name__ == "__main__":
    main()

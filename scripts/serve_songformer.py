#!/usr/bin/env python3
"""Resident MuQ/MusicFM/SongFormer HTTP inference service."""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import math
import os
import sys
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from service_api import (
    BatchItemResult,
    DynamicBatchService,
    ServiceRequest,
    create_service_app,
    run_service,
)


ROOT = Path(__file__).resolve().parents[1]
SONGFORMER_ROOT = ROOT / "SongFormer"


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _load_songformer_module() -> Any:
    """Import the existing implementation without loading any model twice."""

    module_name = "_music_pipeline_songformer_infer"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    sys.path.insert(0, str(SONGFORMER_ROOT))
    path = SONGFORMER_ROOT / "infer_jsonl.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import SongFormer implementation from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    with _working_directory(SONGFORMER_ROOT):
        spec.loader.exec_module(module)
    return module


def _merge_structure_record(
    request: ServiceRequest,
    structure: Sequence[Mapping[str, Any]] | Sequence[Any],
    stage_version: str,
) -> dict[str, Any]:
    record = dict(request.record or {})
    record.setdefault("audio_id", request.audio_id)
    record.setdefault("audio_path", request.audio_path)
    value = list(structure)
    record["structure_raw"] = value
    if str(record.get("content_type", "song")).strip().lower() == "song":
        record["songformer_result"] = value
    status = dict(record.get("stage_status") or {})
    errors = dict(record.get("stage_errors") or {})
    versions = dict(record.get("stage_versions") or {})
    status.pop("structure", None)
    errors.pop("structure", None)
    versions.pop("structure", None)
    status["structure_raw"] = "ok"
    errors.pop("structure_raw", None)
    versions["structure_raw"] = stage_version
    record["stage_status"] = status
    record["stage_errors"] = errors
    record["stage_versions"] = versions
    return record


class SongFormerRuntime:
    """All three heavy models and their immutable inference configuration."""

    def __init__(
        self,
        sf: Any,
        models: tuple[Any, Any, Any, Any, Any, Any],
        args: Namespace,
    ) -> None:
        self.sf = sf
        (
            self.device,
            self.muq,
            self.musicfm,
            self.model,
            self.hp,
            self.dataset_id2label_mask,
        ) = models
        self.args = args
        self._logged_feature_contract = False

    @classmethod
    def load(cls, args: argparse.Namespace) -> "SongFormerRuntime":
        sf = _load_songformer_module()
        worker_args = Namespace(
            model=args.model,
            checkpoint=args.checkpoint,
            config_path=args.config_path,
            num_classes=args.num_classes,
            output_dir=args.scratch_dir,
        )
        try:
            rank = int(str(args.device).split(":", 1)[1])
        except (IndexError, ValueError) as error:
            raise ValueError("SongFormer --device must look like cuda:0") from error
        with _working_directory(SONGFORMER_ROOT):
            models = sf._initialize_worker_models(rank, worker_args)
        return cls(sf, models, args)

    def process_batch(
        self, requests: Sequence[ServiceRequest]
    ) -> list[dict[str, Any] | BatchItemResult]:
        # The model remains strictly serial, but the next track is decoded by one
        # bounded CPU thread while the current track occupies the GPU.
        if not requests:
            return []
        results: list[dict[str, Any] | BatchItemResult] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            next_future = executor.submit(self._decode_request, requests[0])
            for index, request in enumerate(requests):
                current_future = next_future
                if index + 1 < len(requests):
                    next_future = executor.submit(
                        self._decode_request, requests[index + 1]
                    )
                try:
                    wav, sample_rate, content_type = current_future.result()
                    results.append(
                        self._infer_decoded(
                            request, wav, sample_rate, content_type
                        )
                    )
                except Exception as error:
                    results.append(
                        BatchItemResult(record=dict(request.record or {}), error=error)
                    )
        return results

    def infer(self, request: ServiceRequest) -> dict[str, Any]:
        wav, sample_rate, content_type = self._decode_request(request)
        return self._infer_decoded(request, wav, sample_rate, content_type)

    def _decode_request(self, request: ServiceRequest) -> tuple[Any, int, str]:
        record = dict(request.record or {})
        audio_path = str(request.audio_path or record.get("audio_path") or "").strip()
        if not audio_path:
            raise ValueError("SongFormer request is missing audio_path")
        content_type = str(record.get("content_type", "song")).strip().lower()
        if content_type not in {"song", "instrumental"}:
            raise ValueError(f"unsupported content_type={content_type!r}")

        wav, sample_rate = self.sf.librosa.load(
            audio_path, sr=self.sf.INPUT_SAMPLING_RATE
        )
        return wav, int(sample_rate), content_type

    def _infer_decoded(
        self,
        request: ServiceRequest,
        wav: Any,
        sample_rate: int,
        content_type: str,
    ) -> dict[str, Any]:
        if int(sample_rate) != int(self.sf.INPUT_SAMPLING_RATE):
            raise RuntimeError(
                f"unexpected SongFormer sample rate {sample_rate}; "
                f"expected {self.sf.INPUT_SAMPLING_RATE}"
            )
        with self.sf.torch.no_grad():
            audio = self.sf.torch.as_tensor(wav, device=self.device)
            if content_type == "instrumental":
                structure = self.sf.infer_instrumental_structure(
                    audio=audio,
                    muq=self.muq,
                    musicfm=self.musicfm,
                    embedding_batch_size=self.args.embedding_chunk_batch_size,
                )
            else:
                structure = self._infer_song(audio)
        if not structure:
            raise RuntimeError("SongFormer produced no valid structure sections")
        return _merge_structure_record(request, structure, self.args.stage_version)

    def _infer_song(self, audio: Any) -> list[dict[str, Any]]:
        sf = self.sf
        torch = sf.torch
        args = self.args
        total_len = (
            (audio.shape[0] // sf.INPUT_SAMPLING_RATE) // sf.TIME_DUR
        ) * sf.TIME_DUR + sf.TIME_DUR
        total_frames = math.ceil(total_len * sf.AFTER_DOWNSAMPLING_FRAME_RATES)
        logits = {
            "function_logits": np.zeros(
                [total_frames, args.num_classes], dtype=np.float32
            ),
            "boundary_logits": np.zeros([total_frames], dtype=np.float32),
        }
        counts = {
            "function_logits": np.zeros(
                [total_frames, args.num_classes], dtype=np.float32
            ),
            "boundary_logits": np.zeros([total_frames], dtype=np.float32),
        }
        lens = 0
        offset_seconds = 0
        while True:
            start = offset_seconds * sf.INPUT_SAMPLING_RATE
            end = min(
                (offset_seconds + args.win_size) * sf.INPUT_SAMPLING_RATE,
                audio.shape[-1],
            )
            if start >= audio.shape[-1]:
                break
            if end - start <= 1024:
                offset_seconds += args.hop_size
                continue
            segment = audio[start:end]

            muq_output = self.muq(segment.unsqueeze(0), output_hidden_states=True)
            muq_global = muq_output["hidden_states"][10]
            del muq_output
            _, musicfm_states = self.musicfm.get_predictions(segment.unsqueeze(0))
            musicfm_global = musicfm_states[10]
            del musicfm_states

            local_samples = 30 * sf.INPUT_SAMPLING_RATE
            local_chunks = [
                audio[local_start : min(local_start + local_samples, end)]
                for local_start in range(start, end, local_samples)
                if min(local_start + local_samples, end) - local_start > 1024
            ]
            muq_local_parts, musicfm_local_parts = sf.extract_muq_musicfm_chunks(
                local_chunks,
                self.muq,
                self.musicfm,
                batch_size=args.embedding_chunk_batch_size,
                empty_cuda_cache=False,
            )
            if not muq_local_parts or not musicfm_local_parts:
                raise RuntimeError(
                    "local 30-second feature extraction produced no embeddings"
                )
            features = [
                torch.concatenate(musicfm_local_parts, dim=1),
                torch.concatenate(muq_local_parts, dim=1),
                musicfm_global,
                muq_global,
            ]
            feature_lengths = [value.shape[1] for value in features]
            shortest, longest = min(feature_lengths), max(feature_lengths)
            if longest - shortest > 4:
                raise ValueError(
                    f"Embedding shapes differ too much: {longest} vs {shortest}"
                )
            features = [value[:, :shortest, :] for value in features]
            embedding = torch.concatenate(features, axis=-1)
            dataset_ids = torch.tensor(
                sf.DATASET_IDS, device=self.device, dtype=torch.long
            )
            label_mask = torch.tensor(
                self.dataset_id2label_mask[
                    sf.DATASET_LABEL_TO_DATASET_ID[sf.DATASET_LABEL]
                ],
                device=self.device,
                dtype=torch.bool,
            ).unsqueeze(0).unsqueeze(0)
            _, chunk_logits = self.model.infer(
                input_embeddings=embedding,
                dataset_ids=dataset_ids,
                label_id_masks=label_mask,
                with_logits=True,
            )
            start_frame = int(offset_seconds * sf.AFTER_DOWNSAMPLING_FRAME_RATES)
            end_frame = start_frame + min(
                math.ceil(args.hop_size * sf.AFTER_DOWNSAMPLING_FRAME_RATES),
                chunk_logits["boundary_logits"][0].shape[0],
            )
            logits["function_logits"][start_frame:end_frame] += (
                chunk_logits["function_logits"][0].detach().cpu().numpy()
            )
            logits["boundary_logits"][start_frame:end_frame] = (
                chunk_logits["boundary_logits"][0].detach().cpu().numpy()
            )
            counts["function_logits"][start_frame:end_frame] += 1
            counts["boundary_logits"][start_frame:end_frame] += 1
            lens += end_frame - start_frame
            offset_seconds += args.hop_size

        for value in counts.values():
            value[value == 0] = 1
        function_logits = logits["function_logits"][:lens] / counts[
            "function_logits"
        ][:lens]
        boundary_logits = logits["boundary_logits"][:lens] / counts[
            "boundary_logits"
        ][:lens]
        detailed = sf.postprocess_functional_structure_detailed(
            {
                "function_logits": torch.from_numpy(function_logits).unsqueeze(0),
                "boundary_logits": torch.from_numpy(boundary_logits).unsqueeze(0),
            },
            self.hp,
        )
        if not detailed:
            raise RuntimeError("SongFormer produced no valid structure sections")
        endpoints = [(float(item["start"]), str(item["label"])) for item in detailed]
        endpoints.append((float(detailed[-1]["end"]), "end"))
        if not args.no_rule_post_processing:
            endpoints = sf.rule_post_processing(endpoints)

        output: list[dict[str, Any]] = []
        for index in range(len(endpoints) - 1):
            start, label = float(endpoints[index][0]), str(endpoints[index][1])
            end = float(endpoints[index + 1][0])
            midpoint = (start + end) / 2.0
            matching = [
                item
                for item in detailed
                if float(item["start"]) <= midpoint < float(item["end"])
            ]
            detail = max(
                matching or detailed,
                key=lambda item: max(
                    0.0,
                    min(end, float(item["end"]))
                    - max(start, float(item["start"])),
                ),
            )
            output.append(
                {
                    "label": label,
                    "start": start,
                    "end": end,
                    "raw_start": start,
                    "raw_end": end,
                    "start_boundary_confidence": float(
                        detail["start_boundary_confidence"]
                    ),
                    "end_boundary_confidence": float(detail["end_boundary_confidence"]),
                    "boundary_confidence": float(detail["end_boundary_confidence"]),
                    "label_confidence": float(detail["label_confidence"]),
                    "label_source": "songformer",
                }
            )
        return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10101)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default="SongFormer")
    parser.add_argument("--checkpoint", default="SongFormer.safetensors")
    parser.add_argument("--config-path", default="SongFormer.yaml")
    parser.add_argument("--num-classes", type=int, default=128)
    parser.add_argument("--win-size", type=int, default=420)
    parser.add_argument("--hop-size", type=int, default=420)
    parser.add_argument("--embedding-chunk-batch-size", type=int, default=1)
    parser.add_argument("--no-rule-post-processing", action="store_true")
    parser.add_argument(
        "--stage-version", default="dual-structure-local-global-ssm-cbm-v3"
    )
    parser.add_argument(
        "--scratch-dir", default="/tmp/music-data-pipeline-songformer-service"
    )
    parser.add_argument("--queue-size", type=int, default=32)
    parser.add_argument("--max-batch-size", type=int, default=2)
    parser.add_argument("--max-wait-ms", type=int, default=20)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.embedding_chunk_batch_size < 1:
        raise SystemExit("--embedding-chunk-batch-size must be positive")
    if args.max_batch_size < 2 or args.max_wait_ms < 0:
        raise SystemExit("--max-batch-size must be at least 2 and wait non-negative")
    service = DynamicBatchService(
        loader=lambda: SongFormerRuntime.load(args),
        process_batch=lambda runtime, requests: runtime.process_batch(requests),
        stage="songformer",
        device=args.device,
        model_fingerprint=f"{args.model}:{args.checkpoint}:{args.stage_version}",
        max_batch_size=args.max_batch_size,
        max_wait_ms=args.max_wait_ms,
        queue_size=args.queue_size,
    )
    run_service(create_service_app(service), args.host, args.port)


if __name__ == "__main__":
    main()

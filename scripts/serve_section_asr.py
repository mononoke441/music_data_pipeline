#!/usr/bin/env python3
"""Resident Qwen3-ASR + ForcedAligner HTTP inference service."""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from service_api import (
    BatchItemResult,
    DynamicBatchService,
    ServiceError,
    ServiceRequest,
    create_service_app,
    run_service,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"


def _load_section_asr_helpers() -> Any:
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    return importlib.import_module("section_asr_infer")


class SectionASRRuntime:
    """Resident Qwen ASR engine with duration-homogeneous section batching."""

    def __init__(self, helpers: Any, asr: Any, args: argparse.Namespace) -> None:
        self.helpers = helpers
        self.asr = asr
        self.args = args

    @classmethod
    def load(cls, args: argparse.Namespace) -> "SectionASRRuntime":
        helpers = _load_section_asr_helpers()
        import torch
        from qwen_asr import Qwen3ASRModel

        helpers.apply_torch_cuda_memory_limit(
            args.gpu_max_memory_gib,
            0,
            torch_module=torch,
        )
        budget, _free_bytes, total_bytes = helpers.live_asr_vllm_memory_budget(
            torch,
            pipeline_max_memory_gib=args.gpu_max_memory_gib,
            requested_vllm_max_memory_gib=args.vllm_max_memory_gib,
            forced_aligner_reserve_gib=args.forced_aligner_reserve_gib,
            vllm_headroom_gib=args.vllm_headroom_gib,
            minimum_vllm_memory_gib=args.minimum_vllm_memory_gib,
        )
        utilization = helpers.capped_vllm_gpu_memory_utilization(
            budget,
            total_bytes,
            args.gpu_memory_utilization,
        )
        asr = Qwen3ASRModel.LLM(
            model=args.model,
            gpu_memory_utilization=utilization,
            max_inference_batch_size=args.section_batch_size,
            max_new_tokens=args.max_new_tokens,
            forced_aligner=args.forced_aligner,
            forced_aligner_kwargs={"dtype": torch.bfloat16, "device_map": "cuda:0"},
        )
        return cls(helpers, asr, args)

    def process_batch(
        self, requests: Sequence[ServiceRequest]
    ) -> list[dict[str, Any] | BatchItemResult]:
        helpers = self.helpers
        prepared: list[tuple[dict[str, Any], dict[str, Any]]] = []
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for request_index, request in enumerate(requests):
            source = dict(request.record or {})
            source.setdefault("audio_id", request.audio_id)
            source.setdefault("audio_path", request.audio_path)
            audio_id = str(source.get("audio_id") or request.audio_id or "").strip()
            output: dict[str, Any]
            try:
                if not audio_id:
                    raise ValueError("Section ASR request is missing audio_id")
                audio_path = str(
                    source.get("audio_path") or request.audio_path or ""
                ).strip()
                if not audio_path:
                    raise ValueError("Section ASR request is missing audio_path")
                source["audio_id"] = audio_id
                source["audio_path"] = audio_path
                source_sections = list(source.get("sections") or [])
                for section in source_sections:
                    if not isinstance(section, Mapping):
                        raise ValueError("sections must contain JSON objects")
                    if not str(section.get("section_id") or "").strip():
                        raise ValueError("section is missing section_id")
                    start = float(section["start"])
                    end = float(section["end"])
                    float(section.get("voice_coverage", 0.0))
                    if end <= start:
                        raise ValueError("section end must be greater than start")

                plan_hash = helpers.sections_hash(source)
                semantic_fingerprint = helpers.section_asr_input_fingerprint(source)
                section_outputs: list[dict[str, Any]] = []
                for section in source_sections:
                    target = {
                        "section_id": section["section_id"],
                        "start": section["start"],
                        "end": section["end"],
                    }
                    section_outputs.append(target)
                    if str(source.get("content_type", "")).lower() != "song":
                        helpers.set_failure(target, "not_applicable")
                        continue
                    eligible, reason = helpers.should_run_section_asr(section)
                    if not eligible:
                        helpers.set_failure(target, f"skipped_{reason}")
                        continue
                    item = {
                        "record": source,
                        "section": section,
                        "target": target,
                        "request_index": request_index,
                    }
                    duration = float(section["end"]) - float(section["start"])
                    buckets[helpers.duration_bucket(duration)].append(item)
                output = {
                    "audio_id": audio_id,
                    "audio_path": audio_path,
                    "sections": section_outputs,
                    "sections_hash": plan_hash,
                    "section_asr_input_fingerprint": semantic_fingerprint,
                    "semantic_input_fingerprint": semantic_fingerprint,
                    "pipeline_version": helpers.PIPELINE_VERSION,
                }
            except (AttributeError, KeyError, TypeError, ValueError) as error:
                output = {
                    "audio_id": audio_id,
                    "audio_path": source.get("audio_path") or request.audio_path,
                    "sections": [],
                    "pipeline_version": helpers.PIPELINE_VERSION,
                    "stage_status": {"section_asr": "error"},
                    "stage_errors": {
                        "section_asr": f"{type(error).__name__}: {error}"
                    },
                }
            prepared.append((source, output))

        section_batches = helpers.section_batches(
            buckets, self.args.section_batch_size
        )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.args.decode_workers
        ) as executor:
            for batch in section_batches:
                decoded: list[tuple[Mapping[str, Any], tuple[Any, int, float]]] = []
                futures = [
                    (item, executor.submit(helpers.decode_item, item, self.args.padding))
                    for item in batch
                ]
                for item, future in futures:
                    try:
                        decoded.append((item, future.result()))
                    except Exception as error:
                        helpers.set_failure(
                            item["target"],
                            "decode_error",
                            f"{type(error).__name__}: {error}",
                        )
                self._transcribe_decoded(decoded)

        results: list[dict[str, Any] | BatchItemResult] = []
        for source, output in prepared:
            if output.get("stage_status", {}).get("section_asr") == "error":
                error_text = str(
                    (output.get("stage_errors") or {}).get(
                        "section_asr", "Section ASR request failed"
                    )
                )
                results.append(
                    BatchItemResult(
                        record=output,
                        error=ServiceError(error_text),
                    )
                )
                continue
            audio_id = str(output["audio_id"])
            finalized = helpers.finalize_asr_records(
                [source],
                {audio_id: output},
                set(),
                model=self.args.model,
                forced_aligner=self.args.forced_aligner,
            )[0]
            if finalized.get("stage_status", {}).get("section_asr") == "error":
                error_text = str(
                    (finalized.get("stage_errors") or {}).get(
                        "section_asr", "Section ASR inference failed"
                    )
                )
                results.append(
                    BatchItemResult(
                        record=finalized,
                        error=ServiceError(error_text),
                    )
                )
            else:
                results.append(finalized)
        return results

    def _transcribe_decoded(
        self,
        decoded: list[tuple[Mapping[str, Any], tuple[Any, int, float]]],
    ) -> None:
        if not decoded:
            return
        helpers = self.helpers
        try:
            results = helpers.run_batch(self.asr, decoded)
            pairs = list(zip(decoded, results))
        except helpers.BatchCardinalityError as error:
            pairs = []
            for item, _audio in decoded:
                helpers.set_failure(item["target"], "asr_error", str(error))
        except Exception:
            pairs = []
            for value in decoded:
                try:
                    pairs.append((value, helpers.run_batch(self.asr, [value])[0]))
                except Exception as error:
                    helpers.set_failure(
                        value[0]["target"],
                        "asr_error",
                        f"{type(error).__name__}: {error}",
                    )

        for (item, (_audio, _sample_rate, decoded_start)), result in pairs:
            target = item["target"]
            text = str(getattr(result, "text", "") or "").strip()
            language = str(getattr(result, "language", "") or "")
            if not text:
                helpers.set_failure(target, "no_lyrics_detected")
                target["language"] = language
                continue
            tokens = helpers.aligned_items(result)
            if not tokens:
                helpers.set_failure(
                    target,
                    "alignment_error",
                    "ForcedAligner returned no timestamps",
                )
                target.update({"raw_asr_text": text, "language": language})
                continue
            kept = helpers.crop_aligned_tokens(
                tokens,
                decoded_start,
                float(item["section"]["start"]),
                float(item["section"]["end"]),
            )
            target.update(
                {
                    "lyrics": helpers.join_tokens(kept, language) if kept else None,
                    "asr_tokens": kept,
                    "asr_status": "ok" if kept else "no_lyrics_in_core",
                    "language": language,
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10102)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--model",
        default=(
            "/mnt/data/yuyin/datasets/jinwenqing/work/JwqMusic/Evaluation/"
            "Qwen3-ASR-main/Qwen/Qwen3-ASR-1.7B"
        ),
    )
    parser.add_argument(
        "--forced-aligner",
        default=(
            "/mnt/data/yuyin/datasets/jinwenqing/work/JwqMusic/Evaluation/"
            "Qwen3-ASR-main/Qwen/Qwen3-ForcedAligner-0.6B"
        ),
    )
    parser.add_argument("--section-batch-size", type=int, default=4)
    parser.add_argument("--max-wait-ms", type=int, default=200)
    parser.add_argument("--decode-workers", type=int, default=2)
    parser.add_argument("--padding", type=float, default=1.5)
    parser.add_argument("--vllm-max-memory-gib", type=float, default=0.0)
    parser.add_argument("--gpu-max-memory-gib", type=float, default=0.0)
    parser.add_argument("--forced-aligner-reserve-gib", type=float, default=8.0)
    parser.add_argument("--vllm-headroom-gib", type=float, default=4.0)
    parser.add_argument("--minimum-vllm-memory-gib", type=float, default=8.0)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--queue-size", type=int, default=64)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    numeric = (
        args.vllm_max_memory_gib,
        args.gpu_max_memory_gib,
        args.forced_aligner_reserve_gib,
        args.vllm_headroom_gib,
        args.minimum_vllm_memory_gib,
    )
    if not all(math.isfinite(value) and value >= 0 for value in numeric):
        raise SystemExit("GPU memory values must be finite and non-negative")
    if args.minimum_vllm_memory_gib <= 0:
        raise SystemExit("--minimum-vllm-memory-gib must be positive")
    if not 1 <= args.section_batch_size <= 4:
        raise SystemExit("--section-batch-size must be in [1, 4]")
    if args.max_wait_ms < 0 or args.decode_workers <= 0 or args.padding < 0:
        raise SystemExit("wait, decode workers, and padding values are invalid")

    service = DynamicBatchService(
        loader=lambda: SectionASRRuntime.load(args),
        process_batch=lambda runtime, requests: runtime.process_batch(requests),
        stage="section_asr",
        device=args.device,
        model_fingerprint=f"{args.model}:{args.forced_aligner}",
        max_batch_size=4,
        max_wait_ms=args.max_wait_ms,
        queue_size=args.queue_size,
    )
    run_service(create_service_app(service), args.host, args.port)


if __name__ == "__main__":
    main()

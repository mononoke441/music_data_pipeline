#!/usr/bin/env python3
"""Batched padded Song-section ASR with Qwen3 ForcedAligner timestamps."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Sequence,
    Tuple,
)

import numpy as np

from gpu_memory import (
    GIB,
    apply_torch_cuda_memory_limit,
    resolve_asr_vllm_memory_budget,
)

from pipeline_core import (
    PIPELINE_VERSION,
    crop_aligned_tokens,
    decode_audio_range,
    iter_jsonl,
    should_run_section_asr,
    write_jsonl,
)
from pipeline_progress import pipeline_tqdm
from pipeline_state import semantic_input_fingerprint


class BatchCardinalityError(RuntimeError):
    """The backend violated the one-result-per-input contract."""


def capped_vllm_gpu_memory_utilization(
    max_memory_gib: float,
    total_memory_bytes: int,
    requested_utilization: float | None = None,
) -> float:
    """Translate an absolute GiB ceiling to a non-exceeding vLLM fraction."""
    max_memory_gib = float(max_memory_gib)
    total_memory_bytes = int(total_memory_bytes)
    if not math.isfinite(max_memory_gib) or max_memory_gib < 0:
        raise ValueError("vLLM max GPU memory GiB must be finite and non-negative")
    if total_memory_bytes <= 0:
        raise ValueError("total GPU memory must be positive")
    cap = (
        0.99
        if max_memory_gib == 0
        else min(max_memory_gib * (1024**3) / total_memory_bytes, 0.99)
    )
    selected = cap
    if requested_utilization is not None:
        requested = float(requested_utilization)
        if not math.isfinite(requested) or not 0 < requested <= 1:
            raise ValueError("requested vLLM GPU memory utilization must be in (0, 1]")
        selected = min(selected, requested)
    # Truncating to six decimals guarantees the vLLM fraction does not round
    # above the requested absolute ceiling.
    selected = math.floor(selected * 1_000_000) / 1_000_000
    if selected <= 0:
        raise ValueError("resolved vLLM GPU memory utilization is zero")
    return selected


def live_asr_vllm_memory_budget(
    torch_module: Any,
    *,
    pipeline_max_memory_gib: float,
    requested_vllm_max_memory_gib: float,
    forced_aligner_reserve_gib: float,
    vllm_headroom_gib: float,
    minimum_vllm_memory_gib: float,
) -> tuple[float, int, int]:
    """Sample CUDA free memory immediately before constructing Qwen ASR."""

    free_memory_bytes, total_memory_bytes = torch_module.cuda.mem_get_info(0)
    budget = resolve_asr_vllm_memory_budget(
        free_memory_gib=float(free_memory_bytes) / GIB,
        pipeline_max_memory_gib=pipeline_max_memory_gib,
        requested_vllm_max_memory_gib=requested_vllm_max_memory_gib,
        forced_aligner_reserve_gib=forced_aligner_reserve_gib,
        vllm_headroom_gib=vllm_headroom_gib,
        minimum_vllm_memory_gib=minimum_vllm_memory_gib,
    )
    return budget, int(free_memory_bytes), int(total_memory_bytes)


def duration_bucket(duration: float) -> str:
    if duration <= 15:
        return "8-15"
    if duration <= 30:
        return "15-30"
    if duration <= 60:
        return "30-60"
    return "60+"


def sections_hash(record: Mapping[str, Any]) -> str:
    """Bind cached ASR to the exact boundaries and voice-routing inputs."""
    payload = {
        "audio_id": str(record.get("audio_id", "")),
        "content_type": str(record.get("content_type", "")),
        "sections": [
            {
                "section_id": str(section.get("section_id", "")),
                "start": round(float(section.get("start", 0.0)), 6),
                "end": round(float(section.get("end", 0.0)), 6),
                "label": section.get("label"),
                "voice_coverage": section.get("voice_coverage"),
            }
            for section in record.get("sections") or []
        ],
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def section_asr_input_fingerprint(record: Mapping[str, Any]) -> str:
    return semantic_input_fingerprint(record, "section_asr")


def decode_item(
    item: Mapping[str, Any], padding: float
) -> Tuple[np.ndarray, int, float]:
    core_start = float(item["section"]["start"])
    core_end = float(item["section"]["end"])
    decoded_start = max(0.0, core_start - padding)
    decoded_end = min(float(item["record"]["duration"]), core_end + padding)
    raw = decode_audio_range(
        str(item["record"]["audio_path"]),
        decoded_start,
        decoded_end,
        sample_rate=16000,
        output_format="f32le",
    )
    return np.frombuffer(raw, dtype="<f4").copy(), 16000, decoded_start


def aligned_items(result: Any) -> List[Dict[str, Any]]:
    values = getattr(result, "time_stamps", None)
    if values is None:
        return []
    if isinstance(values, list) and len(values) == 1:
        values = values[0]
    values = getattr(values, "items", values)
    output = []
    for value in values or []:
        if isinstance(value, Mapping):
            text = value.get("text", "")
            start = value.get("start_time", value.get("start", 0.0))
            end = value.get("end_time", value.get("end", start))
        else:
            text = getattr(value, "text", "")
            start = getattr(value, "start_time", getattr(value, "start", 0.0))
            end = getattr(value, "end_time", getattr(value, "end", start))
        output.append({"text": str(text), "start": float(start), "end": float(end)})
    return output


def join_tokens(tokens: Iterable[Mapping[str, Any]], language: str) -> str:
    texts = [str(value.get("text", "")).strip() for value in tokens]
    texts = [value for value in texts if value]
    compact = any(
        name in language.lower() for name in ("chinese", "cantonese", "japanese")
    )
    return ("" if compact else " ").join(texts).strip()


def set_failure(target: Dict[str, Any], status: str, error: str | None = None) -> None:
    target.update({"lyrics": None, "asr_tokens": [], "asr_status": status})
    if error:
        target["alignment_error" if status == "alignment_error" else "asr_error"] = (
            error
        )


def run_batch(
    asr: Any, decoded: List[Tuple[Mapping[str, Any], Tuple[np.ndarray, int, float]]]
) -> List[Any]:
    audios = [(audio, sample_rate) for _, (audio, sample_rate, _) in decoded]
    results = asr.transcribe(audio=audios, language=None, return_time_stamps=True)
    if not isinstance(results, (list, tuple)):
        results = [results]
    results = list(results)
    if len(results) != len(decoded):
        raise BatchCardinalityError(
            f"ASR returned {len(results)} result(s) for a batch of {len(decoded)}"
        )
    return results


def section_batches(
    buckets: Mapping[str, Sequence[Dict[str, Any]]],
    batch_size: int,
) -> Iterator[List[Dict[str, Any]]]:
    """Yield duration-homogeneous ASR batches in the fixed bucket order."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for bucket_name in ("8-15", "15-30", "30-60", "60+"):
        items = buckets.get(bucket_name, ())
        for offset in range(0, len(items), batch_size):
            yield list(items[offset : offset + batch_size])


def _submit_decode_batch(
    executor: concurrent.futures.ThreadPoolExecutor,
    batch: Sequence[Dict[str, Any]],
    padding: float,
    decode_function: Callable[
        [Mapping[str, Any], float], Tuple[np.ndarray, int, float]
    ],
) -> Dict[concurrent.futures.Future, Dict[str, Any]]:
    return {executor.submit(decode_function, item, padding): item for item in batch}


def _collect_decoded_batch(
    futures: Mapping[concurrent.futures.Future, Dict[str, Any]],
) -> List[Tuple[Mapping[str, Any], Tuple[np.ndarray, int, float]]]:
    decoded: List[Tuple[Mapping[str, Any], Tuple[np.ndarray, int, float]]] = []
    for future in concurrent.futures.as_completed(futures):
        item = futures[future]
        try:
            decoded.append((item, future.result()))
        except Exception as error:
            set_failure(
                item["target"],
                "decode_error",
                f"{type(error).__name__}: {error}",
            )
    return decoded


def prefetched_decode_batches(
    batches: Iterable[Sequence[Dict[str, Any]]],
    *,
    executor: concurrent.futures.ThreadPoolExecutor,
    padding: float,
    decode_function: Callable[
        [Mapping[str, Any], float], Tuple[np.ndarray, int, float]
    ] = decode_item,
) -> Iterator[List[Tuple[Mapping[str, Any], Tuple[np.ndarray, int, float]]]]:
    """Reuse one pool and decode batch N+1 while the caller infers batch N."""

    iterator = iter(batches)
    try:
        first = next(iterator)
    except StopIteration:
        return
    current = _submit_decode_batch(executor, first, padding, decode_function)
    while current:
        decoded = _collect_decoded_batch(current)
        try:
            following = next(iterator)
        except StopIteration:
            following = None
        next_futures = (
            _submit_decode_batch(executor, following, padding, decode_function)
            if following is not None
            else {}
        )
        # The next ffmpeg jobs are already running while the caller is inside
        # Qwen3-ASR/ForcedAligner for this yielded batch.
        yield decoded
        current = next_futures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="data.sections.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model", required=True, help="Local Qwen3-ASR path or model id"
    )
    parser.add_argument(
        "--forced-aligner", required=True, help="Local ForcedAligner path or model id"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--decode-workers", type=int, default=8)
    parser.add_argument("--padding", type=float, default=1.5)
    parser.add_argument(
        "--vllm-max-memory-gib",
        type=float,
        default=0.0,
        help="absolute vLLM GPU-memory ceiling; 0 means no fixed ceiling",
    )
    parser.add_argument("--gpu-max-memory-gib", type=float, default=0.0)
    parser.add_argument("--forced-aligner-reserve-gib", type=float, default=8.0)
    parser.add_argument("--vllm-headroom-gib", type=float, default=4.0)
    parser.add_argument("--minimum-vllm-memory-gib", type=float, default=8.0)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        help="optional lower fraction; values above --vllm-max-memory-gib are clamped",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0 or args.decode_workers <= 0:
        parser.error("batch size and decode workers must be positive")
    if args.padding < 0:
        parser.error("padding must be non-negative")
    memory_values = (
        args.vllm_max_memory_gib,
        args.gpu_max_memory_gib,
        args.forced_aligner_reserve_gib,
        args.vllm_headroom_gib,
        args.minimum_vllm_memory_gib,
    )
    if not all(math.isfinite(value) and value >= 0 for value in memory_values):
        parser.error("GPU memory values must be finite and non-negative")
    if args.minimum_vllm_memory_gib <= 0:
        parser.error("minimum vLLM memory must be positive")

    records = list(iter_jsonl(args.input))
    existing: Dict[str, Dict[str, Any]] = {}
    if args.resume and Path(args.output).exists():
        for value in iter_jsonl(args.output):
            audio_id = str(value.get("audio_id") or "").strip()
            if not audio_id:
                raise ValueError(f"{args.output}: cached record is missing audio_id")
            if audio_id in existing:
                raise ValueError(f"{args.output}: duplicate audio_id={audio_id}")
            existing[audio_id] = value
    outputs: Dict[str, Dict[str, Any]] = {}
    reused_audio_ids: set[str] = set()
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen_audio_ids: set[str] = set()
    for record in records:
        audio_id = str(record.get("audio_id") or "").strip()
        if not audio_id:
            raise ValueError(f"{args.input}: input record is missing audio_id")
        if audio_id in seen_audio_ids:
            raise ValueError(f"{args.input}: duplicate audio_id={audio_id}")
        seen_audio_ids.add(audio_id)
        semantic_fingerprint = section_asr_input_fingerprint(record)
        try:
            source_sections = list(record.get("sections") or [])
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
            fingerprint = sections_hash(record)
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            outputs[audio_id] = {
                "audio_id": audio_id,
                "audio_path": record.get("audio_path"),
                "sections": [],
                "section_asr_input_fingerprint": semantic_fingerprint,
                "semantic_input_fingerprint": semantic_fingerprint,
                "pipeline_version": PIPELINE_VERSION,
                "stage_status": {"section_asr": "error"},
                "stage_errors": {"section_asr": f"{type(error).__name__}: {error}"},
            }
            continue
        cached = existing.get(audio_id)
        if cached is not None:
            if (
                cached.get("sections_hash") == fingerprint
                and (cached.get("stage_status") or {}).get("section_asr") == "ok"
                and str(
                    cached.get("section_asr_input_fingerprint")
                    or cached.get("semantic_input_fingerprint")
                    or ""
                )
                == semantic_fingerprint
            ):
                outputs[audio_id] = cached
                reused_audio_ids.add(audio_id)
                continue
        section_outputs = []
        for section in source_sections:
            target = {
                "section_id": section["section_id"],
                "start": section["start"],
                "end": section["end"],
            }
            section_outputs.append(target)
            if str(record.get("content_type", "")).lower() != "song":
                set_failure(target, "not_applicable")
                continue
            eligible, reason = should_run_section_asr(section)
            if not eligible:
                set_failure(target, f"skipped_{reason}")
                continue
            item = {"record": record, "section": section, "target": target}
            buckets[
                duration_bucket(float(section["end"]) - float(section["start"]))
            ].append(item)
        outputs[audio_id] = {
            "audio_id": audio_id,
            "audio_path": record.get("audio_path"),
            "sections": section_outputs,
            "sections_hash": fingerprint,
            "section_asr_input_fingerprint": semantic_fingerprint,
            "semantic_input_fingerprint": semantic_fingerprint,
            "pipeline_version": PIPELINE_VERSION,
        }

    pending_sections = sum(len(items) for items in buckets.values())
    progress = pipeline_tqdm(
        total=pending_sections,
        desc="6/7 section ASR + alignment",
        unit="section",
    )
    asr = None
    if any(buckets.values()):
        import torch
        from qwen_asr import Qwen3ASRModel

        torch_memory_fraction = apply_torch_cuda_memory_limit(
            args.gpu_max_memory_gib,
            0,
            torch_module=torch,
        )
        (
            resolved_vllm_max_memory_gib,
            free_memory_bytes,
            total_memory_bytes,
        ) = live_asr_vllm_memory_budget(
            torch,
            pipeline_max_memory_gib=args.gpu_max_memory_gib,
            requested_vllm_max_memory_gib=args.vllm_max_memory_gib,
            forced_aligner_reserve_gib=args.forced_aligner_reserve_gib,
            vllm_headroom_gib=args.vllm_headroom_gib,
            minimum_vllm_memory_gib=args.minimum_vllm_memory_gib,
        )
        resolved_gpu_memory_utilization = capped_vllm_gpu_memory_utilization(
            resolved_vllm_max_memory_gib,
            total_memory_bytes,
            args.gpu_memory_utilization,
        )
        print(
            "[section-asr] "
            f"gpu_max={args.gpu_max_memory_gib:g}GiB "
            f"free={free_memory_bytes / GIB:.6f}GiB "
            f"vllm_max={resolved_vllm_max_memory_gib:g}GiB "
            f"forced_aligner_reserve={args.forced_aligner_reserve_gib:g}GiB "
            f"vllm_headroom={args.vllm_headroom_gib:g}GiB "
            f"minimum_vllm={args.minimum_vllm_memory_gib:g}GiB "
            f"torch_allocator_fraction={torch_memory_fraction:.6f} "
            f"gpu_memory_utilization={resolved_gpu_memory_utilization:.6f}"
        )
        asr = Qwen3ASRModel.LLM(
            model=args.model,
            gpu_memory_utilization=resolved_gpu_memory_utilization,
            max_inference_batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            forced_aligner=args.forced_aligner,
            forced_aligner_kwargs={"dtype": torch.bfloat16, "device_map": "cuda:0"},
        )

    batches = section_batches(buckets, args.batch_size)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.decode_workers
    ) as executor:
        for decoded in prefetched_decode_batches(
            batches,
            executor=executor,
            padding=args.padding,
        ):
            if not decoded:
                continue
            try:
                results = run_batch(asr, decoded)
                pairs = list(zip(decoded, results))
            except BatchCardinalityError as error:
                pairs = []
                for value in decoded:
                    set_failure(value[0]["target"], "asr_error", str(error))
            except Exception:
                pairs = []
                for value in decoded:
                    try:
                        result = run_batch(asr, [value])[0]
                        pairs.append((value, result))
                    except Exception as error:
                        set_failure(
                            value[0]["target"],
                            "asr_error",
                            f"{type(error).__name__}: {error}",
                        )
            for (item, (_, _, decoded_start)), result in pairs:
                target = item["target"]
                text = str(getattr(result, "text", "") or "").strip()
                language = str(getattr(result, "language", "") or "")
                if not text:
                    set_failure(target, "no_lyrics_detected")
                    target["language"] = language
                    continue
                tokens = aligned_items(result)
                if not tokens:
                    set_failure(
                        target,
                        "alignment_error",
                        "ForcedAligner returned no timestamps",
                    )
                    target.update({"raw_asr_text": text, "language": language})
                    continue
                kept = crop_aligned_tokens(
                    tokens,
                    decoded_start,
                    float(item["section"]["start"]),
                    float(item["section"]["end"]),
                )
                target.update(
                    {
                        "lyrics": join_tokens(kept, language) if kept else None,
                        "asr_tokens": kept,
                        "asr_status": "ok" if kept else "no_lyrics_in_core",
                        "language": language,
                    }
                )
            progress.update(len(decoded))
    progress.close()

    final_records = []
    for record in records:
        value = outputs[str(record["audio_id"])]
        lyrics = [
            section["lyrics"] for section in value["sections"] if section.get("lyrics")
        ]
        value["full_transcript"] = "\n".join(lyrics) if lyrics else None
        bad = {"decode_error", "asr_error", "alignment_error"}
        failed_sections = [
            section for section in value["sections"] if section.get("asr_status") in bad
        ]
        value["stage_status"] = {
            "section_asr": "error" if failed_sections or not value["sections"] else "ok"
        }
        if value["stage_status"]["section_asr"] != "ok":
            details = [
                f"{section.get('section_id')}: "
                f"{section.get('alignment_error') or section.get('asr_error') or section.get('asr_status')}"
                for section in failed_sections
            ] or ["input contains no sections"]
            value.setdefault("stage_errors", {}).setdefault(
                "section_asr", "; ".join(details)
            )
        if str(record["audio_id"]) not in reused_audio_ids:
            value["model_versions"] = {
                "section_asr": str(args.model),
                "forced_aligner": str(args.forced_aligner),
            }
        final_records.append(value)
    write_jsonl(Path(args.output), final_records)


if __name__ == "__main__":
    main()

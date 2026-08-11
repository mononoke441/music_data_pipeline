#!/usr/bin/env python3
"""Generate short section captions through an OpenAI-compatible audio API."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, Mapping

import aiohttp

from pipeline_core import PIPELINE_VERSION, decode_audio_range, iter_jsonl, write_jsonl
from pipeline_progress import pipeline_tqdm
from pipeline_state import semantic_input_fingerprint


def sections_hash(record: Mapping[str, Any]) -> str:
    """Fingerprint the exact section plan and caption-relevant routing context."""
    payload = {
        "audio_id": str(record.get("audio_id", "")),
        "content_type": str(record.get("content_type", "")),
        "global_caption": record.get("ALM_Caption") or record.get("global_caption"),
        "global_mir": {
            **(record.get("global_mir") or {}),
            **(record.get("music_cpu") or {}),
        },
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


def section_caption_input_fingerprint(record: Mapping[str, Any]) -> str:
    return semantic_input_fingerprint(record, "section_caption")


class RetryableCaptionError(RuntimeError):
    """A response completed but did not contain a reusable caption."""


def caption_from_response(data: Any) -> str:
    if not isinstance(data, Mapping):
        raise RetryableCaptionError("response is not a JSON object")
    choices = data.get("choices")
    if (
        not isinstance(choices, list)
        or not choices
        or not isinstance(choices[0], Mapping)
    ):
        raise RetryableCaptionError("response is missing choices[0]")
    choice = choices[0]
    if str(choice.get("finish_reason") or "").lower() == "length":
        raise RetryableCaptionError("finish_reason=length")
    message = choice.get("message")
    if not isinstance(message, Mapping) or "content" not in message:
        raise RetryableCaptionError("response is missing message text content")
    content = message.get("content")
    if isinstance(content, list):
        parts = [
            str(item.get("text") or "").strip()
            for item in content
            if isinstance(item, Mapping) and item.get("type") == "text"
        ]
        if not parts:
            raise RetryableCaptionError("response is missing message text content")
        caption = " ".join(part for part in parts if part).strip()
    elif isinstance(content, str):
        caption = content.strip()
    else:
        raise RetryableCaptionError("response is missing message text content")
    if not caption:
        raise RetryableCaptionError("response caption is empty")
    return caption


def reusable_cached_record(
    cached: Mapping[str, Any], current: Mapping[str, Any], section_plan_hash: str
) -> Dict[str, Any] | None:
    if cached.get("sections_hash") != section_plan_hash:
        return None
    current_fingerprint = section_caption_input_fingerprint(current)
    cached_fingerprint = str(cached.get("section_caption_input_fingerprint") or "")
    if cached_fingerprint:
        if cached_fingerprint != current_fingerprint:
            return None
    elif section_caption_input_fingerprint(cached) != current_fingerprint:
        return None
    value = dict(cached)
    value["section_caption_input_fingerprint"] = current_fingerprint
    value["semantic_input_fingerprint"] = current_fingerprint
    return value


def prompt_for(record: Mapping[str, Any], section: Mapping[str, Any]) -> str:
    context = {
        "section_label": section.get("label"),
        "global_caption": record.get("ALM_Caption") or record.get("global_caption"),
        "global_mir": {
            **(record.get("global_mir") or {}),
            **(record.get("music_cpu") or {}),
        },
    }
    if str(record.get("content_type", "")).lower() == "instrumental":
        instruction = (
            "Describe this instrumental section in 1-3 concise sentences. Focus on instruments, "
            "rhythm, harmony, texture, production and its role in the full piece. It has been "
            "classified as instrumental: do not mention singers, vocals or lyrics."
        )
    else:
        instruction = (
            "Describe this song section in 1-3 concise sentences. Focus on local singing style, "
            "instrumentation, rhythm, harmony, texture and dynamics. Do not quote, transcribe or "
            "invent lyrics; lyrics are handled by a separate ASR stage."
        )
    return instruction + "\nContext: " + json.dumps(context, ensure_ascii=False)


async def request_caption(
    session: aiohttp.ClientSession,
    server: str,
    model: str,
    prompt: str,
    audio_bytes: bytes,
    max_tokens: int,
    temperature: float,
) -> str:
    audio_url = "data:audio/wav;base64," + base64.b64encode(audio_bytes).decode("ascii")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio_url", "audio_url": {"url": audio_url}},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    async with session.post(
        server.rstrip("/") + "/v1/chat/completions", json=payload
    ) as response:
        body = await response.text()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {body[:500]}")
    return caption_from_response(json.loads(body))


async def main_async(args: argparse.Namespace) -> None:
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    if args.decode_workers < 1:
        raise ValueError("--decode-workers must be at least 1")
    if args.decoded_buffer < 1:
        raise ValueError("--decoded-buffer must be at least 1")
    if args.track_buffer < 1:
        raise ValueError("--track-buffer must be at least 1")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    input_records = list(iter_jsonl(args.input))
    caption_version = f"{args.model}:section-caption-v1"
    cached_ok: Dict[str, Dict[str, Any]] = {}
    cached_audio_ids: set[str] = set()
    if args.resume and output.exists():
        for record in iter_jsonl(output):
            status = (record.get("stage_status") or {}).get("section_caption")
            audio_id = str(record.get("audio_id", "")).strip()
            valid_sections = record.get("sections") or []
            if not audio_id:
                raise ValueError(f"{output}: cached record is missing audio_id")
            if audio_id in cached_audio_ids:
                raise ValueError(f"{output}: duplicate audio_id={audio_id}")
            cached_audio_ids.add(audio_id)
            if (
                status == "ok"
                and valid_sections
                and all(
                    section.get("status") == "ok"
                    and str(section.get("short_caption") or "").strip()
                    for section in valid_sections
                )
            ):
                cached_ok[audio_id] = record
    # Keep three limits separate.  With a single semaphore and API concurrency
    # 1, section N+1 could not decode until the model had finished generating
    # section N.  The decoded-buffer limit lets those stages overlap without
    # allowing every section in the dataset to retain WAV bytes in memory.
    request_semaphore = asyncio.Semaphore(args.concurrency)
    decode_semaphore = asyncio.Semaphore(args.decode_workers)
    decoded_buffer = asyncio.Semaphore(args.decoded_buffer)
    api_key = os.environ.get(args.api_key_env, "").strip()
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    outputs: Dict[int, Dict[str, Any]] = {}
    progress = pipeline_tqdm(
        total=len(input_records),
        desc="5/7 section caption",
        unit="track",
    )

    async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:

        async def infer_section(
            record: Dict[str, Any], section: Dict[str, Any], index: int
        ) -> Dict[str, Any]:
            result = {
                "section_id": section.get("section_id"),
                "start": section.get("start"),
                "end": section.get("end"),
            }
            async with decoded_buffer:
                try:
                    async with decode_semaphore:
                        audio = await asyncio.to_thread(
                            decode_audio_range,
                            str(record["audio_path"]),
                            float(section["start"]),
                            float(section["end"]),
                            sample_rate=args.sample_rate,
                            output_format="wav",
                        )
                    last_error: Exception | None = None
                    for attempt in range(args.retries + 1):
                        try:
                            server = args.servers[(index + attempt) % len(args.servers)]
                            async with request_semaphore:
                                caption = await request_caption(
                                    session,
                                    server,
                                    args.model,
                                    prompt_for(record, section),
                                    audio,
                                    args.max_tokens,
                                    args.temperature,
                                )
                            if not caption.strip():
                                raise RetryableCaptionError("response caption is empty")
                            result["short_caption"] = caption
                            result["status"] = "ok"
                            return result
                        except Exception as error:
                            last_error = error
                            if attempt < args.retries:
                                await asyncio.sleep(
                                    min(2**attempt + random.random(), 8.0)
                                )
                    raise last_error or RuntimeError("caption request failed")
                except Exception as error:
                    result.update(
                        {
                            "short_caption": None,
                            "status": "error",
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    return result

        async def run_track(
            record_index: int, record: Dict[str, Any]
        ) -> tuple[int, Dict[str, Any]]:
            audio_id = str(record.get("audio_id", "")).strip()
            if not audio_id:
                raise ValueError(
                    f"{args.input}:{record_index + 1}: record is missing audio_id"
                )
            semantic_fingerprint = section_caption_input_fingerprint(record)
            try:
                sections = list(record.get("sections") or [])
                if any(not isinstance(section, Mapping) for section in sections):
                    raise ValueError("sections must contain JSON objects")
                fingerprint = sections_hash(record)
            except (AttributeError, KeyError, TypeError, ValueError) as error:
                progress.update(1)
                return record_index, {
                    "audio_id": audio_id,
                    "audio_path": record.get("audio_path"),
                    "sections": [],
                    "section_caption_input_fingerprint": semantic_fingerprint,
                    "semantic_input_fingerprint": semantic_fingerprint,
                    "stage_status": {"section_caption": "error"},
                    "stage_errors": {
                        "section_caption": f"{type(error).__name__}: {error}"
                    },
                    "pipeline_version": PIPELINE_VERSION,
                    "model_versions": {"section_caption": caption_version},
                }
            cached = cached_ok.get(audio_id)
            if cached is not None:
                reusable = reusable_cached_record(cached, record, fingerprint)
                if reusable is not None:
                    progress.update(1)
                    return record_index, reusable
            results = await asyncio.gather(
                *[
                    infer_section(record, section, index)
                    for index, section in enumerate(sections)
                ]
            )
            failed_sections = [item for item in results if item.get("status") != "ok"]
            value: Dict[str, Any] = {
                "audio_id": audio_id,
                "audio_path": record.get("audio_path"),
                "sections": results,
                "sections_hash": fingerprint,
                "section_caption_input_fingerprint": semantic_fingerprint,
                "semantic_input_fingerprint": semantic_fingerprint,
                "stage_status": {
                    "section_caption": "ok"
                    if results and not failed_sections
                    else "error",
                },
                "pipeline_version": PIPELINE_VERSION,
                "model_versions": {"section_caption": caption_version},
            }
            if value["stage_status"]["section_caption"] != "ok":
                details = [
                    f"{item.get('section_id')}: {item.get('error') or item.get('status')}"
                    for item in failed_sections
                ] or ["input contains no sections"]
                value["stage_errors"] = {"section_caption": "; ".join(details)}
            progress.update(1)
            return record_index, value

        pending: set[asyncio.Task] = set()
        try:
            for record_index, record in enumerate(input_records):
                pending.add(asyncio.create_task(run_track(record_index, record)))
                if len(pending) >= args.track_buffer:
                    completed, pending = await asyncio.wait(
                        pending, return_when=asyncio.FIRST_COMPLETED
                    )
                    for completed_index, value in await asyncio.gather(*completed):
                        outputs[completed_index] = value
            if pending:
                for completed_index, value in await asyncio.gather(*pending):
                    outputs[completed_index] = value
        except BaseException:
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            raise
    progress.close()
    expected = len(input_records)
    if len(outputs) != expected:
        raise RuntimeError(
            f"section caption task coverage mismatch: expected={expected} actual={len(outputs)}"
        )
    write_jsonl(output, (outputs[index] for index in range(expected)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", required=True, help="Sections JSONL enriched with global context"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--servers", nargs="+", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="INF_API_KEY")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--decode-workers", type=int, default=2)
    parser.add_argument(
        "--decoded-buffer",
        type=int,
        default=2,
        help="Maximum decoded sections retained while waiting for API capacity",
    )
    parser.add_argument("--track-buffer", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()

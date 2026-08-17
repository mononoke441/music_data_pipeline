#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1a — 调用 Audio-Language Model (ALM) 服务，为每首歌生成 Base Caption。

兼容任意 OpenAI-compatible 的多模态 chat 接口，典型选项包括：
  - Qwen3-Omni 系列（Qwen3-Omni-30B-A3B-Instruct 等）
  - MusicFlamingo / Audio-Flamingo-3 等专用音乐 / 音频 ALM
  - 其他支持 `audio_url` 的 vLLM / sglang / tgi 服务

依赖：上述 ALM 服务已启动（见 README 部署说明）。
输出字段默认：ALM_Caption（可通过 --output_field 覆盖）。
"""

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
from urllib.parse import urlparse

import aiohttp
from tqdm import tqdm

from pipeline_core import iter_jsonl, write_jsonl
from pipeline_progress import pipeline_tqdm
from pipeline_state import semantic_input_fingerprint

SONG_PROMPT_POOL = [
    "Describe this track in full detail - tell me the genre, tempo, and key, then dive into the instruments, production style, and overall mood it creates.",
    "Write a rich caption that blends the technical details (genre, BPM, key, chords, mix) with how the song feels emotionally and dynamically as it unfolds.",
    "Create a descriptive music caption that combines technical aspects (style, tempo feel, harmony, sound design) with a narrative of the song's emotional arc from beginning to end.",
    "Analyze the track and write a cohesive paragraph describing its musical style, tempo and harmonic character, key instruments and mix, and the mood or atmosphere it conveys as it progresses.",
    "Write a musically informed caption that weaves together genre, rhythmic intensity, harmonic feel, instrumentation, production texture, and the emotional journey of the piece.",
]

INSTRUMENTAL_PROMPT_POOL = [
    "Describe this instrumental track in full detail: genre, tempo feel, key, harmony, instruments, arrangement, production style, dynamics and emotional arc. Do not mention singers, vocals or lyrics.",
    "Write a rich whole-track caption for this instrumental music. Cover rhythm, harmony, instrumentation, sound design, mix, structure and mood. It has no vocals; never invent a singer or lyrics.",
    "Analyze this instrumental piece from beginning to end, including its musical style, pace, tonal character, changing textures and section-level development. Do not describe singing or language.",
]


def is_url(s: str) -> bool:
    try:
        u = urlparse(s)
        return u.scheme in ("http", "https", "file")
    except Exception:
        return False


def to_audio_url(audio_path: str) -> str:
    if is_url(audio_path):
        return audio_path
    p = Path(audio_path).expanduser()
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return p.as_uri()


def build_messages(prompt_text: str, audio_url: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "audio_url", "audio_url": {"url": audio_url}},
            ],
        }
    ]


def collect_jsonl_paths(inputs: List[str]) -> List[Path]:
    paths: List[Path] = []
    for x in inputs:
        p = Path(x)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.jsonl")))
        elif p.is_file() and p.suffix == ".jsonl":
            paths.append(p)
    return sorted(set(paths))


def count_jsonl_lines(paths: List[Path]) -> int:
    total = 0
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    total += 1
    return total


def get_resume_key(
    obj: Dict[str, Any], resume_key: str, audio_field: str
) -> Optional[str]:
    if resume_key:
        v = obj.get(resume_key)
        if isinstance(v, (str, int)):
            return str(v)
    v2 = obj.get(audio_field)
    if isinstance(v2, str) and v2.strip():
        return v2.strip()
    return None


def alm_input_fingerprint(record: Mapping[str, Any]) -> str:
    return semantic_input_fingerprint(record, "alm")


def load_cached_records(
    out_path: Path, resume_key: str, audio_field: str, output_field: str
) -> Dict[str, Dict[str, Any]]:
    done: Dict[str, Dict[str, Any]] = {}
    seen: set[str] = set()
    if not out_path.exists():
        return done
    for obj in iter_jsonl(out_path):
        k = get_resume_key(obj, resume_key, audio_field)
        if k is not None:
            if k in seen:
                raise ValueError(f"{out_path}: duplicate resume key={k!r}")
            seen.add(k)
        caption = obj.get(output_field)
        status = (obj.get("stage_status") or {}).get("alm")
        # Legacy successful records did not have stage_status.  A nonempty
        # caption and no error is sufficient to migrate them safely.
        valid = (
            isinstance(caption, str)
            and bool(caption.strip())
            and not obj.get("_error")
            and status in (None, "ok")
        )
        if k is not None and valid:
            done[k] = obj
    return done


def reusable_cached_record(
    cached: Mapping[str, Any], current: Mapping[str, Any]
) -> Dict[str, Any] | None:
    """Validate and safely backfill fingerprints on legacy successful rows."""

    current_fingerprint = alm_input_fingerprint(current)
    cached_fingerprint = str(cached.get("alm_input_fingerprint") or "")
    if cached_fingerprint:
        if cached_fingerprint != current_fingerprint:
            return None
    elif alm_input_fingerprint(cached) != current_fingerprint:
        return None
    value = dict(cached)
    value["alm_input_fingerprint"] = current_fingerprint
    value["semantic_input_fingerprint"] = current_fingerprint
    mark_alm_success(value)
    return value


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
        text_parts = [
            str(part.get("text") or "").strip()
            for part in content
            if isinstance(part, Mapping) and part.get("type") == "text"
        ]
        if not text_parts:
            raise RetryableCaptionError("response is missing message text content")
        caption = " ".join(part for part in text_parts if part).strip()
    elif isinstance(content, str):
        caption = content.strip()
    else:
        raise RetryableCaptionError("response is missing message text content")
    if not caption:
        raise RetryableCaptionError("response caption is empty")
    return caption


async def post_chat(
    session: aiohttp.ClientSession,
    base_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
    timeout_s: int,
) -> str:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with session.post(url, json=payload, timeout=timeout) as resp:
        text = await resp.text()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {text[:800]}")
        data = json.loads(text)

    return caption_from_response(data)


def mark_alm_error(obj: Dict[str, Any], message: str) -> Dict[str, Any]:
    obj["_error"] = message
    obj.setdefault("stage_status", {})["alm"] = "error"
    obj.setdefault("stage_errors", {})["alm"] = message
    return obj


def mark_alm_success(obj: Dict[str, Any]) -> None:
    obj.pop("_error", None)
    obj.setdefault("stage_status", {})["alm"] = "ok"
    stage_errors = obj.get("stage_errors")
    if isinstance(stage_errors, dict):
        stage_errors.pop("alm", None)
        if not stage_errors:
            obj.pop("stage_errors", None)


def prompt_for_record(obj: Mapping[str, Any]) -> str:
    is_instrumental = str(obj.get("content_type", "song")).lower() == "instrumental"
    prompt = random.choice(
        INSTRUMENTAL_PROMPT_POOL if is_instrumental else SONG_PROMPT_POOL
    )
    if not is_instrumental:
        prompt += " You may describe the voice, singing style and detected language, but do not quote, transcribe or invent lyrics."
    return prompt


def resolve_without_alm_api(
    obj: Dict[str, Any],
    cached_records: Mapping[str, Mapping[str, Any]],
    *,
    resume_key: str,
    audio_field: str,
    output_field: str,
    model: str,
    skip_existing: bool,
) -> Dict[str, Any] | None:
    """Return a final record when ALM inference is unnecessary."""

    key = get_resume_key(obj, resume_key, audio_field)
    if key is not None and key in cached_records:
        cached = reusable_cached_record(cached_records[key], obj)
        if cached is not None:
            return cached

    if (
        skip_existing
        and isinstance(obj.get(output_field), str)
        and obj[output_field].strip()
    ):
        fingerprint = alm_input_fingerprint(obj)
        obj["alm_input_fingerprint"] = fingerprint
        obj["semantic_input_fingerprint"] = fingerprint
        obj.setdefault("model_versions", {})["alm"] = model
        mark_alm_success(obj)
        return obj

    audio_path = obj.get(audio_field)
    if not isinstance(audio_path, str) or not audio_path.strip():
        fingerprint = alm_input_fingerprint(obj)
        obj["alm_input_fingerprint"] = fingerprint
        obj["semantic_input_fingerprint"] = fingerprint
        obj.setdefault("model_versions", {})["alm"] = model
        obj[output_field] = ""
        obj["_prompt"] = prompt_for_record(obj)
        return mark_alm_error(obj, f"missing_{audio_field}")
    return None


async def infer_one(
    session: aiohttp.ClientSession,
    servers: List[str],
    server_rr_lock: asyncio.Lock,
    rr_state: Dict[str, int],
    model: str,
    obj: Dict[str, Any],
    audio_field: str,
    output_field: str,
    max_tokens: int,
    temperature: float,
    timeout_s: int,
    retries: int,
    retry_base_sleep: float,
) -> Dict[str, Any]:
    obj.setdefault("model_versions", {})["alm"] = model
    fingerprint = alm_input_fingerprint(obj)
    obj["alm_input_fingerprint"] = fingerprint
    obj["semantic_input_fingerprint"] = fingerprint
    use_prompt = prompt_for_record(obj)
    audio_path = str(obj[audio_field]).strip()
    audio_url = to_audio_url(audio_path)
    messages = build_messages(use_prompt, audio_url)

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        async with server_rr_lock:
            idx = rr_state["i"]
            rr_state["i"] = (rr_state["i"] + 1) % len(servers)
        base_url = servers[idx]

        try:
            caption = await post_chat(
                session=session,
                base_url=base_url,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout_s=timeout_s,
            )
            if not caption.strip():
                raise RetryableCaptionError("response caption is empty")
            obj[output_field] = caption
            obj["_alm_prompt"] = use_prompt
            mark_alm_success(obj)
            return obj
        except Exception as e:
            last_err = e
            if attempt < retries:
                sleep_s = min(
                    (2**attempt) * retry_base_sleep + random.random() * 0.2, 8.0
                )
                await asyncio.sleep(sleep_s)

    obj[output_field] = ""
    obj["_alm_prompt"] = use_prompt
    return mark_alm_error(obj, f"infer_error: {type(last_err).__name__}: {last_err}")


async def process_one_file(args, session: aiohttp.ClientSession, pbar: tqdm):
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cached_records = (
        load_cached_records(
            out_path, args.resume_key, args.audio_field, args.output_field
        )
        if args.resume
        else {}
    )
    sem = args.sem
    server_rr_lock = args.server_rr_lock
    rr_state = args.rr_state

    error_count = 0
    inputs = list(iter_jsonl(in_path))
    for index, obj in enumerate(inputs, 1):
        if not str(obj.get("audio_id") or "").strip():
            raise ValueError(f"{in_path}:{index}: record is missing audio_id")
    outputs: Dict[int, Dict[str, Any]] = {}

    async def run_one(index: int, obj: Dict[str, Any]):
        nonlocal error_count
        async with sem:
            try:
                resolved = resolve_without_alm_api(
                    obj,
                    cached_records,
                    resume_key=args.resume_key,
                    audio_field=args.audio_field,
                    output_field=args.output_field,
                    model=args.model,
                    skip_existing=args.skip_existing,
                )
                if resolved is not None:
                    return index, resolved

                out_obj = await infer_one(
                    session=session,
                    servers=args.servers,
                    server_rr_lock=server_rr_lock,
                    rr_state=rr_state,
                    model=args.model,
                    obj=obj,
                    audio_field=args.audio_field,
                    output_field=args.output_field,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    timeout_s=args.timeout,
                    retries=args.retries,
                    retry_base_sleep=args.retry_base_sleep,
                )
                if (
                    out_obj.get("_error")
                    or not str(out_obj.get(args.output_field, "")).strip()
                ):
                    error_count += 1
                return index, out_obj
            finally:
                pbar.update(1)

    tasks: set[asyncio.Task] = set()
    try:
        for index, obj in enumerate(inputs):
            tasks.add(asyncio.create_task(run_one(index, obj)))
            if len(tasks) >= args.task_buffer:
                completed, tasks = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for completed_index, value in await asyncio.gather(*completed):
                    outputs[completed_index] = value
        if tasks:
            for completed_index, value in await asyncio.gather(*tasks):
                outputs[completed_index] = value
    except BaseException:
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        raise
    if len(outputs) != len(inputs):
        raise RuntimeError(
            f"ALM task coverage mismatch: expected={len(inputs)} actual={len(outputs)}"
        )
    write_jsonl(out_path, (outputs[index] for index in range(len(inputs))))
    return error_count


def cache_preflight(args: argparse.Namespace) -> int:
    """Print the number of tracks that still need ALM, without opening a session."""

    jsonl_paths = collect_jsonl_paths(args.inputs)
    if not jsonl_paths:
        print("No jsonl files found.", file=sys.stderr)
        return 1

    plans: List[tuple[Path, List[Dict[str, Any] | None]]] = []
    required_count = 0
    out_dir = Path(args.out_dir)
    for in_path in jsonl_paths:
        out_path = out_dir / f"{in_path.stem}.alm.jsonl"
        cached_records = load_cached_records(
            out_path, args.resume_key, args.audio_field, args.output_field
        )
        inputs = list(iter_jsonl(in_path))
        resolved_records: List[Dict[str, Any] | None] = []
        for index, obj in enumerate(inputs, 1):
            if not str(obj.get("audio_id") or "").strip():
                raise ValueError(f"{in_path}:{index}: record is missing audio_id")
            resolved = resolve_without_alm_api(
                obj,
                cached_records,
                resume_key=args.resume_key,
                audio_field=args.audio_field,
                output_field=args.output_field,
                model=args.model,
                skip_existing=args.skip_existing,
            )
            if resolved is None:
                required_count += 1
            resolved_records.append(resolved)
        plans.append((out_path, resolved_records))

    if required_count == 0:
        for out_path, records in plans:
            write_jsonl(out_path, (record for record in records if record is not None))
    print(required_count)
    return 0


async def process_all_async(args):
    if args.concurrency < 1 or args.task_buffer < 1:
        raise ValueError("concurrency and task buffer must be positive")
    if args.retries < 0:
        raise ValueError("retries must be non-negative")
    jsonl_paths = collect_jsonl_paths(args.inputs)
    if not jsonl_paths:
        print("No jsonl files found.", file=sys.stderr)
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_lines = count_jsonl_lines(jsonl_paths)
    pbar = pipeline_tqdm(
        total=total_lines, desc="3/7 whole-track caption", unit="track"
    )

    connector = aiohttp.TCPConnector(
        limit=max(args.concurrency * 2, 64),
        limit_per_host=max(args.concurrency * 2, 64),
        ttl_dns_cache=300,
    )

    api_key = os.environ.get(args.api_key_env, "").strip()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        args.sem = asyncio.Semaphore(args.concurrency)
        args.server_rr_lock = asyncio.Lock()
        args.rr_state = {"i": 0}

        for in_path in jsonl_paths:
            out_path = out_dir / f"{in_path.stem}.alm.jsonl"
            args.in_path = str(in_path)
            args.out_path = str(out_path)
            error_count = await process_one_file(args, session, pbar)
            print(
                f"[caption] file done: {in_path.name} errors={error_count}",
                file=sys.stderr,
            )

    pbar.close()
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="调用通用 Audio-Language Model (Qwen3-Omni 系列 / "
        "MusicFlamingo 等) 为音频生成 Base Caption。",
    )
    ap.add_argument(
        "--inputs",
        "-i",
        nargs="+",
        required=True,
        help="输入 jsonl 文件或包含 jsonl 的目录",
    )
    ap.add_argument("--out_dir", "-o", required=True, help="输出目录")
    ap.add_argument(
        "--servers",
        nargs="+",
        default=[],
        help="ALM 服务 Base URL（支持多个做轮询，OpenAI 兼容接口）",
    )
    ap.add_argument("--api_key_env", default="INF_API_KEY", help="API Key 环境变量名")
    ap.add_argument(
        "--model",
        required=True,
        help="ALM 模型名，例如 Qwen3-Omni-30B-A3B-Instruct / audio-flamingo-3 等",
    )
    ap.add_argument("--audio_field", default="audio_path")
    ap.add_argument("--output_field", default="ALM_Caption")
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--retry_base_sleep", type=float, default=1.0)
    ap.add_argument("--concurrency", type=int, default=2048, help="最大并发请求数")
    ap.add_argument("--task_buffer", type=int, default=2048)
    ap.add_argument(
        "--skip_existing", action="store_true", help="输入已有 output_field 时直接保留"
    )
    ap.add_argument(
        "--resume", action="store_true", help="按 audio_id + ALM model 断点续跑"
    )
    ap.add_argument(
        "--resume_key",
        type=str,
        default="audio_id",
        help="用作断点续跑唯一 key 的字段名",
    )
    ap.add_argument(
        "--cache-preflight",
        action="store_true",
        help="仅检查 resume cache，并输出仍需 ALM 的 track 数",
    )
    args = ap.parse_args()
    if not args.cache_preflight and not args.servers:
        ap.error("--servers is required unless --cache-preflight is used")
    if args.cache_preflight:
        sys.exit(cache_preflight(args))
    rc = asyncio.run(process_all_async(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()

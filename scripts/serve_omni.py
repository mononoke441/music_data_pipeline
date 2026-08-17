#!/usr/bin/env python3
"""Unified whole-track ALM proxy for an existing OpenAI-compatible Omni server."""

from __future__ import annotations

import argparse
import asyncio
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import aiohttp
from fastapi import Request
from fastapi.responses import JSONResponse

from alm_caption_infer import (
    build_messages,
    mark_alm_error,
    mark_alm_success,
    post_chat,
    prompt_for_record,
    to_audio_url,
)
from service_api import (
    BatchItemResult,
    DynamicBatchService,
    ServiceError,
    ServiceRequest,
    create_service_app,
    run_service,
)


@dataclass(frozen=True)
class OmniProxyRuntime:
    upstream: str
    model: str
    max_tokens: int
    temperature: float
    timeout: int
    retries: int
    retry_base_sleep: float

    def health_metadata(self) -> Mapping[str, Any]:
        return {
            "upstream_url": self.upstream,
            "proxy_mode": "external-openai-compatible",
        }


def _probe_upstream(upstream: str, timeout: float = 1.0) -> tuple[bool, str]:
    url = upstream.rstrip("/") + "/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read(4096).decode("utf-8", "replace")
            return 200 <= response.status < 300, body
    except (OSError, urllib.error.URLError, ValueError) as error:
        return False, f"{type(error).__name__}: {error}"


async def _infer_one(
    runtime: OmniProxyRuntime,
    session: aiohttp.ClientSession,
    request: ServiceRequest,
) -> Mapping[str, Any] | BatchItemResult:
    record = dict(request.record or {})
    record["audio_id"] = request.audio_id
    record["audio_path"] = request.audio_path
    record.setdefault("model_versions", {})["alm"] = runtime.model
    record["alm_input_fingerprint"] = request.input_fingerprint
    record["semantic_input_fingerprint"] = request.input_fingerprint
    prompt = str(record.get("_alm_prompt") or "").strip() or prompt_for_record(record)
    messages = build_messages(prompt, to_audio_url(request.audio_path))
    last_error: Exception | None = None
    for attempt in range(runtime.retries + 1):
        try:
            caption = await post_chat(
                session=session,
                base_url=runtime.upstream,
                model=runtime.model,
                messages=messages,
                max_tokens=runtime.max_tokens,
                temperature=runtime.temperature,
                timeout_s=runtime.timeout,
            )
            if not caption.strip():
                raise RuntimeError("upstream returned an empty caption")
            record["ALM_Caption"] = caption
            record["_alm_prompt"] = prompt
            mark_alm_success(record)
            return record
        except Exception as error:
            last_error = error
            if attempt < runtime.retries:
                await asyncio.sleep(min(runtime.retry_base_sleep * 2**attempt, 8.0))
    message = f"infer_error: {type(last_error).__name__}: {last_error}"
    record["ALM_Caption"] = ""
    record["_alm_prompt"] = prompt
    mark_alm_error(record, message)
    return BatchItemResult(record=record, error=ServiceError(message, 502))


async def process_omni_batch(
    runtime: OmniProxyRuntime,
    requests: Sequence[ServiceRequest],
) -> list[Mapping[str, Any] | BatchItemResult]:
    """Proxy whole tracks concurrently; section captioning is intentionally absent."""

    async with aiohttp.ClientSession() as session:
        return list(
            await asyncio.gather(
                *(_infer_one(runtime, session, request) for request in requests)
            )
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream",
        default=os.environ.get("OMNI_UPSTREAM_SERVER", "http://127.0.0.1:10008"),
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("ALM_MODEL", "Qwen3-Omni-30B-A3B-Instruct"),
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-base-sleep", type=float, default=1.0)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-wait-ms", type=int, default=50)
    parser.add_argument("--queue-size", type=int, default=32)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10103)
    return parser


def build_service(args: argparse.Namespace) -> DynamicBatchService:
    if args.concurrency <= 0 or args.queue_size <= 0:
        raise ValueError("concurrency and queue size must be positive")
    if args.max_wait_ms < 0 or args.timeout <= 0 or args.retries < 0:
        raise ValueError("wait/timeout/retry values are invalid")
    if not args.upstream.strip():
        raise ValueError("upstream must not be blank")
    return DynamicBatchService(
        loader=lambda: OmniProxyRuntime(
            upstream=args.upstream,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
            retries=args.retries,
            retry_base_sleep=args.retry_base_sleep,
        ),
        process_batch=process_omni_batch,
        stage="alm",
        device="external",
        model_fingerprint=f"proxy:{args.model}",
        max_batch_size=args.concurrency,
        max_wait_ms=args.max_wait_ms,
        queue_size=args.queue_size,
    )


def create_omni_app(
    service: DynamicBatchService,
    upstream: str,
):
    """Add upstream readiness to the common service API health contract."""

    app = create_service_app(service)

    @app.middleware("http")
    async def upstream_health_guard(request: Request, call_next):
        if request.url.path != "/healthz":
            return await call_next(request)
        snapshot = service.health_snapshot()
        if snapshot.get("status") != "ok":
            return JSONResponse(status_code=503, content={"detail": snapshot})
        healthy, detail = await asyncio.to_thread(_probe_upstream, upstream)
        snapshot["upstream_url"] = upstream.rstrip("/") + "/v1/models"
        snapshot["upstream_healthy"] = healthy
        snapshot["upstream_detail"] = detail[:500]
        if not healthy:
            snapshot["status"] = "upstream_unavailable"
            return JSONResponse(status_code=503, content={"detail": snapshot})
        return JSONResponse(status_code=200, content=snapshot)

    return app


def main() -> None:
    args = build_parser().parse_args()
    service = build_service(args)
    run_service(create_omni_app(service, args.upstream), args.host, args.port)


if __name__ == "__main__":
    main()

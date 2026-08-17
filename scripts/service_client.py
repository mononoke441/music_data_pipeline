#!/usr/bin/env python3
"""Strict client for one streaming inference service."""

from __future__ import annotations

import http.client
import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


class ServiceClientError(RuntimeError):
    pass


class ServiceProtocolError(ServiceClientError):
    pass


class ServiceInferenceError(ServiceClientError):
    def __init__(self, message: str, envelope: "InferEnvelope") -> None:
        super().__init__(message)
        self.envelope = envelope


@dataclass(frozen=True)
class InferEnvelope:
    job_id: str
    request_id: str
    audio_id: str
    input_fingerprint: str
    record: Dict[str, Any]
    stage: str
    model_fingerprint: str
    elapsed_seconds: float


class ServiceClient:
    """HTTP JSON client implementing GET /healthz and POST /v1/infer."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 1800.0,
        retries: int = 3,
        opener: Optional[urllib.request.OpenerDirector] = None,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        if not self.base_url:
            raise ValueError("service base URL must not be empty")
        self.timeout = float(timeout)
        if self.timeout <= 0:
            raise ValueError("service timeout must be positive")
        self.retries = int(retries)
        if self.retries < 0:
            raise ValueError("service retries must not be negative")
        self._opener = opener or urllib.request.build_opener()

    def _json_request(
        self, method: str, path: str, payload: Optional[Mapping[str, Any]] = None
    ) -> Dict[str, Any]:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        raw = b""
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(
                self.base_url + path, data=data, headers=headers, method=method
            )
            try:
                with self._opener.open(request, timeout=self.timeout) as response:
                    raw = response.read()
                break
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")[:1000]
                retryable = error.code in {429, 502, 503, 504}
                if retryable and attempt < self.retries:
                    continue
                raise ServiceClientError(
                    f"{method} {path} returned HTTP {error.code}: {detail}"
                ) from error
            except (
                urllib.error.URLError,
                TimeoutError,
                socket.timeout,
                ConnectionError,
                http.client.HTTPException,
            ) as error:
                if attempt < self.retries:
                    continue
                raise ServiceClientError(f"{method} {path} failed: {error}") from error
        try:
            value = json.loads(raw.decode("utf-8"))
        except Exception as error:
            raise ServiceProtocolError(f"{method} {path} returned invalid JSON") from error
        if not isinstance(value, dict):
            raise ServiceProtocolError(f"{method} {path} response is not an object")
        return value

    def healthz(self) -> Dict[str, Any]:
        value = self._json_request("GET", "/healthz")
        status = str(value.get("status") or "").lower()
        if status not in {"ok", "healthy", "ready"}:
            raise ServiceProtocolError(f"unhealthy service response: {value!r}")
        return value

    def infer(
        self,
        *,
        job_id: str,
        request_id: str,
        audio_id: str,
        audio_path: str,
        input_fingerprint: str,
        record: Mapping[str, Any],
    ) -> InferEnvelope:
        payload = {
            "job_id": job_id,
            "request_id": request_id,
            "audio_id": audio_id,
            "audio_path": audio_path,
            "input_fingerprint": input_fingerprint,
            "record": dict(record),
        }
        value = self._json_request("POST", "/v1/infer", payload)
        expected = {
            "job_id": job_id,
            "request_id": request_id,
            "audio_id": audio_id,
            "input_fingerprint": input_fingerprint,
        }
        for field, wanted in expected.items():
            if value.get(field) != wanted:
                raise ServiceProtocolError(
                    f"response {field} mismatch: got={value.get(field)!r} expected={wanted!r}"
                )
        status = str(value.get("status") or "").lower()
        result = value.get("record")
        if not isinstance(result, dict):
            raise ServiceProtocolError("response lacks object field 'record'")
        result_audio_id = result.get("audio_id")
        if result_audio_id not in (None, audio_id):
            raise ServiceProtocolError(
                f"response record audio_id mismatch: {result_audio_id!r} != {audio_id!r}"
            )
        stage = str(value.get("stage") or "").strip()
        model_fingerprint = str(value.get("model_fingerprint") or "").strip()
        elapsed = value.get("elapsed_seconds", value.get("elapsed"))
        if not stage or not model_fingerprint:
            raise ServiceProtocolError(
                "successful response lacks stage/model_fingerprint metadata"
            )
        try:
            elapsed_seconds = float(elapsed)
        except (TypeError, ValueError) as error:
            raise ServiceProtocolError(
                "successful response lacks numeric elapsed_seconds"
            ) from error
        if elapsed_seconds < 0:
            raise ServiceProtocolError("elapsed_seconds must not be negative")
        envelope = InferEnvelope(
            job_id=job_id,
            request_id=request_id,
            audio_id=audio_id,
            input_fingerprint=input_fingerprint,
            record=dict(result),
            stage=stage,
            model_fingerprint=model_fingerprint,
            elapsed_seconds=elapsed_seconds,
        )
        if status != "ok":
            detail = value.get("error") or value.get("message") or status or "unknown"
            raise ServiceInferenceError(str(detail), envelope)
        return envelope

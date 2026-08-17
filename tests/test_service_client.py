from __future__ import annotations

import json
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from service_client import (  # noqa: E402
    ServiceClient,
    ServiceClientError,
    ServiceProtocolError,
)


class Handler(BaseHTTPRequestHandler):
    requests: list[dict] = []
    mismatch = False
    disconnects = 0
    http_statuses: list[int] = []

    def log_message(self, *_: object) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        assert self.path == "/healthz"
        self._send({"status": "ok"})

    def do_POST(self) -> None:  # noqa: N802
        assert self.path == "/v1/infer"
        size = int(self.headers["Content-Length"])
        value = json.loads(self.rfile.read(size))
        type(self).requests.append(value)
        if type(self).disconnects:
            type(self).disconnects -= 1
            self.close_connection = True
            self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()
            return
        if type(self).http_statuses:
            status = type(self).http_statuses.pop(0)
            self._send({"error": f"HTTP {status}"}, status=status)
            return
        self._send(
            {
                "job_id": value["job_id"],
                "request_id": "wrong" if type(self).mismatch else value["request_id"],
                "audio_id": value["audio_id"],
                "input_fingerprint": value["input_fingerprint"],
                "status": "ok",
                "record": {"audio_id": value["audio_id"], "answer": 42},
                "stage": "fake",
                "model_fingerprint": "model-v1",
                "elapsed_seconds": 0.25,
            }
        )

    def _send(self, value: dict, status: int = 200) -> None:
        data = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def service():
    Handler.requests = []
    Handler.mismatch = False
    Handler.disconnects = 0
    Handler.http_statuses = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_unified_health_and_infer_contract(service: str):
    client = ServiceClient(service, timeout=2)
    assert client.healthz()["status"] == "ok"
    result = client.infer(
        job_id="job",
        request_id="request",
        audio_id="audio",
        audio_path="/audio/a.wav",
        input_fingerprint="fingerprint",
        record={"duration": 3.0},
    )
    assert result.record == {"audio_id": "audio", "answer": 42}
    assert result.stage == "fake"
    assert result.model_fingerprint == "model-v1"
    assert result.elapsed_seconds == 0.25
    assert Handler.requests == [
        {
            "job_id": "job",
            "request_id": "request",
            "audio_id": "audio",
            "audio_path": "/audio/a.wav",
            "input_fingerprint": "fingerprint",
            "record": {"duration": 3.0},
        }
    ]


def test_response_identity_mismatch_fails_closed(service: str):
    Handler.mismatch = True
    with pytest.raises(ServiceProtocolError, match="request_id mismatch"):
        ServiceClient(service, timeout=2).infer(
            job_id="job",
            request_id="request",
            audio_id="audio",
            audio_path="/audio/a.wav",
            input_fingerprint="fingerprint",
            record={},
        )


def test_connection_drop_retries_same_idempotent_request(service: str):
    Handler.disconnects = 1
    result = ServiceClient(service, timeout=2, retries=3).infer(
        job_id="job",
        request_id="stable-request",
        audio_id="audio",
        audio_path="/audio/a.wav",
        input_fingerprint="fingerprint",
        record={"duration": 3.0},
    )
    assert result.record["answer"] == 42
    assert len(Handler.requests) == 2
    assert {value["request_id"] for value in Handler.requests} == {"stable-request"}


@pytest.mark.parametrize("status", [429, 502, 503, 504])
def test_retryable_http_statuses_are_retried(service: str, status: int):
    Handler.http_statuses = [status]
    ServiceClient(service, timeout=2, retries=1).infer(
        job_id="job",
        request_id="stable-request",
        audio_id="audio",
        audio_path="/audio/a.wav",
        input_fingerprint="fingerprint",
        record={},
    )
    assert len(Handler.requests) == 2


def test_conflict_is_not_retried(service: str):
    Handler.http_statuses = [409]
    with pytest.raises(ServiceClientError, match="HTTP 409"):
        ServiceClient(service, timeout=2, retries=3).infer(
            job_id="job",
            request_id="stable-request",
            audio_id="audio",
            audio_path="/audio/a.wav",
            input_fingerprint="fingerprint",
            record={},
        )
    assert len(Handler.requests) == 1

from __future__ import annotations

import asyncio
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
import sys
import threading

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import serve_section_asr  # noqa: E402
import serve_songformer  # noqa: E402
import serve_omni  # noqa: E402


def _request(audio_id: str, record: dict):
    return SimpleNamespace(
        job_id="job",
        request_id=f"request-{audio_id}",
        audio_id=audio_id,
        audio_path=record.get("audio_path", f"/audio/{audio_id}.wav"),
        input_fingerprint=f"fp-{audio_id}",
        record=record,
    )


def test_songformer_merges_structure_without_losing_prior_stage_state():
    request = _request(
        "a",
        {
            "content_type": "song",
            "stage_status": {"gate": "ok", "structure": "old"},
            "stage_errors": {"structure": "old error", "gate": "review"},
            "stage_versions": {"structure": "old", "gate": "v1"},
        },
    )
    structure = [{"label": "verse", "start": 0.0, "end": 10.0}]
    result = serve_songformer._merge_structure_record(request, structure, "sf-v3")

    assert result["audio_id"] == "a"
    assert result["structure_raw"] == structure
    assert result["songformer_result"] == structure
    assert result["stage_status"] == {"gate": "ok", "structure_raw": "ok"}
    assert result["stage_errors"] == {"gate": "review"}
    assert result["stage_versions"] == {"gate": "v1", "structure_raw": "sf-v3"}


def test_songformer_prefetches_next_decode_during_current_gpu_inference():
    runtime = object.__new__(serve_songformer.SongFormerRuntime)
    second_decode_started = threading.Event()
    allow_second_decode = threading.Event()
    inference_order = []

    def decode(request):
        if request.audio_id == "b":
            second_decode_started.set()
            assert allow_second_decode.wait(timeout=1.0)
        return request.audio_id, 24000, "song"

    def infer(request, wav, sample_rate, content_type):
        assert sample_rate == 24000
        assert content_type == "song"
        if request.audio_id == "a":
            assert second_decode_started.wait(timeout=1.0)
            allow_second_decode.set()
        inference_order.append(request.audio_id)
        return {"audio_id": request.audio_id, "decoded": wav}

    runtime._decode_request = decode
    runtime._infer_decoded = infer
    results = runtime.process_batch(
        [_request("a", {}), _request("b", {})]
    )

    assert [result["audio_id"] for result in results] == ["a", "b"]
    assert inference_order == ["a", "b"]


class FakeBatchCardinalityError(RuntimeError):
    pass


class FakeASRHelpers:
    PIPELINE_VERSION = "test-pipeline"
    BatchCardinalityError = FakeBatchCardinalityError

    def __init__(self):
        self.batch_durations = []

    @staticmethod
    def sections_hash(record):
        return "sections-hash"

    @staticmethod
    def section_asr_input_fingerprint(record):
        return "semantic-fingerprint"

    @staticmethod
    def duration_bucket(duration):
        return "short" if duration <= 15 else "long"

    @staticmethod
    def should_run_section_asr(section):
        return True, ""

    @staticmethod
    def set_failure(target, status, error=None):
        target.update(lyrics=None, asr_tokens=[], asr_status=status)
        if error:
            target["asr_error"] = error

    @staticmethod
    def section_batches(buckets, batch_size):
        for name in ("short", "long"):
            values = buckets.get(name, [])
            for offset in range(0, len(values), batch_size):
                yield values[offset : offset + batch_size]

    @staticmethod
    def decode_item(item, padding):
        assert padding == 1.5
        return [0.0], 16000, float(item["section"]["start"])

    def run_batch(self, asr, decoded):
        durations = [
            float(item["section"]["end"]) - float(item["section"]["start"])
            for item, _audio in decoded
        ]
        self.batch_durations.append(durations)
        return [
            SimpleNamespace(text="hello", language="English", time_stamps=[1])
            for _value in decoded
        ]

    @staticmethod
    def aligned_items(result):
        return [{"text": result.text, "start": 0.0, "end": 1.0}]

    @staticmethod
    def crop_aligned_tokens(tokens, decoded_start, core_start, core_end):
        return tokens

    @staticmethod
    def join_tokens(tokens, language):
        return " ".join(token["text"] for token in tokens)

    @staticmethod
    def finalize_asr_records(records, outputs, reused, *, model, forced_aligner):
        output = outputs[str(records[0]["audio_id"])]
        output["stage_status"] = {"section_asr": "ok"}
        output["model_versions"] = {
            "section_asr": model,
            "forced_aligner": forced_aligner,
        }
        return [output]


def test_section_asr_duration_batches_are_homogeneous_and_max_four():
    helpers = FakeASRHelpers()
    args = Namespace(
        section_batch_size=4,
        decode_workers=2,
        padding=1.5,
        model="asr",
        forced_aligner="aligner",
    )
    runtime = serve_section_asr.SectionASRRuntime(helpers, object(), args)
    sections = [
        {
            "section_id": f"a:{index}",
            "start": float(index * 20),
            "end": float(index * 20 + 10),
            "voice_coverage": 1.0,
        }
        for index in range(5)
    ]
    sections.append(
        {
            "section_id": "a:long",
            "start": 200.0,
            "end": 240.0,
            "voice_coverage": 1.0,
        }
    )
    result = runtime.process_batch(
        [
            _request(
                "a",
                {
                    "audio_id": "a",
                    "audio_path": "/audio/a.wav",
                    "duration": 300.0,
                    "content_type": "song",
                    "sections": sections,
                },
            )
        ]
    )[0]

    assert [len(batch) for batch in helpers.batch_durations] == [4, 1, 1]
    assert all(len({"short" if d <= 15 else "long" for d in batch}) == 1 for batch in helpers.batch_durations)
    assert all(section["asr_status"] == "ok" for section in result["sections"])
    assert result["stage_status"] == {"section_asr": "ok"}


def test_section_asr_service_defaults_to_four_requests_and_200ms(monkeypatch):
    captured = {}

    class FakeService:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(serve_section_asr, "DynamicBatchService", FakeService)
    monkeypatch.setattr(serve_section_asr, "create_service_app", lambda service: service)
    monkeypatch.setattr(serve_section_asr, "run_service", lambda app, host, port: None)
    monkeypatch.setattr(sys, "argv", ["serve_section_asr.py"])
    serve_section_asr.main()

    assert captured["max_batch_size"] == 4
    assert captured["max_wait_ms"] == 200


def test_section_asr_returns_structured_error_without_poisoning_batch():
    helpers = FakeASRHelpers()
    runtime = serve_section_asr.SectionASRRuntime(
        helpers,
        object(),
        Namespace(
            section_batch_size=4,
            decode_workers=1,
            padding=1.5,
            model="asr",
            forced_aligner="aligner",
        ),
    )
    invalid = _request(
        "bad",
        {
            "audio_id": "bad",
            "audio_path": "/audio/bad.wav",
            "content_type": "song",
            "sections": [{"section_id": "bad:0", "start": 3.0, "end": 1.0}],
        },
    )
    valid = _request(
        "ok",
        {
            "audio_id": "ok",
            "audio_path": "/audio/ok.wav",
            "content_type": "song",
            "sections": [
                {
                    "section_id": "ok:0",
                    "start": 0.0,
                    "end": 10.0,
                    "voice_coverage": 1.0,
                }
            ],
        },
    )

    bad_result, good_result = runtime.process_batch([invalid, valid])

    assert isinstance(bad_result, serve_section_asr.BatchItemResult)
    assert bad_result.error is not None
    assert bad_result.record["stage_status"] == {"section_asr": "error"}
    assert good_result["audio_id"] == "ok"
    assert good_result["stage_status"] == {"section_asr": "ok"}


def test_omni_proxy_uses_stub_upstream_and_preserves_item_errors(monkeypatch):
    calls = []

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

    async def fake_post_chat(**kwargs):
        calls.append(kwargs)
        audio_url = kwargs["messages"][0]["content"][1]["audio_url"]["url"]
        if "bad.wav" in audio_url:
            raise RuntimeError("stub upstream rejected audio")
        return "A detailed whole-track caption."

    monkeypatch.setattr(serve_omni.aiohttp, "ClientSession", FakeSession)
    monkeypatch.setattr(serve_omni, "post_chat", fake_post_chat)
    runtime = serve_omni.OmniProxyRuntime(
        upstream="http://stub-upstream:10008",
        model="omni-test",
        max_tokens=128,
        temperature=0.1,
        timeout=10,
        retries=0,
        retry_base_sleep=0.0,
    )
    good, bad = asyncio.run(
        serve_omni.process_omni_batch(
            runtime,
            [
                _request("good", {"content_type": "song"}),
                _request(
                    "bad",
                    {"content_type": "song", "audio_path": "/audio/bad.wav"},
                ),
            ],
        )
    )

    assert len(calls) == 2
    assert all(call["base_url"] == "http://stub-upstream:10008" for call in calls)
    assert good["ALM_Caption"] == "A detailed whole-track caption."
    assert good["stage_status"]["alm"] == "ok"
    assert isinstance(bad, serve_omni.BatchItemResult)
    assert bad.record["stage_status"]["alm"] == "error"


def test_omni_proxy_health_requires_upstream_readiness(monkeypatch):
    args = serve_omni.build_parser().parse_args(
        ["--upstream", "http://stub-upstream:10008"]
    )
    service = serve_omni.build_service(args)
    app = serve_omni.create_omni_app(service, args.upstream)
    readiness = {"healthy": False}
    monkeypatch.setattr(
        serve_omni,
        "_probe_upstream",
        lambda upstream: (readiness["healthy"], "stub readiness"),
    )

    with TestClient(app) as client:
        unavailable = client.get("/healthz")
        assert unavailable.status_code == 503
        assert unavailable.json()["detail"]["status"] == "upstream_unavailable"

        readiness["healthy"] = True
        ready = client.get("/healthz")
        assert ready.status_code == 200
        assert ready.json()["upstream_healthy"] is True

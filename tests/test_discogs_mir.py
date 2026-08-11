from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "MusicToolsPipeline"))
sys.path.insert(0, str(ROOT / "scripts"))

import discogs_mir_infer
from discogs_mir_infer import (
    HEAD_NAMES,
    PreparedAudio,
    bounded_prefetch_map,
    build_failure_record,
    build_success_record,
    concatenate_frame_slices,
    discogs_model_version,
    fused_discogs_vocal_score,
    infer_prepared_batch,
    load_resume_state,
    plan_frame_batches,
    prepare_audio,
    record_fingerprint,
    route_discogs_voice,
    scan_pending_inputs,
    stage_version,
    validate_accepted_music,
)
from sub_models.discogs_onnx_model import (
    DiscogsModelPaths,
    bind_cuda_tensor,
    cuda_tensor_binding_spec,
)


def _accepted(audio_id: str = "a") -> dict:
    return {
        "audio_id": audio_id,
        "audio_path": f"/{audio_id}.wav",
        "duration": 12.0,
        "decode_status": "ok",
        "status": "accepted",
        "content_type": "music",
        "music_gate": {"score": 0.91, "coverage": 0.8},
        "stage_versions": {"fast_music_gate": "fast-v"},
        "model_versions": {"fast_music_gate": "fast-v"},
        "stage_status": {"fast_music_gate": "ok"},
        "stage_errors": {},
    }


def test_discogs_stage_version_ignores_runtime_batching():
    base = argparse.Namespace(
        vocal_song=0.7,
        vocal_instrumental=0.3,
        frame_batch_size=256,
        buffered_frames=1024,
    )
    first = stage_version(base, model_version="model-v")
    base.frame_batch_size = 512
    second = stage_version(base, model_version="model-v")

    assert first == second


def test_discogs_semantic_aliases_are_exact():
    assert (
        discogs_mir_infer._model_version_for_digest(
            discogs_mir_infer.CURRENT_MODEL_SEMANTIC_DIGEST
        )
        == discogs_mir_infer.CURRENT_COMPATIBLE_MODEL_VERSION
    )
    assert discogs_mir_infer._model_version_for_digest("f" * 64) == "f" * 16


def _analysis(voice_mean: float, coverage: float, longest: float) -> dict:
    return {
        "voice_analysis": {
            "voice_mean": voice_mean,
            "voice_coverage": coverage,
            "longest_voice_sec": longest,
        },
        "genre": [{"label": "rock", "probability": 0.8}],
        "mood_theme": [{"label": "energetic", "probability": 0.7}],
        "danceability": [{"label": "danceable", "probability": 0.6}],
        "instruments": [{"label": "guitar", "probability": 0.9}],
        "instrument_changes": [{"time": 0.0, "active": ["guitar"]}],
        "discogs_frame_count": 7,
        "discogs_provider": "CUDAExecutionProvider",
    }


def test_discogs_model_version_hashes_each_json_sidecar_presence_and_content(
    tmp_path: Path,
):
    paths = DiscogsModelPaths.from_root(str(tmp_path))
    for index, value in enumerate(paths.values()):
        Path(value).write_bytes(f"onnx-{index}".encode())
    args = argparse.Namespace(discogs_root=str(tmp_path))

    current_version = discogs_model_version(args)
    for index, value in enumerate(paths.values()):
        sidecar = Path(value).with_suffix(".json")
        sidecar.write_text(json.dumps({"classes": [str(index)]}), encoding="utf-8")
        next_version = discogs_model_version(args)
        assert next_version != current_version
        current_version = next_version
    first_sidecar = Path(paths.backbone).with_suffix(".json")
    first_sidecar.write_text('{"classes": ["b"]}', encoding="utf-8")
    changed_sidecar = discogs_model_version(args)

    assert current_version != changed_sidecar


def test_frame_planner_is_bounded_and_covers_each_frame_exactly_once():
    counts = [3, 0, 9, 2]
    batches = plan_frame_batches(counts, max_frames=4)

    assert all(sum(part.frame_count for part in batch) <= 4 for batch in batches)
    covered = {index: [] for index in range(len(counts))}
    for batch in batches:
        for part in batch:
            covered[part.audio_index].extend(range(part.start, part.stop))
    assert covered == {
        0: [0, 1, 2],
        1: [],
        2: list(range(9)),
        3: [0, 1],
    }


class _FakeDevice:
    def __init__(self, index: int):
        self.index = index

    def __str__(self):
        return f"cuda:{self.index}"


class _FakeCudaTensor:
    is_cuda = True
    dtype = torch.float32
    shape = (4, 1280)
    device = _FakeDevice(2)

    @staticmethod
    def is_contiguous():
        return True

    @staticmethod
    def data_ptr():
        return 123456


class _FakeIoBinding:
    def __init__(self):
        self.inputs = []
        self.outputs = []

    def bind_input(self, **kwargs):
        self.inputs.append(kwargs)

    def bind_output(self, **kwargs):
        self.outputs.append(kwargs)


def test_cuda_binding_uses_exact_contiguous_torch_allocation():
    tensor = _FakeCudaTensor()
    spec = cuda_tensor_binding_spec(tensor, device_id=2)
    assert spec == {
        "device_type": "cuda",
        "device_id": 2,
        "element_type": np.float32,
        "shape": (4, 1280),
        "buffer_ptr": 123456,
    }

    binding = _FakeIoBinding()
    bind_cuda_tensor(binding, "input", tensor, device_id=2, output=False)
    bind_cuda_tensor(binding, "embeddings", tensor, device_id=2, output=True)
    assert binding.inputs[0]["name"] == "input"
    assert binding.outputs[0]["name"] == "embeddings"
    with pytest.raises(ValueError, match="device mismatch"):
        cuda_tensor_binding_spec(tensor, device_id=0)
    with pytest.raises(ValueError, match="requires a CUDA tensor"):
        cuda_tensor_binding_spec(torch.zeros(1), device_id=0)


class _FakeSession:
    def __init__(self, name: str):
        self.name = name

    def get_providers(self):
        return ["FakeCUDAProvider"]


class _FakeEngine:
    def __init__(self):
        self.sessions = {name: _FakeSession(name) for name in ("backbone", *HEAD_NAMES)}
        self.backbone_frames = []
        self.head_embeddings = {name: [] for name in HEAD_NAMES}

    def _run_session(self, session, values):
        identities = np.asarray(values)[:, 0].reshape(-1)
        if session.name == "backbone":
            # Input patches are [N, 1, 1]; store their unique identity.
            identities = np.asarray(values)[:, 0, 0]
            self.backbone_frames.extend(identities.tolist())
            return [np.repeat(identities[:, None], 1280, axis=1).astype(np.float32)]
        self.head_embeddings[session.name].extend(identities.tolist())
        return [np.stack([1.0 - identities / 100.0, identities / 100.0], axis=1)]

    @staticmethod
    def _select_embeddings(outputs):
        return outputs[0]

    @staticmethod
    def _voice_summary(predictions):
        return {
            "voice_mean": float(predictions[:, 1].mean()),
            "voice_coverage": float((predictions[:, 1] >= 0.5).mean()),
            "longest_voice_sec": 0.0,
        }

    @staticmethod
    def _top_labels(name, predictions):
        return [{"label": name, "probability": float(predictions[:, 1].mean())}]

    @staticmethod
    def _instrument_changes(predictions, duration):
        return [{"time": 0.0, "active": [], "duration": duration}]


class _FakeFusedEngine(_FakeEngine):
    def __init__(self):
        super().__init__()
        self.fused_batches = []

    def infer_patch_batch(self, values):
        assert isinstance(values, torch.Tensor)
        identities = values[:, 0, 0].to(dtype=torch.float32)
        self.fused_batches.append(identities.tolist())
        self.backbone_frames.extend(identities.tolist())
        embeddings = identities[:, None].repeat(1, 1280).contiguous()
        predictions = {}
        for name in HEAD_NAMES:
            consumed = embeddings[:, 0].numpy()
            self.head_embeddings[name].extend(consumed.tolist())
            predictions[name] = np.stack(
                [1.0 - consumed / 100.0, consumed / 100.0], axis=1
            ).astype(np.float32)
        return embeddings, predictions


def test_cross_audio_inference_reuses_one_backbone_embedding_for_all_heads():
    engine = _FakeEngine()
    first_ids = np.asarray([1, 2, 3], dtype=np.float32)
    second_ids = np.asarray([10, 11, 12, 13, 14], dtype=np.float32)
    prepared = [
        PreparedAudio(_accepted("a"), first_ids[:, None, None], 3.0),
        PreparedAudio(_accepted("b"), second_ids[:, None, None], 5.0),
    ]

    result = infer_prepared_batch(engine, prepared, max_frames=4)

    expected = [1, 2, 3, 10, 11, 12, 13, 14]
    assert engine.backbone_frames == expected
    for name in HEAD_NAMES:
        assert engine.head_embeddings[name] == expected
    assert [item["discogs_frame_count"] for item in result] == [3, 5]
    assert all(item["discogs_provider"] == "FakeCUDAProvider" for item in result)


def test_tensor_patch_batches_stay_tensor_backed_through_fused_engine_path():
    engine = _FakeFusedEngine()
    prepared = [
        PreparedAudio(_accepted("a"), torch.tensor([1, 2, 3])[:, None, None], 3.0),
        PreparedAudio(_accepted("b"), torch.tensor([10, 11, 12])[:, None, None], 3.0),
    ]

    planned = plan_frame_batches([3, 3], max_frames=4)
    packed = concatenate_frame_slices(prepared, planned[0])
    assert isinstance(packed, torch.Tensor)
    assert packed.is_contiguous()
    assert packed.tolist() == [[[1.0]], [[2.0]], [[3.0]], [[10.0]]]

    result = infer_prepared_batch(engine, prepared, max_frames=4)
    assert engine.fused_batches == [[1.0, 2.0, 3.0, 10.0], [11.0, 12.0]]
    assert engine.backbone_frames == [1.0, 2.0, 3.0, 10.0, 11.0, 12.0]
    for name in HEAD_NAMES:
        assert engine.head_embeddings[name] == engine.backbone_frames
    assert [item["discogs_frame_count"] for item in result] == [3, 3]


def test_prepare_audio_keeps_tensor_cache_and_enforces_frame_bound():
    class _Frontend:
        def __call__(self, waveform):
            return torch.arange(6, dtype=torch.float32).reshape(3, 1, 2)

    class _Engine:
        frontend = _Frontend()
        require_cuda = False

    item = prepare_audio(
        _Engine(), _accepted(), torch.zeros(16000), max_cached_frames=3
    )
    assert isinstance(item.patches, torch.Tensor)
    assert item.patches.is_contiguous()
    with pytest.raises(RuntimeError, match="exceeds_buffered_frames"):
        prepare_audio(_Engine(), _accepted(), torch.zeros(16000), max_cached_frames=2)


def test_discogs_routing_uses_no_music_gate_singing_signal():
    assert fused_discogs_vocal_score(1.0, 1.0, 30.0) == 1.0
    assert fused_discogs_vocal_score(0.3333333, 0.3333333, 9.999999) == 0.333333
    assert route_discogs_voice(
        {"voice_mean": 0.9, "voice_coverage": 0.9, "longest_voice_sec": 30.0},
        song_threshold=0.55,
        instrumental_threshold=0.20,
    )[:2] == ("accepted", "song")
    assert route_discogs_voice(
        {"voice_mean": 0.0, "voice_coverage": 0.0, "longest_voice_sec": 0.0},
        song_threshold=0.55,
        instrumental_threshold=0.20,
    )[:2] == ("accepted", "instrumental")
    assert route_discogs_voice(
        {"voice_mean": 0.5, "voice_coverage": 0.3, "longest_voice_sec": 0.0},
        song_threshold=0.55,
        instrumental_threshold=0.20,
    )[:2] == ("review", "unknown")


def test_success_preserves_gate_metadata_and_merges_stage_metadata():
    source = _accepted()
    original_gate = dict(source["music_gate"])
    record = build_success_record(
        source,
        _analysis(0.9, 0.9, 30.0),
        current_version="discogs-v",
        song_threshold=0.55,
        instrumental_threshold=0.20,
    )

    assert record["music_gate"] == original_gate
    assert record["status"] == "accepted"
    assert record["content_type"] == "song"
    assert record["stage_versions"] == {
        "fast_music_gate": "fast-v",
        "discogs_mir": "discogs-v",
    }
    assert record["model_versions"]["discogs_mir"] == "discogs-v"
    assert record["stage_status"]["discogs_mir"] == "ok"
    assert record["global_mir"]["genre"][0]["label"] == "rock"

    quantized = build_success_record(
        source,
        _analysis(0.3333333, 0.3333333, 9.999999),
        current_version="discogs-v",
        song_threshold=0.55,
        instrumental_threshold=0.20,
    )
    assert quantized["discogs_vocal_score"] == 0.333333


def test_review_contains_only_vocal_gray_successes_and_failures_are_separate():
    source = _accepted()
    review = build_success_record(
        source,
        _analysis(0.5, 0.3, 0.0),
        current_version="discogs-v",
        song_threshold=0.55,
        instrumental_threshold=0.20,
    )
    failure = build_failure_record(
        source,
        RuntimeError("onnx failed"),
        current_version="discogs-v",
    )

    assert review["status"] == "review"
    assert "vocal_route_gray" in review["reason_codes"]
    assert review["stage_status"]["discogs_mir"] == "ok"
    assert failure["stage_status"]["discogs_mir"] == "error"
    assert failure["reason_codes"][-1] == "discogs_mir_error"
    assert "vocal_route_gray" not in failure["reason_codes"]


def test_resume_keeps_success_and_clears_retryable_failure(tmp_path: Path):
    source_a = _accepted("a")
    success = {
        "audio_id": "a",
        "stage_input_fingerprint": record_fingerprint(source_a),
        "stage_versions": {"discogs_mir": "older-discogs-v"},
    }
    failure = build_failure_record(
        _accepted("b"),
        RuntimeError("retry me"),
        current_version="discogs-v",
    )
    (tmp_path / "data.song.jsonl").write_text(
        json.dumps(success) + "\n", encoding="utf-8"
    )
    (tmp_path / "failures.jsonl").write_text(
        json.dumps(failure) + "\n", encoding="utf-8"
    )

    assert load_resume_state(tmp_path) == {"a": record_fingerprint(source_a)}
    assert (tmp_path / "failures.jsonl").read_text(encoding="utf-8") == ""
    assert json.loads((tmp_path / "data.song.jsonl").read_text()) == success


def test_pending_scan_rejects_nonaccepted_or_changed_inputs(tmp_path: Path):
    source = _accepted()
    input_path = tmp_path / "accepted.music.jsonl"
    input_path.write_text(json.dumps(source) + "\n", encoding="utf-8")
    assert scan_pending_inputs(input_path, {}) == (1, 1)
    assert scan_pending_inputs(input_path, {"a": record_fingerprint(source)}) == (0, 1)
    with pytest.raises(RuntimeError, match="input metadata changed"):
        scan_pending_inputs(input_path, {"a": "stale"})

    rejected = dict(source, status="review")
    with pytest.raises(ValueError, match="accepted music only"):
        validate_accepted_music(rejected)


def test_bounded_prefetch_yields_ready_decodes_and_captures_item_failures():
    def convert(value: int) -> int:
        if value == 0:
            time.sleep(0.05)
        if value == 2:
            raise RuntimeError("bad item")
        return value * 10

    result = list(bounded_prefetch_map(range(5), convert, max_workers=2, prefetch=3))

    assert result[0].item != 0
    by_item = {item.item: item for item in result}
    assert set(by_item) == set(range(5))
    assert by_item[0].value == 0
    assert by_item[1].value == 10
    assert isinstance(by_item[2].error, RuntimeError)
    assert by_item[3].value == 30
    assert by_item[4].value == 40

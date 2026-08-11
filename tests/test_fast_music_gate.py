from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import fast_music_gate
import fast_gate_core
from fast_gate_core import (
    AudioSetMusicScorer,
    CascadeMusicGate,
    CascadeResult,
    DecisionThresholds,
    STAGE_A_FRACTIONS,
    STAGE_B_FRACTIONS,
    TrackWindowSession,
    build_backend,
    deterministic_offsets,
    load_production_gate_config,
    verify_checkpoint_sha256,
)


class _RecordingBackend:
    name = "fake_embeddings"
    sample_rate = 10

    def __init__(self):
        self.batch_shapes = []

    def embed(self, waveforms: np.ndarray) -> np.ndarray:
        self.batch_shapes.append(waveforms.shape)
        return waveforms.mean(axis=1, keepdims=True)

    def tag_probabilities(self, waveforms: np.ndarray) -> np.ndarray:
        self.batch_shapes.append(waveforms.shape)
        logits = waveforms.mean(axis=1)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        output = np.zeros((len(waveforms), 527), dtype=np.float32)
        output[:, AudioSetMusicScorer.MUSIC_ROOT_INDEX] = probabilities
        return output


def _constant_decoder(
    audio_path: str, start: float, duration: float, sample_rate: int
) -> np.ndarray:
    value = {"music.wav": 10.0, "gray.wav": 0.0, "noise.wav": -10.0}[audio_path]
    return np.full(int(round(duration * sample_rate)), value, dtype=np.float32)


def test_stage_version_preserves_only_the_audited_semantic_digest():
    assert (
        fast_music_gate._stage_version_for_digest(
            fast_music_gate.CURRENT_SEMANTIC_DIGEST
        )
        == fast_music_gate.CURRENT_COMPATIBLE_STAGE_VERSION
    )
    assert fast_music_gate._stage_version_for_digest("f" * 64) == "f" * 16


def test_stage_fingerprint_excludes_operational_source_hashes(tmp_path: Path):
    weight = tmp_path / "model.pt"
    config = tmp_path / "config.json"
    weight.write_bytes(b"weights")
    config.write_text("{}", encoding="utf-8")
    args = types.SimpleNamespace(
        backend_weights=str(weight),
        stage_b_backend_weights=str(weight),
        config=str(config),
        backend="panns_mobilenet",
        backend_repo_id="repo",
        backend_checkpoint_sha256="a" * 64,
        stage_b_backend="panns_mobilenet",
        stage_b_backend_repo_id="repo",
        stage_b_backend_checkpoint_sha256="a" * 64,
        precision="fp32",
        stage_b_precision="fp32",
        batch_size=64,
        stage_b_batch_size=64,
        backend_source_sha256="b" * 64,
        stage_b_backend_source_sha256="b" * 64,
        config_version="v1",
        stage_a_reject=0.05,
        stage_a_accept=0.35,
        stage_b_reject=0.1,
        stage_b_accept=0.3,
    )

    payload = fast_music_gate.build_stage_fingerprint_payload(args)

    assert payload["schema"] == "fast-music-gate-semantic-v2"
    assert payload["decision_contract"] == "sparse-audioset-cascade-v1"
    assert "code" not in payload


def test_deterministic_stage_offsets_are_relative_to_valid_start_range():
    assert deterministic_offsets(100.0, STAGE_A_FRACTIONS) == [9.2, 46.0, 82.8]
    assert deterministic_offsets(100.0, STAGE_B_FRACTIONS) == [
        4.6,
        23.0,
        46.0,
        69.0,
        87.4,
    ]
    assert deterministic_offsets(8.0, STAGE_A_FRACTIONS) == [0.0, 0.0, 0.0]


def test_ffmpeg_range_decode_falls_back_to_accurate_seek_when_fast_seek_is_empty(
    monkeypatch: pytest.MonkeyPatch,
):
    expected = np.asarray([0.25, -0.5], dtype="<f4")
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        payload = b"" if len(commands) == 1 else expected.tobytes()
        return types.SimpleNamespace(returncode=0, stdout=payload, stderr=b"")

    monkeypatch.setattr(fast_gate_core.subprocess, "run", fake_run)

    values = fast_gate_core.ffmpeg_decode_range("no-seek-table.flac", 12.5, 8.0, 32000)

    assert values.tolist() == pytest.approx(expected.tolist())
    assert commands[0].index("-ss") < commands[0].index("-i")
    assert commands[1].index("-ss") > commands[1].index("-i")


def test_short_audio_is_zero_padded_and_decoded_once():
    calls = []

    def decoder(path, start, duration, sample_rate):
        calls.append((path, start, duration, sample_rate))
        return np.ones(int(round(duration * sample_rate)), dtype=np.float32)

    session = TrackWindowSession("short.wav", 3.0, 10, decoder=decoder)
    windows = [session.window(offset) for offset in session.offsets(STAGE_A_FRACTIONS)]

    assert len(calls) == 1
    assert calls[0][1:3] == (0.0, 3.0)
    assert all(window.shape == (80,) for window in windows)
    assert np.all(windows[0][:30] == 1.0)
    assert np.all(windows[0][30:] == 0.0)
    assert windows[0] is windows[1] is windows[2]


def test_tracks_up_to_40_seconds_use_one_full_decode_across_both_stages():
    calls = []

    def decoder(path, start, duration, sample_rate):
        calls.append((start, duration))
        return np.arange(int(round(duration * sample_rate)), dtype=np.float32)

    session = TrackWindowSession("medium.wav", 20.0, 10, decoder=decoder)
    stage_a = [session.window(offset) for offset in session.offsets(STAGE_A_FRACTIONS)]
    stage_b = [session.window(offset) for offset in session.offsets(STAGE_B_FRACTIONS)]

    assert calls == [(0.0, 20.0)]
    assert (
        stage_a[1] is stage_b[2]
    )  # shared 50% window is neither sliced nor inferred twice
    assert stage_a[0][0] == 12.0
    assert stage_b[-1][0] == 114.0


def test_cascade_batches_across_tracks_and_only_expands_gray_stage():
    backend = _RecordingBackend()
    head = AudioSetMusicScorer()
    gate = CascadeMusicGate(
        backend=backend,
        head=head,
        stage_a_thresholds=DecisionThresholds(0.2, 0.8),
        stage_b_thresholds=DecisionThresholds(0.4, 0.6),
        batch_size=32,
        decode_workers=3,
        decoder=_constant_decoder,
    )

    results = gate.classify_records(
        [
            {"audio_path": "music.wav", "duration": 100.0},
            {"audio_path": "gray.wav", "duration": 100.0},
            {"audio_path": "noise.wav", "duration": 100.0},
        ]
    )

    assert [result.decision for result in results] == ["accepted", "review", "rejected"]
    assert backend.batch_shapes == [(9, 80), (4, 80)]
    assert results[0].stage_probabilities["stage_b"] == []
    assert results[1].stage_probabilities["stage_b"] == [0.5] * 5
    assert results[1].offsets["stage_b"] == [4.6, 23.0, 46.0, 69.0, 87.4]
    assert results[1].scoring_version == AudioSetMusicScorer.VERSION


def test_cascade_scores_mean_frozen_embedding_not_mean_window_probability():
    backend = _RecordingBackend()
    head = AudioSetMusicScorer()

    def decoder(path, start, duration, sample_rate):
        value = 10.0 if start < 10.0 else -10.0
        return np.full(int(round(duration * sample_rate)), value, dtype=np.float32)

    gate = CascadeMusicGate(
        backend=backend,
        head=head,
        stage_a_thresholds=DecisionThresholds(0.2, 0.8),
        decoder=decoder,
    )
    result = gate.classify_records(
        [
            {"audio_path": "mixed.wav", "duration": 100.0},
        ]
    )[0]

    # Median native window probability is robust to one musical outlier.
    assert result.decision == "rejected"
    assert result.stage_scores["stage_a"] == pytest.approx(0.000045, abs=1e-6)
    assert result.stage_probabilities["stage_b"] == []
    assert backend.batch_shapes == [(3, 80)]


def test_cascade_supports_different_stage_backends_and_sample_rates():
    class StageBackend(_RecordingBackend):
        def __init__(self, name, sample_rate):
            super().__init__()
            self.name = name
            self.sample_rate = sample_rate

    stage_a = StageBackend("stage_a_10_hz", 10)
    stage_b = StageBackend("stage_b_32k", 20)
    decode_rates = []

    def decoder(path, start, duration, sample_rate):
        decode_rates.append(sample_rate)
        value = 0.0 if sample_rate == 10 else 10.0
        return np.full(int(round(duration * sample_rate)), value, dtype=np.float32)

    head = AudioSetMusicScorer()
    result = CascadeMusicGate(
        backend=stage_a,
        head=head,
        stage_b_backend=stage_b,
        stage_b_head=head,
        decoder=decoder,
    ).classify_records([{"audio_path": "dual.wav", "duration": 100.0}])[0]

    assert result.decision == "accepted"
    assert result.stage_backends == {
        "stage_a": "stage_a_10_hz",
        "stage_b": "stage_b_32k",
    }
    assert result.stage_sample_rates == {"stage_a": 10, "stage_b": 20}
    assert stage_a.batch_shapes == [(3, 80)]
    assert stage_b.batch_shapes == [(5, 160)]
    assert set(decode_rates) == {10, 20}


def test_dual_rate_short_track_decodes_once_at_max_rate_and_resamples_in_memory():
    class StageBackend(_RecordingBackend):
        def __init__(self, name, sample_rate):
            super().__init__()
            self.name = name
            self.sample_rate = sample_rate

    stage_a = StageBackend("stage_a_16k", 16000)
    stage_b = StageBackend("stage_b_32k", 32000)
    decode_calls = []

    def decoder(path, start, duration, sample_rate):
        decode_calls.append((path, start, duration, sample_rate))
        return np.zeros(int(round(duration * sample_rate)), dtype=np.float32)

    head = AudioSetMusicScorer()
    result = CascadeMusicGate(
        backend=stage_a,
        head=head,
        stage_b_backend=stage_b,
        stage_b_head=head,
        decoder=decoder,
        batch_size=8,
    ).classify_records([{"audio_path": "short-dual.wav", "duration": 20.0}])[0]

    assert result.decision == "review"
    assert decode_calls == [("short-dual.wav", 0.0, 20.0, 32000)]
    assert stage_a.batch_shapes == [(3, 8 * 16000)]
    assert stage_b.batch_shapes == [(5, 8 * 32000)]


def test_direct_audioset_scorer_uses_music_labels_without_fitted_parameters():
    scorer = AudioSetMusicScorer()
    tags = np.zeros((4, 527), dtype=np.float32)
    tags[0, 137] = 0.8  # Music root
    tags[1, 27] = 0.7  # Singing
    tags[2, 266] = 0.9  # Song
    tags[3, 140] = 0.95  # Isolated Guitar must not be sufficient by itself

    assert scorer.predict_proba(tags).tolist() == pytest.approx([0.8, 0.7, 0.9, 0.0])
    assert scorer.aggregate([0.9, 0.1, 0.8]) == pytest.approx(0.8)
    with pytest.raises(ValueError, match=r"\[batch, 527\]"):
        scorer.predict_proba(np.zeros((1, 526), dtype=np.float32))


def test_backend_without_weights_has_clear_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="panns_mobilenet weights were not found"):
        build_backend("panns_mobilenet", str(tmp_path / "missing.pt"), device="cpu")


def test_panns_backend_drops_only_foreign_top_level_module_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    panns_root = tmp_path / "panns"
    panns_root.mkdir()
    foreign = types.ModuleType("pytorch")
    foreign.__file__ = str(tmp_path / "foreign" / "pytorch.py")
    child = types.ModuleType("pytorch.models")
    monkeypatch.setitem(sys.modules, "pytorch", foreign)
    monkeypatch.setitem(sys.modules, "pytorch.models", child)

    fast_gate_core._drop_foreign_module_tree("pytorch", panns_root)

    assert "pytorch" not in sys.modules
    assert "pytorch.models" not in sys.modules

    expected = types.ModuleType("pytorch")
    expected.__file__ = str(panns_root / "pytorch" / "__init__.py")
    expected.__path__ = [str((panns_root / "pytorch").resolve())]
    monkeypatch.setitem(sys.modules, "pytorch", expected)
    fast_gate_core._drop_foreign_module_tree("pytorch", panns_root)
    assert sys.modules["pytorch"] is expected


def test_panns_upstream_top_level_pytorch_utils_import(tmp_path: Path, monkeypatch):
    repo = tmp_path / "panns"
    package = repo / "pytorch"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "pytorch_utils.py").write_text("SENTINEL = 7\n", encoding="utf-8")
    (package / "models.py").write_text(
        "from pytorch_utils import SENTINEL\n", encoding="utf-8"
    )
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"weights")
    for name in ("pytorch", "pytorch.models", "pytorch_utils"):
        monkeypatch.delitem(sys.modules, name, raising=False)

    try:
        module = fast_gate_core._import_panns_models(weights, str(repo))
        assert module.SENTINEL == 7
    finally:
        for path in (str(package), str(repo)):
            while path in sys.path:
                sys.path.remove(path)


def _result(decision: str, probability: float) -> CascadeResult:
    return CascadeResult(
        backend="fake",
        scoring_version=AudioSetMusicScorer.VERSION,
        stage_probabilities={"stage_a": [probability] * 3, "stage_b": []},
        stage_scores={"stage_a": probability, "stage_b": None},
        offsets={"stage_a": [1.0, 5.0, 9.0], "stage_b": []},
        decision=decision,
        probability=probability,
        window_seconds=8.0,
        sample_rate=32000,
    )


class _FakeHead:
    scoring_version = "score-v1"


class _FakeBackend:
    name = "fake"


class _FakeGate:
    last_kwargs = None
    call_sizes = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs
        type(self).call_sizes = []

    def classify_records(self, records):
        type(self).call_sizes.append(len(records))
        decisions = {
            "music": _result("accepted", 0.95),
            "gray": _result("review", 0.50),
            "noise": _result("rejected", 0.05),
        }
        return [decisions[record["audio_id"]] for record in records]


def _patch_cli_runtime(monkeypatch):
    artifact = {
        "config_version": "config-v1",
        "scoring": {"version": AudioSetMusicScorer.VERSION},
    }
    monkeypatch.setattr(
        fast_music_gate, "load_production_gate_config", lambda path: artifact
    )

    def resolve(args, payload):
        args.backend = "panns_mobilenet"
        args.stage_b_backend = "panns_mobilenet"
        args.backend_weights = "unused.pt"
        args.stage_b_backend_weights = "unused.pt"
        args.backend_repo = None
        args.stage_b_backend_repo = None
        args.backend_repo_id = "qiuqiangkong/audioset_tagging_cnn"
        args.stage_b_backend_repo_id = "qiuqiangkong/audioset_tagging_cnn"
        args.precision = "fp32"
        args.stage_b_precision = "fp32"
        args.batch_size = 8
        args.stage_b_batch_size = 8
        args.backend_source_sha256 = "1" * 64
        args.stage_b_backend_source_sha256 = "1" * 64
        args.config_version = "config-v1"
        args.stage_a_reject = 0.2
        args.stage_a_accept = 0.8
        args.stage_b_reject = 0.4
        args.stage_b_accept = 0.6

    monkeypatch.setattr(fast_music_gate, "resolve_production_config", resolve)


def test_cli_writes_canonical_outputs_metadata_and_resume(tmp_path: Path, monkeypatch):
    inventory = tmp_path / "inventory.jsonl"
    records = [
        {"audio_id": "music", "audio_path": "/music.wav", "duration": 60.0},
        {"audio_id": "gray", "audio_path": "/gray.wav", "duration": 60.0},
        {"audio_id": "noise", "audio_path": "/noise.wav", "duration": 60.0},
    ]
    inventory.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    output_dir = tmp_path / "out"
    _patch_cli_runtime(monkeypatch)
    monkeypatch.setattr(fast_music_gate, "build_stage_version", lambda args: "stage-v1")
    backend_calls = []

    def build_fake_backend(*args, **kwargs):
        backend_calls.append((args, kwargs))
        return _FakeBackend()

    monkeypatch.setattr(fast_music_gate, "build_backend", build_fake_backend)
    monkeypatch.setattr(fast_music_gate, "CascadeMusicGate", _FakeGate)
    arguments = [
        "--input",
        str(inventory),
        "--output-dir",
        str(output_dir),
        "--backend-weights",
        "unused.pt",
        "--config",
        "unused.json",
        "--device",
        "cpu",
        "--decode-workers",
        "2",
        "--batch-size",
        "8",
    ]

    fast_music_gate.main(arguments)

    assert len(backend_calls) == 1
    assert backend_calls[0][1]["precision"] == "fp32"
    assert _FakeGate.last_kwargs["batch_size"] == 8
    assert _FakeGate.last_kwargs["stage_b_batch_size"] == 8
    assert _FakeGate.call_sizes == [3]

    accepted = json.loads(
        (output_dir / "accepted.music.jsonl").read_text(encoding="utf-8")
    )
    review = json.loads((output_dir / "review.jsonl").read_text(encoding="utf-8"))
    rejected = json.loads((output_dir / "rejected.jsonl").read_text(encoding="utf-8"))
    assert (output_dir / "failures.jsonl").read_text(encoding="utf-8") == ""
    assert accepted["music_gate"]["decision"] == "music"
    assert accepted["stage_status"]["music_gate"] == "ok"
    assert accepted["stage_versions"]["music_gate"] == "stage-v1"
    assert accepted["model_versions"]["music_gate"] == (
        f"fake->fake:{AudioSetMusicScorer.VERSION}"
    )
    assert review["music_gate"]["decision"] == "review"
    assert rejected["music_gate"]["decision"] == "non_music"

    def unexpected_load(*args, **kwargs):
        raise AssertionError("completed resume must not load a backend")

    monkeypatch.setattr(fast_music_gate, "build_backend", unexpected_load)
    monkeypatch.setattr(fast_music_gate, "build_stage_version", lambda args: "stage-v2")
    fast_music_gate.main(arguments + ["--resume"])
    assert len((output_dir / "accepted.music.jsonl").read_text().splitlines()) == 1
    assert (
        json.loads((output_dir / "accepted.music.jsonl").read_text(encoding="utf-8"))[
            "stage_versions"
        ]["music_gate"]
        == "stage-v1"
    )


def test_cli_isolates_asset_failures(tmp_path: Path, monkeypatch):
    inventory = tmp_path / "inventory.jsonl"
    inventory.write_text(
        json.dumps(
            {
                "audio_id": "bad",
                "audio_path": "/bad.wav",
                "duration": 12.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    _patch_cli_runtime(monkeypatch)

    class BrokenGate(_FakeGate):
        def classify_records(self, records):
            raise RuntimeError("decode exploded")

    monkeypatch.setattr(fast_music_gate, "build_stage_version", lambda args: "stage-v1")
    monkeypatch.setattr(
        fast_music_gate, "build_backend", lambda *args, **kwargs: _FakeBackend()
    )
    monkeypatch.setattr(fast_music_gate, "CascadeMusicGate", BrokenGate)

    fast_music_gate.main(
        [
            "--input",
            str(inventory),
            "--output-dir",
            str(output_dir),
            "--backend-weights",
            "unused.pt",
            "--config",
            "unused.json",
        ]
    )

    failure = json.loads((output_dir / "failures.jsonl").read_text(encoding="utf-8"))
    assert failure["music_gate"]["decision"] == "error"
    assert failure["stage_status"]["music_gate"] == "error"
    assert "decode exploded" in failure["stage_errors"]["music_gate"]

    class RecoveredGate(_FakeGate):
        def classify_records(self, records):
            return [_result("accepted", 0.95) for _ in records]

    monkeypatch.setattr(fast_music_gate, "CascadeMusicGate", RecoveredGate)
    fast_music_gate.main(
        [
            "--input",
            str(inventory),
            "--output-dir",
            str(output_dir),
            "--backend-weights",
            "unused.pt",
            "--config",
            "unused.json",
            "--resume",
        ]
    )

    assert (output_dir / "failures.jsonl").read_text(encoding="utf-8") == ""
    accepted = json.loads(
        (output_dir / "accepted.music.jsonl").read_text(encoding="utf-8")
    )
    assert accepted["audio_id"] == "bad"
    assert accepted["stage_status"]["music_gate"] == "ok"


def test_cli_rejects_inventory_decode_failures_without_running_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    inventory = tmp_path / "inventory.jsonl"
    inventory.write_text(
        json.dumps(
            {
                "audio_id": "empty",
                "audio_path": "/empty.flac",
                "duration": None,
                "decode_status": "failed",
                "error": "duration_probe_failed",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    _patch_cli_runtime(monkeypatch)
    monkeypatch.setattr(fast_music_gate, "build_stage_version", lambda args: "stage-v1")
    monkeypatch.setattr(
        fast_music_gate, "build_backend", lambda *args, **kwargs: _FakeBackend()
    )

    class UnexpectedGate(_FakeGate):
        def classify_records(self, records):
            raise AssertionError("invalid inventory asset reached the model")

    monkeypatch.setattr(fast_music_gate, "CascadeMusicGate", UnexpectedGate)
    fast_music_gate.main(
        [
            "--input",
            str(inventory),
            "--output-dir",
            str(output_dir),
            "--backend-weights",
            "unused.pt",
            "--config",
            "unused.json",
        ]
    )

    assert (output_dir / "failures.jsonl").read_text(encoding="utf-8") == ""
    rejected = json.loads((output_dir / "rejected.jsonl").read_text(encoding="utf-8"))
    assert rejected["music_gate"]["decision"] == "invalid_asset"
    assert rejected["stage_status"]["music_gate"] == "skipped_invalid_asset"
    assert rejected["reason_codes"] == ["inventory_decode_failed"]


def test_cli_rejects_deterministically_truncated_audio_without_retry_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    inventory = tmp_path / "inventory.jsonl"
    inventory.write_text(
        json.dumps(
            {
                "audio_id": "truncated",
                "audio_path": "/truncated.flac",
                "duration": 200.0,
                "decode_status": "ok",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    _patch_cli_runtime(monkeypatch)
    monkeypatch.setattr(fast_music_gate, "build_stage_version", lambda args: "stage-v1")
    monkeypatch.setattr(
        fast_music_gate, "build_backend", lambda *args, **kwargs: _FakeBackend()
    )

    class TruncatedGate(_FakeGate):
        def classify_records(self, records):
            raise fast_gate_core.InvalidAudioError("both seeks returned no samples")

    monkeypatch.setattr(fast_music_gate, "CascadeMusicGate", TruncatedGate)
    fast_music_gate.main(
        [
            "--input",
            str(inventory),
            "--output-dir",
            str(output_dir),
            "--backend-weights",
            "unused.pt",
            "--config",
            "unused.json",
        ]
    )

    assert (output_dir / "failures.jsonl").read_text(encoding="utf-8") == ""
    rejected = json.loads((output_dir / "rejected.jsonl").read_text(encoding="utf-8"))
    assert rejected["music_gate"]["decision"] == "invalid_asset"
    assert rejected["reason_codes"] == ["audio_decode_failed"]
    assert rejected["stage_errors"]["music_gate"] == "both seeks returned no samples"


def _pretrained_config_artifact(checkpoint: Path, status: str = "passed"):
    import hashlib

    checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return {
        "schema_version": "music-gate-pretrained-config-v1",
        "selection_status": status,
        "config_version": "production-v1",
        "constraints": {
            "minimum_song_recall": 0.99,
            "maximum_nonmusic_false_accept_rate": 0.02,
            "maximum_review_rate": 0.05,
        },
        "support_constraints": {
            "minimum_total_per_class": 1,
            "minimum_validation_per_class": 1,
            "minimum_test_per_class": 1,
        },
        "support": {
            "total": {"music": 1, "nonmusic": 1},
            "validation": {"music": 1, "nonmusic": 1},
            "test": {"music": 1, "nonmusic": 1},
        },
        "source_support_constraints": {
            "required_sources": ["websource", "musan"],
            "minimum_total_per_source": 1,
        },
        "source_support": {
            "websource": {"train": 1, "val": 0, "test": 0, "total": 1},
            "musan": {"train": 1, "val": 0, "test": 0, "total": 1},
        },
        "metrics": {
            "validation": {
                "song_recall": 1.0,
                "nonmusic_false_accept_rate": 0.0,
                "review_rate": 0.0,
            },
            "test": {
                "song_recall": 1.0,
                "nonmusic_false_accept_rate": 0.0,
                "review_rate": 0.0,
            },
        },
        "backend": "panns_mobilenet",
        "backend_architecture": "mobilenet_v1",
        "backend_repo": "/models/PANNs",
        "backend_checkpoint": str(checkpoint),
        "backend_checkpoint_sha256": checksum,
        "backend_source": {
            "schema": "python-source-tree-v1",
            "sha256": "1" * 64,
            "file_count": 1,
            "root": "/models/PANNs",
        },
        "stage_b_backend": "panns_mobilenet",
        "stage_b_backend_architecture": "mobilenet_v1",
        "stage_b_backend_repo": "/models/PANNs",
        "stage_b_backend_checkpoint": str(checkpoint),
        "stage_b_backend_checkpoint_sha256": checksum,
        "stage_b_backend_source": {
            "schema": "python-source-tree-v1",
            "sha256": "1" * 64,
            "file_count": 1,
            "root": "/models/PANNs",
        },
        "precision": "fp32",
        "stage_b_precision": "fp32",
        "batch_size": 128,
        "stage_b_batch_size": 128,
        "thresholds": {
            "stage_a_reject": 0.2,
            "stage_a_accept": 0.8,
            "stage_b_reject": 0.4,
            "stage_b_accept": 0.6,
        },
        "cascade": {
            "enabled": True,
            "kind": "same_backend_more_windows",
            "stage_b_runs_when": "stage_a_review",
        },
        "sampler": {
            "schema": "uniform-full-track-windows-v2",
            "sample_rate": 32000,
            "short_track_decode_sample_rate": 32000,
            "short_track_resampler": "scipy.signal.resample_poly",
            "full_decode_max_seconds": 40.0,
            "decode_scheduler_schema": "bounded-track-prefetch-v1",
            "window_seconds": 8.0,
            "stage_a_fractions": [0.1, 0.5, 0.9],
            "stage_b_fractions": [0.05, 0.25, 0.5, 0.75, 0.95],
            "aggregation": AudioSetMusicScorer.aggregation,
        },
        "stage_b_sampler": {
            "schema": "uniform-full-track-windows-v2",
            "sample_rate": 32000,
            "short_track_decode_sample_rate": 32000,
            "short_track_resampler": "scipy.signal.resample_poly",
            "full_decode_max_seconds": 40.0,
            "decode_scheduler_schema": "bounded-track-prefetch-v1",
            "window_seconds": 8.0,
            "stage_a_fractions": [0.1, 0.5, 0.9],
            "stage_b_fractions": [0.05, 0.25, 0.5, 0.75, 0.95],
            "aggregation": AudioSetMusicScorer.aggregation,
        },
        "scoring": {
            "method": "native_audioset_posteriors",
            "version": AudioSetMusicScorer.VERSION,
            "class_count": 527,
            "music_indices": list(AudioSetMusicScorer.MUSIC_INDICES),
            "window_aggregation": AudioSetMusicScorer.aggregation,
        },
    }


def test_pretrained_gate_config_rejects_any_fitted_head(tmp_path: Path):
    checkpoint = tmp_path / "panns_mobilenet.pt"
    checkpoint.write_bytes(b"selected checkpoint")
    config = tmp_path / "fast_gate_config.json"
    payload = _pretrained_config_artifact(checkpoint)
    config.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_production_gate_config(str(config))
    assert loaded["config_version"] == "production-v1"
    assert loaded["scoring"]["version"] == AudioSetMusicScorer.VERSION
    verify_checkpoint_sha256(
        str(checkpoint),
        payload["backend_checkpoint_sha256"],
        "Stage A backend",
    )

    payload["head"] = {"weights": [1.0]}
    config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="must not contain fitted parameters"):
        load_production_gate_config(str(config))

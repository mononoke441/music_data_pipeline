from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "SongFormer"))

import instrumental_structure
from instrumental_structure import (
    cosine_self_similarity,
    decode_instrumental_structure,
    extract_shared_embeddings,
    novelty_curve,
)


class _FakeMuQ:
    def __init__(self):
        self.batch_sizes = []

    def __call__(self, audio, output_hidden_states=True):
        del output_hidden_states
        self.batch_sizes.append(int(audio.shape[0]))
        frames = max(2, audio.shape[-1] // 100)
        value = audio.mean(dim=1).reshape(-1, 1, 1).expand(-1, frames, 4)
        return {"hidden_states": [value] * 11}


class _FakeMusicFM:
    def __init__(self):
        self.batch_sizes = []

    def get_predictions(self, audio):
        self.batch_sizes.append(int(audio.shape[0]))
        frames = max(2, audio.shape[-1] // 100)
        value = (audio.mean(dim=1) + 1).reshape(-1, 1, 1).expand(-1, frames, 3)
        return None, [value] * 11


def test_shared_embedding_overlap_add_has_no_chunk_gaps():
    audio = torch.linspace(-1.0, 1.0, 6500)
    result = extract_shared_embeddings(
        audio,
        _FakeMuQ(),
        _FakeMusicFM(),
        sample_rate=100,
        chunk_seconds=30,
        overlap_seconds=5,
        output_frame_rate=2.0,
    )
    assert result.shape == (130, 7)
    assert torch.isfinite(result).all()
    assert torch.all(result.abs().sum(dim=-1) > 0)


def test_shared_embedding_batches_equal_length_chunks_without_padding_tail():
    audio = torch.linspace(-1.0, 1.0, 10000)
    serial = extract_shared_embeddings(
        audio,
        _FakeMuQ(),
        _FakeMusicFM(),
        sample_rate=100,
        chunk_seconds=30,
        overlap_seconds=5,
        output_frame_rate=2.0,
        embedding_batch_size=1,
    )
    batched_muq = _FakeMuQ()
    batched_musicfm = _FakeMusicFM()
    batched = extract_shared_embeddings(
        audio,
        batched_muq,
        batched_musicfm,
        sample_rate=100,
        chunk_seconds=30,
        overlap_seconds=5,
        output_frame_rate=2.0,
        embedding_batch_size=4,
    )

    assert torch.equal(serial, batched)
    assert batched_muq.batch_sizes == [3, 1]
    assert batched_musicfm.batch_sizes == [3, 1]


def test_novelty_and_decoder_cover_duration():
    rng = np.random.default_rng(4)
    a = rng.normal(0, 0.02, (40, 16)) + np.eye(1, 16, 0)
    b = rng.normal(0, 0.02, (40, 16)) + np.eye(1, 16, 8)
    embeddings = np.concatenate([a, b, a], axis=0).astype(np.float32)
    curve = novelty_curve(embeddings, 2.0)
    ssm = cosine_self_similarity(embeddings)
    assert curve.shape == (120,)
    assert ssm.shape == (120, 120)
    assert np.allclose(ssm, ssm.T, atol=1e-6)
    assert float(curve.max()) <= 1.0
    sections = decode_instrumental_structure(embeddings, 60.0)
    assert sections[0]["start"] == 0.0
    assert sections[-1]["end"] == 60.0
    assert all(left["end"] == right["start"] for left, right in zip(sections, sections[1:]))
    assert all(section["label_source"] == "instrumental_ssm_cbm" for section in sections)
    assert all(section["label_confidence"] < 1.0 for section in sections)
    assert len(sections) >= 3
    assert sections[0]["label"] == sections[-1]["label"]


def test_decoder_does_not_assign_single_new_cluster_perfect_confidence():
    rng = np.random.default_rng(12)
    embeddings = rng.normal(size=(24, 8)).astype(np.float32)
    sections = decode_instrumental_structure(embeddings, 12.0)
    assert sections
    assert all(0.0 < section["label_confidence"] < 1.0 for section in sections)


def test_decoder_materializes_cosine_self_similarity_once(monkeypatch):
    rng = np.random.default_rng(14)
    embeddings = rng.normal(size=(48, 12)).astype(np.float32)
    original = instrumental_structure.cosine_self_similarity
    calls = 0

    def counted(values):
        nonlocal calls
        calls += 1
        return original(values)

    monkeypatch.setattr(instrumental_structure, "cosine_self_similarity", counted)
    sections = instrumental_structure.decode_instrumental_structure(embeddings, 24.0)

    assert sections
    assert calls == 1

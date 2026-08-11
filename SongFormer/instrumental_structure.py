"""Instrumental structure decoding from shared MuQ + MusicFM embeddings."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial.distance import squareform

from embedding_batch import extract_muq_musicfm_chunks


def _align_and_concatenate(muq_embedding: torch.Tensor, musicfm_embedding: torch.Tensor) -> torch.Tensor:
    length = min(muq_embedding.shape[1], musicfm_embedding.shape[1])
    muq_embedding = F.normalize(muq_embedding[:, :length].float(), dim=-1)
    musicfm_embedding = F.normalize(musicfm_embedding[:, :length].float(), dim=-1)
    return torch.cat([muq_embedding, musicfm_embedding], dim=-1)


@torch.inference_mode()
def extract_shared_embeddings(
    audio: torch.Tensor,
    muq: Any,
    musicfm: Any,
    *,
    sample_rate: int = 24000,
    chunk_seconds: int = 30,
    overlap_seconds: int = 5,
    output_frame_rate: float = 2.0,
    embedding_batch_size: int = 1,
) -> torch.Tensor:
    """Extract a fused stream with overlap-add to suppress 30-second seams."""

    chunk_samples = int(chunk_seconds * sample_rate)
    overlap_samples = int(overlap_seconds * sample_rate)
    if overlap_samples < 0 or overlap_samples >= chunk_samples:
        raise ValueError("overlap_seconds must be >= 0 and smaller than chunk_seconds")
    hop_samples = chunk_samples - overlap_samples
    total_samples = int(audio.shape[-1])
    total_frames = max(1, int(math.ceil(total_samples / sample_rate * output_frame_rate)))
    accumulated: torch.Tensor | None = None
    accumulated_weight = torch.zeros(total_frames, dtype=torch.float32)
    overlap_frames = max(1, int(round(overlap_seconds * output_frame_rate)))

    chunk_specs = []
    for start in range(0, total_samples, hop_samples):
        finish = min(total_samples, start + chunk_samples)
        value = audio[start:finish]
        if value.numel() <= 1024:
            continue
        chunk_specs.append((start, finish, value))
    muq_embeddings, musicfm_embeddings = extract_muq_musicfm_chunks(
        [value for _, _, value in chunk_specs],
        muq,
        musicfm,
        batch_size=embedding_batch_size,
    )

    for (start, finish, value), muq_embedding, musicfm_embedding in zip(
        chunk_specs, muq_embeddings, musicfm_embeddings
    ):
        combined = _align_and_concatenate(muq_embedding, musicfm_embedding)
        target_frames = max(1, int(math.ceil(value.numel() / sample_rate * output_frame_rate)))
        combined = F.interpolate(
            combined.transpose(1, 2),
            size=target_frames,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        combined_cpu = combined.squeeze(0).cpu()
        global_start = int(round(start / sample_rate * output_frame_rate))
        global_finish = min(total_frames, global_start + combined_cpu.shape[0])
        combined_cpu = combined_cpu[: global_finish - global_start]
        weights = torch.ones(combined_cpu.shape[0], dtype=torch.float32)
        fade = min(overlap_frames, max(0, combined_cpu.shape[0] // 2))
        if fade and start > 0:
            weights[:fade] = torch.linspace(0.05, 1.0, fade)
        if fade and finish < total_samples:
            weights[-fade:] = torch.minimum(weights[-fade:], torch.linspace(1.0, 0.05, fade))
        if accumulated is None:
            accumulated = torch.zeros(
                (total_frames, combined_cpu.shape[-1]), dtype=combined_cpu.dtype
            )
        accumulated[global_start:global_finish] += combined_cpu * weights.unsqueeze(-1)
        accumulated_weight[global_start:global_finish] += weights
        del muq_embedding, musicfm_embedding, combined
    if accumulated is None or not torch.any(accumulated_weight > 0):
        raise RuntimeError("MuQ/MusicFM produced no instrumental embeddings")
    accumulated /= accumulated_weight.clamp_min(1e-8).unsqueeze(-1)
    return accumulated


def cosine_self_similarity(embeddings: np.ndarray) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("embeddings must have shape [frames, dimensions]")
    values = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)
    return np.clip(values @ values.T, -1.0, 1.0).astype(np.float32, copy=False)


def _integral_image(values: np.ndarray) -> np.ndarray:
    return np.pad(values, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)


def _rect_mean(integral: np.ndarray, row0: int, row1: int, col0: int, col1: int) -> float:
    area = max(1, (row1 - row0) * (col1 - col0))
    total = (
        integral[row1, col1] - integral[row0, col1]
        - integral[row1, col0] + integral[row0, col0]
    )
    return float(total / area)


def novelty_curve(
    embeddings: np.ndarray,
    frame_rate: float,
    scales_seconds: Sequence[float] = (4.0, 8.0, 16.0),
) -> np.ndarray:
    """Multi-scale Foote checkerboard novelty over the full cosine SSM."""

    return novelty_curve_from_self_similarity(
        cosine_self_similarity(embeddings), frame_rate, scales_seconds
    )


def novelty_curve_from_self_similarity(
    ssm: np.ndarray,
    frame_rate: float,
    scales_seconds: Sequence[float] = (4.0, 8.0, 16.0),
) -> np.ndarray:
    """Compute novelty from an already materialized cosine SSM."""

    ssm = np.asarray(ssm, dtype=np.float32)
    if ssm.ndim != 2 or ssm.shape[0] == 0 or ssm.shape[0] != ssm.shape[1]:
        raise ValueError("self-similarity matrix must be non-empty and square")
    integral = _integral_image(ssm)
    novelty = np.zeros(len(ssm), dtype=np.float32)
    counts = np.zeros(len(ssm), dtype=np.float32)
    for scale_seconds in scales_seconds:
        radius = max(1, int(round(scale_seconds * frame_rate / 2.0)))
        scale_curve = np.zeros(len(ssm), dtype=np.float32)
        for index in range(radius, len(ssm) - radius):
            left_left = _rect_mean(integral, index - radius, index, index - radius, index)
            right_right = _rect_mean(integral, index, index + radius, index, index + radius)
            left_right = _rect_mean(integral, index - radius, index, index, index + radius)
            right_left = _rect_mean(integral, index, index + radius, index - radius, index)
            scale_curve[index] = max(
                0.0, 0.5 * (left_left + right_right - left_right - right_left)
            )
        positive = scale_curve[scale_curve > 0]
        if positive.size:
            scale_curve /= max(float(np.quantile(positive, 0.95)), 1e-8)
        novelty += np.clip(scale_curve, 0.0, 1.0)
        counts += (scale_curve > 0).astype(np.float32)
    novelty /= np.maximum(counts, 1.0)
    novelty = gaussian_filter1d(novelty, sigma=max(0.5, frame_rate * 0.35))
    edge = min(len(novelty) // 4, max(1, int(round(2.0 * frame_rate))))
    if edge:
        novelty[:edge] = 0.0
        novelty[-edge:] = 0.0
    if float(novelty.max()) > float(novelty.min()):
        novelty = (novelty - novelty.min()) / (novelty.max() - novelty.min())
    return novelty


def _segment_cohesion(integral: np.ndarray, begin: int, finish: int) -> float:
    length = finish - begin
    if length <= 1:
        return 0.0
    block_sum = (
        integral[finish, finish] - integral[begin, finish]
        - integral[finish, begin] + integral[begin, begin]
    )
    # The diagonal is always one and otherwise rewards pathological tiny cuts.
    return float((block_sum - length) / max(1, length * length - length))


def _cbm_boundaries(
    ssm: np.ndarray,
    curve: np.ndarray,
    frame_rate: float,
    minimum_boundary_distance: float,
    maximum_section_duration: float,
) -> np.ndarray:
    """Select a globally optimal candidate subset using a CBM-style DP."""

    length = len(curve)
    minimum_frames = max(1, int(round(minimum_boundary_distance * frame_rate)))
    maximum_frames = max(minimum_frames, int(round(maximum_section_duration * frame_rate)))
    nonzero = curve[curve > 0]
    height = float(np.quantile(nonzero, 0.55)) if nonzero.size else 1.0
    peaks, _ = find_peaks(
        curve,
        distance=max(1, minimum_frames // 2),
        height=height,
        prominence=0.03,
    )
    candidates = {0, length, *(int(value) for value in peaks)}
    # Ensure DP can cover long homogeneous passages without inventing a fixed
    # grid boundary: pick the strongest novelty point in each max-length span.
    anchor = 0
    while length - anchor > maximum_frames:
        lo = anchor + minimum_frames
        hi = min(length, anchor + maximum_frames)
        if hi <= lo:
            break
        selected = lo + int(np.argmax(curve[lo:hi]))
        candidates.add(selected)
        anchor = selected
    points = np.asarray(sorted(candidates), dtype=np.int64)
    integral = _integral_image(ssm)
    best = np.full(len(points), -np.inf, dtype=np.float64)
    previous = np.full(len(points), -1, dtype=np.int64)
    best[0] = 0.0
    section_penalty = 1.5
    boundary_weight = 2.0
    for right_index in range(1, len(points)):
        right = int(points[right_index])
        for left_index in range(right_index):
            left = int(points[left_index])
            frames = right - left
            if frames <= 0 or frames > maximum_frames:
                continue
            if frames < minimum_frames and left != 0 and right != length:
                continue
            duration_seconds = frames / frame_rate
            cohesion = _segment_cohesion(integral, left, right)
            boundary_reward = 0.0 if right == length else boundary_weight * float(curve[right])
            score = best[left_index] + cohesion * duration_seconds - section_penalty + boundary_reward
            if score > best[right_index]:
                best[right_index] = score
                previous[right_index] = left_index
    if previous[-1] < 0:
        return np.asarray([0, length], dtype=np.int64)
    selected = []
    cursor = len(points) - 1
    while cursor >= 0:
        selected.append(int(points[cursor]))
        cursor = int(previous[cursor])
    return np.asarray(list(reversed(selected)), dtype=np.int64)


def _global_cluster_labels(
    segment_embeddings: Sequence[np.ndarray],
    cluster_similarity: float,
) -> tuple[List[str], List[float]]:
    count = len(segment_embeddings)
    if count == 1:
        return ["A"], [0.5]
    values = np.stack(segment_embeddings)
    similarities = np.clip(values @ values.T, -1.0, 1.0)
    distances = np.clip(1.0 - similarities, 0.0, 2.0)
    hierarchy = linkage(squareform(distances, checks=False), method="average")
    raw_clusters = fcluster(hierarchy, t=1.0 - cluster_similarity, criterion="distance")
    remap: Dict[int, int] = {}
    labels: List[str] = []
    for cluster in raw_clusters:
        cluster_id = int(cluster)
        if cluster_id not in remap:
            remap[cluster_id] = len(remap)
        labels.append(_alphabet_label(remap[cluster_id]))

    confidences: List[float] = []
    for index, cluster in enumerate(raw_clusters):
        same = [
            float(similarities[index, other])
            for other in range(count)
            if other != index and raw_clusters[other] == cluster
        ]
        other = [
            float(similarities[index, candidate])
            for candidate in range(count)
            if raw_clusters[candidate] != cluster
        ]
        nearest_other = max(other) if other else 0.0
        if same:
            confidence = 0.5 + 0.5 * max(0.0, float(np.mean(same)) - nearest_other)
        else:
            confidence = 0.5 * max(0.0, 1.0 - nearest_other)
        confidences.append(float(np.clip(confidence, 0.05, 0.95)))
    return labels, confidences


def _alphabet_label(index: int) -> str:
    # A..Z, AA..AZ, BA... for unusually complex works.
    result = ""
    value = index
    while True:
        result = chr(ord("A") + value % 26) + result
        value = value // 26 - 1
        if value < 0:
            return result


def decode_instrumental_structure(
    embeddings: torch.Tensor | np.ndarray,
    duration: float,
    *,
    frame_rate: float = 2.0,
    minimum_boundary_distance: float = 8.0,
    cluster_similarity: float = 0.82,
    maximum_section_duration: float = 120.0,
) -> List[Dict[str, Any]]:
    values = embeddings.detach().cpu().numpy() if isinstance(embeddings, torch.Tensor) else np.asarray(embeddings)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("instrumental embeddings must have shape [frames, dimensions]")
    ssm = cosine_self_similarity(values)
    curve = novelty_curve_from_self_similarity(ssm, frame_rate)
    boundaries = _cbm_boundaries(
        ssm,
        curve,
        frame_rate,
        minimum_boundary_distance,
        maximum_section_duration,
    )

    segment_embeddings: List[np.ndarray] = []
    for begin, finish in zip(boundaries[:-1], boundaries[1:]):
        mean = values[int(begin) : int(finish)].mean(axis=0)
        mean = mean / max(float(np.linalg.norm(mean)), 1e-8)
        segment_embeddings.append(mean)

    labels, label_confidences = _global_cluster_labels(segment_embeddings, cluster_similarity)

    occurrence = {label: labels.count(label) for label in set(labels)}
    if labels:
        first_duration = (boundaries[1] - boundaries[0]) / frame_rate
        if occurrence[labels[0]] == 1 and first_duration <= min(30.0, duration * 0.15):
            labels[0] = "Intro"
        last_duration = (boundaries[-1] - boundaries[-2]) / frame_rate
        if len(labels) > 1 and occurrence[labels[-1]] == 1 and last_duration <= min(30.0, duration * 0.15):
            labels[-1] = "Outro"

    output: List[Dict[str, Any]] = []
    for index, (begin, finish) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        end_index = min(int(finish), max(0, len(curve) - 1))
        confidence = 1.0 if finish >= len(curve) else float(curve[end_index])
        output.append({
            "label": labels[index],
            "label_source": "instrumental_ssm_cbm",
            "start": float(begin / frame_rate),
            "end": min(float(duration), float(finish / frame_rate)),
            "raw_start": float(begin / frame_rate),
            "raw_end": min(float(duration), float(finish / frame_rate)),
            "start_boundary_confidence": 1.0 if begin == 0 else float(curve[min(int(begin), len(curve) - 1)]),
            "end_boundary_confidence": confidence,
            "boundary_confidence": confidence,
            "label_confidence": float(label_confidences[index]),
        })
    if output:
        output[0]["start"] = output[0]["raw_start"] = 0.0
        output[-1]["end"] = output[-1]["raw_end"] = float(duration)
    return output


@torch.inference_mode()
def infer_instrumental_structure(
    audio: torch.Tensor,
    muq: Any,
    musicfm: Any,
    *,
    embedding_batch_size: int = 1,
) -> List[Dict[str, Any]]:
    duration = audio.numel() / 24000.0
    embeddings = extract_shared_embeddings(
        audio, muq, musicfm, embedding_batch_size=embedding_batch_size
    )
    return decode_instrumental_structure(embeddings, duration)

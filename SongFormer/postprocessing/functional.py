# This file contains code adapted from the following sources:
# [MIT license] https://github.com/mir-aidj/all-in-one/blob/main/src/allin1/postprocessing/functional.py

import numpy as np
import torch
from .helpers import (
    local_maxima,
    peak_picking,
    # event_frames_to_time,
)
from dataset.label2id import LABEL_TO_ID, ID_TO_LABEL
from dataset.custom_types import MsaInfo


def event_frames_to_time(frame_rates, boundary: np.array):
    boundary = np.array(boundary)
    boundary_times = boundary / frame_rates
    return boundary_times


def postprocess_functional_structure(
    logits,
    config,
):
    # pdb.set_trace()
    boundary_logits = logits["boundary_logits"]
    function_logits = logits["function_logits"]

    assert boundary_logits.shape[0] == 1 and function_logits.shape[0] == 1, (
        "Only batch size 1 is supported"
    )
    raw_prob_sections = torch.sigmoid(boundary_logits[0])
    raw_prob_functions = torch.softmax(function_logits[0].transpose(0, 1), dim=0)

    # filter_size=4 * cfg.min_hops_per_beat + 1
    prob_sections, _ = local_maxima(
        raw_prob_sections, filter_size=config.local_maxima_filter_size
    )
    prob_sections = prob_sections.cpu().numpy()

    prob_functions = raw_prob_functions.cpu().numpy()

    boundary_candidates = peak_picking(
        boundary_activation=prob_sections,
        window_past=int(12 * config.frame_rates),  # 原来是fps
        window_future=int(12 * config.frame_rates),
    )
    boundary = boundary_candidates > 0.0

    duration = len(prob_sections) / config.frame_rates
    pred_boundary_times = event_frames_to_time(
        frame_rates=config.frame_rates, boundary=np.flatnonzero(boundary)
    )
    if pred_boundary_times[0] != 0:
        pred_boundary_times = np.insert(pred_boundary_times, 0, 0)
    if pred_boundary_times[-1] != duration:
        pred_boundary_times = np.append(pred_boundary_times, duration)
    pred_boundaries = np.stack([pred_boundary_times[:-1], pred_boundary_times[1:]]).T

    pred_boundary_indices = np.flatnonzero(boundary)
    pred_boundary_indices = pred_boundary_indices[pred_boundary_indices > 0]
    prob_segment_function = np.split(prob_functions, pred_boundary_indices, axis=1)
    pred_labels = [p.mean(axis=1).argmax().item() for p in prob_segment_function]

    segments: MsaInfo = []
    for (start, end), label in zip(pred_boundaries, pred_labels):
        segment = (float(start), str(ID_TO_LABEL[label]))
        segments.append(segment)

    segments.append((float(pred_boundary_times[-1]), "end"))
    return segments


def postprocess_functional_structure_detailed(logits, config):
    """Return sections with boundary and function confidence.

    The original helper reduces logits to `(start, label)` tuples and loses the
    evidence required by downbeat snapping.  This variant keeps the exact raw
    boundaries and probabilities while using the same peak picker.
    """

    boundary_logits = logits["boundary_logits"]
    function_logits = logits["function_logits"]
    assert boundary_logits.shape[0] == 1 and function_logits.shape[0] == 1, (
        "Only batch size 1 is supported"
    )

    raw_boundary_probability = torch.sigmoid(boundary_logits[0])
    raw_function_probability = torch.softmax(function_logits[0], dim=-1)
    filtered, _ = local_maxima(
        raw_boundary_probability,
        filter_size=config.local_maxima_filter_size,
    )
    filtered_np = filtered.detach().cpu().numpy()
    candidates = peak_picking(
        boundary_activation=filtered_np,
        window_past=int(12 * config.frame_rates),
        window_future=int(12 * config.frame_rates),
    )
    internal_indices = np.flatnonzero(candidates > 0.0)
    internal_indices = internal_indices[
        (internal_indices > 0) & (internal_indices < len(filtered_np))
    ]
    boundary_indices = np.concatenate(
        [np.array([0], dtype=np.int64), internal_indices, np.array([len(filtered_np)], dtype=np.int64)]
    )
    boundary_indices = np.unique(boundary_indices)
    boundary_probability = raw_boundary_probability.detach().cpu().numpy()
    function_probability = raw_function_probability.detach().cpu().numpy()

    output = []
    for section_index in range(len(boundary_indices) - 1):
        begin = int(boundary_indices[section_index])
        finish = int(boundary_indices[section_index + 1])
        if finish <= begin:
            continue
        segment_probability = function_probability[begin:finish]
        means = segment_probability.mean(axis=0)
        label_index = int(means.argmax())
        start_confidence = 1.0 if begin == 0 else float(boundary_probability[min(begin, len(boundary_probability) - 1)])
        end_confidence = 1.0 if finish >= len(boundary_probability) else float(boundary_probability[finish])
        output.append({
            "label": str(ID_TO_LABEL[label_index]),
            "start": float(begin / config.frame_rates),
            "end": float(finish / config.frame_rates),
            "start_boundary_confidence": start_confidence,
            "end_boundary_confidence": end_confidence,
            "boundary_confidence": end_confidence,
            "label_confidence": float(means[label_index]),
        })
    return output

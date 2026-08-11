#!/usr/bin/env python3
"""Numerically compare the torch frontend with pinned TensorflowInputMusiCNN."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "MusicToolsPipeline"))

from sub_models.discogs_onnx_model import DiscogsMelFrontend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", help="Optional audio; otherwise use deterministic noise")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import essentia.standard as es

    if args.audio:
        waveform = es.MonoLoader(filename=args.audio, sampleRate=16000)().astype(np.float32)
    else:
        waveform = np.random.default_rng(20260805).normal(0, 0.1, 16000 * 30).astype(np.float32)
    extractor = es.TensorflowInputMusiCNN()
    expected = np.stack([
        extractor(frame)
        for frame in es.FrameGenerator(
            waveform,
            frameSize=512,
            hopSize=256,
            startFromZero=False,
        )
    ])
    actual = (
        DiscogsMelFrontend(torch.device(args.device))
        .mel_frames(torch.from_numpy(waveform))
        .detach().cpu().numpy()
    )
    frames = min(len(expected), len(actual))
    delta = expected[:frames] - actual[:frames]
    cosine = np.sum(expected[:frames] * actual[:frames], axis=1) / np.maximum(
        np.linalg.norm(expected[:frames], axis=1) * np.linalg.norm(actual[:frames], axis=1),
        1e-12,
    )
    print(f"frames: essentia={len(expected)} torch={len(actual)} compared={frames}")
    print(f"mean_abs_error={np.abs(delta).mean():.9f}")
    print(f"max_abs_error={np.abs(delta).max():.9f}")
    print(f"mean_cosine={cosine.mean():.9f}")


if __name__ == "__main__":
    main()

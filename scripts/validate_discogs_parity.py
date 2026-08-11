#!/usr/bin/env python3
"""Compare ONNX instrument activations with the pinned Essentia PB pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "MusicToolsPipeline"))

from sub_models.discogs_onnx_model import DiscogsModelPaths, DiscogsOnnxEngine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", nargs="+", required=True)
    parser.add_argument("--pb-backbone", required=True)
    parser.add_argument("--pb-instrument", required=True)
    parser.add_argument("--onnx-root", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--minimum-agreement", type=float, default=0.99)
    args = parser.parse_args()

    import essentia.standard as es

    pb_backbone = es.TensorflowPredictEffnetDiscogs(
        graphFilename=args.pb_backbone,
        output="PartitionedCall:1",
    )
    pb_head = es.TensorflowPredict2D(graphFilename=args.pb_instrument)
    onnx = DiscogsOnnxEngine(DiscogsModelPaths.from_root(args.onnx_root))
    agreements = []
    for path in args.audio:
        waveform = es.MonoLoader(filename=path, sampleRate=16000)().astype(np.float32)
        expected = np.asarray(pb_head(pb_backbone(waveform)), dtype=np.float32)
        _, heads, _ = onnx.predict_frames(torch.from_numpy(waveform), 16000)
        actual = np.asarray(heads["instrument"], dtype=np.float32)
        frames = min(len(expected), len(actual))
        classes = min(expected.shape[1], actual.shape[1])
        if frames == 0 or classes == 0:
            raise RuntimeError(f"no comparable predictions for {path}")
        expected_active = expected[:frames, :classes] >= args.threshold
        actual_active = actual[:frames, :classes] >= args.threshold
        agreement = float((expected_active == actual_active).mean())
        agreements.append(agreement)
        print(f"{path}: active_label_agreement={agreement:.6f} frames={frames}")
    total = float(np.mean(agreements))
    print(f"mean_active_label_agreement={total:.6f}")
    if total < args.minimum_agreement:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

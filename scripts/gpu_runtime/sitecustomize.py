"""Apply runner-provided CUDA budgets before target modules load models.

This directory is placed on ``PYTHONPATH`` only for managed GPU subprocesses.
Keeping resource policy outside inference modules keeps inference code focused;
stage/model versions remain provenance only and never control cache reuse.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))

from gpu_memory import (  # noqa: E402
    apply_torch_cuda_memory_limit,
    capped_onnx_providers,
)


torch_max = os.environ.get("PIPELINE_TORCH_GPU_MAX_MEMORY_GIB", "").strip()
if torch_max and float(torch_max) > 0:
    import torch

    device = os.environ.get("PIPELINE_TORCH_CUDA_DEVICE", "0")
    fraction = apply_torch_cuda_memory_limit(
        float(torch_max),
        int(device),
        torch_module=torch,
    )
    print(
        "[gpu-memory] "
        f"torch_max={float(torch_max):g}GiB allocator_fraction={fraction:.6f}",
        file=sys.stderr,
        flush=True,
    )


onnx_max = os.environ.get("PIPELINE_ORT_GPU_MAX_MEMORY_GIB", "").strip()
if onnx_max and float(onnx_max) > 0:
    import onnxruntime as ort

    original_inference_session = ort.InferenceSession

    def capped_inference_session(*args, **kwargs):
        providers = kwargs.get("providers")
        if providers is not None:
            kwargs["providers"] = capped_onnx_providers(
                providers,
                float(onnx_max),
            )
        return original_inference_session(*args, **kwargs)

    ort.InferenceSession = capped_inference_session
    print(
        f"[gpu-memory] onnx_max={float(onnx_max):g}GiB",
        file=sys.stderr,
        flush=True,
    )

"""Shared GPU-memory budget helpers for runner-managed PyTorch stages."""

from __future__ import annotations

import math
from typing import Any, Sequence


GIB = 1024 ** 3


def capped_memory_fraction(max_memory_gib: float, total_memory_bytes: int) -> float:
    """Return a six-decimal CUDA allocator fraction that never exceeds the cap."""

    maximum = float(max_memory_gib)
    total = int(total_memory_bytes)
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("GPU max memory GiB must be finite and positive")
    if total <= 0:
        raise ValueError("total GPU memory must be positive")
    fraction = min(maximum * GIB / total, 0.99)
    fraction = math.floor(fraction * 1_000_000) / 1_000_000
    if fraction <= 0:
        raise ValueError("resolved GPU memory fraction is zero")
    return fraction


def cuda_device_index(device: str | int) -> int:
    if isinstance(device, int):
        return device
    value = str(device).strip().lower()
    if value == "cuda":
        return 0
    if value.startswith("cuda:"):
        return int(value.split(":", 1)[1])
    raise ValueError(f"expected a CUDA device, got: {device!r}")


def apply_torch_cuda_memory_limit(
    max_memory_gib: float,
    device: str | int = 0,
    *,
    torch_module: Any | None = None,
) -> float:
    """Apply a per-process PyTorch allocator ceiling before model loading."""

    if torch_module is None:
        import torch as torch_module

    maximum = float(max_memory_gib)
    if not math.isfinite(maximum) or maximum < 0:
        raise ValueError("GPU max memory GiB must be finite and non-negative")
    index = cuda_device_index(device)
    if not torch_module.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable")
    if maximum == 0:
        # Zero is the runner's explicit unlimited sentinel. Never install a
        # positive allocator ceiling in this mode.
        return 0.0
    total = int(torch_module.cuda.get_device_properties(index).total_memory)
    fraction = capped_memory_fraction(maximum, total)
    torch_module.cuda.set_per_process_memory_fraction(fraction, index)
    return fraction


def resolve_asr_vllm_memory_budget(
    *,
    free_memory_gib: float,
    pipeline_max_memory_gib: float,
    requested_vllm_max_memory_gib: float,
    forced_aligner_reserve_gib: float,
    vllm_headroom_gib: float,
    minimum_vllm_memory_gib: float,
) -> float:
    """Resolve the ASR vLLM budget from live free memory and positive caps."""

    values = {
        "free GPU memory": float(free_memory_gib),
        "pipeline GPU cap": float(pipeline_max_memory_gib),
        "ASR vLLM cap": float(requested_vllm_max_memory_gib),
        "ForcedAligner reserve": float(forced_aligner_reserve_gib),
        "vLLM headroom": float(vllm_headroom_gib),
        "minimum vLLM memory": float(minimum_vllm_memory_gib),
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("ASR GPU memory limits must be finite")
    if values["free GPU memory"] <= 0:
        raise ValueError("free GPU memory must be positive")
    if any(value < 0 for name, value in values.items() if name != "free GPU memory"):
        raise ValueError("ASR GPU memory limits must be non-negative")

    candidates = [values["free GPU memory"]]
    for name in ("pipeline GPU cap", "ASR vLLM cap"):
        if values[name] > 0:
            candidates.append(values[name])
    budget = min(candidates) - values["ForcedAligner reserve"] - values["vLLM headroom"]
    if budget < values["minimum vLLM memory"]:
        raise ValueError(
            "insufficient GPU memory for ASR vLLM: "
            f"resolved={budget:.6f}GiB minimum={values['minimum vLLM memory']:.6f}GiB "
            f"free={values['free GPU memory']:.6f}GiB "
            f"aligner_reserve={values['ForcedAligner reserve']:.6f}GiB "
            f"headroom={values['vLLM headroom']:.6f}GiB"
        )
    return budget


def capped_onnx_providers(
    providers: Sequence[Any],
    max_memory_gib: float,
) -> list[Any]:
    """Add a CUDAExecutionProvider arena limit without changing other providers."""

    maximum = float(max_memory_gib)
    if not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("ONNX GPU max memory GiB must be finite and positive")
    limit = int(maximum * GIB)
    output: list[Any] = []
    for provider in providers:
        if isinstance(provider, str):
            if provider == "CUDAExecutionProvider":
                output.append((provider, {"gpu_mem_limit": limit}))
            else:
                output.append(provider)
            continue
        name, raw_options = provider
        options = dict(raw_options or {})
        if name == "CUDAExecutionProvider":
            existing = options.get("gpu_mem_limit")
            options["gpu_mem_limit"] = (
                min(int(existing), limit) if existing is not None else limit
            )
        output.append((name, options))
    return output

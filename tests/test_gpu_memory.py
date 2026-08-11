from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from gpu_memory import (  # noqa: E402
    GIB,
    apply_torch_cuda_memory_limit,
    capped_onnx_providers,
    capped_memory_fraction,
    cuda_device_index,
    resolve_asr_vllm_memory_budget,
)


def test_capped_memory_fraction_never_exceeds_absolute_limit():
    fraction = capped_memory_fraction(32, 81559 * 1024**2)
    assert fraction == pytest.approx(0.401770)
    assert fraction * 81559 * 1024**2 <= 32 * GIB


def test_cuda_device_index_parses_visible_process_device():
    assert cuda_device_index("cuda") == 0
    assert cuda_device_index("cuda:3") == 3
    assert cuda_device_index(2) == 2
    with pytest.raises(ValueError):
        cuda_device_index("cpu")


def test_apply_torch_limit_calls_allocator_before_model_loading():
    class Properties:
        total_memory = 80 * GIB

    class FakeCuda:
        calls = []

        @staticmethod
        def is_available():
            return True

        @staticmethod
        def get_device_properties(index):
            assert index == 0
            return Properties()

        @classmethod
        def set_per_process_memory_fraction(cls, fraction, index):
            cls.calls.append((fraction, index))

    class FakeTorch:
        cuda = FakeCuda

    fraction = apply_torch_cuda_memory_limit(32, "cuda:0", torch_module=FakeTorch)
    assert fraction == 0.4
    assert FakeCuda.calls == [(0.4, 0)]


def test_zero_torch_limit_does_not_install_allocator_ceiling():
    class FakeCuda:
        properties_called = False
        calls = []

        @staticmethod
        def is_available():
            return True

        @classmethod
        def get_device_properties(cls, index):
            cls.properties_called = True
            raise AssertionError("unlimited mode must not query allocator capacity")

        @classmethod
        def set_per_process_memory_fraction(cls, fraction, index):
            cls.calls.append((fraction, index))

    class FakeTorch:
        cuda = FakeCuda

    assert apply_torch_cuda_memory_limit(0, 0, torch_module=FakeTorch) == 0.0
    assert FakeCuda.properties_called is False
    assert FakeCuda.calls == []


def test_asr_budget_uses_live_free_memory_positive_caps_and_12_gib_reserve():
    base = {
        "forced_aligner_reserve_gib": 8,
        "vllm_headroom_gib": 4,
        "minimum_vllm_memory_gib": 8,
    }
    assert (
        resolve_asr_vllm_memory_budget(
            free_memory_gib=70,
            pipeline_max_memory_gib=0,
            requested_vllm_max_memory_gib=0,
            **base,
        )
        == 58
    )
    assert (
        resolve_asr_vllm_memory_budget(
            free_memory_gib=70,
            pipeline_max_memory_gib=40,
            requested_vllm_max_memory_gib=32,
            **base,
        )
        == 20
    )
    assert (
        resolve_asr_vllm_memory_budget(
            free_memory_gib=70,
            pipeline_max_memory_gib=40,
            requested_vllm_max_memory_gib=20,
            **base,
        )
        == 8
    )
    with pytest.raises(ValueError, match="insufficient GPU memory"):
        resolve_asr_vllm_memory_budget(
            free_memory_gib=19,
            pipeline_max_memory_gib=0,
            requested_vllm_max_memory_gib=0,
            **base,
        )


def test_onnx_cuda_provider_limit_preserves_other_options_and_providers():
    providers = [
        ("CUDAExecutionProvider", {"device_id": 0}),
        "CPUExecutionProvider",
    ]
    assert capped_onnx_providers(providers, 28) == [
        (
            "CUDAExecutionProvider",
            {"device_id": 0, "gpu_mem_limit": 28 * GIB},
        ),
        "CPUExecutionProvider",
    ]

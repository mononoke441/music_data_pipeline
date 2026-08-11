from __future__ import annotations

import concurrent.futures
import threading
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from section_asr_infer import (  # noqa: E402
    capped_vllm_gpu_memory_utilization,
    live_asr_vllm_memory_budget,
    prefetched_decode_batches,
    section_batches,
)


def _item(value: int) -> dict:
    return {"value": value, "target": {"section_id": str(value)}}


def test_section_batches_never_mix_duration_buckets():
    buckets = {
        "8-15": [_item(1), _item(2), _item(3)],
        "30-60": [_item(4), _item(5)],
    }
    assert [
        [item["value"] for item in batch]
        for batch in section_batches(buckets, batch_size=2)
    ] == [[1, 2], [3], [4, 5]]


def test_asr_vllm_memory_is_capped_at_32_gib_and_allows_only_lower_override():
    total = 80 * (1024**3)
    assert capped_vllm_gpu_memory_utilization(32, total) == 0.4
    assert capped_vllm_gpu_memory_utilization(32, total, 0.3) == 0.3
    assert capped_vllm_gpu_memory_utilization(32, total, 0.8) == 0.4
    resolved = capped_vllm_gpu_memory_utilization(32, 81559 * (1024**2))
    assert resolved * 81559 <= 32 * 1024
    assert capped_vllm_gpu_memory_utilization(0, total) == 0.99


def test_asr_samples_live_free_memory_and_reserves_8_plus_4_gib():
    gib = 1024**3

    class FakeCuda:
        @staticmethod
        def mem_get_info(index):
            assert index == 0
            return 72 * gib, 80 * gib

    class FakeTorch:
        cuda = FakeCuda

    budget, free_bytes, total_bytes = live_asr_vllm_memory_budget(
        FakeTorch,
        pipeline_max_memory_gib=0,
        requested_vllm_max_memory_gib=0,
        forced_aligner_reserve_gib=8,
        vllm_headroom_gib=4,
        minimum_vllm_memory_gib=8,
    )
    assert budget == 60
    assert free_bytes == 72 * gib
    assert total_bytes == 80 * gib
    utilization = capped_vllm_gpu_memory_utilization(budget, total_bytes)
    assert utilization == 0.75
    assert utilization * total_bytes <= free_bytes - 12 * gib


def test_decode_pool_prefetches_next_batch_and_isolates_failures():
    next_batch_started = threading.Event()

    def decode(item, padding):
        assert padding == 1.5
        if item["value"] == 2:
            next_batch_started.set()
        if item["value"] == 3:
            raise RuntimeError("broken audio")
        return np.asarray([item["value"]], dtype=np.float32), 16000, 0.0

    batches = [[_item(1)], [_item(2), _item(3)]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        iterator = prefetched_decode_batches(
            batches,
            executor=executor,
            padding=1.5,
            decode_function=decode,
        )
        first = next(iterator)
        # Batch 2 was submitted before batch 1 was yielded to GPU inference.
        assert next_batch_started.wait(timeout=1.0)
        second = next(iterator)

    assert first[0][0]["value"] == 1
    assert [item[0]["value"] for item in second] == [2]
    assert batches[1][1]["target"]["asr_status"] == "decode_error"
    assert "broken audio" in batches[1][1]["target"]["asr_error"]

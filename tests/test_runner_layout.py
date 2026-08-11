from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_runner_has_valid_shell_syntax_and_two_positional_paths():
    runner = ROOT / "run_pipeline.sh"
    assert runner.is_file()
    subprocess.run(["bash", "-n", str(runner)], check=True)

    result = subprocess.run(["bash", str(runner)], capture_output=True, text=True)
    assert result.returncode == 2
    assert "INPUT_DIR RESULT_DIR" in result.stderr


def test_runner_separates_final_and_intermediate_outputs():
    source = (ROOT / "run_pipeline.sh").read_text(encoding="utf-8")
    assert 'FINAL_DIR="$RESULT_DIR/final"' in source
    assert 'INTERMEDIATE_DIR="$RESULT_DIR/intermediate"' in source
    assert 'INVENTORY_DIR="$INTERMEDIATE_DIR/inventory"' in source
    assert 'ROUTING_DIR="$INTERMEDIATE_DIR/routing"' in source
    assert 'SECTIONS_DIR="$INTERMEDIATE_DIR/sections"' in source
    assert 'LOG_DIR="$INTERMEDIATE_DIR/logs"' in source
    assert '--output-dir "$FINAL_DIR"' in source
    assert '--annotations-dir "$FINAL_DIR/annotations"' in source
    assert "Annotations:   $FINAL_DIR/annotations" in source
    assert "Main JSONL:" not in source
    assert 'INVENTORY_HASH_JOBS="${INVENTORY_HASH_JOBS:-8}"' in source
    assert '--hash-jobs "$INVENTORY_HASH_JOBS"' in source
    assert '--hash-cache "$INVENTORY_DIR/data.hash_cache.jsonl"' in source


def test_runner_uses_sparse_gate_then_discogs_and_not_legacy_beats_gate():
    source = (ROOT / "run_pipeline.sh").read_text(encoding="utf-8")
    assert "scripts/fast_music_gate.py" in source
    assert "scripts/discogs_mir_infer.py" in source
    assert "FAST_GATE_CONFIG=" in source
    assert 'FAST_GATE_DECODE_WORKERS="${FAST_GATE_DECODE_WORKERS:-16}"' in source
    assert "FAST_GATE_BATCH_SIZE=" not in source
    assert '--batch-size "$FAST_GATE_BATCH_SIZE"' not in source
    assert "--beats-checkpoint" not in source
    assert '--vocal-song "$DISCOGS_VOCAL_SONG"' in source
    assert '--vocal-instrumental "$DISCOGS_VOCAL_INSTRUMENTAL"' in source
    assert source.index("scripts/fast_music_gate.py") < source.index(
        "scripts/discogs_mir_infer.py"
    )


def test_external_alm_can_overlap_section_asr_without_duplicate_run():
    source = (ROOT / "run_pipeline.sh").read_text(encoding="utf-8")
    assert (
        'PARALLEL_ASR_WITH_EXTERNAL_ALM="${PARALLEL_ASR_WITH_EXTERNAL_ALM:-1}"'
        in source
    )
    assert "run_section_asr() {" in source
    assert '"$ALM_MODE" == external' in source
    assert 'add_pipeline_job "$!" "section ASR and alignment"' in source
    assert 'if [[ "$asr_parallel_started" == 1 ]]; then' in source
    assert source.count('"$PIPELINE_ROOT/scripts/section_asr_infer.py"') == 1


def test_runner_disables_the_global_gpu_ceiling_by_default():
    source = (ROOT / "run_pipeline.sh").read_text(encoding="utf-8")
    assert 'PIPELINE_GPU_MAX_MEMORY_GIB="${PIPELINE_GPU_MAX_MEMORY_GIB:-0}"' in source
    assert (
        'OMNI_VLLM_MAX_MEMORY_GIB="${OMNI_VLLM_MAX_MEMORY_GIB:-$PIPELINE_GPU_MAX_MEMORY_GIB}"'
        in source
    )
    assert (
        'ASR_VLLM_MAX_MEMORY_GIB="${ASR_VLLM_MAX_MEMORY_GIB:-$PIPELINE_GPU_MAX_MEMORY_GIB}"'
        in source
    )
    assert "resolve_vllm_gpu_memory_utilization() {" in source
    assert "wait_for_gpu_capacity() {" in source
    assert '--gpu-memory-utilization "$resolved_gpu_memory_utilization"' in source
    assert "scripts/gpu_runtime${PYTHONPATH:+:$PYTHONPATH}" in source
    assert source.count("PIPELINE_TORCH_GPU_MAX_MEMORY_GIB=") >= 3
    assert 'PIPELINE_ORT_GPU_MAX_MEMORY_GIB="$DISCOGS_ORT_GPU_MEMORY_GIB"' in source
    assert '--gpu-max-memory-gib "$PIPELINE_GPU_MAX_MEMORY_GIB"' in source
    assert '--forced-aligner-reserve-gib "$ASR_FORCED_ALIGNER_RESERVE_GIB"' in source
    assert "no runner memory ceiling" in source
    assert 'VLLM_GPU_HEADROOM_GIB="${VLLM_GPU_HEADROOM_GIB:-4}"' in source
    assert (
        'ASR_FORCED_ALIGNER_RESERVE_GIB="${ASR_FORCED_ALIGNER_RESERVE_GIB:-8}"'
        in source
    )
    assert 'ASR_MIN_VLLM_MEMORY_GIB="${ASR_MIN_VLLM_MEMORY_GIB:-8}"' in source
    assert '--vllm-headroom-gib "$VLLM_GPU_HEADROOM_GIB"' in source
    assert '--minimum-vllm-memory-gib "$ASR_MIN_VLLM_MEMORY_GIB"' in source
    assert "--query-gpu=memory.free,memory.total" in source
    assert "--cfg reset_incompatible_output=true" in source


def test_cpu_mir_is_bounded_and_cannot_initialize_cuda():
    source = (ROOT / "run_pipeline.sh").read_text(encoding="utf-8")
    assert 'CPU_MIR_WORKERS="${CPU_MIR_WORKERS:-8}"' in source
    assert 'export CUDA_VISIBLE_DEVICES=""' in source

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


def test_runner_uses_resident_services_in_stage_order():
    source = (ROOT / "run_pipeline.sh").read_text(encoding="utf-8")
    for name in (
        "FAST_GATE_SERVICE_URL",
        "DISCOGS_MIR_SERVICE_URL",
        "MUSIC_CPU_SERVICE_URL",
        "STRUCTURE_RAW_SERVICE_URL",
        "SECTION_ASR_SERVICE_URL",
        "ALM_SERVICE_URL",
    ):
        assert name in source
    assert "scripts/service_batch_infer.py" in source
    assert "scripts/service_healthcheck.py" in source
    assert source.index("service_batch fast_gate") < source.index(
        "service_batch discogs_mir"
    )
    assert "scripts/fast_music_gate.py" not in source
    assert "scripts/discogs_mir_infer.py" not in source
    assert "MusicToolsPipeline/ray_inference.py" not in source
    assert "SongFormer/infer_jsonl.py" not in source
    assert "scripts/section_asr_infer.py" not in source


def test_runner_has_no_section_key_or_section_caption_stage():
    source = (ROOT / "run_pipeline.sh").read_text(encoding="utf-8")
    assert "section_key" not in source.lower()
    assert "section caption" not in source.lower()
    assert "section_caption" not in source.lower()


def test_runner_parallelizes_three_whole_track_service_calls():
    source = (ROOT / "run_pipeline.sh").read_text(encoding="utf-8")
    block = source[source.index('echo "[2-3/7]'): source.index('echo "[4/7]')]
    assert "service_batch music_cpu" in block
    assert "service_batch structure_raw" in block
    assert "service_batch alm" in block
    assert block.count(" &") >= 2
    assert "wait \"$cpu_pid\"" in block
    assert "wait \"$structure_pid\"" in block


def test_runner_never_manages_models_or_gpu_memory():
    source = (ROOT / "run_pipeline.sh").read_text(encoding="utf-8")
    for forbidden in (
        "CUDA_VISIBLE_DEVICES",
        "nvidia-smi",
        "torch",
        "onnxruntime",
        "start_local_alm",
        "stop_local_alm",
        "vllm serve",
    ):
        assert forbidden not in source

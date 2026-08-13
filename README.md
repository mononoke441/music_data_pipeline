# Music-Data-Pipeline

A dirty-media-to-training-data pipeline for mixed YouTube audio collections. It inventories and content-deduplicates audio/video assets, rejects non-music, routes accepted music into Song and Instrumental tracks, reuses shared whole-track analysis, applies two structure decoders, dynamically decodes sections in memory, and emits one validated JSON annotation per accepted track.

For a copy-paste Linux/NVIDIA setup, including the validated single-80GB path and a separate 2x48GB Qwen3-Omni service, see the [Chinese deployment guide](docs/DEPLOYMENT_ZH.md).

The recommended entry point is:

```bash
bash run_pipeline.sh /path/to/raw_media /path/to/result
```

Model paths, devices, concurrency, and structure thresholds are grouped at the top of `run_pipeline.sh` and remain environment-overridable. Music-gate thresholds are frozen in the versioned zero-training config at `MusicToolsPipeline/checkpoints/fast_gate_config.json`. Song/Instrumental routing uses the Discogs vocal score with explicit `DISCOGS_VOCAL_SONG=0.55` and `DISCOGS_VOCAL_INSTRUMENTAL=0.20` defaults. In the default `ALM_MODE=local`, the runner temporarily starts Qwen3-Omni after the music-analysis GPU stages, stops it after captioning, and then loads Qwen3-ASR plus ForcedAligner. Set `ALM_MODE=external` and `ALM_SERVER` to use an existing OpenAI-compatible audio service. The runner never creates persistent audio slices.

Runner-managed GPU stages are unlimited by default (`PIPELINE_GPU_MAX_MEMORY_GIB=0`); zero never installs a positive PyTorch allocator ceiling. Set a positive GiB value to restore a shared ceiling. Omni dynamically budgets vLLM from current free memory with `VLLM_GPU_HEADROOM_GIB=4` GiB by default. ASR uses the minimum of current free memory and any positive total/ASR caps, then reserves 8 GiB for ForcedAligner plus the same 4 GiB headroom. It waits instead of starting until at least `ASR_MIN_VLLM_MEMORY_GIB=8` GiB remains for vLLM. External ALM services and unrelated jobs remain outside the runner's control.

CPU MIR defaults to eight Ray workers and runs with CUDA hidden, so Chordino, BeatNet, and key extraction cannot create CUDA contexts while overlapping SongFormer.

With `ALM_MODE=external`, section key, remote section caption, and section ASR/alignment run concurrently by default. The ASR runner reuses one ffmpeg decode pool and prefetches the next batch during current-batch GPU inference. Set `PARALLEL_ASR_WITH_EXTERNAL_ALM=0` if the external ALM service actually shares the same local GPU.

The production music gate uses PANNs MobileNet with sparse 3×8-second sampling and a SHA-locked `fast_gate_config.json`; only gray-zone tracks run the 5×8-second second stage. The fixed config uses native 527-class AudioSet posteriors and contains no logistic head, fine-tuning, scaler, or probability calibrator. Accepted music alone proceeds to the separate cross-track-batched Discogs MIR and Song/Instrumental routing stage.

Gate weights can be downloaded or verified without starting services via `bash scripts/download_gate_assets.sh all`; observed SHA256 values are recorded, and trusted checksum mismatches fail closed. The runner validates the fixed config, checkpoint SHA256, and bundled PANNs source fingerprint before inference. Per-run gate and Discogs throughput is written to each stage's `runtime_metrics.json`. The runner also writes `intermediate/logs/pipeline_runtime.json` and `stage_timings.jsonl`, including whole-pipeline seconds per input/accepted track and every logical stage's elapsed time.

Consumable artifacts form four complete, mutually exclusive partitions: successful records under `RESULT_DIR/final/annotations/<source-relative-path>.json`, routing uncertainty in `review.jsonl`, deterministic media rejection in `rejected.jsonl`, and retryable per-item failures in `retry.jsonl`. A complete run with retry records exits normally and reports `partial_success`; service death, model initialization failure, corrupt non-tail JSON, or incomplete coverage still fails nonzero. Inventory, routing, whole-track analysis, section analysis, caches, and logs are grouped under `RESULT_DIR/intermediate/` for resume and diagnosis.

Key integrity and analysis behavior:

- `audio_id` is a streaming SHA256 of file content. Exact copies collapse to one canonical `audio_path`, with `duplicate_paths` and `duplicate_count` retained in inventory metadata.
- User-facing lookup is path-derived through `source_relpath`; it never hashes or scans audio. Use `scripts/find_annotation.py --input-root ROOT --result-dir RESULT --audio FILE`.
- Instrumental structure uses the full cosine self-similarity matrix, multi-scale Foote novelty, global CBM-style boundary selection, and global hierarchical section clustering.
- Confidence-aware post-processing preserves meaningful 2–8 second functional sections and strong-boundary short sections while still removing sub-2-second glitches.
- Section-key metadata includes duration-weighted `diatonic_chord_duration_ratio` and `tonic_chord_duration_ratio`.
- `SongFormer/infer_jsonl.py --save-frame-logits-dir DIR` optionally writes compressed per-frame boundary/function-logit NPZ sidecars; it is off by default.
- The runner finishes with `scripts/validate_pipeline_output.py`. Disabled stages never consume stale files and keep nullable fields with explicit statuses.
- Resume caches are bound to semantic input fingerprints and required payload completeness. Stage/model versions remain provenance only; section-derived caches additionally bind to the section plan/hash. Failed items remain retryable while completed items continue to be reused.

See [README_zh.md](README_zh.md) for architecture, model paths, commands, outputs, and validation.

# Music-Data-Pipeline

A resident-service audio annotation pipeline. It inventories and hashes media,
rejects non-music, routes Song versus Instrumental, runs global MIR and structure
analysis, and publishes one validated JSON annotation per accepted track.

The runners never import or construct inference models:

- `run_pipeline.sh` keeps stage barriers for batch comparison and resume.
- `run_stream_pipeline.sh` advances each audio independently and publishes it as
  soon as its own dependencies finish.

All model-bearing components stay resident behind `GET /healthz` and
`POST /v1/infer`: Fast Gate, Discogs ONNX, CPU MIR, SongFormer, Section ASR, and
the whole-track Omni proxy. Section key and section caption inference have been
removed. The annotation schema keeps global key, chords, beats/downbeats,
whole-track caption, structure, lyrics, ASR tokens, and alignment status.

Start and inspect services:

```bash
bash manage_model_services.sh start all --profile 80
bash manage_model_services.sh status all --profile 80
```

The supervisor supports `24`, `48`, and `80` GiB deployment profiles. Omni's
vLLM engine may run on an independent GPU node; `serve_omni.py` exposes the same
pipeline API while forwarding to that existing OpenAI-compatible engine.

Run either scheduler:

```bash
bash run_pipeline.sh /path/to/raw_media /path/to/result
bash run_stream_pipeline.sh /path/to/raw_media /path/to/result
```

Both produce four mutually exclusive final partitions:

- `final/annotations/<source-relative-path>.json`
- `final/review.jsonl`
- `final/rejected.jsonl`
- `final/retry.jsonl`

Streaming recovery lives in `intermediate/stream/state.sqlite3` (SQLite WAL).
Requests use deterministic idempotency IDs, successful matching stages are
reused, and annotations are written atomically per item. Services receive shared
audio paths and JSON only; the pipeline does not persist section audio slices.

See [README_zh.md](README_zh.md) and
[docs/DEPLOYMENT_ZH.md](docs/DEPLOYMENT_ZH.md) for configuration and deployment.

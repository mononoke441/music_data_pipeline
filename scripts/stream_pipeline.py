#!/usr/bin/env python3
"""Per-audio streaming orchestrator backed by idempotent inference services."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional

from annotation_storage import annotation_path, atomic_write_json, normalize_source_relpath
from calc_duration import (
    decode_probe,
    ffprobe_duration_seconds,
    inventory_stage_version,
    scan_media,
)
from pipeline_core import (
    ANNOTATION_SCHEMA_VERSION,
    PIPELINE_VERSION,
    extract_downbeats,
    iter_jsonl,
    postprocess_sections,
    stable_audio_id,
    write_jsonl,
)
from service_client import InferEnvelope, ServiceClient
from stream_state import StreamState, canonical_fingerprint
from structure_postprocess import voice_metrics


REMOTE_STAGES = (
    "fast_gate",
    "discogs_mir",
    "music_cpu",
    "structure_raw",
    "asr",
    "alm",
)
SERVICE_URL_ENV = {
    "fast_gate": "FAST_GATE_SERVICE_URL",
    "discogs_mir": "DISCOGS_MIR_SERVICE_URL",
    "music_cpu": "MUSIC_CPU_SERVICE_URL",
    "structure_raw": "STRUCTURE_RAW_SERVICE_URL",
    "asr": "SECTION_ASR_SERVICE_URL",
    "alm": "ALM_SERVICE_URL",
}
FULL_TRACK_STAGES = ("music_cpu", "structure_raw", "alm")
FINAL_STAGE_NAMES = (
    "music_gate",
    "discogs_mir",
    "alm",
    "music_cpu",
    "structure_raw",
    "structure_postprocess",
    "section_asr",
)


def _deep_overlay(target: MutableMapping[str, Any], source: Mapping[str, Any]) -> None:
    merge_fields = {
        "stage_status",
        "stage_errors",
        "stage_versions",
        "model_versions",
        "global_mir",
    }
    for key, value in source.items():
        if key in merge_fields and isinstance(value, Mapping):
            merged = dict(target.get(key) or {})
            merged.update(value)
            target[key] = merged
        else:
            target[key] = value


def _service_input(
    base: Mapping[str, Any], results: Iterable[Mapping[str, Any]]
) -> Dict[str, Any]:
    value = dict(base)
    for result in results:
        _deep_overlay(value, result)
    return value


def _gate_terminal(record: Mapping[str, Any]) -> Optional[str]:
    status = str(record.get("status") or "").lower()
    if status in {"review", "rejected"}:
        return status
    decision = str((record.get("music_gate") or {}).get("decision") or "").lower()
    if decision == "review":
        return "review"
    if decision in {"non_music", "rejected"}:
        return "rejected"
    return None


def _accepted_route(record: Mapping[str, Any]) -> Optional[str]:
    status = str(record.get("status") or "").lower()
    content_type = str(record.get("content_type") or "").lower()
    if status == "accepted" and content_type in {"song", "instrumental"}:
        return content_type
    return None


def _stage_result(
    state: StreamState, job_id: str, audio_id: str, stage: str
) -> Optional[Dict[str, Any]]:
    row = state.stage(job_id, audio_id, stage)
    if row is None or row.status != "succeeded" or row.result is None:
        return None
    return dict(row.result)


def _result_record(
    state: StreamState,
    job_id: str,
    source: Mapping[str, Any],
    stages: Iterable[str],
) -> Dict[str, Any]:
    results = []
    audio_id = str(source["audio_id"])
    for stage in stages:
        value = _stage_result(state, job_id, audio_id, stage)
        if value is not None:
            results.append(value)
    return _service_input(source, results)


def build_processed_record(
    source: Mapping[str, Any], stage_results: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    """Create the model-free structure result used as the Song ASR input."""

    routed = _service_input(
        source,
        (stage_results["fast_gate"], stage_results["discogs_mir"]),
    )
    duration = float(routed.get("duration") or 0.0)
    cpu_value = dict(stage_results["music_cpu"])
    structure_value = dict(stage_results["structure_raw"])
    raw_structure = (
        structure_value.get("structure_raw")
        or structure_value.get("songformer_result")
        or []
    )
    if duration <= 0 or not isinstance(raw_structure, list) or not raw_structure:
        raise ValueError("structure postprocess requires duration and structure_raw")
    joined = _service_input(routed, (cpu_value, structure_value))
    sections = postprocess_sections(
        raw_structure, duration, extract_downbeats(joined)
    )
    voice = routed.get("voice_analysis") or {}
    if not isinstance(voice, Mapping):
        voice = {}
    for section in sections:
        section.update(voice_metrics(section, voice))
    return _service_input(
        routed,
        (
            cpu_value,
            structure_value,
            {
                "sections": sections,
                "stage_status": {"structure_postprocess": "ok"},
            },
        ),
    )


def build_final_annotation(
    source: Mapping[str, Any], stage_results: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    """Run model-free structure postprocess, merge, and strict local validation."""

    routed = _service_input(
        source,
        (stage_results["fast_gate"], stage_results["discogs_mir"]),
    )
    content_type = _accepted_route(routed)
    if content_type is None:
        raise ValueError("final annotation input is not an accepted song/instrumental")
    duration = float(routed.get("duration") or 0.0)
    if duration <= 0:
        raise ValueError("final annotation duration must be positive")

    cpu_value = dict(stage_results["music_cpu"])
    structure_value = dict(stage_results["structure_raw"])
    alm_value = dict(stage_results["alm"])
    cpu_features = cpu_value.get("music_cpu") or cpu_value
    if not isinstance(cpu_features, Mapping):
        raise ValueError("music_cpu service returned no object payload")
    missing_cpu = [name for name in ("chords", "beatnet", "key") if not cpu_features.get(name)]
    if missing_cpu:
        raise ValueError(f"music_cpu payload is incomplete: missing={missing_cpu}")
    raw_structure = (
        structure_value.get("structure_raw")
        or structure_value.get("songformer_result")
        or []
    )
    if not isinstance(raw_structure, list) or not raw_structure:
        raise ValueError("structure service returned no structure_raw")
    processed = build_processed_record(source, stage_results)
    sections = [dict(section) for section in processed["sections"]]

    asr_value = dict(stage_results.get("asr") or {})
    asr_by_id = {
        str(section.get("section_id") or ""): dict(section)
        for section in asr_value.get("sections") or []
        if isinstance(section, Mapping)
    }
    for section in sections:
        if content_type == "instrumental":
            section["lyrics"] = None
            section["asr_tokens"] = []
            section["asr_status"] = "not_applicable"
            section["asr_error"] = None
            section["alignment_error"] = None
        else:
            section_id = str(section.get("section_id") or "")
            extra = asr_by_id.get(section_id)
            if extra is None:
                raise ValueError(f"ASR result is missing section_id={section_id}")
            for key, value in extra.items():
                if key not in {"section_id", "start", "end"}:
                    section[key] = value
            if section.get("asr_status") in {
                None,
                "not_run",
                "error",
                "decode_error",
                "asr_error",
                "alignment_error",
            }:
                raise ValueError(f"ASR result failed for section_id={section_id}")
            section.setdefault("asr_tokens", [])
            section.setdefault("lyrics", None)
            section.setdefault("asr_error", None)
            section.setdefault("alignment_error", None)

    global_mir = dict(routed.get("global_mir") or {})
    for name in ("chords", "beatnet", "key"):
        global_mir[name] = cpu_features[name]
    caption = alm_value.get("ALM_Caption") or alm_value.get("global_caption")
    if not str(caption or "").strip():
        raise ValueError("ALM service returned an empty global caption")
    if content_type == "song" and "asr" not in stage_results:
        raise ValueError("song is missing the whole-track ASR result")

    versions: Dict[str, Any] = {}
    for value in (routed, cpu_value, structure_value, alm_value, asr_value):
        versions.update(value.get("stage_versions") or {})
        versions.update(value.get("model_versions") or {})
    for name in ("alm", "section_asr", "forced_aligner"):
        versions.setdefault(name, None)
    statuses = {
        "music_gate": "ok",
        "discogs_mir": "ok",
        "alm": "ok",
        "music_cpu": "ok",
        "structure_raw": "ok",
        "structure_postprocess": "ok",
        "section_asr": "ok" if content_type == "song" else "not_run",
    }
    annotation = {
        "audio_id": str(source["audio_id"]),
        "audio_path": source.get("audio_path"),
        "source_relpath": normalize_source_relpath(str(source.get("source_relpath") or "")),
        "duration": duration,
        "status": "accepted",
        "content_type": content_type,
        "content_confidence": routed.get("content_confidence"),
        "music_gate": routed.get("music_gate") or {},
        "voice_analysis": routed.get("voice_analysis") or {},
        "global_caption": str(caption),
        "global_mir": global_mir,
        "raw_structure": raw_structure,
        "full_transcript": asr_value.get("full_transcript") if content_type == "song" else None,
        "sections": sections,
        "stage_status": statuses,
        "stage_errors": {name: None for name in FINAL_STAGE_NAMES},
        "model_versions": versions,
        "pipeline_version": PIPELINE_VERSION,
        "annotation_schema_version": ANNOTATION_SCHEMA_VERSION,
    }
    validate_stream_annotation(annotation)
    return annotation


def validate_stream_annotation(record: Mapping[str, Any]) -> None:
    required = {
        "audio_id",
        "audio_path",
        "source_relpath",
        "duration",
        "status",
        "content_type",
        "global_caption",
        "global_mir",
        "raw_structure",
        "full_transcript",
        "sections",
        "stage_status",
        "stage_errors",
        "model_versions",
        "pipeline_version",
        "annotation_schema_version",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"final annotation missing fields={sorted(missing)}")
    if record.get("status") != "accepted" or record.get("content_type") not in {
        "song",
        "instrumental",
    }:
        raise ValueError("final annotation has invalid route")
    if record.get("annotation_schema_version") != ANNOTATION_SCHEMA_VERSION:
        raise ValueError("final annotation has invalid annotation schema")
    duration = float(record.get("duration") or 0.0)
    previous = 0.0
    sections = list(record.get("sections") or [])
    if not sections:
        raise ValueError("final annotation has no sections")
    for section in sections:
        start, end = float(section["start"]), float(section["end"])
        if abs(start - previous) > 0.01 or end <= start:
            raise ValueError("final annotation section coverage is invalid")
        previous = end
        removed = {
            "key",
            "key_status",
            "key_error",
            "short_caption",
            "caption_status",
            "caption_error",
        } & set(section)
        if removed:
            raise ValueError(f"streaming annotation contains removed fields={sorted(removed)}")
    if abs(previous - duration) > 0.01:
        raise ValueError("final annotation sections do not cover the track")


def iter_inventory(input_root: str | Path, jobs: int = 8) -> Iterable[Dict[str, Any]]:
    """Yield each local no-model inventory result as soon as it completes."""
    root = Path(input_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"input root is not a directory: {root}")
    paths = scan_media(root)
    worker_count = max(1, min(int(jobs), len(paths) or 1))
    stage_version = inventory_stage_version(1.0)
    def inspect(path: str) -> Dict[str, Any]:
        audio_id = stable_audio_id(path, str(root))
        duration = ffprobe_duration_seconds(path)
        decode_ok, decode_error = decode_probe(path, 1.0) if duration else (False, "duration_probe_failed")
        return {
            "audio_id": audio_id,
            "audio_path": path,
            "source_relpath": Path(path).relative_to(root).as_posix(),
            "duration": round(float(duration), 6) if duration else None,
            "decode_status": "ok" if decode_ok else "failed",
            "error": None if decode_ok else decode_error,
            "pipeline_version": PIPELINE_VERSION,
            "stage_versions": {"inventory": stage_version},
        }

    seen: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(inspect, path): path for path in paths}
        for future in as_completed(futures):
            record = future.result()
            audio_id = str(record["audio_id"])
            previous = seen.get(audio_id)
            if previous is not None:
                raise ValueError(
                    f"duplicate SHA256 audio_id={audio_id}: "
                    f"paths={sorted([previous, str(record['audio_path'])])!r}; "
                    "streaming input must be pre-deduplicated"
                )
            seen[audio_id] = str(record["audio_path"])
            yield record


def build_inventory(input_root: str | Path, jobs: int = 8) -> list[Dict[str, Any]]:
    return list(iter_inventory(input_root, jobs))


class StreamingPipeline:
    def __init__(
        self,
        *,
        job_id: str,
        result_dir: str | Path,
        state: StreamState,
        clients: Mapping[str, Any],
        max_inflight: int = 64,
        on_publish: Optional[Callable[[Mapping[str, Any], Path], None]] = None,
    ) -> None:
        if not 1 <= int(max_inflight) <= 64:
            raise ValueError("max_inflight must be between 1 and 64")
        missing = set(REMOTE_STAGES) - set(clients)
        if missing:
            raise ValueError(f"missing service clients={sorted(missing)}")
        self.job_id = str(job_id)
        self.result_dir = Path(result_dir)
        self.final_dir = self.result_dir / "final"
        self.annotations_dir = self.final_dir / "annotations"
        self.state = state
        self.clients = dict(clients)
        self.max_inflight = int(max_inflight)
        self.on_publish = on_publish
        self._health_checked: set[str] = set()

    def _health(self, stage: str) -> None:
        if stage not in self._health_checked:
            self.clients[stage].healthz()
            self._health_checked.add(stage)

    def _submit(
        self,
        executor: ThreadPoolExecutor,
        inflight: MutableMapping[Future[Any], tuple[str, str, str]],
        stage: str,
        source: Mapping[str, Any],
        service_record: Mapping[str, Any],
    ) -> bool:
        audio_id = str(source["audio_id"])
        fingerprint = canonical_fingerprint(stage, service_record)
        row = self.state.prepare_stage(
            self.job_id, audio_id, stage, fingerprint
        )
        if row.status != "pending":
            return False
        claimed = self.state.claim_stage(
            self.job_id, audio_id, stage, fingerprint
        )
        if claimed is None:
            return False
        self._health(stage)
        future = executor.submit(
            self.clients[stage].infer,
            job_id=self.job_id,
            request_id=claimed.request_id,
            audio_id=audio_id,
            audio_path=str(source.get("audio_path") or ""),
            input_fingerprint=fingerprint,
            record=dict(service_record),
        )
        inflight[future] = (audio_id, stage, claimed.request_id)
        return True

    def _ready_stages(
        self, source: Mapping[str, Any]
    ) -> tuple[list[tuple[str, Dict[str, Any]]], Optional[str]]:
        audio_id = str(source["audio_id"])
        gate = self.state.stage(self.job_id, audio_id, "fast_gate")
        if gate is None or gate.status == "pending":
            return [("fast_gate", dict(source))], None
        if gate.status == "failed":
            return [], "failed"
        if gate.status != "succeeded" or gate.result is None:
            return [], None
        after_gate = _service_input(source, (gate.result,))
        terminal = _gate_terminal(after_gate)
        if terminal:
            return [], terminal

        discogs = self.state.stage(self.job_id, audio_id, "discogs_mir")
        if discogs is None or discogs.status == "pending":
            return [("discogs_mir", after_gate)], None
        if discogs.status == "failed":
            return [], "failed"
        if discogs.status != "succeeded" or discogs.result is None:
            return [], None
        routed = _service_input(after_gate, (discogs.result,))
        terminal = _gate_terminal(routed)
        if terminal:
            return [], terminal
        content_type = _accepted_route(routed)
        if content_type is None:
            raise ValueError(
                f"Discogs service returned invalid route for audio_id={audio_id}"
            )

        ready = []
        for stage in FULL_TRACK_STAGES:
            row = self.state.stage(self.job_id, audio_id, stage)
            if row is None or row.status == "pending":
                ready.append((stage, routed))
            elif row.status == "failed":
                return [], "failed"
        if ready:
            return ready, None
        if any(
            (row := self.state.stage(self.job_id, audio_id, stage)) is None
            or row.status != "succeeded"
            for stage in FULL_TRACK_STAGES
        ):
            return [], None
        if content_type == "song":
            asr = self.state.stage(self.job_id, audio_id, "asr")
            if asr is not None and asr.status == "failed":
                return [], "failed"
            if asr is None or asr.status == "pending":
                results = {
                    stage: self.state.stage(self.job_id, audio_id, stage).result
                    for stage in ("fast_gate", "discogs_mir", *FULL_TRACK_STAGES)
                }
                processed = build_processed_record(source, results)  # type: ignore[arg-type]
                return [("asr", processed)], None
        return ready, None

    def _can_publish(self, source: Mapping[str, Any]) -> bool:
        audio_id = str(source["audio_id"])
        routed = _result_record(
            self.state, self.job_id, source, ("fast_gate", "discogs_mir")
        )
        content_type = _accepted_route(routed)
        if content_type is None:
            return False
        required = list(FULL_TRACK_STAGES)
        if content_type == "song":
            required.append("asr")
        return all(
            (row := self.state.stage(self.job_id, audio_id, stage)) is not None
            and row.status == "succeeded"
            for stage in required
        )

    def _publish(self, source: Mapping[str, Any]) -> Path:
        audio_id = str(source["audio_id"])
        results: Dict[str, Mapping[str, Any]] = {}
        for stage in ("fast_gate", "discogs_mir", *FULL_TRACK_STAGES, "asr"):
            value = _stage_result(self.state, self.job_id, audio_id, stage)
            if value is not None:
                results[stage] = value
        annotation = build_final_annotation(source, results)
        target = annotation_path(self.annotations_dir, str(annotation["source_relpath"]))
        atomic_write_json(target, annotation)
        if self.on_publish is not None:
            self.on_publish(annotation, target)
        self.state.set_item_status(self.job_id, audio_id, "published")
        return target

    def run(self, inventory: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.final_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        self.state.recover_running(self.job_id)

        # A committed state row without its atomically published file is repaired.
        for item in self.state.items(self.job_id):
            if item["status"] != "published":
                continue
            target = annotation_path(
                self.annotations_dir, str(item["record"].get("source_relpath") or "")
            )
            if not target.is_file():
                self.state.set_item_status(self.job_id, item["audio_id"], "pending")

        inventory_queue: queue.Queue[object] = queue.Queue(self.max_inflight)
        inventory_done = object()

        def produce_inventory() -> None:
            try:
                for value in inventory:
                    inventory_queue.put(dict(value))
            except BaseException as error:
                inventory_queue.put(error)
            finally:
                inventory_queue.put(inventory_done)

        producer = threading.Thread(target=produce_inventory, daemon=True)
        producer.start()
        producer_finished = False
        seen_inventory_ids: set[str] = set()
        inflight: Dict[Future[Any], tuple[str, str, str]] = {}
        with ThreadPoolExecutor(max_workers=self.max_inflight) as executor:
            while True:
                progress = False
                while True:
                    try:
                        incoming = inventory_queue.get_nowait()
                    except queue.Empty:
                        break
                    if incoming is inventory_done:
                        producer_finished = True
                        progress = True
                        continue
                    if isinstance(incoming, BaseException):
                        raise incoming
                    if not isinstance(incoming, Mapping):
                        raise TypeError("inventory iterator yielded a non-object")
                    record = dict(incoming)
                    audio_id = str(record.get("audio_id") or "")
                    if audio_id in seen_inventory_ids:
                        raise ValueError(
                            f"duplicate SHA256 audio_id in input: {audio_id}"
                        )
                    seen_inventory_ids.add(audio_id)
                    self.state.register_items(self.job_id, [record])
                    if record.get("decode_status") != "ok":
                        self.state.set_item_status(
                            self.job_id,
                            audio_id,
                            "failed",
                            str(record.get("error") or "inventory_decode_failed"),
                        )
                    progress = True

                all_items = self.state.items(self.job_id)
                if producer_finished:
                    stale_ids = {
                        str(item["audio_id"])
                        for item in all_items
                        if str(item["audio_id"]) not in seen_inventory_ids
                    }
                    if stale_ids:
                        raise ValueError(
                            "stream state contains audio_id values absent from the "
                            f"current inventory: {sorted(stale_ids)[:10]}"
                        )
                items = [
                    item
                    for item in all_items
                    if str(item["audio_id"]) in seen_inventory_ids
                ]
                for item in items:
                    if len(inflight) >= self.max_inflight:
                        break
                    if item["status"] in {"published", "review", "rejected", "failed"}:
                        continue
                    source = item["record"]
                    ready, terminal = self._ready_stages(source)
                    if terminal in {"review", "rejected"}:
                        self.state.set_item_status(
                            self.job_id, item["audio_id"], terminal
                        )
                        progress = True
                        continue
                    if terminal == "failed":
                        self.state.set_item_status(
                            self.job_id,
                            item["audio_id"],
                            "failed",
                            "one or more inference stages failed",
                        )
                        progress = True
                        continue
                    if self._can_publish(source):
                        try:
                            self._publish(source)
                        except Exception as error:
                            self.state.set_item_status(
                                self.job_id,
                                item["audio_id"],
                                "failed",
                                f"{type(error).__name__}: {error}",
                            )
                        progress = True
                        continue
                    for stage, service_record in ready:
                        if len(inflight) >= self.max_inflight:
                            break
                        progress = (
                            self._submit(
                                executor, inflight, stage, source, service_record
                            )
                            or progress
                        )

                if inflight:
                    done, _ = wait(
                        tuple(inflight), timeout=0.05, return_when=FIRST_COMPLETED
                    )
                    for future in done:
                        audio_id, stage, request_id = inflight.pop(future)
                        try:
                            response = future.result()
                            if isinstance(response, InferEnvelope):
                                result = response.record
                                model_fingerprint = response.model_fingerprint
                                elapsed_seconds = response.elapsed_seconds
                            elif isinstance(response, Mapping):
                                result = dict(response)
                                model_fingerprint = None
                                elapsed_seconds = None
                            else:
                                raise TypeError("service client returned an invalid response")
                            self.state.finish_stage(
                                self.job_id,
                                audio_id,
                                stage,
                                request_id,
                                result,
                                model_fingerprint,
                                elapsed_seconds,
                            )
                        except Exception as error:
                            error_envelope = getattr(error, "envelope", None)
                            self.state.fail_stage(
                                self.job_id,
                                audio_id,
                                stage,
                                request_id,
                                f"{type(error).__name__}: {error}",
                                getattr(error_envelope, "model_fingerprint", None),
                                getattr(error_envelope, "elapsed_seconds", None),
                            )
                        progress = True
                    if done:
                        continue
                pending = [
                    item
                    for item in items
                    if item["status"] == "pending"
                ]
                if producer_finished and not pending and not inflight:
                    break
                if not progress and producer_finished and not inflight:
                    raise RuntimeError(
                        f"stream scheduler stalled with {len(pending)} pending items"
                    )
                if not progress and not inflight and not producer_finished:
                    try:
                        incoming = inventory_queue.get(timeout=0.05)
                        inventory_queue.put(incoming)
                    except queue.Empty:
                        pass

        producer.join(timeout=1.0)

        final_items = self.state.items(self.job_id)
        partitions: Dict[str, list[Dict[str, Any]]] = {
            "review": [],
            "rejected": [],
            "retry": [],
        }
        accepted: list[Dict[str, Any]] = []
        for item in final_items:
            status = item["status"]
            source = item["record"]
            routed = _result_record(
                self.state,
                self.job_id,
                source,
                ("fast_gate", "discogs_mir"),
            )
            if _accepted_route(routed) is not None:
                accepted.append(routed)
            if status in {"review", "rejected"}:
                partitions[status].append(routed)
            elif status == "failed":
                audio_id = str(source["audio_id"])
                failed_stage = "inventory"
                detail = item.get("error") or "streaming item failed"
                for stage in REMOTE_STAGES:
                    row = self.state.stage(self.job_id, audio_id, stage)
                    if row is not None and row.status == "failed":
                        failed_stage = stage
                        detail = row.error or detail
                        break
                partitions["retry"].append(
                    {
                        "audio_id": audio_id,
                        "audio_path": source.get("audio_path"),
                        "source_relpath": source.get("source_relpath"),
                        "failure_stage": failed_stage,
                        "retryable": True,
                        "stage_status": {failed_stage: "error"},
                        "stage_errors": {failed_stage: detail},
                        "pipeline_version": PIPELINE_VERSION,
                        "semantic_input_fingerprint": canonical_fingerprint(
                            failed_stage, source
                        ),
                    }
                )
        for name, values in partitions.items():
            write_jsonl(self.final_dir / f"{name}.jsonl", values)
        stream_dir = self.result_dir / "intermediate" / "stream"
        inventory_path = stream_dir / "inventory.jsonl"
        accepted_path = stream_dir / "accepted.jsonl"
        write_jsonl(inventory_path, (item["record"] for item in final_items))
        write_jsonl(accepted_path, accepted)
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("validate_pipeline_output.py")),
                "--inventory",
                str(inventory_path),
                "--base",
                str(accepted_path),
                "--annotations-dir",
                str(self.annotations_dir),
                "--review",
                str(self.final_dir / "review.jsonl"),
                "--rejected",
                str(self.final_dir / "rejected.jsonl"),
                "--retry",
                str(self.final_dir / "retry.jsonl"),
                "--alm-enabled",
                "--section-asr-enabled",
            ],
            check=True,
        )
        return {
            "total": len(final_items),
            "published": sum(item["status"] == "published" for item in final_items),
            "review": len(partitions["review"]),
            "rejected": len(partitions["rejected"]),
            "retry": len(partitions["retry"]),
        }


def _default_job_id(input_root: str | Path) -> str:
    value = os.path.abspath(os.path.expanduser(os.fspath(input_root)))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _load_inventory(path: str | Path) -> list[Dict[str, Any]]:
    return [dict(value) for value in iter_jsonl(path)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("result_dir")
    parser.add_argument("--job-id")
    parser.add_argument("--state-db")
    parser.add_argument("--inventory-jsonl", help="Testing/import hook for a prebuilt inventory")
    parser.add_argument("--inventory-jobs", type=int, default=8)
    parser.add_argument("--max-inflight", type=int, default=64)
    parser.add_argument("--service-timeout", type=float, default=1800.0)
    parser.add_argument("--service-retries", type=int, default=3)
    for stage in REMOTE_STAGES:
        environment_name = SERVICE_URL_ENV[stage]
        parser.add_argument(
            f"--{stage.replace('_', '-')}-url",
            default=os.environ.get(environment_name),
        )
    args = parser.parse_args()

    missing = [
        stage
        for stage in REMOTE_STAGES
        if not getattr(args, f"{stage}_url")
    ]
    if missing:
        parser.error(f"missing service URL(s): {', '.join(missing)}")
    input_root = Path(args.input_dir).expanduser().resolve()
    result_dir = Path(args.result_dir).expanduser().resolve()
    inventory = (
        _load_inventory(args.inventory_jsonl)
        if args.inventory_jsonl
        else iter_inventory(input_root, args.inventory_jobs)
    )

    clients = {
        stage: ServiceClient(
            getattr(args, f"{stage}_url"),
            timeout=args.service_timeout,
            retries=args.service_retries,
        )
        for stage in REMOTE_STAGES
    }
    state_path = (
        Path(args.state_db)
        if args.state_db
        else result_dir / "intermediate" / "stream" / "state.sqlite3"
    )
    job_id = args.job_id or _default_job_id(input_root)
    with StreamState(state_path) as state:
        summary = StreamingPipeline(
            job_id=job_id,
            result_dir=result_dir,
            state=state,
            clients=clients,
            max_inflight=args.max_inflight,
        ).run(inventory)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    if summary["retry"]:
        sys.exit(2)


if __name__ == "__main__":
    main()

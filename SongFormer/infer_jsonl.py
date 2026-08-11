#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import importlib
import json
import math
import multiprocessing as mp
import os
import queue as queue_module
import time
from argparse import Namespace
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any

# monkey patch to fix issues in msaf
import scipy
import numpy as np

scipy.inf = np.inf

import librosa
import torch
from ema_pytorch import EMA
from loguru import logger
from muq import MuQ
from musicfm.model.musicfm_25hz import MusicFM25Hz
from omegaconf import OmegaConf

PIPELINE_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(PIPELINE_SCRIPTS_DIR))
from pipeline_progress import pipeline_tqdm  # noqa: E402

if os.environ.get("PIPELINE_QUIET_LOGS", "1") == "1":
    logger.remove()
    logger.add(sys.stderr, level="ERROR")

MUSICFM_HOME_PATH = os.path.join("ckpts", "MusicFM")

BEFORE_DOWNSAMPLING_FRAME_RATES = 25
AFTER_DOWNSAMPLING_FRAME_RATES = 8.333

DATASET_LABEL = "SongForm-HX-8Class"
DATASET_IDS = [5]

TIME_DUR = 420
INPUT_SAMPLING_RATE = 24000

from dataset.label2id import DATASET_ID_ALLOWED_LABEL_IDS, DATASET_LABEL_TO_DATASET_ID
from postprocessing.functional import (
    postprocess_functional_structure_detailed,
)
from instrumental_structure import infer_instrumental_structure
from audio_prefetch import iter_prefetched_queue
from embedding_batch import extract_muq_musicfm_chunks


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            yield line_no, json.loads(line)


def uid_from_obj(obj: dict, line_no: int) -> str:
    """Return a filename-safe stable uid, preferring the pipeline audio_id."""

    audio_id = str(obj.get("audio_id", "")).strip()
    if not audio_id:
        return f"line-{line_no:09d}"
    return hashlib.sha1(audio_id.encode("utf-8")).hexdigest()


def count_lines(path: str) -> int:
    return sum(1 for _line_no, _record in iter_jsonl(path))


def load_checkpoint(checkpoint_path, device=None):
    """Load checkpoint from path"""
    if device is None:
        device = "cpu"

    if checkpoint_path.endswith(".pt"):
        checkpoint = torch.load(checkpoint_path, map_location=device)
    elif checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        checkpoint = {"model_ema": load_file(checkpoint_path, device=device)}
    else:
        raise ValueError("Unsupported checkpoint format. Use .pt or .safetensors")
    return checkpoint


def rule_post_processing(msa_list):
    if len(msa_list) <= 2:
        return msa_list

    result = msa_list.copy()

    while len(result) > 2:
        first_duration = result[1][0] - result[0][0]
        if first_duration < 1.0 and len(result) > 2:
            result[0] = (result[0][0], result[1][1])
            result = [result[0]] + result[2:]
        else:
            break

    while len(result) > 2:
        last_label_duration = result[-1][0] - result[-2][0]
        if last_label_duration < 1.0:
            result = result[:-2] + [result[-1]]
        else:
            break

    while len(result) > 2:
        if result[0][1] == result[1][1] and result[1][0] <= 10.0:
            result = [(result[0][0], result[0][1])] + result[2:]
        else:
            break

    while len(result) > 2:
        last_duration = result[-1][0] - result[-2][0]
        if result[-2][1] == result[-3][1] and last_duration <= 10.0:
            result = result[:-2] + [result[-1]]
        else:
            break

    return result


def _cache_meta_path(prediction_path: str) -> str:
    return prediction_path[:-5] + ".meta.json"


def _fsync_directory(path: str | Path) -> None:
    """Durably publish a prior atomic rename when the filesystem supports it."""

    directory = Path(path)
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_json(path: str | Path, value: Any, *, indent: int | None = None) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                indent=indent,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_prediction_cache(
    prediction_path: str,
    structure: Sequence[Any],
    metadata: Mapping[str, Any],
) -> None:
    """Commit the prediction first and its metadata as the completion marker."""

    _atomic_write_json(prediction_path, list(structure), indent=4)
    _atomic_write_json(_cache_meta_path(prediction_path), dict(metadata))


def _load_cache_metadata(prediction_path: str) -> dict:
    try:
        with open(_cache_meta_path(prediction_path), "r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _load_cached_structure(prediction_path: str) -> list[Any] | None:
    """Return a non-empty, parseable structure cache or ``None``."""

    try:
        with open(prediction_path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(value, list) or not value:
        return None
    for section in value:
        if isinstance(section, Mapping):
            label = str(section.get("label", "")).strip()
            try:
                start = float(section["start"])
                end = float(section["end"])
            except (KeyError, TypeError, ValueError):
                return None
            if not label or not math.isfinite(start) or not math.isfinite(end) or end <= start:
                return None
        elif isinstance(section, (list, tuple)) and len(section) >= 2:
            try:
                start = float(section[0])
            except (TypeError, ValueError):
                return None
            if not math.isfinite(start) or not str(section[1]).strip():
                return None
        else:
            return None
    return value


def _cache_is_reusable(
    prediction_path: str,
    content_type: str | None = None,
    require_frame_logits: bool = False,
) -> bool:
    if not os.path.isfile(prediction_path) or _load_cached_structure(prediction_path) is None:
        return False
    metadata = _load_cache_metadata(prediction_path)
    cached_content_type = str(metadata.get("content_type", "")).strip().lower()
    if cached_content_type not in {"song", "instrumental"}:
        return False
    if metadata.get("state") not in (None, "ok"):
        return False
    if content_type is not None and cached_content_type != content_type:
        return False
    if require_frame_logits and content_type == "song":
        artifact = metadata.get("frame_logits") or {}
        if not artifact.get("path") or not os.path.isfile(str(artifact["path"])):
            return False
    return True


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _save_frame_logits(
    output_dir: str,
    uid: str,
    function_logits: np.ndarray,
    boundary_logits: np.ndarray,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    target = os.path.abspath(os.path.join(output_dir, f"{uid}.npz"))
    temporary = target + ".tmp.npz"
    np.savez_compressed(
        temporary,
        function_logits=np.asarray(function_logits, dtype=np.float16),
        boundary_logits=np.asarray(boundary_logits, dtype=np.float16),
        frame_rate=np.asarray(AFTER_DOWNSAMPLING_FRAME_RATES, dtype=np.float32),
    )
    with open(temporary, "rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    _fsync_directory(Path(target).parent)
    return {
        "path": target,
        "sha256": _sha256_file(target),
        "format": "npz",
        "dtype": "float16",
        "frame_rate": AFTER_DOWNSAMPLING_FRAME_RATES,
        "function_shape": list(function_logits.shape),
        "boundary_shape": list(boundary_logits.shape),
    }


def get_processed_uids(output_dir: str):
    """uids that already have uid.json in output_dir"""
    p = Path(output_dir)
    if not p.exists():
        return set()
    ret = set()
    for x in p.iterdir():
        if (
            x.is_file()
            and x.suffix == ".json"
            and not x.name.endswith(".meta.json")
            and _cache_is_reusable(str(x))
        ):
            ret.add(x.stem)
    return ret


def build_tasks_from_jsonl(
    input_jsonl: str,
    audio_key: str,
    output_dir: str,
    require_frame_logits: bool = False,
):
    """
    Return list of tasks: (uid, audio_path, content_type).
    """
    tasks = []
    seen_uids: set[str] = set()

    for line_no, obj in iter_jsonl(input_jsonl):
        uid = uid_from_obj(obj, line_no)
        if uid in seen_uids:
            raise ValueError(
                f"duplicate SongFormer task identity at line {line_no}: uid={uid}"
            )
        seen_uids.add(uid)
        audio_path = obj.get(audio_key, None)
        if not audio_path:
            # still keep progress stable, but skip
            logger.warning(f"missing key '{audio_key}' at line {line_no}, skip")
            continue
        content_type = str(obj.get("content_type", "song")).strip().lower()
        if content_type not in {"song", "instrumental"}:
            logger.warning(
                f"unsupported content_type={content_type!r} at line {line_no}, skip"
            )
            continue
        out_path = os.path.join(output_dir, f"{uid}.json")
        if _cache_is_reusable(
            out_path,
            content_type,
            require_frame_logits=require_frame_logits,
        ):
            continue
        tasks.append((uid, audio_path, content_type))

    return tasks


def _initialize_worker_models(rank: int, args: Namespace):
    device = f"cuda:{rank}"

    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    muq = MuQ.from_pretrained("ckpts/MuQ-large-msd-iter")
    muq = muq.to(device).eval()

    musicfm = MusicFM25Hz(
        is_flash=False,
        stat_path=os.path.join(MUSICFM_HOME_PATH, "msd_stats.json"),
        model_path=os.path.join(MUSICFM_HOME_PATH, "pretrained_msd.pt"),
    )
    musicfm = musicfm.to(device).eval()

    module = importlib.import_module("models." + str(args.model))
    model_class = getattr(module, "Model")
    hp = OmegaConf.load(os.path.join("configs", args.config_path))
    model = model_class(hp)

    ckpt = load_checkpoint(
        checkpoint_path=os.path.join("ckpts", args.checkpoint), device="cpu"
    )
    if ckpt.get("model_ema", None) is not None:
        logger.info(f"[rank {rank}] Loading EMA model parameters")
        model_ema = EMA(model, include_online_model=False)
        model_ema.load_state_dict(ckpt["model_ema"])
        model.load_state_dict(model_ema.ema_model.state_dict())
    else:
        logger.info(
            f"[rank {rank}] No EMA model parameters found, using original model"
        )
        model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    dataset_id2label_mask = {}
    for key, allowed_ids in DATASET_ID_ALLOWED_LABEL_IDS.items():
        dataset_id2label_mask[key] = np.ones(args.num_classes, dtype=bool)
        dataset_id2label_mask[key][allowed_ids] = False

    os.makedirs(args.output_dir, exist_ok=True)
    return device, muq, musicfm, model, hp, dataset_id2label_mask


def inference_worker(
    worker_id: int,
    rank: int,
    queue_input: mp.Queue,
    queue_output: mp.Queue,
    args: Namespace,
):
    """
    Each task is (uid, audio_path, content_type).
    Writes output to output_dir/uid.json
    Always puts one message into queue_output per task:
      {"uid": uid, "ok": True} or {"uid": uid, "ok": False, "error": "..."}
    """
    try:
        (
            device,
            muq,
            musicfm,
            model,
            hp,
            dataset_id2label_mask,
        ) = _initialize_worker_models(rank, args)
    except BaseException as error:
        queue_output.put(
            {
                "type": "worker_init_error",
                "worker_id": worker_id,
                "rank": rank,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        logger.exception(f"worker {worker_id} rank {rank} failed to initialize")
        raise

    queue_output.put(
        {"type": "worker_ready", "worker_id": worker_id, "rank": rank}
    )
    num_classes = args.num_classes

    def send_result(uid: str, ok: bool, **extra: Any) -> None:
        queue_output.put(
            {
                "type": "result",
                "worker_id": worker_id,
                "rank": rank,
                "uid": uid,
                "ok": bool(ok),
                **extra,
            }
        )

    def load_audio_task(item):
        uid, audio_path, content_type = item
        out_path = os.path.join(args.output_dir, f"{uid}.json")
        if _cache_is_reusable(
            out_path,
            content_type,
            require_frame_logits=bool(args.save_frame_logits_dir),
        ):
            return {
                "item": item,
                "state": "skipped",
                "wav": None,
                "sample_rate": None,
                "error": None,
            }
        try:
            wav, sample_rate = librosa.load(audio_path, sr=INPUT_SAMPLING_RATE)
            return {
                "item": item,
                "state": "loaded",
                "wav": wav,
                "sample_rate": sample_rate,
                "error": None,
            }
        except Exception as error:
            return {
                "item": item,
                "state": "error",
                "wav": None,
                "sample_rate": None,
                "error": f"{type(error).__name__}: {error}",
            }

    logged_feature_contract = False
    with torch.no_grad():
        for loaded in iter_prefetched_queue(
            queue_input,
            load_audio_task,
            prefetch=args.decode_prefetch,
        ):
            item = loaded["item"]
            uid, audio_path, content_type = item
            out_path = os.path.join(args.output_dir, f"{uid}.json")

            if loaded["state"] == "skipped":
                send_result(uid, True, skipped=True)
                continue
            if loaded["state"] == "error":
                send_result(uid, False, error=loaded["error"])
                logger.error(
                    f"process {rank} decode error\nuid={uid}\naudio={audio_path}\n"
                    f"{loaded['error']}"
                )
                continue

            try:
                audio = torch.tensor(loaded["wav"]).to(device)

                if content_type == "instrumental":
                    structure = infer_instrumental_structure(
                        audio=audio,
                        muq=muq,
                        musicfm=musicfm,
                        embedding_batch_size=args.embedding_chunk_batch_size,
                    )
                    _write_prediction_cache(
                        out_path,
                        structure,
                        {
                            "state": "ok",
                            "stage_version": args.stage_version,
                            "content_type": content_type,
                        },
                    )
                    send_result(uid, True)
                    continue

                win_size = args.win_size
                hop_size = args.hop_size

                total_len = (
                    (audio.shape[0] // INPUT_SAMPLING_RATE) // TIME_DUR
                ) * TIME_DUR + TIME_DUR
                total_frames = math.ceil(total_len * AFTER_DOWNSAMPLING_FRAME_RATES)

                logits = {
                    "function_logits": np.zeros(
                        [total_frames, num_classes], dtype=np.float32
                    ),
                    "boundary_logits": np.zeros([total_frames], dtype=np.float32),
                }
                logits_num = {
                    "function_logits": np.zeros(
                        [total_frames, num_classes], dtype=np.float32
                    ),
                    "boundary_logits": np.zeros([total_frames], dtype=np.float32),
                }

                lens = 0
                i = 0
                while True:
                    start_idx = i * INPUT_SAMPLING_RATE
                    end_idx = min((i + win_size) * INPUT_SAMPLING_RATE, audio.shape[-1])
                    if start_idx >= audio.shape[-1]:
                        break
                    if end_idx - start_idx <= 1024:
                        i += hop_size
                        continue

                    audio_seg = audio[start_idx:end_idx]

                    # MuQ embedding (420s)
                    muq_output = muq(audio_seg.unsqueeze(0), output_hidden_states=True)
                    muq_embd_420s = muq_output["hidden_states"][10]
                    del muq_output

                    # MusicFM embedding (420s)
                    _, musicfm_hidden_states = musicfm.get_predictions(
                        audio_seg.unsqueeze(0)
                    )
                    musicfm_embd_420s = musicfm_hidden_states[10]
                    del musicfm_hidden_states

                    # Training used 30-second local embeddings alongside the
                    # 420-second global embeddings.  Use the same resident
                    # MuQ/MusicFM instances for both contexts; repeating the
                    # global tensors into the local slots is not equivalent.
                    local_chunks = []
                    local_samples = 30 * INPUT_SAMPLING_RATE
                    for local_start in range(start_idx, end_idx, local_samples):
                        local_end = min(local_start + local_samples, end_idx)
                        local_audio = audio[local_start:local_end]
                        if local_audio.numel() <= 1024:
                            continue
                        local_chunks.append(local_audio)
                    wrapped_muq_30s, wrapped_musicfm_30s = extract_muq_musicfm_chunks(
                        local_chunks,
                        muq,
                        musicfm,
                        batch_size=args.embedding_chunk_batch_size,
                    )
                    if not wrapped_muq_30s or not wrapped_musicfm_30s:
                        raise RuntimeError(
                            "local 30-second feature extraction produced no embeddings"
                        )
                    muq_embd_30s = torch.concatenate(wrapped_muq_30s, dim=1)
                    musicfm_embd_30s = torch.concatenate(wrapped_musicfm_30s, dim=1)

                    all_embds = [
                        musicfm_embd_30s,
                        muq_embd_30s,
                        musicfm_embd_420s,
                        muq_embd_420s,
                    ]

                    # align lengths
                    if len(all_embds) > 1:
                        embd_lens = [x.shape[1] for x in all_embds]
                        max_embd_len = max(embd_lens)
                        min_embd_len = min(embd_lens)
                        if abs(max_embd_len - min_embd_len) > 4:
                            raise ValueError(
                                f"Embedding shapes differ too much: {max_embd_len} vs {min_embd_len}"
                            )
                        for idx in range(len(all_embds)):
                            all_embds[idx] = all_embds[idx][:, :min_embd_len, :]

                    if not logged_feature_contract:
                        logger.info(
                            f"[rank {rank}] SongFormer feature contract "
                            f"local/global shapes={[tuple(value.shape) for value in all_embds]} "
                            f"concat_dim={sum(int(value.shape[-1]) for value in all_embds)}"
                        )
                        logged_feature_contract = True

                    embd = torch.concatenate(all_embds, axis=-1)

                    dataset_label = DATASET_LABEL
                    dataset_ids = torch.Tensor(DATASET_IDS).to(device, dtype=torch.long)

                    msa_info, chunk_logits = model.infer(
                        input_embeddings=embd,
                        dataset_ids=dataset_ids,
                        label_id_masks=torch.Tensor(
                            dataset_id2label_mask[
                                DATASET_LABEL_TO_DATASET_ID[dataset_label]
                            ]
                        )
                        .to(device, dtype=bool)
                        .unsqueeze(0)
                        .unsqueeze(0),
                        with_logits=True,
                    )

                    start_frame = int(i * AFTER_DOWNSAMPLING_FRAME_RATES)
                    end_frame = start_frame + min(
                        math.ceil(hop_size * AFTER_DOWNSAMPLING_FRAME_RATES),
                        chunk_logits["boundary_logits"][0].shape[0],
                    )

                    logits["function_logits"][start_frame:end_frame, :] += (
                        chunk_logits["function_logits"][0].detach().cpu().numpy()
                    )
                    logits["boundary_logits"][start_frame:end_frame] = (
                        chunk_logits["boundary_logits"][0].detach().cpu().numpy()
                    )
                    logits_num["function_logits"][start_frame:end_frame, :] += 1
                    logits_num["boundary_logits"][start_frame:end_frame] += 1
                    lens += end_frame - start_frame

                    i += hop_size

                # avoid divide-by-zero
                logits_num["function_logits"][logits_num["function_logits"] == 0] = 1
                logits_num["boundary_logits"][logits_num["boundary_logits"] == 0] = 1

                logits["function_logits"] /= logits_num["function_logits"]
                logits["boundary_logits"] /= logits_num["boundary_logits"]

                function_logits_np = logits["function_logits"][:lens]
                boundary_logits_np = logits["boundary_logits"][:lens]
                frame_logits_artifact = None
                if args.save_frame_logits_dir:
                    frame_logits_artifact = _save_frame_logits(
                        args.save_frame_logits_dir,
                        uid,
                        function_logits_np,
                        boundary_logits_np,
                    )

                logits["function_logits"] = torch.from_numpy(
                    function_logits_np
                ).unsqueeze(0)
                logits["boundary_logits"] = torch.from_numpy(
                    boundary_logits_np
                ).unsqueeze(0)

                detailed_output = postprocess_functional_structure_detailed(logits, hp)
                if not detailed_output:
                    raise RuntimeError(
                        "SongFormer produced no valid structure sections"
                    )
                msa_infer_output = [
                    (float(item["start"]), str(item["label"]))
                    for item in detailed_output
                ]
                msa_infer_output.append((float(detailed_output[-1]["end"]), "end"))
                if not args.no_rule_post_processing:
                    msa_infer_output = rule_post_processing(msa_infer_output)

                msa_json = []
                for idx in range(len(msa_infer_output) - 1):
                    start = float(msa_infer_output[idx][0])
                    end = float(msa_infer_output[idx + 1][0])
                    midpoint = (start + end) / 2.0
                    matching = [
                        item
                        for item in detailed_output
                        if float(item["start"]) <= midpoint < float(item["end"])
                    ]
                    if detailed_output:
                        detail = max(
                            matching or detailed_output,
                            key=lambda item: max(
                                0.0,
                                min(end, float(item["end"]))
                                - max(start, float(item["start"])),
                            ),
                        )
                    else:
                        detail = {
                            "start_boundary_confidence": 0.0,
                            "end_boundary_confidence": 0.0,
                            "label_confidence": 0.0,
                        }
                    msa_json.append(
                        {
                            "label": msa_infer_output[idx][1],
                            "start": start,
                            "end": end,
                            "raw_start": start,
                            "raw_end": end,
                            "start_boundary_confidence": float(
                                detail["start_boundary_confidence"]
                            ),
                            "end_boundary_confidence": float(
                                detail["end_boundary_confidence"]
                            ),
                            "boundary_confidence": float(
                                detail["end_boundary_confidence"]
                            ),
                            "label_confidence": float(detail["label_confidence"]),
                            "label_source": "songformer",
                        }
                    )

                metadata = {
                    "state": "ok",
                    "stage_version": args.stage_version,
                    "content_type": content_type,
                }
                if frame_logits_artifact is not None:
                    metadata["frame_logits"] = frame_logits_artifact
                _write_prediction_cache(out_path, msa_json, metadata)
                send_result(uid, True)

            except Exception as e:
                send_result(uid, False, error=f"{type(e).__name__}: {e}")
                logger.error(
                    f"process {rank} error\nuid={uid}\naudio={audio_path}\n{e}"
                )


def merge_back_to_jsonl(
    input_jsonl: str,
    output_dir: str,
    output_jsonl: str,
    audio_key: str,
    stage_version: str,
    failures: Mapping[str, str] | None = None,
):
    """Rebuild JSONL from current inputs and reusable per-track predictions."""
    failure_by_uid = {str(key): str(value) for key, value in (failures or {}).items()}
    target = Path(output_jsonl)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for line_no, obj in iter_jsonl(input_jsonl):
                uid = uid_from_obj(obj, line_no)
                pred_path = os.path.join(output_dir, f"{uid}.json")
                content_type = str(obj.get("content_type", "song")).strip().lower()
                structure = (
                    _load_cached_structure(pred_path)
                    if _cache_is_reusable(pred_path, content_type)
                    else None
                )
                stage_status = dict(obj.get("stage_status") or {})
                stage_errors = dict(obj.get("stage_errors") or {})
                stage_versions = dict(obj.get("stage_versions") or {})
                stage_status.pop("structure", None)
                stage_errors.pop("structure", None)
                stage_versions.pop("structure", None)
                if structure is not None:
                    obj["structure_raw"] = structure
                    metadata = _load_cache_metadata(pred_path)
                    if metadata.get("frame_logits"):
                        obj.setdefault("structure_artifacts", {})["frame_logits"] = (
                            metadata["frame_logits"]
                        )
                    stage_status["structure_raw"] = "ok"
                    stage_errors.pop("structure_raw", None)
                    structure_version = metadata.get("stage_version") or stage_version
                else:
                    obj["structure_raw"] = None
                    if uid in failure_by_uid:
                        error = failure_by_uid[uid]
                    elif not obj.get(audio_key):
                        error = f"missing_{audio_key}"
                    elif content_type not in {"song", "instrumental"}:
                        error = f"unsupported_content_type:{content_type}"
                    else:
                        error = "missing_or_invalid_structure_prediction"
                    stage_status["structure_raw"] = "error"
                    stage_errors["structure_raw"] = error
                    structure_version = stage_version
                stage_versions["structure_raw"] = structure_version
                obj["stage_status"] = stage_status
                obj["stage_errors"] = stage_errors
                obj["stage_versions"] = stage_versions

                # Compatibility for downstream code that still reads this field.
                if content_type == "song":
                    obj["songformer_result"] = obj["structure_raw"]

                handle.write(json.dumps(obj, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _worker_snapshot(processes: Sequence[Any]) -> list[tuple[Any, Any, bool]]:
    return [
        (getattr(process, "pid", None), getattr(process, "exitcode", None), process.is_alive())
        for process in processes
    ]


def _raise_for_crashed_workers(processes: Sequence[Any], context: str) -> None:
    crashed = [
        (getattr(process, "pid", None), getattr(process, "exitcode", None))
        for process in processes
        if getattr(process, "exitcode", None) not in (None, 0)
    ]
    if crashed:
        raise RuntimeError(f"SongFormer worker crashed during {context}: {crashed}")


def _wait_for_workers_ready(
    processes: Sequence[Any],
    queue_output: Any,
    *,
    timeout: float,
) -> None:
    """Do not admit any task until every resident model reports readiness."""

    deadline = time.monotonic() + max(0.0, float(timeout))
    ready: set[int] = set()
    expected = set(range(len(processes)))
    while ready != expected:
        _raise_for_crashed_workers(processes, "initialization")
        exited_before_ready = [
            (index, getattr(process, "pid", None), getattr(process, "exitcode", None))
            for index, process in enumerate(processes)
            if index not in ready and not process.is_alive()
        ]
        if exited_before_ready:
            raise RuntimeError(
                "SongFormer worker exited before ready: "
                f"{exited_before_ready}"
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                "SongFormer workers were not ready within "
                f"{timeout:g}s; ready={sorted(ready)} snapshot={_worker_snapshot(processes)}"
            )
        try:
            message = queue_output.get(timeout=min(0.25, remaining))
        except queue_module.Empty:
            continue
        if not isinstance(message, Mapping):
            raise RuntimeError(f"invalid SongFormer worker message before ready: {message!r}")
        message_type = message.get("type")
        if message_type == "worker_init_error":
            raise RuntimeError(
                f"SongFormer worker {message.get('worker_id')} initialization failed: "
                f"{message.get('error')}"
            )
        if message_type != "worker_ready":
            raise RuntimeError(f"unexpected SongFormer message before ready: {message!r}")
        worker_id = int(message.get("worker_id", -1))
        if worker_id not in expected:
            raise RuntimeError(f"invalid SongFormer ready worker_id={worker_id}")
        if worker_id in ready:
            raise RuntimeError(f"duplicate SongFormer ready worker_id={worker_id}")
        ready.add(worker_id)


def _put_with_worker_supervision(
    queue_input: Any,
    item: Any,
    processes: Sequence[Any],
    *,
    poll_timeout: float,
    max_wait: float,
) -> None:
    deadline = time.monotonic() + max(0.0, float(max_wait))
    while True:
        _raise_for_crashed_workers(processes, "queue submission")
        if not any(process.is_alive() for process in processes):
            raise RuntimeError("all SongFormer workers exited during queue submission")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                "SongFormer input queue remained full for "
                f"{max_wait:g}s; snapshot={_worker_snapshot(processes)}"
            )
        try:
            queue_input.put(item, timeout=min(max(0.01, poll_timeout), remaining))
            return
        except queue_module.Full:
            continue


def _dispatch_tasks(
    tasks: Sequence[tuple[str, str, str]],
    processes: Sequence[Any],
    queue_input: Any,
    *,
    poll_timeout: float,
    max_wait: float,
) -> None:
    for item in tasks:
        _put_with_worker_supervision(
            queue_input,
            item,
            processes,
            poll_timeout=poll_timeout,
            max_wait=max_wait,
        )
    for _ in processes:
        _put_with_worker_supervision(
            queue_input,
            None,
            processes,
            poll_timeout=poll_timeout,
            max_wait=max_wait,
        )


def _admit_and_dispatch(
    tasks: Sequence[tuple[str, str, str]],
    processes: Sequence[Any],
    queue_input: Any,
    queue_output: Any,
    *,
    ready_timeout: float,
    poll_timeout: float,
    max_wait: float,
) -> None:
    _wait_for_workers_ready(processes, queue_output, timeout=ready_timeout)
    _dispatch_tasks(
        tasks,
        processes,
        queue_input,
        poll_timeout=poll_timeout,
        max_wait=max_wait,
    )


def _collect_worker_results(
    tasks: Sequence[tuple[str, str, str]],
    processes: Sequence[Any],
    queue_output: Any,
    *,
    stall_timeout: float,
    progress: Any = None,
) -> tuple[int, int, int, dict[str, str]]:
    expected = {str(item[0]) for item in tasks}
    if len(expected) != len(tasks):
        raise RuntimeError("SongFormer task uids must be unique")
    received: set[str] = set()
    failures: dict[str, str] = {}
    ok = skipped = 0
    deadline = time.monotonic() + max(0.0, float(stall_timeout))
    while received != expected:
        _raise_for_crashed_workers(processes, "result collection")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            missing = sorted(expected - received)[:10]
            raise TimeoutError(
                "SongFormer produced no result within "
                f"{stall_timeout:g}s; missing={missing} snapshot={_worker_snapshot(processes)}"
            )
        try:
            message = queue_output.get(timeout=min(0.5, remaining))
        except queue_module.Empty:
            if not any(process.is_alive() for process in processes):
                raise RuntimeError(
                    "all SongFormer workers exited before returning every result: "
                    f"missing={sorted(expected - received)[:10]}"
                )
            continue
        if not isinstance(message, Mapping):
            raise RuntimeError(f"invalid SongFormer result message: {message!r}")
        if message.get("type") == "worker_init_error":
            raise RuntimeError(
                f"SongFormer worker {message.get('worker_id')} initialization failed: "
                f"{message.get('error')}"
            )
        if message.get("type") != "result":
            raise RuntimeError(f"unexpected SongFormer result message: {message!r}")
        uid = str(message.get("uid", ""))
        if uid not in expected or uid in received:
            raise RuntimeError(f"unexpected/duplicate SongFormer result uid={uid!r}")
        received.add(uid)
        deadline = time.monotonic() + max(0.0, float(stall_timeout))
        if message.get("ok", False):
            ok += 1
            if message.get("skipped", False):
                skipped += 1
        else:
            failures[uid] = str(message.get("error") or "unknown_structure_error")
        if progress is not None:
            progress.update(1)
            progress.set_postfix(
                {"ok": ok, "fail": len(failures), "skipped": skipped},
                refresh=False,
            )
    return ok, len(failures), skipped, failures


def _terminate_workers(processes: Sequence[Any], *, timeout: float = 10.0) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    deadline = time.monotonic() + max(0.0, float(timeout))
    for process in processes:
        remaining = max(0.0, deadline - time.monotonic())
        process.join(timeout=remaining)
    for process in processes:
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(timeout=1.0)


def _join_workers(processes: Sequence[Any], *, timeout: float) -> None:
    deadline = time.monotonic() + max(0.0, float(timeout))
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    alive = [getattr(process, "pid", None) for process in processes if process.is_alive()]
    if alive:
        _terminate_workers(processes, timeout=min(10.0, max(0.0, timeout)))
        raise TimeoutError(f"SongFormer workers did not exit within {timeout:g}s: {alive}")
    _raise_for_crashed_workers(processes, "shutdown")


def main():
    parser = argparse.ArgumentParser()

    # jsonl-only inputs
    parser.add_argument(
        "--input_jsonl", type=str, required=True, help="Input jsonl path"
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        required=True,
        help="Output jsonl path (with structure_raw)",
    )
    parser.add_argument(
        "--audio_key", type=str, default="audio_path", help="audio path key in jsonl"
    )

    # inference outputs
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        required=True,
        help="Directory to store per-item uid.json results",
    )

    # parallelism
    parser.add_argument("--gpu_num", "-gn", type=int, default=1, help="Number of GPUs")
    parser.add_argument(
        "--num_thread_per_gpu",
        "-tn",
        type=int,
        default=1,
        help="Processes per GPU (must be 1; models are resident and shared work is queued)",
    )
    parser.add_argument(
        "--decode-prefetch",
        type=int,
        choices=(0, 1),
        default=1,
        help="Decode one next track on CPU while current GPU inference runs; 0 disables",
    )
    parser.add_argument(
        "--embedding-chunk-batch-size",
        type=int,
        default=1,
        help="Equal-length 30-second MuQ/MusicFM chunks per GPU call",
    )
    parser.add_argument("--worker-ready-timeout", type=float, default=600.0)
    parser.add_argument("--queue-put-timeout", type=float, default=1.0)
    parser.add_argument("--queue-put-max-wait", type=float, default=3600.0)
    parser.add_argument("--worker-stall-timeout", type=float, default=3600.0)
    parser.add_argument("--worker-join-timeout", type=float, default=60.0)

    # model config
    parser.add_argument(
        "--model", type=str, required=True, help="Model to use, e.g., SongFormer"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint filename under ckpts/, e.g., SongFormer.safetensors",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Config filename under configs/, e.g., SongFormer.yaml",
    )
    parser.add_argument("--num_classes", type=int, default=128)
    parser.add_argument(
        "--stage_version",
        default="dual-structure-local-global-ssm-cbm-v3",
        help="Structure provenance label; it does not control cache reuse",
    )
    parser.add_argument(
        "--save-frame-logits-dir",
        default="",
        help="Optional directory for compressed SongFormer frame-logit NPZ sidecars",
    )

    # behavior
    parser.add_argument(
        "--no_rule_post_processing",
        action="store_true",
        help="Disable rule-based post-processing",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Debug mode: run only 1 item on GPU 0"
    )
    parser.add_argument(
        "--force_merge_only",
        action="store_true",
        help="Skip inference, only merge uid.json -> output_jsonl",
    )

    args = parser.parse_args()
    if args.embedding_chunk_batch_size < 1:
        raise ValueError("embedding_chunk_batch_size must be at least 1")
    for name in (
        "worker_ready_timeout",
        "queue_put_timeout",
        "queue_put_max_wait",
        "worker_stall_timeout",
        "worker_join_timeout",
    ):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")

    os.makedirs(args.output_dir, exist_ok=True)

    # merge-only mode
    if args.force_merge_only:
        merge_back_to_jsonl(
            args.input_jsonl,
            args.output_dir,
            args.output_jsonl,
            args.audio_key,
            args.stage_version,
        )
        return

    processed_uids = get_processed_uids(args.output_dir)
    tasks = build_tasks_from_jsonl(
        args.input_jsonl,
        args.audio_key,
        args.output_dir,
        require_frame_logits=bool(args.save_frame_logits_dir),
    )

    logger.info(f"output_dir: {args.output_dir}")
    logger.info(f"already processed: {len(processed_uids)}")
    logger.info(f"to process now: {len(tasks)}")
    total_tracks = count_lines(args.input_jsonl)

    if not tasks:
        progress = pipeline_tqdm(
            total=total_tracks,
            initial=total_tracks,
            desc="2/7 structure inference",
            unit="track",
        )
        progress.close()
        merge_back_to_jsonl(
            args.input_jsonl,
            args.output_dir,
            args.output_jsonl,
            args.audio_key,
            args.stage_version,
        )
        logger.info(f"nothing to infer. merged jsonl: {args.output_jsonl}")
        return

    init_args = Namespace(
        output_dir=args.output_dir,
        win_size=420,
        hop_size=420,
        num_classes=args.num_classes,
        model=args.model,
        checkpoint=args.checkpoint,
        config_path=args.config_path,
        no_rule_post_processing=args.no_rule_post_processing,
        stage_version=args.stage_version,
        save_frame_logits_dir=args.save_frame_logits_dir,
        decode_prefetch=args.decode_prefetch,
        embedding_chunk_batch_size=args.embedding_chunk_batch_size,
    )

    if args.debug:
        if len(tasks) == 0:
            logger.warning("no tasks to run (all processed). will still merge.")
        else:
            queue_input: mp.Queue = mp.Queue()
            queue_output: mp.Queue = mp.Queue()
            queue_input.put(tasks[0])
            queue_input.put(None)
            inference_worker(0, 0, queue_input, queue_output, init_args)
            ready = queue_output.get()
            if ready.get("type") != "worker_ready":
                raise RuntimeError(f"unexpected debug readiness message: {ready!r}")
            result = queue_output.get()
            debug_failures = (
                {}
                if result.get("ok", False)
                else {str(result.get("uid")): str(result.get("error"))}
            )
        merge_back_to_jsonl(
            args.input_jsonl,
            args.output_dir,
            args.output_jsonl,
            args.audio_key,
            args.stage_version,
            failures=debug_failures if tasks else None,
        )
        return

    gpu_num = args.gpu_num
    num_thread_per_gpu = args.num_thread_per_gpu
    if num_thread_per_gpu != 1:
        raise ValueError(
            "num_thread_per_gpu must be 1: each process loads MuQ, MusicFM and SongFormer; "
            "use more GPUs or input shards instead of duplicate model replicas"
        )
    num_workers = gpu_num * num_thread_per_gpu

    if num_workers <= 0:
        raise ValueError("gpu_num * num_thread_per_gpu must be > 0")

    queue_input: mp.Queue = mp.Queue(maxsize=2048)
    queue_output: mp.Queue = mp.Queue()

    processes = []
    for worker_idx in range(num_workers):
        rank = worker_idx % gpu_num
        logger.info(f"spawn worker {worker_idx} on GPU {rank}")
        time.sleep(0.05)
        p = mp.Process(
            target=inference_worker,
            args=(worker_idx, rank, queue_input, queue_output, init_args),
            daemon=True,
        )
        p.start()
        processes.append(p)
    progress = None
    try:
        _admit_and_dispatch(
            tasks,
            processes,
            queue_input,
            queue_output,
            ready_timeout=args.worker_ready_timeout,
            poll_timeout=args.queue_put_timeout,
            max_wait=args.queue_put_max_wait,
        )
        progress = pipeline_tqdm(
            total=total_tracks,
            initial=total_tracks - len(tasks),
            desc="2/7 structure inference",
            unit="track",
        )
        ok, fail, skipped, failures = _collect_worker_results(
            tasks,
            processes,
            queue_output,
            stall_timeout=args.worker_stall_timeout,
            progress=progress,
        )
        _join_workers(processes, timeout=args.worker_join_timeout)
    except Exception:
        _terminate_workers(processes)
        raise
    finally:
        if progress is not None:
            progress.close()

    # merge to output jsonl
    merge_back_to_jsonl(
        args.input_jsonl,
        args.output_dir,
        args.output_jsonl,
        args.audio_key,
        args.stage_version,
        failures=failures,
    )
    logger.info(
        f"all done. ok={ok} fail={fail} skipped={skipped} "
        f"merged jsonl: {args.output_jsonl}"
    )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

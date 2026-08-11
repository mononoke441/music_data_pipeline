"""
Ray分布式推理实现
"""

import glob
import hashlib
import json
import logging
import os
import queue as queue_module
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple
import argparse

import ray
from ray.util.queue import Queue as RayQueue

from config import cfg, parse_cfg_overrides
from dataloader import create_dataloader
from task_tracker import TaskTracker
from workers import DataLoaderWorker, create_worker, QueueMonitor, SaveWorker

PIPELINE_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(PIPELINE_SCRIPTS_DIR))
from pipeline_progress import pipeline_tqdm  # noqa: E402
from runtime_integrity import (  # noqa: E402
    CPU_MIR_SEMANTIC_INPUT_FIELDS,
    build_stage_fingerprint_payload,
    build_task_manifest,
    normalize_results_file,
    reset_incompatible_stage_state,
    validate_existing_results,
    validate_result_tracker_coverage,
    write_stage_fingerprint_manifest,
)


# 设置日志
def setup_logging(log_file=None):
    """设置日志配置"""
    if log_file:
        # 重置已有处理器，避免重复日志
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        # 创建文件处理器
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)

        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(
            logging.WARNING
            if os.environ.get("PIPELINE_QUIET_LOGS", "1") == "1"
            else logging.INFO
        )

        # 设置格式
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 配置根日志器
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
    else:
        logging.basicConfig(level=logging.INFO)


logger = logging.getLogger(__name__)


def prepare_data_paths(data_path, output_path, group_by_segment=False):
    """
    准备数据路径列表
    如果输入是目录，返回每个segment文件的路径和对应的输出路径
    如果输入是文件，返回单个文件的路径和原始输出路径

    Returns:
        List[Dict]: 每个元素包含 'data_path' 和 'output_path'
    """
    if os.path.isdir(data_path):
        # 目录模式：查找所有 segment*.jsonl 文件
        segment_pattern = os.path.join(data_path, "segment*.jsonl")
        segment_files = glob.glob(segment_pattern)

        if not segment_files:
            logger.error(f"No segment*.jsonl files found in {data_path}")
            return []

        # 排序确保顺序一致
        segment_files.sort()

        logger.info(f"Found {len(segment_files)} segment files:")
        for i, seg_file in enumerate(segment_files):
            logger.info(f"  {i + 1}. {os.path.basename(seg_file)}")

        # 为每个segment文件准备输出路径
        path_list = []
        for seg_file in segment_files:
            seg_name = os.path.splitext(os.path.basename(seg_file))[0]

            if group_by_segment:
                # 分组模式：每个segment有独立的输出目录
                seg_output_path = os.path.join(output_path, seg_name)
            else:
                # 非分组模式：使用原始输出路径
                seg_output_path = output_path

            path_list.append(
                {
                    "data_path": seg_file,
                    "output_path": seg_output_path,
                    "segment_name": seg_name,
                }
            )

        return path_list
    else:
        # 文件模式：直接返回原始路径
        logger.info(f"Input is a single file: {data_path}")
        return [
            {
                "data_path": data_path,
                "output_path": output_path,
                "segment_name": os.path.splitext(os.path.basename(data_path))[0],
            }
        ]


def generate_all_lance_paths(data_path, output_path, group_by_segment=False):
    """
    生成所有 Lance 数据集的路径列表（不进行分片）

    命名规则：path + segment_xx.lance (xx: 00-ff)

    Returns:
        List[Dict]: 包含所有 segment 路径的列表
    """
    # 如果传入的是单个 .lance 文件，直接返回
    if data_path.endswith(".lance"):
        segment_name = os.path.splitext(os.path.basename(data_path))[0]
        segment_output = (
            os.path.join(output_path, segment_name) if group_by_segment else output_path
        )
        return [
            {
                "data_path": data_path,
                "output_path": segment_output,
                "segment_name": segment_name,
            }
        ]

    # 目录或前缀模式：根据命名规则生成 segment_00.lance ~ segment_ff.lance
    base_path = data_path.rstrip("/") + "/"
    path_list = []
    for i in range(256):
        segment_name = f"segment_{i:02x}"
        segment_file = f"{base_path}{segment_name}.lance"
        segment_output = (
            os.path.join(output_path, segment_name) if group_by_segment else output_path
        )
        path_list.append(
            {
                "data_path": segment_file,
                "output_path": segment_output,
                "segment_name": segment_name,
            }
        )

    return path_list


def filter_completed_paths(path_list):
    """
    过滤掉已完成的路径（检查 success.jsonl 文件）

    Args:
        path_list: 路径列表

    Returns:
        List[Dict]: 未完成的路径列表
    """
    # A bare success marker has no knowledge of the current input, code or
    # model fingerprint. Validation therefore happens inside run_inference;
    # pre-filtering here would silently skip stale results.
    return list(path_list)


def prepare_lance_paths(
    data_path,
    output_path,
    group_by_segment=False,
    worker_rank: int = 0,
    world_size: int = 1,
):
    """
    根据命名规则生成 Lance 数据集的分片路径列表

    命名规则：path + segment_xx.lance (xx: 00-ff)

    注意：此函数会先过滤掉已完成的路径，然后再进行分片
    """
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got: {world_size}")
    if worker_rank < 0 or worker_rank >= world_size:
        raise ValueError(
            f"worker_rank must be in [0, {world_size - 1}], got: {worker_rank}"
        )

    # 先生成所有路径
    all_paths = generate_all_lance_paths(data_path, output_path, group_by_segment)

    # 过滤掉已完成的路径
    remaining_paths = filter_completed_paths(all_paths)

    # 对剩余路径进行分片
    total_segments = len(remaining_paths)
    if total_segments == 0:
        return []

    chunk_size = total_segments // world_size
    remainder = total_segments % world_size

    start = worker_rank * chunk_size + min(worker_rank, remainder)
    end = start + chunk_size + (1 if worker_rank < remainder else 0)

    return remaining_paths[start:end]


def _ensure_model_workers(workers, config, model_path):
    if workers:
        return workers
    if not ray.is_initialized():
        quiet = os.environ.get("PIPELINE_QUIET_LOGS", "1") == "1"
        ray.init(
            log_to_driver=not quiet,
            logging_level=logging.ERROR if quiet else logging.INFO,
        )
    available_resources = ray.available_resources()
    available_gpus = available_resources.get("GPU", 0)
    gpu_per_worker = getattr(config, "gpu_per_worker", 0.05) or 0.05
    num_workers = config.num_workers or (
        max(1, int(available_gpus / gpu_per_worker)) if gpu_per_worker > 0 else 1
    )
    logger.info(
        "num_workers=%s, gpus=%s, gpu_per_worker=%s, configured=%s",
        num_workers,
        available_gpus,
        gpu_per_worker,
        config.num_workers,
    )
    created = [
        create_worker(model_type=config.model_type, model_path=model_path)
        for _ in range(num_workers)
    ]
    ray.get([worker.get_model_info.remote() for worker in created])
    logger.info("All workers are ready")
    return created


def _write_success_marker(output_path: str, payload: Mapping[str, Any]) -> None:
    success_path = os.path.join(output_path, "success.jsonl")
    tmp_path = success_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, success_path)
    logger.info("Success marker saved to: %s", success_path)


@dataclass(frozen=True)
class ResumeState:
    tracker: TaskTracker
    task_map: Dict[int, str]
    input_fingerprint: str
    stage_fingerprint: str
    run_fingerprint: str
    completed_count: int


def _required_payload_fields(config: Any, stage_name: str) -> Tuple[str, ...]:
    if stage_name != "music_cpu":
        return ()
    return ("chords", "beatnet", "key")


def _is_missing_dataset_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(
        fragment in message
        for fragment in ("not found", "was not found", "does not exist", "no such file")
    )


def _load_data_loader(
    config: Any, data_path: str
) -> Optional[Tuple[Any, int, Dict[str, Any]]]:
    """Create the driver-side loader, returning None for a missing shard."""

    try:
        dataloader_kwargs = config.get_dataloader_kwargs()
        data_loader = create_dataloader(
            dataloader_type=config.dataloader_type,
            data_path=data_path,
            batch_size=config.batch_size,
            **dataloader_kwargs,
        )
        total_samples = len(data_loader)
    except Exception as error:
        if _is_missing_dataset_error(error):
            logger.warning("Dataset file not found, skipping %s: %s", data_path, error)
            return None
        logger.exception("Failed to create dataloader for %s", data_path)
        raise

    logger.info("Loaded %s samples", total_samples)
    return data_loader, total_samples, dataloader_kwargs


def _configure_resume_state(
    *,
    config: Any,
    data_loader: Any,
    total_samples: int,
    output_path: str,
    db_path: str,
    model_path: Optional[str],
    stage_name: str,
) -> ResumeState:
    """Bind progress to stable inputs and record current stage provenance."""

    tracker = TaskTracker(db_path)
    had_versioned_progress = bool(tracker.run_fingerprint)
    results_path = os.path.join(output_path, "results.jsonl")
    may_reset = bool(getattr(config, "reset_incompatible_output", False))

    if (
        os.path.exists(results_path)
        and os.path.getsize(results_path) > 0
        and not had_versioned_progress
    ):
        if not may_reset:
            raise RuntimeError(
                f"{results_path} predates safe versioned progress tracking; "
                "use a new output directory"
            )
        removed = reset_incompatible_stage_state(output_path, db_path)
        logger.warning("Reset incompatible runtime artifacts: %s", removed)
        tracker = TaskTracker(db_path)
        had_versioned_progress = False

    semantic_fields = (
        CPU_MIR_SEMANTIC_INPUT_FIELDS if stage_name == "music_cpu" else None
    )
    task_map, input_fingerprint = build_task_manifest(
        data_loader,
        semantic_fields=semantic_fields,
    )
    if len(task_map) != total_samples:
        raise RuntimeError(
            f"Dataloader counted {total_samples} samples but exposed "
            f"{len(task_map)} manifest records"
        )

    payload = build_stage_fingerprint_payload(config, model_path=model_path)
    stage_fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    if (
        stage_name == "music_cpu"
        and tracker.input_fingerprint
        and tracker.input_fingerprint != input_fingerprint
        and tracker.input_fingerprint_schema != "cpu_mir_semantic_v1"
        and set(tracker._task_id_to_key.values()) == set(task_map.values())
    ):
        tracker.migrate_semantic_input_fingerprint(
            input_fingerprint=input_fingerprint,
            task_map=task_map,
        )
        logger.info(
            "Migrated CPU MIR progress to the semantic input fingerprint without invalidating tasks"
        )

    try:
        run_fingerprint = tracker.configure_run(
            input_fingerprint=input_fingerprint,
            stage_fingerprint=stage_fingerprint,
            task_map=task_map,
            input_fingerprint_schema=(
                "cpu_mir_semantic_v1" if stage_name == "music_cpu" else None
            ),
        )
    except RuntimeError:
        if not may_reset:
            raise
        removed = reset_incompatible_stage_state(output_path, db_path)
        logger.warning("Reset incompatible runtime artifacts: %s", removed)
        tracker = TaskTracker(db_path)
        had_versioned_progress = False
        run_fingerprint = tracker.configure_run(
            input_fingerprint=input_fingerprint,
            stage_fingerprint=stage_fingerprint,
            task_map=task_map,
            input_fingerprint_schema=(
                "cpu_mir_semantic_v1" if stage_name == "music_cpu" else None
            ),
        )

    write_stage_fingerprint_manifest(output_path, stage_fingerprint, payload)
    required_fields = _required_payload_fields(config, stage_name)
    existing_records = normalize_results_file(
        results_path,
        task_order={task_key: task_id for task_id, task_key in task_map.items()},
    )
    unexpected_results = set(existing_records) - set(task_map.values())
    if unexpected_results:
        raise RuntimeError(
            "Progress/results coverage mismatch: "
            f"unexpected_results={sorted(unexpected_results)[:5]}"
        )
    successful_result_keys = validate_existing_results(
        results_path,
        stage_name,
        had_versioned_progress,
        required_fields,
    )
    try:
        tracker.reset_incomplete_allocations()
    except Exception:
        logger.warning("Failed to reset incomplete allocations; continuing")

    completed_tasks = tracker.get_completed_tasks()
    completed_keys = {task_map[task_id] for task_id in completed_tasks}
    missing_results = completed_keys - successful_result_keys
    if missing_results:
        retry_ids = sorted(
            task_id
            for task_id, task_key in task_map.items()
            if task_key in missing_results
        )
        tracker.mark_tasks_finished(retry_ids, ["error"] * len(retry_ids))
        logger.warning(
            "Demoted %s incomplete cached successes to retryable errors",
            len(retry_ids),
        )
        completed_tasks = tracker.get_completed_tasks()

    return ResumeState(
        tracker=tracker,
        task_map=task_map,
        input_fingerprint=input_fingerprint,
        stage_fingerprint=stage_fingerprint,
        run_fingerprint=run_fingerprint,
        completed_count=len(completed_tasks),
    )


def _finish_completed_resume(
    *,
    state: ResumeState,
    output_path: str,
    data_path: str,
    total_samples: int,
    workers: Optional[List[Any]],
    stage_name: str,
    required_payload_fields: Tuple[str, ...],
) -> None:
    progress = pipeline_tqdm(
        total=total_samples,
        initial=total_samples,
        desc="2/7 CPU MIR",
        unit="track",
    )
    progress.close()
    stats = state.tracker.get_progress_stats()
    if (
        stats.get("failed", 0)
        or stats.get("unallocated", 0)
        or stats.get("allocated", 0)
    ):
        raise RuntimeError(f"No pending rows but tracker is inconsistent: {stats}")

    coverage = validate_result_tracker_coverage(
        results_path=os.path.join(output_path, "results.jsonl"),
        stage_name=stage_name,
        tracker=state.tracker,
        task_map=state.task_map,
        required_payload_fields=required_payload_fields,
    )
    _write_success_marker(
        output_path,
        {
            "status": "success",
            "total_saved": 0,
            "loader_batches": 0,
            "model_batches": 0,
            "num_workers": len(workers or []),
            "run_fingerprint": state.run_fingerprint,
            "input_fingerprint": state.input_fingerprint,
            "stage_fingerprint": state.stage_fingerprint,
            "failed": coverage["error"],
            "resume_status": "already_complete",
        },
    )
    logger.info(
        "No pending tasks for %s; model and Ray actors were not initialized",
        data_path,
    )


def _create_loader_workers(
    *,
    config: Any,
    data_path: str,
    db_path: str,
    total_samples: int,
    dataloader_kwargs: Dict[str, Any],
) -> List[Any]:
    requested = int(getattr(config, "num_dataloader_workers", 1))
    if requested <= 1 or config.dataloader_type != "lance":
        if requested > 1:
            logger.warning(
                "Multiple DataLoaderWorkers are unsupported for %s; using one",
                config.dataloader_type,
            )
        return [
            DataLoaderWorker.remote(
                dataloader_type=config.dataloader_type,
                data_path=data_path,
                db_path=db_path,
                batch_size=config.batch_size,
                worker_id=0,
                **dataloader_kwargs,
            )
        ]

    worker_count = min(requested, total_samples)
    samples_per_worker, remainder = divmod(total_samples, worker_count)
    base_offset = int(dataloader_kwargs.get("offset") or 0)
    logger.info(
        "Creating %s sharded DataLoaderWorkers (%s base samples, %s remainder, base offset %s)",
        worker_count,
        samples_per_worker,
        remainder,
        base_offset,
    )
    workers = []
    for worker_id in range(worker_count):
        relative_offset = worker_id * samples_per_worker + min(worker_id, remainder)
        offset = base_offset + relative_offset
        limit = samples_per_worker + (1 if worker_id < remainder else 0)
        worker_kwargs = dict(dataloader_kwargs)
        worker_kwargs.pop("offset", None)
        worker_kwargs.pop("limit", None)
        workers.append(
            DataLoaderWorker.remote(
                dataloader_type=config.dataloader_type,
                data_path=data_path,
                db_path=db_path,
                batch_size=config.batch_size,
                worker_id=worker_id,
                offset=offset,
                limit=limit,
                **worker_kwargs,
            )
        )
    return workers


def _supervise_worker_refs(
    *,
    loader_refs: List[Any],
    model_refs: List[Any],
    save_ref: Any,
    input_queue: RayQueue,
) -> Tuple[List[int], List[int], Any]:
    """Supervise every producer/consumer and coordinate exactly one sentinel/model."""

    loader_indices = {ref: index for index, ref in enumerate(loader_refs)}
    model_indices = {ref: index for index, ref in enumerate(model_refs)}
    loader_results: Dict[int, int] = {}
    model_results: Dict[int, int] = {}
    save_result: Any = None
    pending = set(loader_refs + model_refs + [save_ref])
    sentinels_remaining = len(model_refs)

    while pending:
        if len(loader_results) == len(loader_refs) and sentinels_remaining:
            try:
                input_queue.put(None, block=True, timeout=1.0)
                sentinels_remaining -= 1
            except queue_module.Full:
                pass

        ready, _ = ray.wait(list(pending), num_returns=1, timeout=1.0)
        if not ready:
            continue
        for ref in ready:
            value = ray.get(ref)
            pending.remove(ref)
            if ref in loader_indices:
                loader_results[loader_indices[ref]] = int(value)
            elif ref in model_indices:
                if len(loader_results) != len(loader_refs):
                    raise RuntimeError(
                        "Model worker exited before all loader workers completed"
                    )
                model_results[model_indices[ref]] = int(value)
            elif ref == save_ref:
                save_result = value

    if sentinels_remaining:
        raise RuntimeError(
            f"Worker graph exited with {sentinels_remaining} unsent model sentinels"
        )
    return (
        [loader_results[index] for index in range(len(loader_refs))],
        [model_results[index] for index in range(len(model_refs))],
        save_result,
    )


def _run_worker_graph(
    *,
    loader_workers: List[Any],
    model_workers: List[Any],
    save_worker: Any,
    queue_monitor: Any,
    input_queue: RayQueue,
    result_queue: RayQueue,
    db_path: str,
    total_samples: int,
    progress: Any,
) -> Tuple[List[int], List[int], Any]:
    loader_refs = [
        worker.run.remote(
            input_queue,
            db_queue=None,
            num_model_workers=len(model_workers),
            num_loader_workers=len(loader_workers),
        )
        for worker in loader_workers
    ]
    model_refs = [
        worker.run.remote(input_queue, result_queue) for worker in model_workers
    ]
    save_ref = save_worker.run.remote(
        result_queue,
        db_queue=None,
        total_tasks=total_samples,
        num_model_workers=len(model_workers),
    )
    monitor_ref = queue_monitor.run.remote()

    stop_progress = threading.Event()

    def refresh_progress() -> None:
        while not stop_progress.wait(1.0):
            stats = TaskTracker(db_path).get_progress_stats()
            terminal = min(
                total_samples,
                int(stats.get("completed", 0)) + int(stats.get("failed", 0)),
            )
            if terminal > progress.n:
                progress.update(terminal - progress.n)

    progress_thread = threading.Thread(target=refresh_progress, daemon=True)
    progress_thread.start()
    logger.info("Starting queue pipeline")

    try:
        return _supervise_worker_refs(
            loader_refs=loader_refs,
            model_refs=model_refs,
            save_ref=save_ref,
            input_queue=input_queue,
        )
    except Exception:
        logger.exception("Queue pipeline failed")
        for ref in loader_refs + model_refs + [save_ref]:
            try:
                ray.cancel(ref, force=True)
            except Exception:
                logger.debug("Failed to cancel Ray reference", exc_info=True)
        raise
    finally:
        ray.cancel(monitor_ref)
        stop_progress.set()
        progress_thread.join(timeout=2.0)
        final_stats = TaskTracker(db_path).get_progress_stats()
        terminal = min(
            total_samples,
            int(final_stats.get("completed", 0)) + int(final_stats.get("failed", 0)),
        )
        if terminal > progress.n:
            progress.update(terminal - progress.n)
        progress.close()


def _validate_and_publish_success(
    *,
    state: ResumeState,
    output_path: str,
    db_path: str,
    total_samples: int,
    loader_results: List[int],
    model_results: List[int],
    save_result: Any,
    stage_name: str,
    required_payload_fields: Tuple[str, ...],
) -> None:
    current_tracker = TaskTracker(db_path)
    stats = current_tracker.get_progress_stats()
    total_model_batches = sum(model_results)
    total_saved = (
        int(save_result.get("persisted", 0))
        if isinstance(save_result, dict)
        else int(save_result or 0)
    )
    logger.info(
        "CPU MIR complete: saved=%s model_batches=%s stats=%s",
        total_saved,
        total_model_batches,
        stats,
    )

    if stats.get("unallocated", 0) or stats.get("allocated", 0):
        raise RuntimeError(
            f"Inference persisted terminal records but is incomplete: {stats}. "
            "Failed tasks remain retryable on the next run."
        )
    if stats.get("completed", 0) + stats.get("failed", 0) != total_samples:
        raise RuntimeError(f"Expected {total_samples} terminal tasks, got {stats}")

    coverage = validate_result_tracker_coverage(
        results_path=os.path.join(output_path, "results.jsonl"),
        stage_name=stage_name,
        tracker=current_tracker,
        task_map=state.task_map,
        required_payload_fields=required_payload_fields,
    )
    if coverage["ok"] != stats.get("completed", 0) or coverage[
        "error"
    ] != stats.get("failed", 0):
        raise RuntimeError(
            f"Result/tracker terminal counts disagree: coverage={coverage}, stats={stats}"
        )

    _write_success_marker(
        output_path,
        {
            "status": "partial_success" if coverage["error"] else "success",
            "total_saved": total_saved,
            "loader_batches": sum(loader_results),
            "model_batches": total_model_batches,
            "num_workers": len(model_results),
            "run_fingerprint": state.run_fingerprint,
            "input_fingerprint": state.input_fingerprint,
            "stage_fingerprint": state.stage_fingerprint,
            "failed": coverage["error"],
        },
    )


def run_inference(
    data_path: str,
    output_path: str,
    workers: Optional[List[Any]] = None,
    model_path: Optional[str] = None,
    db_path: Optional[str] = None,
    group_id: Optional[str] = None,
    config: Any = None,
) -> Optional[List[Any]]:
    """Run one resumable Ray inference stage."""

    del group_id
    config = config or cfg
    os.makedirs(output_path, exist_ok=True)
    db_path = db_path or os.path.join(output_path, "progress.jsonl")
    log_path = os.path.join(output_path, "inference.log")
    setup_logging(log_path)

    loaded = _load_data_loader(config, data_path)
    if loaded is None:
        return workers
    data_loader, total_samples, dataloader_kwargs = loaded
    stage_name = (
        "music_cpu"
        if str(config.model_type).startswith("music_cpu")
        else str(config.model_type)
    )
    state = _configure_resume_state(
        config=config,
        data_loader=data_loader,
        total_samples=total_samples,
        output_path=output_path,
        db_path=db_path,
        model_path=model_path,
        stage_name=stage_name,
    )
    required_fields = _required_payload_fields(config, stage_name)
    remaining_tasks = total_samples - state.completed_count
    logger.info(
        "Resume state: completed=%s remaining=%s",
        state.completed_count,
        remaining_tasks,
    )

    if remaining_tasks == 0:
        _finish_completed_resume(
            state=state,
            output_path=output_path,
            data_path=data_path,
            total_samples=total_samples,
            workers=workers,
            stage_name=stage_name,
            required_payload_fields=required_fields,
        )
        return workers

    progress = pipeline_tqdm(
        total=total_samples,
        initial=state.completed_count,
        desc="2/7 CPU MIR",
        unit="track",
    )
    workers = _ensure_model_workers(workers, config, model_path)
    input_queue = RayQueue(maxsize=100)
    result_queue = RayQueue(maxsize=1000)
    loader_workers = _create_loader_workers(
        config=config,
        data_path=data_path,
        db_path=db_path,
        total_samples=total_samples,
        dataloader_kwargs=dataloader_kwargs,
    )
    save_worker = SaveWorker.remote(
        output_path,
        db_path,
        worker_id=0,
        buffer_size=256,
        log_path=log_path,
        stage_name=stage_name,
        stage_fingerprint=state.stage_fingerprint,
    )
    queue_monitor = QueueMonitor.remote(
        queues={"input_queue": input_queue, "result_queue": result_queue},
        interval=10.0,
        log_path=log_path,
    )
    loader_results, model_results, save_result = _run_worker_graph(
        loader_workers=loader_workers,
        model_workers=workers,
        save_worker=save_worker,
        queue_monitor=queue_monitor,
        input_queue=input_queue,
        result_queue=result_queue,
        db_path=db_path,
        total_samples=total_samples,
        progress=progress,
    )
    _validate_and_publish_success(
        state=state,
        output_path=output_path,
        db_path=db_path,
        total_samples=total_samples,
        loader_results=loader_results,
        model_results=model_results,
        save_result=save_result,
        stage_name=stage_name,
        required_payload_fields=required_fields,
    )
    logger.info("Results: %s", os.path.join(output_path, "results.jsonl"))
    return workers


if __name__ == "__main__":
    # --- 常用配置默认值（可以被命令行覆盖） ---
    cfg.data_path = "s3://embodied-multimodality/speech/processed_datasets/stage1/audio_b1_4_0/"  # 数据或目录
    cfg.output_path = "./outputs"  # 推理结果目录
    cfg.model_type = "music_cpu_pipeline"
    cfg.dataloader_type = "lance"  # 可选: 'oss', 'audio_jsonl', 'jsonl', 'lance'
    cfg.lance_prompt_key = "audio_flac"
    cfg.batch_size = 128
    cfg.num_dataloader_workers = 10
    cfg.gpu_per_worker = 0.05  # 自动计算 num_workers 时使用
    cfg.num_workers = 1
    cfg.group_by_segment = True

    # ========== 命令行参数解析 ==========
    parser = argparse.ArgumentParser()
    # rank / world size 属于执行环境，而不是 cfg 本身
    parser.add_argument("--worker-rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    # 通用 cfg 覆盖：可以多次传入
    parser.add_argument(
        "--cfg",
        action="append",
        default=[],
        help="覆盖 cfg 中的任意字段，如: --cfg model_type=qwen3omni --cfg batch_size=64",
    )

    args = parser.parse_args()

    # 通用 --cfg 覆盖（自动按类型转换）
    if args.cfg:
        overrides = parse_cfg_overrides(args.cfg)
        cfg.update(**overrides)

    worker_rank = args.worker_rank
    world_size = args.world_size

    # 使用（可能被覆盖后的）配置中的数据路径
    data_path = cfg.data_path
    output_path = cfg.output_path
    group_by_segment = cfg.group_by_segment

    if cfg.dataloader_type != "lance" and not os.path.exists(data_path):
        print(f"Error: Data path not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    # 准备数据路径列表
    if cfg.dataloader_type == "lance":
        # 先生成所有路径并过滤已完成的，然后按 worker_rank 和 world_size 分片
        all_paths = generate_all_lance_paths(data_path, output_path, group_by_segment)
        remaining_paths = filter_completed_paths(all_paths)
        logger.info(
            "Lance paths: total=%s, completed=%s, remaining=%s",
            len(all_paths),
            len(all_paths) - len(remaining_paths),
            len(remaining_paths),
        )

        # 对剩余路径进行分片
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got: {world_size}")
        if worker_rank < 0 or worker_rank >= world_size:
            raise ValueError(
                f"worker_rank must be in [0, {world_size - 1}], got: {worker_rank}"
            )

        total_segments = len(remaining_paths)
        if total_segments == 0:
            path_list = []
        else:
            chunk_size = total_segments // world_size
            remainder = total_segments % world_size
            start = worker_rank * chunk_size + min(worker_rank, remainder)
            end = start + chunk_size + (1 if worker_rank < remainder else 0)
            path_list = remaining_paths[start:end]

        logger.info(
            "Lance sharding: worker_rank=%s, world_size=%s, assigned_segments=%s",
            worker_rank,
            world_size,
            len(path_list),
        )
    else:
        path_list = prepare_data_paths(data_path, output_path, group_by_segment)
    if not path_list:
        print("No valid data paths found", file=sys.stderr)
        sys.exit(1)

    # Delay Ray and model actors until a version-validated manifest contains
    # unfinished tasks. Once created, workers are reused across segments.
    workers = None
    model_path = (
        getattr(cfg, "model_path", None) or None
    )  # 允许某些模型不需要 model_path

    # 直接使用所有 workers 处理每个 segment
    for path_info in path_list:
        data_file = path_info["data_path"]
        output_dir = path_info["output_path"]
        segment_name = path_info["segment_name"]

        workers = run_inference(
            data_path=data_file,
            output_path=output_dir,
            workers=workers,
            model_path=model_path,
            group_id=None,
            config=cfg,
        )

    if ray.is_initialized():
        ray.shutdown()

# -*- coding: utf-8 -*-
"""
DataLoader Worker: 从 TaskTracker 获取任务，加载数据，放入队列
"""

import ray
import logging
from typing import List
from ray.util.queue import Queue as RayQueue

from dataloader import create_dataloader
from task_tracker import TaskTracker
from audio_info import AudioInfo

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1)
class DataLoaderWorker:
    """数据加载 Worker，从 TaskTracker 获取任务并加载数据到队列"""

    def __init__(
        self,
        dataloader_type: str,
        data_path: str,
        db_path: str,
        batch_size: int,
        worker_id: int = 0,
        offset: int = None,
        limit: int = None,
        **dataloader_kwargs,
    ):
        """
        初始化 DataLoader Worker

        Args:
            dataloader_type: 数据加载器类型（'oss'/'audio_jsonl'/'jsonl'/'lance'）
            data_path: 数据路径
            db_path: 数据库路径（用于创建 TaskTracker）
            batch_size: 批次大小
            worker_id: Worker ID（用于多 worker 场景）
            offset: 数据起始偏移量（用于数据分片，仅 Lance 支持）
            limit: 数据限制数量（用于数据分片，仅 Lance 支持）
            **dataloader_kwargs: 传递给 create_dataloader 的其他参数
        """
        self.worker_id = worker_id

        # 如果指定了 offset 和 limit，更新 dataloader_kwargs（仅 Lance 支持）
        if offset is not None or limit is not None:
            if dataloader_type == "lance":
                if offset is not None:
                    dataloader_kwargs["offset"] = offset
                if limit is not None:
                    dataloader_kwargs["limit"] = limit
                logger.info(
                    f"DataLoaderWorker {worker_id}: offset={offset}, limit={limit}"
                )
            else:
                logger.warning(
                    f"DataLoaderWorker {worker_id}: offset/limit not supported for {dataloader_type}, ignoring"
                )

        # 在 Actor 内部构建 dataloader，避免序列化复杂对象（如 Minio 客户端）
        try:
            self.data_loader = create_dataloader(
                dataloader_type=dataloader_type,
                data_path=data_path,
                batch_size=batch_size,
                **dataloader_kwargs,
            )
        except Exception as e:
            logger.error(f"DataLoaderWorker {worker_id}: Create dataloader failed: {e}")
            raise
        self.db_path = db_path
        self.batch_size = batch_size
        # 在 Actor 内部创建 TaskTracker（避免序列化问题）
        self.task_tracker = TaskTracker(db_path)

    def run(
        self,
        input_queue: RayQueue,
        db_queue: RayQueue = None,
        num_model_workers: int = 1,
        num_loader_workers: int = 1,
    ):
        """
        顺序迭代数据加载器，跳过已完成的任务，放入队列

        Args:
            input_queue: 输入队列，放入 AudioInfo 批次
            db_queue: 数据库操作队列（不再使用，保留以兼容接口）
            num_model_workers: 保留的兼容参数；结束信号由 driver 协调端发送
            num_loader_workers: 保留的兼容参数；结束信号由 driver 协调端发送

        Returns:
            加载的总批次数量
        """
        del db_queue, num_model_workers, num_loader_workers

        # 启动时查询一次已完成的任务，并更新 dataloader
        completed_tasks = self.task_tracker.get_completed_tasks()
        logger.info(
            f"DataLoaderWorker {self.worker_id}: Found {len(completed_tasks)} completed tasks, will skip them"
        )

        # 更新 dataloader 的 completed_tasks（如果支持）
        if hasattr(self.data_loader, "completed_tasks"):
            self.data_loader.completed_tasks = completed_tasks

        batch_count = 0
        current_batch: List[AudioInfo] = []
        current_batch_indices: List[int] = []
        items_processed = 0

        logger.info("DataLoaderWorker %s started", self.worker_id)

        # 使用迭代方式（IterableDataset 的标准用法）
        # dataloader 直接返回 (idx, AudioInfo) 元组
        try:
            for idx, audio_info in self.data_loader:
                current_batch.append(audio_info)
                current_batch_indices.append(idx)
                items_processed += 1

                # 达到批次大小时，放入队列
                if len(current_batch) >= self.batch_size:
                    input_queue.put((current_batch, current_batch_indices))
                    batch_count += 1
                    if batch_count % 1000 == 0:
                        logger.debug(
                            "DataLoaderWorker %s loaded %s batches (%s items)",
                            self.worker_id,
                            batch_count,
                            items_processed,
                        )
                    current_batch = []
                    current_batch_indices = []
        except Exception as e:
            logger.error(
                f"DataLoaderWorker {self.worker_id}: Error during data loading iteration: {e}"
            )
            raise

        # 处理剩余的不足一个批次的数据
        if current_batch:
            input_queue.put((current_batch, current_batch_indices))
            batch_count += 1

        logger.info(
            "DataLoaderWorker %s completed: batches=%s items=%s",
            self.worker_id,
            batch_count,
            items_processed,
        )
        return batch_count

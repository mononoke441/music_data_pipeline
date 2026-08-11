# -*- coding: utf-8 -*-
"""
Ray Model Worker
简化的模型工作器，只需要继承基础模型类即可
"""
import ray
import logging
from typing import List, Dict, Any
from ray.util.queue import Queue as RayQueue
from audio_info import AudioInfo

logger = logging.getLogger(__name__)


@ray.remote(num_gpus=0, num_cpus=1)
class ModelWorker:
    """
    Ray Actor，每个GPU一个worker
    简化的实现，只需要指定模型类即可
    """
    
    def __init__(self, model_class, model_name: str, model_path: str = None, **kwargs):
        """
        初始化模型工作器
        
        Args:
            model_class: BaseModel 子类
            model_name: 模型名称
            model_path: 模型路径
            **kwargs: 其他参数
        """
        self.model_class = model_class
        self.model_name = model_name
        self.model_path = model_path
        self.kwargs = kwargs
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        try:
            # 创建模型实例，让模型类自己处理设备分配
            self.model = self.model_class(
                model_name=self.model_name,
                model_path=self.model_path,
                **self.kwargs
            )
            logger.info(f"Model {self.model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def generate_batch(self, batch_data: List[AudioInfo]) -> List[AudioInfo]:
        """
        处理一批数据
        
        Args:
            batch_data: AudioInfo 对象列表
            
        Returns:
            AudioInfo 对象列表（包含预测结果）
        """
        try:
            # 使用模型的generate_batch方法
            results = self.model.generate_batch(batch_data)
            
            # 添加worker信息到 _extra 字段（用于向后兼容）
            for result in results:
                if not hasattr(result, '_extra'):
                    result._extra = {}
                result._extra["worker_processed"] = True
                # 使用 model_name 作为 worker 标识（如果没有 idx 属性）
                worker_id = getattr(self, 'idx', None) or getattr(self, 'model_name', 'unknown')
                result._extra["worker_id"] = worker_id
            
            return results
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if self.model:
            info = self.model.get_model_info()
            info["worker_name"] = self.model_name
            return info
        return {"worker_name": self.model_name, "status": "unloaded"}
    
    def run(self, input_queue: RayQueue, result_queue: RayQueue):
        """
        从输入队列读取数据，处理，放入结果队列
        
        Args:
            input_queue: 输入队列，每个元素为 (batch_data: List[AudioInfo], task_ids: List[int]) 或 None
            result_queue: 结果队列，放入处理后的 AudioInfo 批次
        
        Returns:
            处理的总批次数量
        """
        batch_count = 0
        try:
            actor_id = ray.get_runtime_context().get_actor_id()
        except Exception:
            actor_id = id(self)
        worker_id = f"{self.model_name}:{actor_id}"
        try:
            while True:
                # Queue failures are actor failures. Retrying forever here hides
                # the cause and leaves SaveWorker blocked indefinitely.
                item = input_queue.get()
                if item is None:
                    break

                batch_data, task_ids = item
                if not batch_data:
                    continue
                if len(batch_data) != len(task_ids):
                    raise RuntimeError(
                        f"Input batch has {len(batch_data)} items but {len(task_ids)} task ids"
                    )
                try:
                    results = self.generate_batch(batch_data)
                    if len(results) != len(task_ids):
                        raise RuntimeError(
                            f"Model returned {len(results)} results for {len(task_ids)} tasks"
                        )
                    result_queue.put((results, task_ids))
                    batch_count += 1
                except Exception as error:
                    logger.exception("Error processing batch %s", task_ids)
                    error_results = []
                    for audio_info in batch_data:
                        audio_info.error = f"{type(error).__name__}: {error}"
                        audio_info.predictions = []
                        error_results.append(audio_info)
                    # Every input task produces exactly one durable output,
                    # including failed tasks.
                    result_queue.put((error_results, task_ids))
        finally:
            result_queue.put({"type": "worker_done", "worker_id": worker_id})

        logger.info("ModelWorker %s completed, processed %s batches", self.model_name, batch_count)
        return batch_count
    
    def cleanup(self):
        """清理资源"""
        if self.model:
            self.model.cleanup()
            self.model = None


def create_model_worker(model_class, model_name: str, model_path: str = None, **kwargs):
    """创建通用模型工作器"""
    return ModelWorker.remote(model_class, model_name, model_path, **kwargs)



def create_worker(model_type: str = None, model_name: str = None, model_path: str = None, **kwargs):
    """
    工厂函数：根据类型创建对应的模型工作器
    
    Args:
        model_type: 模型类型：'music_cpu_pipeline'，以及组成它的
                    'chordino'、'beatnet'、'essentia' 调试入口
                   如果为 None，从环境变量 MODEL_TYPE 读取
        model_name: 模型名称
        model_path: 模型路径
        **kwargs: 其他参数（传递给具体的模型）
        
    Returns:
        ModelWorker 实例
    """
    import os
    from sub_models import (
        ChordinoModel,
        EssentiaModel,
        BeatNetModel,
        MusicCpuPipelineModel,
    )

    if model_type is None:
        model_type = os.environ.get('MODEL_TYPE', 'music_cpu_pipeline').lower()

    model_map = {
        'chordino': {
            'cls': ChordinoModel,
            'default_name': 'Chordino',
            'options': {'num_gpus': 0, 'num_cpus': kwargs.pop('num_cpus', 1.0)},
        },
        'essentia': {
            'cls': EssentiaModel,
            'default_name': 'Essentia',
            'options': {'num_gpus': 0, 'num_cpus': kwargs.pop('num_cpus', 1.0)},
        },
        'beatnet': {
            'cls': BeatNetModel,
            'default_name': 'BeatNet',
            'options': {'num_gpus': 0, 'num_cpus': kwargs.pop('num_cpus', 1.0)},
        },
        'music_cpu_pipeline': {
            'cls': MusicCpuPipelineModel,
            'default_name': 'MusicCpuPipeline',
            'options': {'num_gpus': 0, 'num_cpus': kwargs.pop('num_cpus', 2.0)},
        },
    }

    if model_type not in model_map:
        raise ValueError(
            f"Unknown model_type: {model_type}. Supported: "
            f"'chordino', 'essentia', 'beatnet', 'music_cpu_pipeline'"
        )

    entry = model_map[model_type]
    model_cls = entry['cls']
    if model_name is None:
        model_name = entry['default_name']

    options = entry['options']
    if options:
        return ModelWorker.options(**options).remote(model_cls, model_name, model_path, **kwargs)
    else:
        return ModelWorker.remote(model_cls, model_name, model_path, **kwargs)


# 兼容旧接口：为特定模型类型提供便捷工厂函数
def create_chordino_worker(model_name: str = None, model_path: str = None, **kwargs):
    """创建 Chordino 模型 Worker"""
    return create_worker(model_type="chordino", model_name=model_name, model_path=model_path, **kwargs)


def create_beatnet_worker(model_name: str = None, model_path: str = None, **kwargs):
    """创建 BeatNet 模型 Worker"""
    return create_worker(model_type="beatnet", model_name=model_name, model_path=model_path, **kwargs)


def create_essentia_worker(model_name: str = None, model_path: str = None, **kwargs):
    """创建 Essentia 模型 Worker"""
    return create_worker(model_type="essentia", model_name=model_name, model_path=model_path, **kwargs)

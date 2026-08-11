# -*- coding: utf-8 -*-
"""
DataLoader类定义，基于 PyTorch IterableDataset 实现
支持流式数据加载和断点续跑
"""
import json
import logging
import os
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Set, Tuple

if TYPE_CHECKING:
    import numpy as np

try:
    from torch.utils.data import IterableDataset
except ImportError:
    raise ImportError("PyTorch is required. Please install: pip install torch")

from audio_info import AudioInfo

logger = logging.getLogger(__name__)


class BaseDataLoader(IterableDataset):
    """
    基础数据加载器类，基于 IterableDataset
    提供通用的数据访问和断点续跑支持
    """

    def __init__(
        self,
        data_path: str,
        batch_size: int = 1,
        shuffle: bool = False,
        completed_tasks: Optional[Set[int]] = None,
        **kwargs
    ):
        """
        初始化数据加载器
        
        Args:
            data_path: 数据文件路径
            batch_size: 批处理大小
            shuffle: 是否打乱数据
            completed_tasks: 已完成的任务索引集合（用于跳过）
            **kwargs: 其他参数
        """
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.config = kwargs
        self.completed_tasks = completed_tasks or set()
        
        # 缓存总数（延迟计算）
        self._total_count: Optional[int] = None

    def _count_items(self) -> int:
        """计算数据总数（子类需要实现）"""
        raise NotImplementedError("Subclass must implement _count_items")

    def __len__(self) -> int:
        """返回数据总数（延迟计算）"""
        if self._total_count is None:
            self._total_count = self._count_items()
        return self._total_count

    def _raw_iter(self) -> Iterator[Tuple[int, Dict[str, Any]]]:
        """
        原始迭代器，返回 (index, item) 元组
        子类需要实现此方法
        """
        raise NotImplementedError("Subclass must implement _raw_iter")

    def __iter__(self) -> Iterator[Tuple[int, AudioInfo]]:
        """
        迭代器，跳过已完成的任务
        返回 (index, AudioInfo) 元组
        
        所有 dataloader 统一返回 AudioInfo 格式，确保数据流动的一致性
        """
        for idx, item_dict in self._raw_iter():
            if idx not in self.completed_tasks:
                try:
                    # 转换为 AudioInfo 对象
                    audio_info = AudioInfo.from_dict(item_dict)
                    
                    # 验证 AudioInfo 至少有一个音频路径或音频数据（用于后续处理）
                    if not (audio_info.audio_path or audio_info.url or audio_info.path or
                            audio_info.audio_bytes or
                            (audio_info.bucket and audio_info.object_key)):
                        audio_info.error = "input_validation_failed:missing_audio_source"
                        logger.warning(
                            f"Item at index {idx} has no audio path, audio_bytes or OSS params; "
                            "emitting a retryable error result"
                        )
                    
                    yield (idx, audio_info)
                except Exception as e:
                    logger.warning(
                        f"Failed to convert item at index {idx} to AudioInfo: {e}. "
                        "Emitting a retryable error result."
                    )
                    fallback = AudioInfo(
                        error=f"input_conversion_failed:{type(e).__name__}:{e}"
                    )
                    if isinstance(item_dict, dict):
                        fallback._extra = {
                            key: value
                            for key, value in item_dict.items()
                            if key not in {"audio_bytes", "access_key", "secret_key"}
                        }
                    yield (idx, fallback)

    def get_item(self, index: int) -> Dict[str, Any]:
        """
        根据索引获取单个数据项（向后兼容方法）
        注意：对于流式数据源，此方法可能效率较低
        """
        for idx, item in self._raw_iter():
            if idx == index:
                return item
        raise IndexError(f"Index {index} out of range")


class JSONLDataLoader(BaseDataLoader):
    """JSONL格式数据加载器"""

    def _count_items(self) -> int:
        """统计JSONL文件行数"""
        if not os.path.exists(self.data_path):
            return 0
        count = 0
        with open(self.data_path, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def _raw_iter(self) -> Iterator[Tuple[int, Dict[str, Any]]]:
        """流式读取JSONL文件，确保返回的字典适配 AudioInfo 格式"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        with open(self.data_path, encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if line:  # 跳过空行
                    try:
                        item = json.loads(line)
                        # 确保至少有一个音频路径字段（适配 AudioInfo）
                        if 'audio_path' not in item:
                            # 尝试从其他字段推断 audio_path
                            if 'url' in item:
                                item['audio_path'] = item['url']
                            elif 'path' in item:
                                item['audio_path'] = item['path']
                            elif 'file' in item:
                                item['audio_path'] = item['file']
                        yield (idx, item)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {idx}: {e}")
                        continue


class AudioJSONLDataLoader(BaseDataLoader):
    """音频JSONL数据加载器，专门处理音频文件路径"""

    def _count_items(self) -> int:
        """统计JSONL文件行数"""
        if not os.path.exists(self.data_path):
            return 0
        count = 0
        with open(self.data_path, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def _raw_iter(self) -> Iterator[Tuple[int, Dict[str, Any]]]:
        """流式读取音频JSONL文件"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        with open(self.data_path, encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if line:  # 跳过空行
                    try:
                        item = json.loads(line)
                        # 确保有audio_path字段
                        if 'audio_path' not in item:
                            if 'prompt' in item:
                                item['audio_path'] = item['prompt']
                            elif 'path' in item:
                                item['audio_path'] = item['path']
                            elif 'file' in item:
                                item['audio_path'] = item['file']
                            else:
                                logger.warning(f"No audio_path found in item at line {idx}")
                        yield (idx, item)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {idx}: {e}")
                        continue


def _sanitize_endpoint(ep: str) -> str:
    """清理endpoint字符串"""
    ep = ep.strip()
    if ep.startswith("http://"):
        ep = ep[len("http://"):]
    elif ep.startswith("https://"):
        ep = ep[len("https://"):]
    return ep.split("/")[0]


class OSSDataloader(BaseDataLoader):
    """
    读取 segment jsonl + metadata_{subset}.jsonl，
    通过 MinIO 预签名 + ffmpeg 裁剪音频，产出音频样本。
    
    注意：由于需要加载metadata，此实现仍需要一次性加载segment数据到内存
    """

    SRC_PREFIX = "s3://archive-oss/nginx/data"
    DST_PREFIX = "qz_oss://embodied-multimodality-datasets/speech/datasets/Podcast"
    DST_BUCKET = "embodied-multimodality-datasets"

    def __init__(self, data_path: str, batch_size: int = 1, shuffle: bool = False, **kwargs):
        # 延迟导入以避免对未使用场景的依赖
        try:
            import ffmpeg  # noqa: F401
            import numpy  # noqa: F401
            from minio import Minio  # noqa: F401
        except Exception as e:
            logger.warning(f"依赖缺失：{e}. 需要安装 ffmpeg-python、numpy、minio 并确保系统有 ffmpeg/ffprobe")

        self.meta_dir: str = kwargs.get("meta_dir")
        self.endpoint: str = _sanitize_endpoint(kwargs.get("endpoint", ""))
        self.access_key: str = kwargs.get("access_key", "")
        self.secret_key: str = kwargs.get("secret_key", "")
        self.secure: bool = bool(kwargs.get("secure", False))
        self.presign_secs: int = int(kwargs.get("presign_secs", 600))
        self.margin: float = float(kwargs.get("margin", 30.0))
        self.output_sample_rate: int = int(kwargs.get("output_sample_rate", 16000))
        self.load_workers: int = int(kwargs.get("load_workers", min(4, os.cpu_count() or 1)))
        self.show_load_progress: bool = bool(kwargs.get("show_load_progress", True))

        if not self.meta_dir:
            raise ValueError("OSSDataloader 需要提供 meta_dir")
        if not self.endpoint or not self.access_key or not self.secret_key:
            raise ValueError("OSSDataloader 需要提供 endpoint/access_key/secret_key")

        # 缓存：subset -> { oid -> meta_obj }
        self._meta_cache: dict[str, dict[str, dict]] = {}

        # MinIO 客户端
        from minio import Minio
        self._minio_client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure
        )

        # 加载segment数据到内存（OSSDataloader需要metadata，无法完全流式）
        self._segment_data: List[Dict[str, Any]] = []
        self._load_segment_data(data_path)

        # 提取completed_tasks并调用父类初始化
        completed_tasks = kwargs.pop("completed_tasks", None)
        super().__init__(data_path, batch_size, shuffle, completed_tasks, **kwargs)

    def _load_segment_data(self, data_path: str):
        """加载segment数据到内存"""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _gather_candidates(dir_or_file: str) -> List[str]:
            if os.path.isdir(dir_or_file):
                candidates = [
                    os.path.join(dir_or_file, name)
                    for name in os.listdir(dir_or_file)
                    if name.startswith("segment") and name.endswith(".jsonl")
                ]
                candidates.sort()
                return candidates
            else:
                return [dir_or_file]

        def load_one_file(fp: str) -> List[Dict[str, Any]]:
            seg_group = os.path.splitext(os.path.basename(fp))[0]
            items = []
            try:
                with open(fp, encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                        except Exception:
                            continue
                        if not all(k in item for k in ["start", "end", "subset_name", "metadata__id"]):
                            continue
                        if not isinstance(item["metadata__id"], dict) or "$oid" not in item["metadata__id"]:
                            continue
                        item["segment_group"] = seg_group
                        items.append(item)
            except Exception as e:
                logger.warning(f"Failed to load {fp}: {e}")
            return items

        segment_files = _gather_candidates(data_path)
        if not segment_files:
            logger.warning("No segment files found")
            return

        pbar = None
        if self.show_load_progress:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=len(segment_files), desc='Loading segments', unit='file')
            except ImportError:
                pass

        with ThreadPoolExecutor(max_workers=self.load_workers) as executor:
            futures = {executor.submit(load_one_file, fp): fp for fp in segment_files}
            for future in as_completed(futures):
                items = future.result()
                self._segment_data.extend(items)
                if pbar:
                    pbar.update(1)

        if pbar:
            pbar.close()
        logger.info(f"Loaded {len(self._segment_data)} segments")

    def _count_items(self) -> int:
        """返回segment数据总数"""
        return len(self._segment_data)

    def _load_meta_subset(self, subset: str) -> dict:
        """加载metadata子集"""
        if subset in self._meta_cache:
            return self._meta_cache[subset]

        meta_file = os.path.join(self.meta_dir, f"metadata_{subset}.jsonl")
        if not os.path.exists(meta_file):
            logger.warning(f"Metadata file not found: {meta_file}")
            self._meta_cache[subset] = {}
            return {}

        idx = {}
        with open(meta_file, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    meta = json.loads(line)
                    oid = meta.get("_id", {}).get("$oid") or meta.get("id")
                    if oid:
                        idx[oid] = meta
                except Exception:
                    continue
            
        self._meta_cache[subset] = idx
        return idx

    def _raw_iter(self) -> Iterator[Tuple[int, Dict[str, Any]]]:
        """迭代segment数据并加载metadata"""
        indices = list(range(len(self._segment_data)))
        if self.shuffle:
            import random
            random.shuffle(indices)

        for idx in indices:
            seg = self._segment_data[idx]
            start = float(seg["start"])
            end = float(seg["end"])
            duration = max(end - start, 0.0)
            subset = str(seg["subset_name"])
            meta_oid = seg.get("metadata__id", {}).get("$oid")

            try:
                subset_index = self._load_meta_subset(subset)
                meta = subset_index.get(meta_oid)
                if meta is None:
                    logger.warning(f"subset {subset} 未找到 oid {meta_oid}, 跳过")
                    continue
                src_path = meta.get("path")
                if not src_path:
                    logger.warning(f"meta 缺少 path 字段，跳过：{meta}")
                    continue

                # 构建完整样本信息
                qz_uri = self._replace_prefix_path(src_path)
                _bucket_from_uri, object_key = self._qz_to_bucket_and_key(qz_uri)
                bucket = self.DST_BUCKET
                sample = dict(seg)
                sample.update({
                    "start": start,
                    "end": end,
                    "duration": duration,
                    "subset_name": subset,
                    "metadata__id": meta_oid,
                    "segment_id": seg.get("id"),
                    "bucket": bucket,
                    "object_key": object_key,
                    "target_sample_rate": self.output_sample_rate,
                    "endpoint": self.endpoint,
                    "access_key": self.access_key,
                    "secret_key": self.secret_key,
                    "secure": self.secure,
                    "presign_secs": self.presign_secs,
                })
                yield (idx, sample)
            except Exception as e:
                logger.warning(f"Failed to process segment {idx}: {e}")
                continue
            
    def _replace_prefix_path(self, path: str) -> str:
        """替换路径前缀"""
        if path.startswith(self.SRC_PREFIX):
            return path.replace(self.SRC_PREFIX, self.DST_PREFIX, 1)
        return path

    def _qz_to_bucket_and_key(self, qz_uri: str) -> Tuple[str, str]:
        """解析qz_uri为bucket和key"""
        if qz_uri.startswith("qz_oss://"):
            uri = qz_uri[9:]  # 移除 "qz_oss://"
            parts = uri.split("/", 1)
            if len(parts) == 2:
                return parts[0], parts[1]
        return "", ""

class LanceDataLoader(BaseDataLoader):
    """
    Lance 数据集加载器
    支持从 S3 或本地加载 .lance 格式的数据集
    """
    
    def __init__(
        self,
        data_path: str,
        batch_size: int = 1,
        shuffle: bool = False,
        completed_tasks: Optional[Set[int]] = None,
        **kwargs
    ):
        """
        初始化 Lance 数据加载器
        
        Args:
            data_path: Lance 数据集路径（.lance 文件）
            batch_size: 批次大小
            shuffle: 是否打乱（Lance 数据集通常不支持打乱）
            completed_tasks: 已完成的任务索引集合
            **kwargs: 其他参数
                - prompt_key: 要读取的列名，默认 "prompt" 或 "audio_flac"
                - offset: 起始偏移量，默认 0
                - limit: 限制读取数量，默认 None（读取全部）
                - io_config: daft.io.IOConfig 对象（用于 S3 访问）
                - resume_from_offset: 恢复偏移量，默认 None
        """
        if not data_path.endswith(".lance"):
            raise ValueError(
                f"LanceDataLoader only supports .lance files, got: {data_path}"
            )
        
        self.prompt_key = kwargs.get("prompt_key", "audio_flac")
        self.offset = kwargs.get("offset", 0)
        self.limit = kwargs.get("limit", None)
        self.resume_from_offset = kwargs.get("resume_from_offset", None)
        self.io_config = kwargs.get("io_config", None)
        
        # 计算 resume_from_offset（基于 completed_tasks）
        if completed_tasks and len(completed_tasks) > 0:
            # 找到最大的已完成任务索引
            max_completed = max(completed_tasks)
            resume_offset = max_completed + 1
            if self.resume_from_offset is None:
                self.resume_from_offset = resume_offset
            else:
                # 取两者中的较大值
                self.resume_from_offset = max(self.resume_from_offset, resume_offset)
        
        # 如果设置了 resume_from_offset，更新 offset
        if self.resume_from_offset is not None:
            self.offset = self.resume_from_offset
        
        super().__init__(
            data_path=data_path,
            batch_size=batch_size,
            shuffle=shuffle,
            completed_tasks=completed_tasks,
            **kwargs
        )
        
        # 延迟加载数据集（避免在序列化时出现问题）
        self._dataset = None
    
    def _load_dataset(self):
        """延迟加载数据集（在 Actor 内部调用）"""
        if self._dataset is not None:
            return
        
        try:
            import lance
            # 尝试导入 daft 的配置转换工具
            try:
                from daft.io.object_store_options import io_config_to_storage_options
                storage_options = io_config_to_storage_options(self.io_config, self.data_path)
            except ImportError:
                # 如果 daft 不可用，手动构建 storage_options
                logger.warning("daft not available, using manual storage_options")
                storage_options = self._build_storage_options_manually()
            
            self._dataset = lance.dataset(
                self.data_path,
                storage_options=storage_options,
            )
        except Exception as e:
            logger.error(f"Failed to load Lance dataset from {self.data_path}: {e}")
            raise
    
    def _build_storage_options_manually(self):
        """手动构建 storage_options（当 daft 不可用时）"""
        if self.io_config is None:
            return {}
        
        storage_options = {}
        
        # 处理 S3 配置
        if hasattr(self.io_config, 's3') and self.io_config.s3:
            s3_config = self.io_config.s3
            storage_options.update({
                'key': getattr(s3_config, 'key_id', None) or getattr(s3_config, 'access_key_id', None),
                'secret': getattr(s3_config, 'access_key', None) or getattr(s3_config, 'secret_access_key', None),
                'endpoint_override': getattr(s3_config, 'endpoint_url', None),
                'scheme': 'https' if getattr(s3_config, 'use_ssl', True) else 'http',
            })
        
        return storage_options
    
    def _count_items(self) -> int:
        """计算数据总数"""
        self._load_dataset()
        total_rows = self._dataset.count_rows()
        if self.limit is None:
            return max(total_rows - self.offset, 0)
        else:
            return min(self.limit, max(total_rows - self.offset, 0))
    
    def _raw_iter(self) -> Iterator[Tuple[int, Dict[str, Any]]]:
        """原始迭代器，返回 (index, item_dict) 元组"""
        self._load_dataset()
        
        # 每次调用 _raw_iter 时重新创建迭代器（因为迭代器只能迭代一次）
        # 计算实际要读取的数量
        # 注意：如果 limit 已经设置，直接使用，避免重复调用 count_rows()（可能有性能开销）
        if self.limit is not None:
            effective_limit = self.limit
        else:
            # 只有在 limit 为 None 时才需要调用 count_rows()
            total_rows = self._dataset.count_rows()
            effective_limit = max(total_rows - self.offset, 0)
        
        import time
        iter_start_time = time.time()
        logger.info(f"LanceDataLoader: Creating iterator with offset={self.offset}, limit={effective_limit}, batch_size={self.batch_size}")
        
        # 读取多个列：audio_flac 和主键字段
        columns_to_read = [
            self.prompt_key,  # audio_flac
            '_id',
            'path',
            'metadata_oid',
            'segment_key'
        ]
        
        # 使用 batch_size 参数控制每次读取的批次大小，避免 byte array offset overflow
        to_batches_start = time.time()
        record_batches = self._dataset.to_batches(
            columns=columns_to_read,
            offset=self.offset,
            limit=effective_limit,
            scan_in_order=True,
            batch_size=self.batch_size,
        )
        to_batches_time = time.time() - to_batches_start
        logger.info(f"LanceDataLoader: to_batches() took {to_batches_time:.2f}s")
        
        current_idx = self.offset
        batch_count = 0
        first_batch_time = None
        
        # 直接使用 Arrow API 访问数据，避免 to_pylist() 的开销
        for record_batch in record_batches:
            if first_batch_time is None:
                first_batch_time = time.time()
                time_to_first_batch = first_batch_time - iter_start_time
                logger.info(f"LanceDataLoader: Time to first batch: {time_to_first_batch:.2f}s")
            
            batch_count += 1
            if batch_count % 100 == 0:  # 每 100 个批次输出一次日志
                elapsed = time.time() - iter_start_time
                logger.info(f"LanceDataLoader: Processed {batch_count} record batches, current_idx={current_idx}, elapsed={elapsed:.2f}s")
            
            # 获取所有列数据
            audio_column = record_batch[self.prompt_key]
            # 安全地获取其他列（如果存在）
            try:
                id_column = record_batch['_id']
            except (KeyError, IndexError):
                id_column = None
            try:
                path_column = record_batch['path']
            except (KeyError, IndexError):
                path_column = None
            try:
                metadata_oid_column = record_batch['metadata_oid']
            except (KeyError, IndexError):
                metadata_oid_column = None
            try:
                segment_key_column = record_batch['segment_key']
            except (KeyError, IndexError):
                segment_key_column = None
            
            batch_size = len(audio_column)
            
            # 批量处理，减少循环开销
            for i in range(batch_size):
                # 跳过已完成的任务
                if current_idx in self.completed_tasks:
                    current_idx += 1
                    continue
                
                # 直接从 Arrow 数组获取值（避免转换为 Python list）
                audio_value = audio_column[i].as_py()
                
                # 构建数据字典
                item_dict = {self.prompt_key: audio_value}
                
                # 如果值是 bytes，将其映射到 audio_bytes
                if isinstance(audio_value, bytes):
                    item_dict['audio_bytes'] = audio_value
                
                # 读取主键字段
                if id_column is not None:
                    try:
                        id_value = id_column[i].as_py()
                        # _id 是 struct<$oid: string>，需要提取 $oid
                        if isinstance(id_value, dict) and '$oid' in id_value:
                            item_dict['_id'] = id_value['$oid']
                        elif isinstance(id_value, str):
                            item_dict['_id'] = id_value
                        else:
                            item_dict['_id'] = str(id_value) if id_value is not None else None
                    except Exception as e:
                        logger.warning(f"Failed to extract _id at index {current_idx}: {e}")
                        item_dict['_id'] = None
                
                if path_column is not None:
                    try:
                        path_value = path_column[i].as_py()
                        item_dict['path'] = path_value
                    except Exception as e:
                        logger.warning(f"Failed to extract path at index {current_idx}: {e}")
                        item_dict['path'] = None
                
                if metadata_oid_column is not None:
                    try:
                        metadata_oid_value = metadata_oid_column[i].as_py()
                        item_dict['metadata_oid'] = metadata_oid_value
                    except Exception as e:
                        logger.warning(f"Failed to extract metadata_oid at index {current_idx}: {e}")
                        item_dict['metadata_oid'] = None
                
                if segment_key_column is not None:
                    try:
                        segment_key_value = segment_key_column[i].as_py()
                        item_dict['segment_key'] = segment_key_value
                    except Exception as e:
                        logger.warning(f"Failed to extract segment_key at index {current_idx}: {e}")
                        item_dict['segment_key'] = None
                
                # 添加索引信息（用于断点续跑）
                item_dict['_index'] = current_idx
                
                yield (current_idx, item_dict)
                current_idx += 1

def decode_oss_audio(item, target_sr: int = 16000) -> "np.ndarray":
    """
    从 OSS 字典描述符或 AudioInfo 对象解码音频，返回 numpy 数组
    
    Args:
        item: AudioInfo 对象或包含 bucket, object_key, start, end 等信息的字典
        target_sr: 目标采样率，默认 16000
        
    Returns:
        audio_array: float32 numpy 数组
    """
    import ffmpeg
    import numpy as np
    from minio import Minio
    from audio_info import AudioInfo
    
    # 支持 AudioInfo 对象或字典
    if isinstance(item, AudioInfo):
        endpoint = item.endpoint or ""
        access_key = item.access_key or ""
        secret_key = item.secret_key or ""
        secure = item.secure
        bucket = item.bucket or ""
        object_key = item.object_key or ""
        start = float(item.start or 0.0)
        end = float(item.end or 0.0)
        presign_secs = item.presign_secs or 1800
    elif isinstance(item, dict):
        endpoint = item.get('endpoint')
        access_key = item.get('access_key')
        secret_key = item.get('secret_key')
        secure = item.get('secure', False)
        bucket = item.get('bucket')
        object_key = item.get('object_key')
        start = float(item.get('start', 0.0))
        end = float(item.get('end', 0.0))
        presign_secs = int(item.get('presign_secs', 1800))
    else:
        raise TypeError(f"item must be AudioInfo or dict, got {type(item)}")
    
    if not (endpoint and access_key and secret_key and bucket and object_key):
        logger.error("Missing required OSS parameters for presigning")
        return np.array([], dtype=np.float32)
    
    try:
        client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=bool(secure))
        url = client.get_presigned_url("GET", bucket, object_key, expires=timedelta(seconds=int(presign_secs)))
        
        # 解码音频
        ss = float(start)
        t = max(0.0, float(end) - float(start)) if end is not None else 0.0
        
        out, err = (
            ffmpeg
            .input(url, ss=f"{ss:.9f}", seekable=1, rw_timeout="30M")
            .filter("aresample", int(target_sr))
            .filter('atrim', duration=t)
            .filter('asetpts', 'PTS-STARTPTS')
            .output('pipe:', format='f32le', acodec='pcm_f32le', ac=1, ar=int(target_sr))
            .global_args(
                '-loglevel', 'error',
                '-probesize', '1M',
                '-analyzeduration', '5M',
                '-reconnect', '1',
                '-reconnect_streamed', '1',
                '-reconnect_delay_max', '2'
            )
            .run(capture_stdout=True, capture_stderr=True)
        )
        return np.frombuffer(out, dtype=np.float32)
    except ffmpeg.Error as e:
        stderr_msg = e.stderr.decode('utf-8', errors='ignore') if hasattr(e, 'stderr') and e.stderr else str(e)
        logger.error(f"ffmpeg decode failed: url={url[:100]}... start={ss} duration={t} sr={target_sr} err={stderr_msg[:500]}")
        return np.array([], dtype=np.float32)
    except Exception as e:
        logger.error(f"decode_oss_audio failed: url={url[:100] if 'url' in locals() else 'N/A'}... error={e}")
        return np.array([], dtype=np.float32)


def create_dataloader(
    dataloader_type: str = None,
    data_path: str = None,
    batch_size: int = 1,
    shuffle: bool = False,
    completed_tasks: Optional[Set[int]] = None,
    **kwargs
):
    """
    工厂函数：根据类型创建对应的 DataLoader
    
    Args:
        dataloader_type: DataLoader 类型，可选值：'oss', 'audio_jsonl', 'jsonl', 'lance'
                        如果为 None，从环境变量 DATALOADER_TYPE 读取
        data_path: 数据路径
        batch_size: 批次大小
        shuffle: 是否打乱
        completed_tasks: 已完成的任务索引集合
        **kwargs: 其他参数（传递给具体的 DataLoader）
        
    Returns:
        DataLoader 实例
    """
    if dataloader_type is None:
        dataloader_type = os.environ.get('DATALOADER_TYPE', 'oss').lower()
    
    if dataloader_type == 'oss' or dataloader_type == 'osdataloader':
        return OSSDataloader(
            data_path=data_path,
            batch_size=batch_size,
            shuffle=shuffle,
            completed_tasks=completed_tasks,
            **kwargs
        )
    elif dataloader_type == 'audio_jsonl' or dataloader_type == 'audiojsonldataloader':
        return AudioJSONLDataLoader(
            data_path=data_path,
            batch_size=batch_size,
            shuffle=shuffle,
            completed_tasks=completed_tasks,
            **kwargs
        )
    elif dataloader_type == 'jsonl' or dataloader_type == 'jsonldataloader':
        return JSONLDataLoader(
            data_path=data_path,
            batch_size=batch_size,
            shuffle=shuffle,
            completed_tasks=completed_tasks,
            **kwargs
        )
    elif dataloader_type == 'lance' or dataloader_type == 'lancedataloader':
        return LanceDataLoader(
            data_path=data_path,
            batch_size=batch_size,
            shuffle=shuffle,
            completed_tasks=completed_tasks,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown dataloader_type: {dataloader_type}. Supported: 'oss', 'audio_jsonl', 'jsonl', 'lance'")


if __name__ == '__main__':
    """
    测试 dataloader 是否正常工作
    用法示例：
        # 测试 AudioJSONL 数据加载器
        python dataloader.py --type audio_jsonl --path /path/to/audio.jsonl
        
        # 测试 OSS 数据加载器
        python dataloader.py --type oss --path /path/to/segment_dir
        
        # 测试 Lance 数据加载器（本地文件）
        python dataloader.py --type lance --path /path/to/data.lance --prompt-key audio_flac
        
        # 测试 Lance 数据加载器（S3，使用默认配置）
        python dataloader.py --type lance --path s3://bucket/path/to/data.lance --prompt-key audio_flac
        
        # 测试 Lance 数据加载器（S3，使用自定义配置）
        python dataloader.py --type lance --path s3://bucket/path/to/data.lance \\
            --prompt-key audio_flac \\
            --access-key-id YOUR_ACCESS_KEY \\
            --secret-access-key YOUR_SECRET_KEY \\
            --endpoint-url http://oss.example.com:8009
    """
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Test dataloader')
    parser.add_argument('--type', type=str, default='audio_jsonl', 
                       choices=['audio_jsonl', 'oss', 'lance'],
                       help='Dataloader type to test (audio_jsonl, oss, or lance)')
    parser.add_argument('--path', type=str, required=True,
                       help='Data file path (for audio_jsonl/lance) or segment directory (for oss)')
    parser.add_argument('--max-items', type=int, default=5,
                       help='Maximum number of items to test')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size')
    parser.add_argument('--prompt-key', type=str, default='audio_flac',
                       help='Prompt key for Lance dataloader (default: audio_flac)')
    parser.add_argument('--access-key-id', type=str, default=None,
                       help='S3 access key ID (for Lance dataloader)')
    parser.add_argument('--secret-access-key', type=str, default=None,
                       help='S3 secret access key (for Lance dataloader)')
    parser.add_argument('--endpoint-url', type=str, default=None,
                       help='S3 endpoint URL (for Lance dataloader)')
    
    args = parser.parse_args()
    
    print(f"Testing {args.type} dataloader with path: {args.path}")
    print("=" * 60)
    
    try:
        # 从 config 获取参数
        from config import cfg
        
        # 准备 dataloader kwargs
        if args.type == 'oss':
            # OSSDataloader 需要额外参数（包含 shuffle）
            dataloader_kwargs = cfg.get_dataloader_kwargs()
            if not dataloader_kwargs.get('meta_dir'):
                print("Error: OSSDataloader requires meta_dir in config")
                print(f"Current config meta_dir: {cfg.meta_dir}")
                sys.exit(1)
            if not dataloader_kwargs.get('endpoint'):
                print("Error: OSSDataloader requires endpoint in config")
                print(f"Current config endpoint: {cfg.endpoint}")
                sys.exit(1)
            
            print("Using config:")
            print(f"  - meta_dir: {dataloader_kwargs['meta_dir']}")
            print(f"  - endpoint: {dataloader_kwargs['endpoint']}")
            print(f"  - output_sample_rate: {dataloader_kwargs['output_sample_rate']}")
            print(f"  - presign_secs: {dataloader_kwargs['presign_secs']}")
            print(f"  - shuffle: {dataloader_kwargs['shuffle']}")
            print("=" * 60)
        elif args.type == 'lance':
            # LanceDataLoader 需要额外参数
            dataloader_kwargs = {
                'prompt_key': args.prompt_key,
                'offset': 0,
                'limit': None,
            }
            
            # S3 配置（硬编码）
            ENDPOINT_URL = "http://oss.sii.shaipower.online:8009"
            ACCESS_KEY_ID = "KDHLDKB84RDW4VE7P5KI"
            SECRET_ACCESS_KEY = "oWPvR7UJqkLirm36uTgHqUbKGe8Hbk30BvK5PpVc"
            
            # 如果提供了命令行参数，优先使用命令行参数
            endpoint_url = args.endpoint_url or ENDPOINT_URL
            access_key_id = args.access_key_id or ACCESS_KEY_ID
            secret_access_key = args.secret_access_key or SECRET_ACCESS_KEY
            
            # 创建 IOConfig
            try:
                from daft.io import IOConfig, S3Config
                dataloader_kwargs['io_config'] = IOConfig(
                    s3=S3Config(
                        key_id=access_key_id,
                        access_key=secret_access_key,
                        endpoint_url=endpoint_url,
                        use_ssl=False,
                    )
                )
                print("Using S3 config:")
                print(f"  - endpoint_url: {endpoint_url}")
                print(f"  - access_key_id: {access_key_id[:10]}...")
                print("=" * 60)
            except ImportError:
                print("Warning: daft not available, will use manual storage_options")
                print("=" * 60)
            
            print("Using Lance config:")
            print(f"  - prompt_key: {args.prompt_key}")
            print("  - offset: 0")
            print("  - limit: None (read all)")
            print("=" * 60)
        else:
            # AudioJSONLDataLoader 只需要基本参数
            dataloader_kwargs = {
                'shuffle': cfg.shuffle,
            }
            print("Using config:")
            print(f"  - shuffle: {cfg.shuffle}")
            print("=" * 60)
        
        # 创建 dataloader（shuffle 已经在 dataloader_kwargs 中，不需要单独传）
        data_loader = create_dataloader(
            dataloader_type=args.type,
            data_path=args.path,
            batch_size=args.batch_size,
            **dataloader_kwargs
        )
        
        # 测试 __len__
        total = len(data_loader)
        print(f"Total items: {total}")
        print("=" * 60)
        
        # 测试迭代
        print(f"Testing iteration (showing first {args.max_items} items):")
        count = 0
        for idx, audio_info in data_loader:
            print(f"\n[{idx}] AudioInfo:")
            print(f"  - audio_path: {audio_info.audio_path}")
            if audio_info.url:
                print(f"  - url: {audio_info.url}")
            if audio_info.path:
                print(f"  - path: {audio_info.path}")
            if audio_info.audio_bytes is not None:
                # 显示音频 bytes 信息
                audio_size_mb = len(audio_info.audio_bytes) / (1024 * 1024)
                print(f"  - audio_bytes: {len(audio_info.audio_bytes)} bytes ({audio_size_mb:.2f} MB)")
                # 尝试读取音频信息（如果是 FLAC 格式）
                try:
                    import io
                    import soundfile as sf
                    audio_bytes_io = io.BytesIO(audio_info.audio_bytes)
                    info = sf.info(audio_bytes_io)
                    print(f"    - format: {info.format}")
                    print(f"    - sample_rate: {info.samplerate} Hz")
                    print(f"    - channels: {info.channels}")
                    print(f"    - duration: {info.duration:.2f} seconds")
                    print(f"    - frames: {info.frames}")
                except Exception as e:
                    print(f"    - (could not read audio info: {e})")
            if audio_info.start is not None:
                print(f"  - start: {audio_info.start}")
            if audio_info.end is not None:
                print(f"  - end: {audio_info.end}")
            if audio_info.duration is not None:
                print(f"  - duration: {audio_info.duration}")
            if audio_info.has_oss_params():
                print(f"  - OSS: {audio_info.bucket}/{audio_info.object_key}")
                print(f"  - endpoint: {audio_info.endpoint}")
            if audio_info.subset_name:
                print(f"  - subset_name: {audio_info.subset_name}")
            if audio_info.segment_id:
                print(f"  - segment_id: {audio_info.segment_id}")
            print(f"  - target_sample_rate: {audio_info.target_sample_rate}")
            
            # 验证 AudioInfo 对象
            if not isinstance(audio_info, AudioInfo):
                print(f"  ✗ ERROR: Not an AudioInfo object! Type: {type(audio_info)}")
                sys.exit(1)
            
            count += 1
            if count >= args.max_items:
                print(f"\n... (stopped after {args.max_items} items)")
                break
        
        print("=" * 60)
        print(f"✓ Test passed! Successfully loaded {count} items")
        print("✓ All items are AudioInfo objects")
        print("✓ Dataloader is working correctly")
        
    except FileNotFoundError as e:
        print(f"✗ Error: File not found - {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

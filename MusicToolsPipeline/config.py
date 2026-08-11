"""
配置管理模块
集中管理所有配置参数，方便查看和修改
"""
import logging
import os
from typing import Optional, Any, List, Dict

logger = logging.getLogger(__name__)


class Config:
    """推理配置类，集中管理所有配置参数"""
    
    def __init__(self):
        # ========== 数据加载器配置 ==========
        # OSS 相关配置（仅在 OSSDataloader 时使用）
        self.meta_dir: Optional[str] = os.environ.get('OSS_META_DIR', '')
        self.endpoint: Optional[str] = os.environ.get('OSS_ENDPOINT', '')
        self.access_key: Optional[str] = os.environ.get('OSS_ACCESS_KEY', '')
        self.secret_key: Optional[str] = os.environ.get('OSS_SECRET_KEY', '')
        self.secure: bool = False
        self.presign_secs: int = 3600
        
        # 数据加载器通用配置
        self.output_sample_rate: int = 16000
        self.shuffle: bool = False
        
        # Lance 数据集相关配置
        self.lance_prompt_key: str = "audio_flac"  # Lance 数据集中要读取的列名
        self.lance_io_config: Optional[Any] = None  # daft.io.IOConfig 对象（用于 S3 访问）
        self.lance_offset: int = 0  # 起始偏移量
        self.lance_limit: Optional[int] = None  # 限制读取数量，None 表示读取全部
        
        # Lance S3 配置（如果数据在 S3 上）
        self.lance_s3_endpoint: Optional[str] = os.environ.get('LANCE_S3_ENDPOINT', '')
        self.lance_s3_access_key_id: Optional[str] = os.environ.get('LANCE_S3_ACCESS_KEY_ID', '')
        self.lance_s3_secret_access_key: Optional[str] = os.environ.get('LANCE_S3_SECRET_ACCESS_KEY', '')
        self.lance_s3_use_ssl: bool = False
        
        # ========== 模型配置 ==========
        self.model_type: str = 'music_cpu_pipeline'
        self.dataloader_type: str = 'jsonl'
        
        # ========== 推理配置 ==========
        self.batch_size: int = 4
        
        # ========== 数据路径 / 模型路径配置 ==========
        self.data_path: str = ''
        self.output_path: str = './outputs'
        # 模型路径（用于 CLI 通过 --cfg 传入）
        self.model_path: str = ''
        
        # ========== 其他配置 ==========
        self.group_by_segment: bool = True
        self.num_dataloader_workers: int = 1  # DataLoaderWorker 数量，用于并行加载数据
        self.num_workers: int = 1
        # Dedicated stage directories may discard unsafe legacy state or state
        # whose input manifest cannot be mapped to the current task identities.
        self.reset_incompatible_output: bool = False
    
    def get_dataloader_kwargs(self) -> dict:
        """获取数据加载器所需的 kwargs"""
        kwargs = {
            'meta_dir': self.meta_dir,
            'endpoint': self.endpoint,
            'access_key': self.access_key,
            'secret_key': self.secret_key,
            'secure': self.secure,
            'output_sample_rate': self.output_sample_rate,
            'presign_secs': self.presign_secs,
            'shuffle': self.shuffle,
        }
        
        # 如果是 Lance 数据加载器，添加 Lance 特定配置
        if self.dataloader_type == 'lance':
            # 如果 io_config 未设置，从 S3 配置创建
            io_config = self.lance_io_config
            if io_config is None and self.lance_s3_access_key_id and self.lance_s3_secret_access_key:
                try:
                    from daft.io import IOConfig, S3Config
                    io_config = IOConfig(
                        s3=S3Config(
                            key_id=self.lance_s3_access_key_id,
                            access_key=self.lance_s3_secret_access_key,
                            endpoint_url=self.lance_s3_endpoint,
                            use_ssl=self.lance_s3_use_ssl,
                        )
                    )
                except ImportError:
                    logger.warning("daft not available, will use manual storage_options")
                    io_config = None
            
            kwargs.update({
                'prompt_key': self.lance_prompt_key,
                'io_config': io_config,
                'offset': self.lance_offset,
                'limit': self.lance_limit,
            })
        
        return kwargs
    
    def update(self, **kwargs):
        """更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config key: {key}")


# 全局配置实例
cfg = Config()


def _cast_to_type(key: str, value: str):
    """
    根据 cfg 中已有字段的类型，把字符串转成正确的类型。
    仅用于命令行传入的字符串覆盖配置。
    """
    if not hasattr(cfg, key):
        raise ValueError(f"Unknown config key: {key}")
    orig = getattr(cfg, key)

    # bool 单独处理
    if isinstance(orig, bool):
        return value.lower() in ("1", "true", "yes", "y", "on")

    if isinstance(orig, int):
        return int(value)
    if isinstance(orig, float):
        return float(value)

    # 其他直接当作字符串
    return value


def parse_cfg_overrides(overrides: List[str]) -> Dict[str, Any]:
    """
    解析命令行传入的 --cfg key=value 形式的覆盖项，并根据 cfg 中字段类型做转换。
    
    Args:
        overrides: 形如 ["batch_size=4", "model_type=music_cpu_pipeline"] 的列表
    """
    updates: Dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--cfg 参数格式错误，应为 key=value，实际得到: {item}")
        key, val = item.split("=", 1)
        key = key.strip()
        val = val.strip()
        updates[key] = _cast_to_type(key, val)
    return updates

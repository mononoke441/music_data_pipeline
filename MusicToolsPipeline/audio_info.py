"""
音频信息数据类
统一数据流动时的格式
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class AudioInfo:
    """
    音频信息数据类，统一数据流动格式
    
    包含音频路径、OSS参数、时间戳、推理结果等信息
    """
    # ========== 音频路径相关 ==========
    audio_path: Optional[str] = None  # 音频文件路径或URL
    url: Optional[str] = None  # URL（兼容字段）
    path: Optional[str] = None  # 路径（兼容字段）
    audio_bytes: Optional[bytes] = None  # 音频数据（bytes格式，用于 Lance 数据集）
    
    # ========== OSS 相关参数（仅在 OSSDataloader 时使用）==========
    bucket: Optional[str] = None  # OSS bucket
    object_key: Optional[str] = None  # OSS object key
    endpoint: Optional[str] = None  # OSS endpoint
    access_key: Optional[str] = None  # OSS access key
    secret_key: Optional[str] = None  # OSS secret key
    secure: bool = False  # 是否使用 HTTPS
    presign_secs: int = 3600  # 预签名有效期（秒）
    
    # ========== 时间戳相关 ==========
    start: Optional[float] = None  # 开始时间（秒）
    end: Optional[float] = None  # 结束时间（秒）
    duration: Optional[float] = None  # 时长（秒）
    
    # ========== 元数据相关 ==========
    subset_name: Optional[str] = None  # 数据集子集名称
    metadata__id: Optional[str] = None  # 元数据ID
    segment_id: Optional[str] = None  # 片段ID
    segment_group: Optional[str] = None  # 片段组
    
    # ========== Lance 主键字段 ==========
    _id: Optional[str] = None  # MongoDB ObjectId（从 Lance 数据集读取）
    metadata_oid: Optional[str] = None  # 元数据 OID（从 Lance 数据集读取）
    segment_key: Optional[str] = None  # 段键（从 Lance 数据集读取）
    
    # ========== 音频处理相关 ==========
    target_sample_rate: int = 16000  # 目标采样率
    
    # ========== 推理结果相关 ==========
    predictions: List[Dict[str, Any]] = field(default_factory=list)  # 预测结果
    error: Optional[str] = None  # 错误信息
    
    # ========== 其他字段 ==========
    # 允许存储额外的字段（用于向后兼容）
    _extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {}
        for key, value in self.__dict__.items():
            if key == '_extra':
                result.update(value)
            elif key == 'audio_bytes':
                # 排除 audio_bytes，避免 JSON 序列化错误和节省存储空间
                continue
            elif value is not None:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AudioInfo':
        """从字典创建 AudioInfo"""
        # 提取已知字段
        known_fields = {
            'audio_path', 'url', 'path', 'audio_bytes', 'bucket', 'object_key', 'endpoint',
            'access_key', 'secret_key', 'secure', 'presign_secs',
            'start', 'end', 'duration', 'subset_name', 'metadata__id',
            'segment_id', 'segment_group', 'target_sample_rate',
            'predictions', 'error',
            # Lance 主键字段
            '_id', 'metadata_oid', 'segment_key'
        }
        
        kwargs = {}
        extra = {}
        
        for key, value in data.items():
            if key in known_fields:
                kwargs[key] = value
            else:
                extra[key] = value
        
        if extra:
            kwargs['_extra'] = extra
        
        return cls(**kwargs)
    
    def get_audio_identifier(self) -> str:
        """获取音频标识符（用于日志和错误信息）"""
        return (
            self.audio_path or 
            self.url or 
            self.path or 
            f"OSS:{self.bucket}/{self.object_key}" if self.bucket and self.object_key else 
            ""
        )
    
    def has_oss_params(self) -> bool:
        """判断是否有 OSS 参数"""
        return self.bucket is not None and self.object_key is not None


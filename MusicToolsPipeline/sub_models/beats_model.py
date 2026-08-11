# -*- coding: utf-8 -*-
"""
BEATs模型实现
"""
import torch
import torchaudio
import numpy as np
from typing import List
from .base_model import BaseModel
from dataloader import decode_oss_audio
from audio_info import AudioInfo
import logging

logger = logging.getLogger(__name__)


class BEATsModel(BaseModel):
    """BEATs音频分类模型"""
    
    def __init__(self, model_name, device="cuda", model_path=None, **kwargs):
        """
        初始化BEATs模型
        
        Args:
            model_name: 模型名称
            device: 设备
            model_path: 模型文件路径
            **kwargs: 其他参数
        """
        self.device = device
        super().__init__(model_name, model_path, **kwargs)
    
    def _load_model(self):
        """加载BEATs模型"""
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), 'beats'))
        from BEATs import BEATs, BEATsConfig
        
        # 加载checkpoint
        checkpoint = torch.load(self.model_path)
        
        # 创建模型
        cfg = BEATsConfig(checkpoint['cfg'])
        self.model = BEATs(cfg)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        self.model.eval()
        
        # 保存标签字典
        self.label_dict = checkpoint['label_dict']
        
        logger.info(f"BEATs model loaded from {self.model_path}")
    
    def load_audio(self, audio_path, target_sample_rate=16000):
        """
        加载音频文件
        
        Args:
            audio_path: 音频文件路径
            target_sample_rate: 目标采样率
            
        Returns:
            audio_tensor: 音频张量
        """
        # 加载音频
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # 转换为单声道
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # 重采样到目标采样率
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
            waveform = resampler(waveform)
        
        return waveform.squeeze(0)  # 移除batch维度
    
    def generate(self, inputs: List[AudioInfo], top_k=5, **kwargs) -> List[AudioInfo]:
        """
        对音频进行分类推理
        
        Args:
            inputs: AudioInfo 对象列表
            top_k: 返回top-k预测结果
            **kwargs: 其他参数
            
        Returns:
            AudioInfo 对象列表（包含预测结果）
        """
        results = []
        for audio_info in inputs:
            # 判断音频数据来源：优先使用 audio_bytes（Lance 数据集）
            if audio_info.audio_bytes is not None:
                # 从 bytes 加载音频
                try:
                    import io
                    import soundfile as sf
                    # 将 bytes 转换为音频数组
                    audio_bytes_io = io.BytesIO(audio_info.audio_bytes)
                    audio_array, sample_rate = sf.read(audio_bytes_io)
                    
                    # 转换为单声道
                    if len(audio_array.shape) > 1:
                        audio_array = np.mean(audio_array, axis=1)
                    
                    # 转换为 torch tensor
                    audio_tensor = torch.from_numpy(audio_array.copy()).float()
                    
                    # 重采样到目标采样率
                    if sample_rate != audio_info.target_sample_rate:
                        # 使用 torchaudio 重采样
                        audio_tensor = audio_tensor.unsqueeze(0)  # 添加通道维度
                        resampler = torchaudio.transforms.Resample(sample_rate, audio_info.target_sample_rate)
                        audio_tensor = resampler(audio_tensor)
                        audio_tensor = audio_tensor.squeeze(0)  # 移除通道维度
                except Exception as e:
                    logger.error(f"Failed to load audio from bytes: {e}")
                    audio_info.error = f"Failed to load audio from bytes: {e}"
                    audio_info.predictions = []
                    results.append(audio_info)
                    continue
            elif audio_info.has_oss_params():
                # 有 OSS 参数：使用 decode_oss_audio 解码
                audio_array = decode_oss_audio(audio_info, audio_info.target_sample_rate)
                # 检查是否为空数组（解码失败）
                if audio_array.size == 0:
                    audio_info.error = "ffmpeg decode failed"
                    audio_info.predictions = []
                    results.append(audio_info)
                    continue
                # 复制数组以确保可写，避免 PyTorch 警告
                audio_tensor = torch.from_numpy(audio_array.copy()).float()
            else:
                # 没有 OSS 参数：使用 load_audio 加载本地文件或 URL
                audio_path = audio_info.audio_path or audio_info.url or audio_info.path
                if not audio_path:
                    logger.error(f"No audio_path found in AudioInfo: {audio_info}")
                    audio_info.error = "No audio_path in input"
                    audio_info.predictions = []
                    results.append(audio_info)
                    continue
                audio_tensor = self.load_audio(audio_path, audio_info.target_sample_rate)
            
            # 添加batch维度
            audio_input = audio_tensor.unsqueeze(0)  # [1, length]
            padding_mask = torch.zeros(1, audio_input.shape[1]).bool().to(self.device)
            
            # 推理
            with torch.no_grad():
                probs = self.model.extract_features(audio_input.to(self.device), padding_mask=padding_mask)[0]
            
            # 获取top-k结果
            top_k_probs, top_k_indices = probs.topk(k=top_k)
            top_k_labels = [self.label_dict[idx.item()] for idx in top_k_indices[0]]
            top_k_prob_values = top_k_probs[0].cpu().numpy().tolist()
            
            # 更新 AudioInfo 的预测结果
            audio_info.predictions = [
                {"label": label, "probability": float(prob)}
                for label, prob in zip(top_k_labels, top_k_prob_values)
            ]
            audio_info.audio_bytes = None
            results.append(audio_info)
        
        return results
    
    def generate_batch(self, batch_data: List[AudioInfo], **kwargs) -> List[AudioInfo]:
        """
        处理一批数据，覆盖父类实现
        
        Args:
            batch_data: AudioInfo 对象列表
            **kwargs: 其他参数
            
        Returns:
            AudioInfo 对象列表（包含预测结果）
        """
        # 直接调用 generate，传递 AudioInfo 列表
        return self.generate(batch_data, **kwargs)
    
    def cleanup(self):
        """清理模型资源"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

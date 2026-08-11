# -*- coding: utf-8 -*-
"""
BeatNet 节拍/速度检测模型
"""
from typing import List
from .base_model import BaseModel
from audio_info import AudioInfo
import logging
import numpy as np

logger = logging.getLogger(__name__)


class _MadmomNumpyCompat:
    """NumPy facade for madmom releases that still target NumPy < 1.24."""

    def __getattr__(self, name):
        if name == "int":
            return int
        return getattr(np, name)

    @staticmethod
    def asarray(value, *args, **kwargs):
        try:
            return np.asarray(value, *args, **kwargs)
        except ValueError:
            # DBNDownBeatTrackingProcessor stores variable-length state paths
            # beside scalar log probabilities. New NumPy versions require the
            # object dtype to construct this intentionally ragged 2-D array.
            if "dtype" not in kwargs and not args:
                return np.asarray(value, dtype=object)
            raise


class BeatNetModel(BaseModel):
    """使用 BeatNet 检测节拍与估计 BPM"""

    def _load_model(self):
        try:
            from madmom.features import downbeats as madmom_downbeats
            from BeatNet.BeatNet import BeatNet
            madmom_downbeats.np = _MadmomNumpyCompat()
            # offline + DBN 与 tools/run.py 保持一致
            self.estimator = BeatNet(1, mode='offline', inference_model='DBN', plot=[], thread=False)
        except Exception as e:
            logger.error(f"Failed to init BeatNet: {e}")
            raise

    def generate(self, inputs: List[AudioInfo], **kwargs) -> List[AudioInfo]:
        results: List[AudioInfo] = []
        for audio_info in inputs:
            # 优先使用 audio_bytes（Lance 数据集）
            audio_input = None
            if audio_info.audio_bytes is not None and isinstance(audio_info.audio_bytes, bytes):
                audio_input = audio_info.audio_bytes
            else:
                # 回退到文件路径
                audio_path = audio_info.audio_path or audio_info.url or audio_info.path
                if not audio_path:
                    audio_info.error = audio_info.error or "No audio_path or audio_bytes in input"
                    results.append(audio_info)
                    continue
                audio_input = audio_path
            
            try:
                # 尝试直接传 bytes 或路径给 process 方法
                beat_output = self.estimator.process(audio_input)
                beats = []
                timestamps = []
                max_beat_number = 0
                for b in beat_output:
                    ts = float(b[0])
                    beat_num = int(b[1]) if len(b) > 1 else 0
                    beats.append({"timestamp": ts, "beat": beat_num})
                    timestamps.append(ts)
                    max_beat_number = max(max_beat_number, beat_num)
                bpm = None
                if len(timestamps) > 1:
                    intervals = np.diff(timestamps)
                    intervals = intervals[(intervals > 0.1) & (intervals < 3.0)]
                    if len(intervals) > 0:
                        bpm = 60.0 / float(np.mean(intervals))
                audio_info._extra["beatnet"] = {
                    "values": beats,
                    "max_beat_number": max_beat_number,
                    "bpm": float(bpm) if bpm is not None else None,
                }
            except Exception as e:
                audio_info._extra["beatnet_error"] = str(e)
            results.append(audio_info)
        return results

    def generate_batch(self, batch_data: List[AudioInfo]) -> List[AudioInfo]:  # type: ignore[override]
        """BeatNet 直接接受 AudioInfo 列表，不使用基类的字符串提取逻辑。"""
        normalized_inputs: List[AudioInfo] = []
        for item in batch_data:
            if isinstance(item, AudioInfo):
                audio_info = item
            elif isinstance(item, dict):
                audio_info = AudioInfo.from_dict(item)
            else:
                # 假设是音频路径字符串
                audio_info = AudioInfo(audio_path=str(item))
            if audio_info._extra is None:
                audio_info._extra = {}
            normalized_inputs.append(audio_info)
        return self.generate(normalized_inputs)
    
    def cleanup(self):
        # 无需特殊清理
        pass


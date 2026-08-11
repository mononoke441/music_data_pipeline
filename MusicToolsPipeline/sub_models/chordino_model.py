# -*- coding: utf-8 -*-
"""
Chordino 和弦检测模型
"""
from typing import List
from .base_model import BaseModel
from audio_info import AudioInfo
# from .chord_extractor.extractors import Chordino
import logging

logger = logging.getLogger(__name__)


class ChordinoModel(BaseModel):
    """使用 Chordino 进行和弦检测"""

    def _load_model(self):
        try:
            from chord_extractor.extractors import Chordino
            # 缓存提取器
            self.extractor = Chordino()
        except Exception as e:
            logger.error(f"Failed to init Chordino: {e}")
            raise

    def generate(self, inputs: List[AudioInfo], **kwargs) -> List[AudioInfo]:
        results: List[AudioInfo] = []
        for audio_info in inputs:
            audio_source = None
            if getattr(audio_info, "audio_bytes", None):
                # extractor 已支持直接传入 bytes
                audio_source = audio_info.audio_bytes
            else:
                audio_source = audio_info.audio_path or audio_info.url or audio_info.path

            if not audio_source:
                audio_info.error = audio_info.error or "No audio data available"
                results.append(audio_info)
                continue
            try:
                chords = self.extractor.extract(audio_source)
                chords_filtered = [c for c in chords if getattr(c, "chord", None) != "N"]
                chords_with_time = [
                    {"chord": c.chord, "timestamp": float(c.timestamp)}
                    for c in chords_filtered
                ]
                # 写入到 _extra，避免与通用 predictions 冲突
                audio_info._extra["chords"] = {
                    "values": chords_with_time,
                    "progression": [c["chord"] for c in chords_with_time],
                }
            except Exception as e:
                audio_info._extra["chords_error"] = str(e)
            results.append(audio_info)
        return results
    
    def generate_batch(self, batch_data: List[AudioInfo]) -> List[AudioInfo]:  # type: ignore[override]
        """Chordino 直接接受 AudioInfo 列表，不使用基类的字符串提取逻辑。"""
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



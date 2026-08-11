# -*- coding: utf-8 -*-
"""
音乐分析 CPU 流水线模型

提供生产 CPU MIR Pipeline：
- MusicCpuPipelineModel: 全曲处理（Chordino + BeatNet + Essentia），输出 chords/beatnet/key
"""
from typing import List
from .base_model import BaseModel
from audio_info import AudioInfo
import logging

logger = logging.getLogger(__name__)


def _publish_stage_status(audio_info: AudioInfo, required_fields: List[str]) -> None:
    """Expose nested CPU MIR failures in the canonical stage envelope."""
    music_cpu = audio_info._extra.get("music_cpu") or {}
    errors = {
        key: value
        for key, value in music_cpu.items()
        if key.endswith("_error") and value
    }
    error_key_by_field = {"chords": "chords_error", "beatnet": "beatnet_error", "key": "essentia_error"}
    for field in required_fields:
        known_error = error_key_by_field.get(field, f"{field}_error")
        if field not in music_cpu and known_error not in music_cpu:
            errors[f"{field}_error"] = f"missing_{field}_output"
    stage_status = dict(audio_info._extra.get("stage_status") or {})
    stage_errors = dict(audio_info._extra.get("stage_errors") or {})
    if errors:
        stage_status["music_cpu"] = "partial_error"
        stage_errors["music_cpu"] = errors
    else:
        stage_status["music_cpu"] = "ok"
        stage_errors.pop("music_cpu", None)
    audio_info._extra["stage_status"] = stage_status
    audio_info._extra["stage_errors"] = stage_errors


class MusicCpuPipelineModel(BaseModel):
    """
    全曲 CPU 流水线：Chordino → BeatNet → Essentia
    输出 music_cpu: {chords, beatnet, key}

    用于 Step 1b 全曲 MIR 特征提取。
    """

    def _load_model(self):
        try:
            # MusicCpuPipelineModel already runs inside one Ray ModelWorker.
            # Loading another set of Ray actors here creates a nested Ray graph,
            # duplicates process state and can mix incompatible Python runtimes.
            # Keep the three CPU models in the owning actor instead.
            from .chordino_model import ChordinoModel
            from .beatnet_model import BeatNetModel
            from .essentia_model import EssentiaModel

            self.chordino_model = ChordinoModel(model_name="Chordino")
            self.beatnet_model = BeatNetModel(model_name="BeatNet")
            self.essentia_model = EssentiaModel(model_name="Essentia")
        except Exception as e:
            logger.error(f"Failed to init MusicCpuPipelineModel: {e}")
            raise

    def generate(self, inputs: List[AudioInfo], **kwargs) -> List[AudioInfo]:
        for audio_info in inputs:
            if audio_info._extra is None:
                audio_info._extra = {}

        current_results = inputs

        # 1) Chordino
        try:
            current_results = self.chordino_model.generate_batch(current_results)
            logger.debug(f"Chordino processed {len(current_results)} items")
        except Exception as e:
            logger.warning(f"Chordino failed: {e}, continuing")
            for ai in current_results:
                ai._extra.setdefault("chords_error", f"Chordino failed: {e}")

        # 2) BeatNet
        try:
            current_results = self.beatnet_model.generate_batch(current_results)
            logger.debug(f"BeatNet processed {len(current_results)} items")
        except Exception as e:
            logger.warning(f"BeatNet failed: {e}, continuing")
            for ai in current_results:
                ai._extra.setdefault("beatnet_error", f"BeatNet failed: {e}")

        # 3) Essentia
        try:
            current_results = self.essentia_model.generate_batch(current_results)
            logger.debug(f"Essentia processed {len(current_results)} items")
        except Exception as e:
            logger.warning(f"Essentia failed: {e}")
            for ai in current_results:
                ai._extra.setdefault("essentia_error", f"Essentia failed: {e}")

        for audio_info in current_results:
            if audio_info._extra is None:
                audio_info._extra = {}

            music_cpu = {}

            if "chords" in audio_info._extra:
                music_cpu["chords"] = audio_info._extra["chords"]
            if "chords_error" in audio_info._extra:
                music_cpu["chords_error"] = audio_info._extra["chords_error"]

            if "beatnet" in audio_info._extra:
                music_cpu["beatnet"] = audio_info._extra["beatnet"]
            if "beatnet_error" in audio_info._extra:
                music_cpu["beatnet_error"] = audio_info._extra["beatnet_error"]

            if "key" in audio_info._extra:
                music_cpu["key"] = audio_info._extra["key"]
            if "essentia_error" in audio_info._extra:
                music_cpu["essentia_error"] = audio_info._extra["essentia_error"]

            audio_info._extra["music_cpu"] = music_cpu

            audio_info._extra.pop("chords", None)
            audio_info._extra.pop("chords_error", None)
            audio_info._extra.pop("beatnet", None)
            audio_info._extra.pop("beatnet_error", None)
            audio_info._extra.pop("key", None)
            audio_info._extra.pop("essentia_error", None)
            _publish_stage_status(audio_info, ["chords", "beatnet", "key"])

        return current_results

    def generate_batch(self, batch_data: List[AudioInfo]) -> List[AudioInfo]:
        normalized_inputs: List[AudioInfo] = []
        for item in batch_data:
            if isinstance(item, AudioInfo):
                audio_info = item
            elif isinstance(item, dict):
                audio_info = AudioInfo.from_dict(item)
            else:
                audio_info = AudioInfo(audio_path=str(item))
            if audio_info._extra is None:
                audio_info._extra = {}
            normalized_inputs.append(audio_info)
        return self.generate(normalized_inputs)

    def cleanup(self):
        pass

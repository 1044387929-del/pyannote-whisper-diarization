"""说话人分割"""
import math
import tempfile
from pathlib import Path
from typing import Generator

import torch
import torchaudio

from pyannote.audio import Pipeline
from pyannote.core import Annotation

from .config import CONFIG_PATH


class DiarizationPipeline:
    """pyannote 说话人分割流水线，封装模型加载与设备选择。"""

    def __init__(self, config_path: str | Path | None = None, device: torch.device | None = None):
        self.config_path = Path(config_path or CONFIG_PATH)
        self._pipeline: Pipeline | None = None
        self._device = device

    @property
    def pipeline(self) -> Pipeline:
        if self._pipeline is None:
            self._pipeline = Pipeline.from_pretrained(str(self.config_path))
            device = self._device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
            self._pipeline.to(device)
        return self._pipeline

    def diarize(self, audio_path: str | Path) -> Annotation:
        """对整段音频做说话人分割，返回标注。"""
        result = self.pipeline(str(audio_path))
        return result.speaker_diarization

    def diarize_chunked(
        self,
        audio_path: str | Path,
        chunk_duration: float = 15,
    ) -> Generator[tuple[float, float, str], None, None]:
        """按块切分：每块算完就 yield (start, end, speaker)。"""
        waveform, sample_rate = torchaudio.load(str(audio_path))
        duration_s = waveform.shape[1] / sample_rate
        num_chunks = max(1, math.ceil(duration_s / chunk_duration))

        for i in range(num_chunks):
            start_s = i * chunk_duration
            end_s = min(start_s + chunk_duration, duration_s)
            if start_s >= duration_s:
                break
            start_sample = int(start_s * sample_rate)
            end_sample = int(end_s * sample_rate)
            chunk_wav = waveform[:, start_sample:end_sample]
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
            try:
                torchaudio.save(tmp_path, chunk_wav, sample_rate)
                result = self.pipeline(tmp_path)
                anno = result.speaker_diarization
                for turn, _, speaker in anno.itertracks(yield_label=True):
                    yield start_s + turn.start, start_s + turn.end, speaker
            finally:
                Path(tmp_path).unlink(missing_ok=True)


_default_pipeline: DiarizationPipeline | None = None


def _get_pipeline() -> Pipeline:
    """兼容旧接口：返回 pyannote Pipeline 实例。"""
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = DiarizationPipeline()
    return _default_pipeline.pipeline


def diarize_whole(audio_path: str | Path) -> Annotation:
    """整体切分：对整段音频一次性做说话人分割。"""
    return DiarizationPipeline().diarize(audio_path)


def diarize_chunked(
    audio_path: str | Path,
    chunk_duration: float = 15,
) -> Generator[tuple[float, float, str], None, None]:
    """按块切分：每块算完就 yield 该块结果。"""
    yield from DiarizationPipeline().diarize_chunked(audio_path, chunk_duration)

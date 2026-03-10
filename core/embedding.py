"""
本脚本用于提取音频的声纹 embedding
提取器：封装 pyannote 模型
方法：从音频文件提取 embedding 或从波形数组提取 embedding
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torchaudio

from core.config import EMBEDDING_MODEL_PATH

if TYPE_CHECKING:
    pass


class EmbeddingExtractor:
    """声纹 embedding 提取器，封装 pyannote 模型。"""

    def __init__(
        self,
        # 模型路径
        model_path: str | Path | None = None,
        # 设备
        device: torch.device | None = None,
    ):
        self.model_path = Path(model_path or EMBEDDING_MODEL_PATH)
        self._device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None

    @property
    def _extractor(self):
        if self._model is None:
            # 从 pyannote 模型库加载 speaker verification 模型，
            # 这个模型是用于提取音频的声纹 embedding 的
            from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
            # 加载模型
            self._model = PretrainedSpeakerEmbedding({"checkpoint": str(self.model_path)}, device=self._device)
        return self._model

    def from_path(self, audio_path: str | Path) -> np.ndarray:
        """从音频文件提取 embedding。"""
        fn = self._extractor
        waveform, sr = torchaudio.load(str(audio_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != fn.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, fn.sample_rate)
        waveform = waveform.unsqueeze(0)
        emb = fn(waveform)[0]
        return emb.cpu().numpy() if hasattr(emb, "cpu") else np.asarray(emb)

    def from_waveform(
        self,
        waveform: np.ndarray | torch.Tensor,
        sample_rate: int,
    ) -> np.ndarray:
        """从波形数组提取 embedding，shape: (C, T)。"""
        fn = self._extractor
        w = torch.as_tensor(waveform, dtype=torch.float32)
        if w.dim() == 1:
            w = w.unsqueeze(0)
        if w.shape[0] > 1:
            w = w.mean(dim=0, keepdim=True)
        if sample_rate != fn.sample_rate:
            w = torchaudio.functional.resample(w, sample_rate, fn.sample_rate)
        w = w.unsqueeze(0)
        emb = fn(w)[0]
        return emb.cpu().numpy() if hasattr(emb, "cpu") else np.asarray(emb)


def get_embedding(
    audio_path: str | Path,
    model_path: str | Path | None = None,
    device: torch.device | None = None,
) -> np.ndarray:
    """对整段音频提取一条 speaker embedding 向量。"""
    return EmbeddingExtractor(model_path=model_path, device=device).from_path(audio_path)


def get_embedding_from_waveform(
    waveform: np.ndarray | torch.Tensor,
    sample_rate: int,
    model_path: str | Path | None = None,
    device: torch.device | None = None,
) -> np.ndarray:
    """从波形数组提取 embedding。"""
    return EmbeddingExtractor(model_path=model_path, device=device).from_waveform(waveform, sample_rate)

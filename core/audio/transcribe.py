"""Whisper 转写"""
import tempfile
from pathlib import Path
from typing import Iterator

import torch
from faster_whisper import WhisperModel

from .config import WHISPER_MODEL_PATH


class WhisperTranscriber:
    """Faster Whisper 转写器。"""

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str | None = None,
        compute_type: str | None = None,
    ):
        self.model_path = Path(model_path or WHISPER_MODEL_PATH)
        self._model: WhisperModel | None = None
        self._device = device
        self._compute_type = compute_type

    @property
    def model(self) -> WhisperModel:
        if self._model is None:
            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            compute_type = self._compute_type or ("float16" if device == "cuda" else "int8")
            self._model = WhisperModel(str(self.model_path), device=device, compute_type=compute_type)
        return self._model

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
        vad_filter: bool = True,
    ) -> str:
        """对音频文件做转写，返回全文。"""
        segments, _ = self.model.transcribe(str(audio_path), language=language, vad_filter=vad_filter)
        return "".join(s.text for s in segments).strip()

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        language: str | None = None,
    ) -> str:
        """对 wav 字节流做转写，用于 Live 模式。"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            tmp_path = f.name
        try:
            return self.transcribe(tmp_path, language=language)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def transcribe_stream(
        self,
        audio_path: str | Path,
        language: str | None = None,
    ) -> Iterator[tuple[float, float, str]]:
        """流式转写：yield (start, end, text)。"""
        segments, _ = self.model.transcribe(str(audio_path), language=language, vad_filter=True)
        for s in segments:
            yield s.start, s.end, s.text or ""


_default_transcriber: WhisperTranscriber | None = None


def _get_whisper_model() -> WhisperModel:
    global _default_transcriber
    if _default_transcriber is None:
        _default_transcriber = WhisperTranscriber()
    return _default_transcriber.model


def transcribe_chunk(audio_bytes: bytes, language: str | None = None) -> str:
    """对单块音频（wav）做 Whisper 转写，用于 live 模式。"""
    return WhisperTranscriber().transcribe_bytes(audio_bytes, language=language)


def transcribe_audio(
    audio_path: str,
    model_path: str | Path | None = None,
    device: str = "cuda",
    compute_type: str = "float16",
):
    """使用 Faster Whisper 转写音频，逐段打印。"""
    t = WhisperTranscriber(model_path=model_path, device=device, compute_type=compute_type)
    segments, _ = t.model.transcribe(audio_path)
    for seg in segments:
        print(f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}")


def transcribe_to_text(
    audio_path: str,
    model_path: str | Path | None = None,
    language: str | None = None,
) -> str:
    """转写音频并返回全文。"""
    return WhisperTranscriber(model_path=model_path).transcribe(audio_path, language=language)

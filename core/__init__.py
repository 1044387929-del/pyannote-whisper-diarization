"""
pyannote_diarization 核心模块：说话人分割、声纹 embedding、转写
"""
from pathlib import Path

from .config import BASE_DIR, EMBEDDING_MODEL_PATH
from .embedding import get_embedding, get_embedding_from_waveform
from .diarize import diarize_whole, diarize_chunked
from .transcribe import transcribe_audio

__all__ = [
    "BASE_DIR",
    "EMBEDDING_MODEL_PATH",
    "get_embedding",
    "get_embedding_from_waveform",
    "diarize_whole",
    "diarize_chunked",
    "transcribe_audio",
]

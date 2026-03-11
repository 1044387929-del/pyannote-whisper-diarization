"""
core：音频转录（audio）+ LLM（llm）

- 音频转录：embedding、diarize、transcribe、pipeline
- LLM：llm_client、refine_pipeline
"""
from pathlib import Path

# 兼容旧导入：从 core.audio 转发
from core.audio.config import BASE_DIR, CONFIG_PATH, EMBEDDING_MODEL_PATH
from core.audio.embedding import get_embedding, get_embedding_from_waveform, EmbeddingExtractor
from core.audio.diarize import diarize_whole, diarize_chunked
from core.audio.transcribe import transcribe_audio, transcribe_chunk
from core.audio.pipeline import (
    transcribe_with_speakers,
    transcribe_chunk_with_speakers,
    transcribe_with_speakers_stream,
)

__all__ = [
    "BASE_DIR",
    "CONFIG_PATH",
    "EMBEDDING_MODEL_PATH",
    "get_embedding",
    "get_embedding_from_waveform",
    "EmbeddingExtractor",
    "diarize_whole",
    "diarize_chunked",
    "transcribe_audio",
    "transcribe_chunk",
    "transcribe_with_speakers",
    "transcribe_chunk_with_speakers",
    "transcribe_with_speakers_stream",
]

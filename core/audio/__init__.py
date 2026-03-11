"""音频转录相关：配置、embedding、diarize、transcribe、pipeline"""
from .config import (
    BASE_DIR,
    CONFIG_PATH,
    DATA_DIR,
    EMBEDDING_MODEL_PATH,
    PLDA_DIR,
    WHISPER_MODEL_PATH,
)
from .embedding import EmbeddingExtractor, get_embedding, get_embedding_from_waveform
from .diarize import DiarizationPipeline, diarize_chunked, diarize_whole
from .transcribe import (
    WhisperTranscriber,
    transcribe_audio,
    transcribe_chunk,
    transcribe_to_text,
)
from .pipeline import (
    DiarizedTranscriber,
    Utterance,
    transcribe_with_speakers,
    transcribe_chunk_with_speakers,
    transcribe_with_speakers_stream,
)

__all__ = [
    "BASE_DIR",
    "CONFIG_PATH",
    "DATA_DIR",
    "EMBEDDING_MODEL_PATH",
    "PLDA_DIR",
    "WHISPER_MODEL_PATH",
    "EmbeddingExtractor",
    "get_embedding",
    "get_embedding_from_waveform",
    "DiarizationPipeline",
    "diarize_whole",
    "diarize_chunked",
    "WhisperTranscriber",
    "transcribe_audio",
    "transcribe_chunk",
    "transcribe_to_text",
    "DiarizedTranscriber",
    "Utterance",
    "transcribe_with_speakers",
    "transcribe_chunk_with_speakers",
    "transcribe_with_speakers_stream",
]

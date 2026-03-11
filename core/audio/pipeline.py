"""综合流程：说话人分割 + 声纹匹配 + Whisper 转写"""
from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import numpy as np
import torch
import torchaudio

from speakers import SpeakerRegistry

from .diarize import DiarizationPipeline
from .embedding import EmbeddingExtractor
from .transcribe import WhisperTranscriber

MIN_DURATION_FOR_EMBEDDING = 0.5


@dataclass
class Utterance:
    start: float
    end: float
    speaker: str
    student_id: str
    text: str

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "speaker": self.speaker,
            "student_id": self.student_id,
            "text": self.text,
        }


class DiarizedTranscriber:
    """带说话人识别的转写：pyannote 分割 + 声纹匹配 + Whisper 转写。"""

    def __init__(self, language: str | None = None, match_threshold: float = 0.5):
        self.language = language
        self.match_threshold = match_threshold
        self._diarize = DiarizationPipeline()
        self._embedding = EmbeddingExtractor()
        self._whisper = WhisperTranscriber()

    def transcribe(self, audio_path: str | Path, speakers: list[dict]) -> list[Utterance]:
        audio_path = Path(audio_path)
        registry = SpeakerRegistry.from_speakers_list(speakers)
        anno = self._diarize.diarize(audio_path)
        waveform, sample_rate = torchaudio.load(str(audio_path))
        utterances: list[Utterance] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            for turn, _, _ in anno.itertracks(yield_label=True):
                t_start, t_end = turn.start, turn.end
                speaker, student_id = self._match_speaker(turn, waveform, sample_rate, registry)
                text = self._transcribe_segment(turn, waveform, sample_rate, tmpdir)
                utterances.append(Utterance(start=t_start, end=t_end, speaker=speaker, student_id=student_id, text=text))
        return utterances

    def transcribe_stream(self, audio_path: str | Path, speakers: list[dict]) -> Generator[dict, None, None]:
        audio_path = Path(audio_path)
        registry = SpeakerRegistry.from_speakers_list(speakers)
        t0_d = time.perf_counter()
        anno = self._diarize.diarize(audio_path)
        diarization_seconds = round(time.perf_counter() - t0_d, 2)
        waveform, sample_rate = torchaudio.load(str(audio_path))
        turns = list(anno.itertracks(yield_label=True))
        total = len(turns)
        whisper_total_seconds = 0.0
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            for idx, (turn, _, _) in enumerate(turns):
                t_start, t_end = turn.start, turn.end
                speaker, student_id = self._match_speaker(turn, waveform, sample_rate, registry)
                t0_w = time.perf_counter()
                text = self._transcribe_segment(turn, waveform, sample_rate, tmpdir)
                whisper_total_seconds += time.perf_counter() - t0_w
                index = idx + 1
                progress = round(index / total * 100, 1) if total else 100.0
                yield {
                    "start": t_start, "end": t_end, "speaker": speaker, "student_id": student_id, "text": text,
                    "index": index, "total": total, "progress": progress,
                }
        yield {
            "status": "done", "total": total, "progress": 100.0,
            "diarization_seconds": diarization_seconds, "whisper_seconds": round(whisper_total_seconds, 2),
        }

    def _match_speaker(self, turn, waveform: torch.Tensor, sample_rate: int, registry: SpeakerRegistry) -> tuple[str, str]:
        t_start, t_end = turn.start, turn.end
        if not registry.name_to_emb or (t_end - t_start) < MIN_DURATION_FOR_EMBEDDING:
            return "unknown", "unknown"
        try:
            seg_wav = waveform[:, int(t_start * sample_rate) : int(t_end * sample_rate)].numpy()
            seg_emb = self._embedding.from_waveform(seg_wav, sample_rate)
            return registry.match(seg_emb, threshold=self.match_threshold)
        except (AssertionError, ValueError):
            return "unknown", "unknown"

    def _transcribe_segment(self, turn, waveform: torch.Tensor, sample_rate: int, tmpdir: Path) -> str:
        t_start, t_end = turn.start, turn.end
        seg_path = tmpdir / f"seg_{t_start:.1f}.wav"
        seg_audio = waveform[:, int(t_start * sample_rate) : int(t_end * sample_rate)]
        torchaudio.save(str(seg_path), seg_audio, sample_rate)
        return self._whisper.transcribe(seg_path, language=self.language)


def transcribe_with_speakers(
    audio_path: str | Path,
    speakers: list[dict],
    language: str | None = None,
    match_threshold: float = 0.5,
) -> list[dict]:
    transcriber = DiarizedTranscriber(language=language, match_threshold=match_threshold)
    return [u.to_dict() for u in transcriber.transcribe(audio_path, speakers)]


def transcribe_chunk_with_speakers(
    audio_bytes: bytes,
    speakers: list[dict],
    language: str | None = None,
    match_threshold: float = 0.5,
) -> list[dict]:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        return transcribe_with_speakers(tmp_path, speakers, language=language, match_threshold=match_threshold)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def transcribe_with_speakers_stream(
    audio_path: str | Path,
    speakers: list[dict],
    language: str | None = None,
    match_threshold: float = 0.5,
):
    transcriber = DiarizedTranscriber(language=language, match_threshold=match_threshold)
    yield from transcriber.transcribe_stream(audio_path, speakers)

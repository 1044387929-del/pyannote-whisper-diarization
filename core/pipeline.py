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
    """转写输出：单个说话片段"""
    # 说话片段开始时间，单位：秒
    start: float
    # 说话片段结束时间，单位：秒
    end: float
    # 说话人姓名
    speaker: str
    # 说话人学号
    student_id: str
    # 说话片段文本
    text: str

    # 将 Utterance 对象转换为字典·
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

    def __init__(
        self,
        # Whisper 语言
        language: str | None = None,
        # 声纹匹配阈值
        match_threshold: float = 0.5,
    ):
        self.language = language
        self.match_threshold = match_threshold
        self._diarize = DiarizationPipeline()
        self._embedding = EmbeddingExtractor()
        self._whisper = WhisperTranscriber()

    def transcribe(
        self,
        audio_path: str | Path,
        speakers: list[dict],
    ) -> list[Utterance]:
        """对音频做完整转写，用提供的声纹匹配说话人。"""
        audio_path = Path(audio_path)
        registry = SpeakerRegistry.from_speakers_list(speakers)
        anno = self._diarize.diarize(audio_path)
        waveform, sample_rate = torchaudio.load(str(audio_path))

        utterances: list[Utterance] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            for turn, _, _ in anno.itertracks(yield_label=True):
                t_start, t_end = turn.start, turn.end
                speaker, student_id = self._match_speaker(
                    turn, waveform, sample_rate, registry,
                )
                text = self._transcribe_segment(
                    turn, waveform, sample_rate, tmpdir,
                )
                utterances.append(Utterance(
                    start=t_start,
                    end=t_end,
                    speaker=speaker,
                    student_id=student_id,
                    text=text,
                ))
        return utterances

    def transcribe_stream(
        self,
        audio_path: str | Path,
        speakers: list[dict],
    ) -> Generator[dict, None, None]:
        """流式版本：每完成一个说话片段就 yield 一个结果。"""
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
                speaker, student_id = self._match_speaker(
                    turn, waveform, sample_rate, registry,
                )
                t0_w = time.perf_counter()
                text = self._transcribe_segment(
                    turn, waveform, sample_rate, tmpdir,
                )
                whisper_total_seconds += time.perf_counter() - t0_w

                index = idx + 1
                progress = round(index / total * 100, 1) if total else 100.0
                yield {
                    "start": t_start,
                    "end": t_end,
                    "speaker": speaker,
                    "student_id": student_id,
                    "text": text,
                    "index": index,
                    "total": total,
                    "progress": progress,
                }

        yield {
            "status": "done",
            "total": total,
            "progress": 100.0,
            "diarization_seconds": diarization_seconds,
            "whisper_seconds": round(whisper_total_seconds, 2),
        }

    def _match_speaker(
        self,
        turn,
        waveform: torch.Tensor,
        sample_rate: int,
        registry: SpeakerRegistry,
    ) -> tuple[str, str]:
        """对单个 turn 做声纹匹配，返回 (speaker, student_id)。"""
        t_start, t_end = turn.start, turn.end
        duration = t_end - t_start
        if not registry.name_to_emb or duration < MIN_DURATION_FOR_EMBEDDING:
            return "unknown", "unknown"
        try:
            seg_wav = waveform[
                :,
                int(t_start * sample_rate) : int(t_end * sample_rate),
            ].numpy()
            seg_emb = self._embedding.from_waveform(seg_wav, sample_rate)
            return registry.match(seg_emb, threshold=self.match_threshold)
        except (AssertionError, ValueError):
            return "unknown", "unknown"

    def _transcribe_segment(
        self,
        turn,
        waveform: torch.Tensor,
        sample_rate: int,
        tmpdir: Path,
    ) -> str:
        """对单个 turn 做 Whisper 转写。"""
        t_start, t_end = turn.start, turn.end
        seg_path = tmpdir / f"seg_{t_start:.1f}.wav"
        seg_audio = waveform[
            :,
            int(t_start * sample_rate) : int(t_end * sample_rate),
        ]
        torchaudio.save(str(seg_path), seg_audio, sample_rate)
        return self._whisper.transcribe(seg_path, language=self.language)


# ---------- 兼容旧接口：返回 dict 格式 ----------


def transcribe_with_speakers(
    audio_path: str | Path,
    speakers: list[dict],
    language: str | None = None,
    match_threshold: float = 0.5,
) -> list[dict]:
    """对音频做完整转写（兼容旧接口，返回 list[dict]）。"""
    transcriber = DiarizedTranscriber(language=language, match_threshold=match_threshold)
    utterances = transcriber.transcribe(audio_path, speakers)
    return [u.to_dict() for u in utterances]


def transcribe_chunk_with_speakers(
    audio_bytes: bytes,
    speakers: list[dict],
    language: str | None = None,
    match_threshold: float = 0.5,
) -> list[dict]:
    """对单块音频做转写，用于 Live WebSocket（兼容旧接口）。"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        return transcribe_with_speakers(
            tmp_path,
            speakers,
            language=language,
            match_threshold=match_threshold,
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def transcribe_with_speakers_stream(
    audio_path: str | Path,
    speakers: list[dict],
    language: str | None = None,
    match_threshold: float = 0.5,
):
    """流式版本（兼容旧接口）。"""
    transcriber = DiarizedTranscriber(language=language, match_threshold=match_threshold)
    yield from transcriber.transcribe_stream(audio_path, speakers)

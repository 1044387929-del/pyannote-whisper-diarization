"""pyannote_diarization 复用工具：时间格式化、音频转换、后缀校验"""
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Final
from pydub import AudioSegment

# 支持的音频后缀
ALLOWED_EXT: Final= {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm"}


def secs_to_hms(secs: float) -> str:
    """秒数转为 小时:分钟:秒.毫秒，如 01:23:45.678"""
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = int(secs % 60)
    ms = int((secs % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def get_audio_suffix(filename: str | None) -> str:
    """从文件名取音频后缀，无后缀则返回 .wav"""
    if not filename:
        return ".wav"
    suf = Path(filename).suffix.lower()
    return suf if suf else ".wav"


def webm_to_wav(audio_bytes: bytes) -> bytes:
    """将 webm 转为 wav，供 Whisper 使用。需要 pydub 和 ffmpeg。"""
    webm_path: str | None = None
    wav_path: str | None = None
    try:
        # 将 bytes 写入临时文件
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
            f.write(audio_bytes)
            webm_path = f.name
        # 使用 pydub 将 webm 转为 wav
        seg = AudioSegment.from_file(webm_path, format="webm")
        # 将 wav 写入临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as w:
            seg.export(w.name, format="wav")
            wav_path = w.name
        return Path(wav_path).read_bytes()
    finally:
        # 删除临时的 webm 文件
        if webm_path:
            Path(webm_path).unlink(missing_ok=True)
        # 删除临时的 wav 文件
        if wav_path:
            Path(wav_path).unlink(missing_ok=True)


@contextmanager
def temp_audio_file(content: bytes, suffix: str = ".wav"):
    """上下文管理器：将 bytes 写入临时音频文件，用毕自动删除。"""
    path = None
    try:
        # 将 bytes 写入临时文件
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(content)
            path = f.name
        yield path
    finally:
        if path:
            Path(path).unlink(missing_ok=True)

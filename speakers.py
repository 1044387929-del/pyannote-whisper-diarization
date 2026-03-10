"""声纹加载与说话人匹配"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from core.embedding import EmbeddingExtractor


class SpeakerRegistry:
    """说话人注册表：name_to_emb + 匹配逻辑。"""

    def __init__(
        self,
        # 说话人姓名到 embedding 的映射
        name_to_emb: dict[str, np.ndarray] | None = None,
        # 说话人姓名到学号的映射
        name_to_student_id: dict[str, str] | None = None,
    ):
        self.name_to_emb = name_to_emb or {}
        self.name_to_student_id = name_to_student_id or {k: k for k in self.name_to_emb}
        # fronzenset 是不可变集合，用于存储允许的说话人姓名
        self._allowed_names = frozenset(self.name_to_emb.keys())

    # 类方法，从说话人列表构建 SpeakerRegistry
    # 类方法可以被类直接调用，不需要实例化
    # 这里用类方法更好，是因为我们不需要实例化 SpeakerRegistry，只需要构建一个 SpeakerRegistry 对象
    @classmethod
    # 
    def from_speakers_list(cls, speakers: list[dict]) -> "SpeakerRegistry":
        """
        从 [{"student_id","name","embedding"}, ...] 构建。
        @param
        speakers: list[dict] 说话人列表，每个说话人是一个字典，包含 student_id、name、embedding 字段
        @return
        SpeakerRegistry 说话人注册表
        @raise
        ValueError: 如果说话人列表为空
        @raise
        ValueError: 如果说话人列表中的说话人姓名为空
        @raise
        ValueError: 如果说话人列表中的说话人embedding为空
        @raise
        ValueError: 如果说话人列表中的说话人embedding为空
        """
        name_to_emb: dict[str, np.ndarray] = {}
        name_to_student_id: dict[str, str] = {}
        for s in speakers:
            name = (s.get("name") or s.get("student_id") or "").strip()
            sid = (s.get("student_id") or name or "unknown").strip()
            emb = s.get("embedding")
            if not name or emb is None:
                continue
            arr = np.array(emb, dtype=np.float32)
            if arr.size == 0:
                continue
            name_to_emb[name] = arr
            name_to_student_id[name] = sid
        return cls(name_to_emb=name_to_emb, name_to_student_id=name_to_student_id)

    def match(
        self,
        seg_emb: np.ndarray,
        threshold: float = 0.5,
    ) -> tuple[str, str]:
        """
        匹配最近说话人，返回 (speaker_name, student_id)。
        @param
        seg_emb: np.ndarray 说话人embedding
        @param
        threshold: float 相似度阈值
        @return
        tuple[str, str] (speaker_name, student_id)
        """
        if not self.name_to_emb:
            return "unknown", "unknown"
        best_name, best_sim = "unknown", threshold
        n = np.linalg.norm(seg_emb)
        if n < 1e-8:
            return "unknown", "unknown"
        for name, emb in self.name_to_emb.items():
            sim = np.dot(seg_emb, emb) / (n * np.linalg.norm(emb) + 1e-8)
            if sim > best_sim:
                best_sim, best_name = sim, name
        sid = self.name_to_student_id.get(best_name, "unknown")
        if best_name not in self._allowed_names:
            best_name, sid = "unknown", "unknown"
        return best_name, sid


def load_speaker_embeddings(embeddings_dir: str | Path) -> dict[str, np.ndarray]:
    """
    从目录加载所有 .pkl 说话人 embedding，文件名为姓名。
    @param
    embeddings_dir: str | Path 说话人 embedding 目录
    @return
    dict[str, np.ndarray] 说话人姓名到 embedding 的映射
    """
    d = Path(embeddings_dir)
    if not d.exists():
        return {}
    out = {}
    for p in d.glob("*.pkl"):
        with open(p, "rb") as f:
            out[p.stem] = pickle.load(f)
    return out


def load_speakers_from_wav(
    wav_dir: str | Path,
    model_path: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """
    从 .wav 目录加载说话人 embedding（现场计算），文件名为姓名。
    @param
    wav_dir: str | Path 说话人 wav 目录
    @param
    model_path: str | Path | None 模型路径
    @return
    dict[str, np.ndarray] 说话人姓名到 embedding 的映射
    """
    d = Path(wav_dir)
    if not d.exists():
        return {}
    extractor = EmbeddingExtractor(model_path=model_path)
    out = {}
    for p in d.glob("*.wav"):
        emb = extractor.from_path(p)
        out[p.stem] = emb
    return out


def load_speakers_from_mapping(
    mapping: dict[str, str] | dict[str, Path],
    model_path: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """
    从「姓名 -> 音频路径」键值对加载声纹。
    @param
    mapping: dict[str, str] | dict[str, Path] 姓名到音频路径的映射
    @param
    model_path: str | Path | None 模型路径
    @return
    dict[str, np.ndarray] 说话人姓名到 embedding 的映射
    """
    out: dict[str, np.ndarray] = {}
    extractor = EmbeddingExtractor(model_path=model_path)
    for name, audio_path in mapping.items():
        p = Path(audio_path)
        if not p.exists():
            continue
        out[name] = extractor.from_path(p)
    return out


def load_speakers_from_json(
    json_path: str | Path,
    model_path: str | Path | None = None,
    base_dir: str | Path | None = None,
) -> dict[str, np.ndarray]:
    """
    从 JSON 加载声纹。格式 {"姓名": "音频路径", ...}
    @param
    json_path: str | Path JSON 文件路径
    @param
    model_path: str | Path | None 模型路径
    @param
    base_dir: str | Path | None 基础目录
    @return
    dict[str, np.ndarray] 说话人姓名到 embedding 的映射
    """
    p = Path(json_path)
    base = Path(base_dir) if base_dir else p.parent
    with open(p, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    if not isinstance(mapping, dict):
        return {}
    resolved = {
        name: str(base / Path(path)) if not Path(path).is_absolute() else str(path)
        for name, path in mapping.items()
    }
    return load_speakers_from_mapping(resolved, model_path=model_path)


def load_speakers(speakers_dir: str | Path, prefer_pkl: bool = True) -> dict[str, np.ndarray]:
    """
    从目录加载声纹：优先 .pkl，若无则从 .wav 现场计算。
    @param
    speakers_dir: str | Path 说话人目录
    @param
    prefer_pkl: bool 优先加载 .pkl 文件
    @return
    dict[str, np.ndarray] 说话人姓名到 embedding 的映射
    """
    d = Path(speakers_dir)
    if not d.exists():
        return {}
    if prefer_pkl and list(d.glob("*.pkl")):
        return load_speaker_embeddings(d)
    if list(d.glob("*.wav")):
        return load_speakers_from_wav(d)
    return load_speaker_embeddings(d)


def match_speaker(
    seg_emb: np.ndarray,
    name_to_emb: dict[str, np.ndarray],
    threshold: float = 0.5,
) -> str:
    """
    余弦相似度匹配最近说话人
    @param
    seg_emb: np.ndarray 说话人embedding
    @param
    name_to_emb: dict[str, np.ndarray] 说话人姓名到 embedding 的映射
    @param
    threshold: float 相似度阈值，如果相似度大于阈值，则认为匹配成功
    @return
    str 说话人姓名
    """
    reg = SpeakerRegistry(name_to_emb=name_to_emb)
    name, _ = reg.match(seg_emb, threshold=threshold)
    return name

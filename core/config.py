"""路径与模型配置（兼容旧导入：从 core.audio 转发）"""
from core.audio.config import (
    BASE_DIR,
    CONFIG_PATH,
    DATA_DIR,
    EMBEDDING_MODEL_PATH,
    PLDA_DIR,
    WHISPER_MODEL_PATH,
)

__all__ = [
    "BASE_DIR",
    "CONFIG_PATH",
    "DATA_DIR",
    "EMBEDDING_MODEL_PATH",
    "PLDA_DIR",
    "WHISPER_MODEL_PATH",
]

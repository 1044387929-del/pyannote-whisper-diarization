"""音频转录相关：路径与模型配置"""
from pathlib import Path

# 项目根目录（core/audio/config.py -> 向上 3 层）
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_DIR / "config.yaml"
PLDA_DIR = BASE_DIR / "plda"
DATA_DIR = BASE_DIR / "data"

EMBEDDING_MODEL_PATH = "/root/autodl-tmp/modelscope/hub/models/pyannote/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin"
WHISPER_MODEL_PATH = "/root/autodl-tmp/modelscope/hub/models/openai-mirror/faster-whisper-large-v3"

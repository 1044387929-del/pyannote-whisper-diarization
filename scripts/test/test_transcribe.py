"""测试 /transcribe 接口。在项目根目录运行: python scripts/test_transcribe.py"""
import json
import sys
from pathlib import Path

# 确保项目根在 path 中
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import requests

SPEAKERS_JSON = Path(r"/root/autodl-tmp/django_hmxy/pyannote_diarization/data/json/speakers_embedding.json")
AUDIO_PATH = Path(r"/root/autodl-tmp/django_hmxy/pyannote_diarization/data/audio/audio_all.wav")
URL = "http://127.0.0.1:8001/transcribe"

speakers = json.loads(SPEAKERS_JSON.read_text(encoding="utf-8"))
data = [("language", "zh")]
for s in speakers:
    data.append(("student_id", s["student_id"]))
    data.append(("name", s["name"]))
    data.append(("embedding", json.dumps(s["embedding"], ensure_ascii=False)))

with open(AUDIO_PATH, "rb") as f:
    r = requests.post(f"{URL}", data=data, files={"audio": (AUDIO_PATH.name, f, "audio/wav")}, timeout=300)
print(r.status_code)
print(json.dumps(r.json(), ensure_ascii=False, indent=2))

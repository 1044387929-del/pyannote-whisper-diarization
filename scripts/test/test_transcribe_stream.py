"""测试 /transcribe/stream 流式接口。先启动 uvicorn app:app，然后运行: python pyannote_diarization/scripts/test/test_transcribe_stream.py"""
import json
import sys
from pathlib import Path
from utils.common import secs_to_hms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import requests


SPEAKERS_JSON = Path(r"/root/autodl-tmp/django_hmxy/pyannote_diarization/data/json/speakers_embedding.json")
AUDIO_PATH = Path(r"/root/autodl-tmp/django_hmxy/pyannote_diarization/data/audio/audio_all.wav")
URL = "http://127.0.0.1:8001/transcribe/stream"

speakers = json.loads(SPEAKERS_JSON.read_text(encoding="utf-8"))
data = [("language", "zh")]
for s in speakers:
    data.append(("student_id", s["student_id"]))
    data.append(("name", s["name"]))
    data.append(("embedding", json.dumps(s["embedding"], ensure_ascii=False)))

print("[客户端] POST /transcribe/stream 流式转录，开始请求...")
print("-" * 50)

with open(AUDIO_PATH, "rb") as f:
    r = requests.post(
        URL,
        data=data,
        files={"audio": (AUDIO_PATH.name, f, "audio/wav")},
        stream=True,
        timeout=600,
    )

if r.status_code != 200:
    print(f"[客户端] 错误: status={r.status_code}, {r.text}")
    sys.exit(1)

print("[客户端] 连接成功，开始接收流式结果...")
utterances: list = []
diarization_seconds = 0.0
whisper_seconds = 0.0
buffer = ""

for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
    if not chunk:
        continue
    buffer += chunk
    while "\n\n" in buffer:
        line, buffer = buffer.split("\n\n", 1)
        line = line.strip()
        if line.startswith("data: "):
            try:
                obj = json.loads(line[6:])
                if obj.get("status") == "done":
                    diarization_seconds = obj.get("diarization_seconds", 0)
                    whisper_seconds = obj.get("whisper_seconds", 0)
                    continue
                utterances.append(obj)
                count = obj.get("index", len(utterances))
                total = obj.get("total", 0)
                progress = obj.get("progress", 0)
                t0, t1 = obj.get("start", 0), obj.get("end", 0)
                speaker = obj.get("speaker", "?")
                text = obj.get("text", "").strip()
                ts = f"{secs_to_hms(t0)} - {secs_to_hms(t1)}"
                pct = f" ({progress}%)" if total else ""
                print(f"[客户端] 收到第 {count}/{total} 句{pct} | {ts} | {speaker}: {text}")
            except json.JSONDecodeError:
                pass

total = len(utterances)
print("-" * 50)
print(f"[客户端] 接收完成，共 {total} 句 (100%)")
print(f"[客户端] pyannote 分割: {diarization_seconds}s | Whisper 转写: {whisper_seconds}s")
print()
print("[客户端] 返回 JSON:")
print(json.dumps({
    "utterances": utterances,
    "total": total,
    "progress": 100.0,
    "diarization_seconds": diarization_seconds,
    "whisper_seconds": whisper_seconds,
}, ensure_ascii=False, indent=2))

"""测试 POST /transcriptions/stream 流式接口。先启动 uvicorn app:app，再运行本脚本。"""
import json
import sys
from pathlib import Path


# 项目根（scripts/test/api/ -> 上溯 4 层）
ROOT = Path(__file__).resolve().parent.parent.parent.parent
TEST_DATA = Path(__file__).resolve().parent.parent / "data"
sys.path.insert(0, str(ROOT))
from utils.common import secs_to_hms
import requests

# 脚本内参数（按需修改）
REFINE = True  # True：流中同时做精修
SPEAKERS_JSON = TEST_DATA / "json" / "speakers_embedding.json"
AUDIO_PATH = TEST_DATA / "audio" / "audio_all.wav"
URL = r"http://127.0.0.1:8001/transcriptions/stream"

# 如果说话人 JSON 不存在，则退出
if not SPEAKERS_JSON.exists():
    print(f"[客户端] 未找到说话人 JSON: {SPEAKERS_JSON}")
    sys.exit(1)
# 如果音频文件不存在，则退出
if not AUDIO_PATH.exists():
    print(f"[客户端] 未找到音频文件: {AUDIO_PATH}")
    sys.exit(1)

speakers = json.loads(SPEAKERS_JSON.read_text(encoding="utf-8"))
data = [("language", "zh")]
if REFINE:
    data.append(("refine", "true"))
for s in speakers:
    data.append(("student_id", s["student_id"]))
    data.append(("name", s["name"]))
    data.append(("embedding", json.dumps(s["embedding"], ensure_ascii=False)))

print("[客户端] POST /transcriptions/stream 流式转录，开始请求..." + (" [refine=true]" if REFINE else ""))
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
                ev_type = obj.get("type")
                t0, t1 = obj.get("start", 0), obj.get("end", 0)
                speaker = obj.get("speaker", "?")
                text = (obj.get("text") or "").strip()
                ts = f"{secs_to_hms(t0)} - {secs_to_hms(t1)}"
                if REFINE and ev_type == "raw":
                    print(f"[客户端] 修正前 | {ts} | {speaker}: {text}")
                    continue
                if REFINE and ev_type == "refined":
                    print(f"[客户端] 修正后 | {ts} | {speaker}: {text}")
                    utterances.append({k: v for k, v in obj.items() if k != "type"})
                    continue
                # 未开 refine 或旧格式：直接当作一条结果
                utterances.append(obj)
                count = obj.get("index", len(utterances))
                total = obj.get("total", 0)
                progress = obj.get("progress", 0)
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

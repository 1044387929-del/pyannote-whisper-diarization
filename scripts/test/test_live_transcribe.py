"""测试 WebSocket /ws/live-transcribe 实时转写接口。先启动 uvicorn，再运行本脚本。"""
import asyncio
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

try:
    import websockets
except ImportError:
    print("请先安装 websockets: pip install websockets")
    sys.exit(1)

try:
    import torchaudio
except ImportError:
    torchaudio = None

AUDIO_PATH = Path(r"/root/autodl-tmp/django_hmxy/pyannote_diarization/data/audio/audio_all.wav")
SPEAKERS_JSON = Path(r"/root/autodl-tmp/django_hmxy/pyannote_diarization/data/json/speakers_embedding.json")
URL = "ws://127.0.0.1:8001/ws/live_transcribe"
CHUNK_SECONDS = 10


def secs_to_hms(secs: float) -> str:
    """秒数转为 小时:分钟:秒.毫秒"""
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = int(secs % 60)
    ms = int((secs % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def split_audio_into_chunks(audio_path: Path, chunk_seconds: float = 10) -> list[bytes]:
    """将 wav 文件按时间切分成块（每块独立 wav）。"""
    if not torchaudio:
        data = audio_path.read_bytes()
        return [data] if len(data) > 100 else []

    import tempfile
    waveform, sample_rate = torchaudio.load(str(audio_path))
    chunk_samples = int(sample_rate * chunk_seconds)
    total_samples = waveform.shape[1]
    chunks = []
    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        seg = waveform[:, start:end]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            torchaudio.save(f.name, seg, sample_rate)
            chunks.append(Path(f.name).read_bytes())
    return chunks


async def main():
    if not AUDIO_PATH.exists():
        print(f"音频不存在: {AUDIO_PATH}")
        sys.exit(1)

    chunks = split_audio_into_chunks(AUDIO_PATH, CHUNK_SECONDS)
    if not chunks:
        print("无法切分音频")
        sys.exit(1)

    speakers = []
    if SPEAKERS_JSON.exists():
        speakers = json.loads(SPEAKERS_JSON.read_text(encoding="utf-8"))
        for s in speakers:
            s["embedding"] = s.get("embedding", [])
        print(f"[客户端] 加载说话人 {len(speakers)} 个")
    else:
        print("[客户端] 未找到 speakers_embedding.json，将仅转写不分人")

    print(f"[客户端] 连接 ws://127.0.0.1:8001/ws/live_transcribe")
    print(f"[客户端] 音频 {AUDIO_PATH.name} 切分为 {len(chunks)} 块")
    print("-" * 50)

    async with websockets.connect(URL) as ws:
        # init
        init_msg = {"type": "init", "language": "zh"}
        if speakers:
            init_msg["speakers"] = speakers
        await ws.send(json.dumps(init_msg))
        resp = json.loads(await ws.recv())
        if resp.get("type") == "ready":
            print(f"[客户端] 连接就绪 | has_speakers={resp.get('has_speakers', False)}")
        else:
            print(f"[客户端] 意外: {resp}")
            return

        for i, chunk in enumerate(chunks):
            data_b64 = base64.b64encode(chunk).decode()
            await ws.send(json.dumps({"type": "audio", "data": data_b64, "chunk_index": i + 1}))
            resp = json.loads(await ws.recv())
            if resp.get("type") == "transcript":
                utterances = resp.get("utterances") or []
                idx = resp.get("chunk_index", i + 1)
                if utterances:
                    chunk_offset = (idx - 1) * CHUNK_SECONDS
                    for u in utterances:
                        speaker = u.get("speaker", "?")
                        text = (u.get("text") or "").strip()
                        t0, t1 = u.get("start", 0), u.get("end", 0)
                        abs_t0 = chunk_offset + t0
                        abs_t1 = chunk_offset + t1
                        ts = f"{secs_to_hms(abs_t0)} - {secs_to_hms(abs_t1)}"
                        if text:
                            print(f"[客户端] 第 {idx} 块 | {ts} | {speaker}: {text}")
                else:
                    text = (resp.get("text") or "").strip()
                    if text:
                        print(f"[客户端] 第 {idx} 块: {text}")
            elif resp.get("type") == "error":
                print(f"[客户端] 错误: {resp.get('message')}")
                break

        # end
        await ws.send(json.dumps({"type": "end"}))
        resp = json.loads(await ws.recv())
        if resp.get("type") == "done":
            print("-" * 50)
            print(f"[客户端] 完成，共 {resp.get('total_chunks', 0)} 块")


if __name__ == "__main__":
    asyncio.run(main())

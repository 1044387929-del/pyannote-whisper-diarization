"""
测试：客户端中途断开连接时，FastAPI 是否中止 GPU 转录。

用法（先启动服务）:
  uvicorn app:app --host 0.0.0.0 --port 8001

  cd pyannote_diarization
  python scripts/test/api/test_client_disconnect.py [音频文件] [断开秒数]

示例:
  python scripts/test/api/test_client_disconnect.py scripts/test/data/audio/audio_all.wav 5

会发起 POST /transcriptions，在指定秒数后杀掉客户端进程以关闭连接（模拟用户取消）。
观察 FastAPI 终端是否打印「取消转录」且转录未跑完。
"""
import json
import subprocess
import sys
import time
from pathlib import Path

# 项目根（scripts/test/api/ -> 上溯 4 层）
ROOT = Path(__file__).resolve().parent.parent.parent.parent
TEST_DATA = Path(__file__).resolve().parent.parent / "data"
SPEAKERS_JSON = TEST_DATA / "json" / "speakers_embedding.json"
URL = "http://127.0.0.1:8001/transcriptions"


def run(audio_path: Path, disconnect_after_secs: float = 5.0) -> None:
    if not audio_path.exists():
        print(f"音频文件不存在: {audio_path}")
        sys.exit(1)

    if SPEAKERS_JSON.exists():
        speakers = json.loads(SPEAKERS_JSON.read_text(encoding="utf-8"))
        s = speakers[0] if isinstance(speakers, list) else list(speakers.values())[0]
        student_id = s.get("student_id", "test")
        name = s.get("name", "test")
        emb = s.get("embedding", [0.0] * 256)
    else:
        student_id, name = "test", "test"
        emb = [0.0] * 256

    emb_str = json.dumps(emb, ensure_ascii=False)
    # 用 curl 发请求，指定秒数后 kill 进程即可关闭连接（服务端会看到 client disconnect）
    cmd = [
        "curl", "-s", "-N", "-X", "POST", URL,
        "-F", "language=zh",
        "-F", f"student_id={student_id}",
        "-F", f"name={name}",
        "-F", f"embedding={emb_str}",
        "-F", f"audio=@{audio_path.resolve()}",
    ]
    print(f"[客户端] 发起转录请求，{disconnect_after_secs}s 后断开连接…")
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(disconnect_after_secs)
    p.terminate()
    try:
        p.wait(timeout=2)
    except subprocess.TimeoutExpired:
        p.kill()
    print(f"[客户端] 已断开连接（进程已终止）")
    print("\n请查看 FastAPI 终端是否出现:「取消转录：收到客户端断开连接」且转录未完成即停止。")

if __name__ == "__main__":
    audio = Path(sys.argv[1]) if len(sys.argv) > 1 else TEST_DATA / "audio" / "audio_all.wav"
    secs = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    print(f"音频: {audio} | {secs}s 后断开")
    run(audio, secs)

"""
FastAPI 接口：声纹 embedding、转录

HTTP 测试示例（先启动: uvicorn api:app --host 0.0.0.0 --port 8001）:

  # 1. 健康检查
  curl http://127.0.0.1:8001/health

  # 2. POST /embedding - 上传学号+音频，返回 embedding
  curl -X POST http://127.0.0.1:8001/embedding \\
    -F "student_id=2021001" \\
    -F "name=张三" \\
    -F "audio=@data/embedding_audios/peppa.wav"

  # 3. POST /transcribe - 提交若干个（学号、姓名、向量）三元组 + 音频
  curl -X POST http://127.0.0.1:8001/transcribe \\
    -F "student_id=2021001" -F "student_id=2021002" \\
    -F "name=张三" -F "name=李四" \\
    -F "embedding=[-0.13,0.12,...]" -F "embedding=[0.2,-0.1,...]" \\
    -F "audio=@data/audio/audio_all.wav" -F "language=zh"

  # 4. POST /transcribe/stream - 流式转录，SSE 实时推送每句结果
  curl -N -X POST http://127.0.0.1:8001/transcribe/stream \\
    -F "student_id=2021001" -F "student_id=2021002" \\
    -F "name=张三" -F "name=李四" \\
    -F "embedding=[-0.13,0.12,...]" -F "embedding=[0.2,-0.1,...]" \\
    -F "audio=@data/audio/audio_all.wav" -F "language=zh"
"""
import asyncio
import base64
import json
import tempfile
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from core.embedding import get_embedding
from core.pipeline import transcribe_chunk_with_speakers, transcribe_with_speakers, transcribe_with_speakers_stream
from core.transcribe import transcribe_chunk
from utils.common import ALLOWED_EXT, get_audio_suffix, secs_to_hms, webm_to_wav
from utils.errors import (
    ERR_AUDIO_EMPTY,
    err_embedding_extract,
    err_embedding_format,
    err_embedding_not_array,
    err_speakers_mismatch,
    err_transcribe,
    err_unsupported_format,
    ws_error,
)


app = FastAPI(title="声纹 API", description="embedding 提取、转录（学号+姓名+向量 -> 转写结果）")

STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/live")
    async def live_page():
        """Live 实时转写前端页面"""
        return FileResponse(STATIC_DIR / "live.html")


@app.post("/embedding")
async def upload_audio_embedding(
    student_id: str = Form(..., description="学号"),
    name: str = Form("", description="姓名（可选）"),
    audio: UploadFile = File(..., description="音频文件"),
):
    """
    上传学号 + 音频，返回 256 维 embedding 向量。

    curl 示例:
      curl -X POST http://127.0.0.1:8001/embedding \\
        -F "student_id=2021001" -F "name=张三" -F "audio=@peppa.wav"
    """
    suf = Path(audio.filename or "").suffix.lower()
    if suf and suf not in ALLOWED_EXT:
        raise HTTPException(**err_unsupported_format(suf, ", ".join(ALLOWED_EXT)))

    content = await audio.read()
    if not content:
        raise HTTPException(**ERR_AUDIO_EMPTY)

    suffix = get_audio_suffix(audio.filename)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(content)
        tmp_path = f.name

    try:
        emb = get_embedding(tmp_path)
        emb_list = emb.cpu().numpy().tolist() if hasattr(emb, "cpu") else emb.tolist()
        dim = len(emb_list)
    except Exception as e:
        raise HTTPException(**err_embedding_extract(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {
        "student_id": student_id,
        "name": name or student_id,
        "embedding": emb_list,
        "embedding_dim": dim,
    }


@app.post("/transcribe")
async def transcribe_audio(
    student_id: List[str] = Form(..., description="学号，可多个，按顺序与 name、embedding 对应"),
    name: List[str] = Form(..., description="姓名，可多个"),
    embedding: List[str] = Form(..., description="256 维向量 JSON 字符串，如 [0.1,-0.2,...]"),
    audio: UploadFile = File(..., description="待转写音频"),
    language: str = Form("", description="Whisper 语言，如 zh、en，空则自动检测"),
):
    """
    提交若干个（学号、姓名、向量）三元组 + 待测音频，返回转录 JSON。
    三元组用重复的 form 字段传递，三列按索引一一对应。

    curl 示例（2 个说话人）:
      curl -X POST http://127.0.0.1:8001/transcribe \\
        -F "student_id=2021001" -F "student_id=2021002" \\
        -F "name=张三" -F "name=李四" \\
        -F "embedding=[-0.13,0.12,...]" -F "embedding=[0.2,-0.1,...]" \\
        -F "audio=@audio.wav" -F "language=zh"
    """
    suf = Path(audio.filename or "").suffix.lower()
    if suf and suf not in ALLOWED_EXT:
        raise HTTPException(**err_unsupported_format(suf, ", ".join(ALLOWED_EXT)))

    n = len(student_id)
    if n != len(name) or n != len(embedding) or n == 0:
        raise HTTPException(**err_speakers_mismatch(len(student_id), len(name), len(embedding)))

    speakers_list = []
    for i in range(n):
        try:
            emb = json.loads(embedding[i])
        except json.JSONDecodeError as e:
            raise HTTPException(**err_embedding_format(i + 1, e))
        if not isinstance(emb, list):
            raise HTTPException(**err_embedding_not_array(i + 1))
        speakers_list.append({"student_id": student_id[i], "name": name[i], "embedding": emb})

    content = await audio.read()
    if not content:
        raise HTTPException(**ERR_AUDIO_EMPTY)

    suffix = get_audio_suffix(audio.filename)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(content)
        tmp_path = f.name

    try:
        lang = language.strip() or None
        print(f"[服务端] /transcribe 转录开始 | 说话人数: {n} | 音频: {audio.filename}")
        loop = asyncio.get_event_loop()
        utterances = await loop.run_in_executor(
            None, lambda: transcribe_with_speakers(tmp_path, speakers_list, language=lang)
        )
        print(f"[服务端] /transcribe 转录完成 | 共 {len(utterances)} 句")
        return {"utterances": utterances}
    except Exception as e:
        raise HTTPException(**err_transcribe(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


async def _stream_transcribe_events(tmp_path: str, speakers_list: list, lang: str | None):
    """SSE 流式生成：每完成一句转写就 yield 一个 data 事件。"""
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def run():
        try:
            for utt in transcribe_with_speakers_stream(tmp_path, speakers_list, language=lang):
                loop.call_soon_threadsafe(queue.put_nowait, utt)
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, e)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    loop.run_in_executor(None, run)

    count = 0
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            if item.get("status") == "done":
                d = item.get("diarization_seconds", 0)
                w = item.get("whisper_seconds", 0)
                print(f"[服务端] 流式转录完成 | pyannote: {d}s, Whisper: {w}s")
            else:
                count += 1
                t0, t1 = item.get("start", 0), item.get("end", 0)
                speaker = item.get("speaker", "?")
                text = (item.get("text") or "").strip()
                preview = (text[:30] + "..") if len(text) > 30 else text
                ts = f"{secs_to_hms(t0)} - {secs_to_hms(t1)}"
                print(f"[服务端] 进度 第 {count} 句 | {ts} | {speaker}: {preview}")
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
    finally:
        Path(tmp_path).unlink(missing_ok=True)
        print(f"[服务端] 流式转录结束 | 共 {count} 句")


@app.post("/transcribe/stream")
async def transcribe_audio_stream(
    student_id: List[str] = Form(..., description="学号，可多个，按顺序与 name、embedding 对应"),
    name: List[str] = Form(..., description="姓名，可多个"),
    embedding: List[str] = Form(..., description="256 维向量 JSON 字符串，如 [0.1,-0.2,...]"),
    audio: UploadFile = File(..., description="待转写音频"),
    language: str = Form("", description="Whisper 语言，如 zh、en，空则自动检测"),
):
    """
    流式转录：参数与 /transcribe 相同，每完成一句即通过 SSE 推送，无需等待全部完成。

    curl 示例（SSE 流式接收）:
      curl -N -X POST http://127.0.0.1:8001/transcribe/stream \\
        -F "student_id=2021001" -F "student_id=2021002" \\
        -F "name=张三" -F "name=李四" \\
        -F "embedding=[-0.13,0.12,...]" -F "embedding=[0.2,-0.1,...]" \\
        -F "audio=@audio.wav" -F "language=zh"
    """
    suf = Path(audio.filename or "").suffix.lower()
    if suf and suf not in ALLOWED_EXT:
        raise HTTPException(**err_unsupported_format(suf, ", ".join(ALLOWED_EXT)))

    n = len(student_id)
    if n != len(name) or n != len(embedding) or n == 0:
        raise HTTPException(**err_speakers_mismatch(len(student_id), len(name), len(embedding)))

    speakers_list = []
    for i in range(n):
        try:
            emb = json.loads(embedding[i])
        except json.JSONDecodeError as e:
            raise HTTPException(**err_embedding_format(i + 1, e))
        if not isinstance(emb, list):
            raise HTTPException(**err_embedding_not_array(i + 1))
        speakers_list.append({"student_id": student_id[i], "name": name[i], "embedding": emb})

    content = await audio.read()
    if not content:
        raise HTTPException(**ERR_AUDIO_EMPTY)

    suffix = get_audio_suffix(audio.filename)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(content)
        tmp_path = f.name

    print(f"[服务端] 流式转录请求开始 | 说话人数: {n} | 音频: {audio.filename}")
    lang = language.strip() or None
    return StreamingResponse(
        _stream_transcribe_events(tmp_path, speakers_list, lang),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.websocket("/ws/live_transcribe")
async def websocket_live_transcribe(websocket: WebSocket):
    """
    WebSocket 实时转写：客户端持续发送音频块，服务端近似实时返回转写结果。
    init 传入 speakers 时做 diarization + 声纹匹配（分人），否则仅 Whisper 转写。

    消息协议：
    - 客户端 -> init: {"type": "init", "language": "zh", "speakers": [{"student_id","name","embedding"}, ...]}
    - 服务端 -> ready: {"type": "ready", "language": "zh", "has_speakers": true}
    - 客户端 -> audio: {"type": "audio", "data": "<base64>", "chunk_index": 1}
    - 服务端 -> transcript: {"type": "transcript", "utterances": [...], "text": "...", "chunk_index": 1}
    - 客户端 -> end: {"type": "end"}
    - 服务端 -> done: {"type": "done", "total_chunks": N}
    """
    await websocket.accept()
    lang: str | None = None
    speakers_list: list[dict] = []
    loop = asyncio.get_event_loop()
    chunk_count = 0

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json(ws_error("Invalid JSON"))
                continue

            msg_type = msg.get("type", "")

            if msg_type == "init":
                lang = (msg.get("language") or "").strip() or None
                speakers_raw = msg.get("speakers")
                speakers_list = []
                if isinstance(speakers_raw, list):
                    for s in speakers_raw:
                        if isinstance(s, dict) and s.get("embedding") is not None:
                            speakers_list.append(s)
                print(f"[服务端] Live 转写连接建立 | language={lang} | speakers={len(speakers_list)}")
                await websocket.send_json({
                    "type": "ready",
                    "language": lang or "auto",
                    "has_speakers": len(speakers_list) > 0,
                })

            elif msg_type == "audio":
                data_b64 = msg.get("data")
                chunk_index = msg.get("chunk_index", chunk_count + 1)
                if not data_b64:
                    await websocket.send_json(ws_error("Missing audio data"))
                    continue
                try:
                    audio_bytes = base64.b64decode(data_b64)
                except Exception as e:
                    await websocket.send_json(ws_error(f"Invalid base64: {e}"))
                    continue
                if len(audio_bytes) < 100:
                    await websocket.send_json({
                        "type": "transcript",
                        "utterances": [],
                        "text": "",
                        "chunk_index": chunk_index,
                    })
                    continue
                fmt = (msg.get("format") or "wav").lower()
                if fmt == "webm":
                    try:
                        audio_bytes = webm_to_wav(audio_bytes)
                    except Exception as e:
                        await websocket.send_json(ws_error(f"webm 转 wav 失败: {e}", chunk_index=chunk_index))
                        continue
                try:
                    if speakers_list:
                        utterances = await loop.run_in_executor(
                            None,
                            lambda: transcribe_chunk_with_speakers(audio_bytes, speakers_list, language=lang),
                        )
                        text = " ".join(u.get("text", "").strip() for u in utterances).strip()
                        for u in utterances:
                            s, t = u.get("speaker", "?"), u.get("text", "").strip()
                            if t:
                                print(f"[服务端] Live 第 {chunk_count+1} 块 | {s}: {t[:40]}..")
                    else:
                        text = await loop.run_in_executor(
                            None, lambda: transcribe_chunk(audio_bytes, language=lang)
                        )
                        utterances = [{"speaker": "unknown", "student_id": "unknown", "text": text}] if text else []
                        if text:
                            print(f"[服务端] Live 第 {chunk_count+1} 块 | {text[:50]}..")
                except Exception as e:
                    print(f"[服务端] Live 转写失败 chunk {chunk_index}: {e}")
                    await websocket.send_json(ws_error(str(e), chunk_index=chunk_index))
                    continue
                chunk_count += 1
                await websocket.send_json({
                    "type": "transcript",
                    "utterances": utterances,
                    "text": text,
                    "chunk_index": chunk_index,
                })

            elif msg_type == "end":
                print(f"[服务端] Live 转写结束 | 共 {chunk_count} 块")
                await websocket.send_json({"type": "done", "total_chunks": chunk_count})
                break

            else:
                await websocket.send_json(ws_error(f"Unknown type: {msg_type}"))

    except WebSocketDisconnect:
        print(f"[服务端] Live 转写 WebSocket 断开")
    except Exception as e:
        print(f"[服务端] Live 转写异常: {e}")
        try:
            await websocket.send_json(ws_error(str(e)))
        except Exception:
            pass


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)

"""Live 实时转写：WebSocket + 前端页面"""
import asyncio
import base64
import json
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse
from fastapi.websockets import WebSocket, WebSocketDisconnect

from core.pipeline import transcribe_chunk_with_speakers
from core.transcribe import transcribe_chunk
from utils.errors import ws_error
from utils.common import webm_to_wav

router = APIRouter(tags=["live"])

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@router.get("/live")
async def live_page():
    """Live 实时转写前端页面"""
    return FileResponse(STATIC_DIR / "live.html")


@router.websocket("/ws/live_transcribe")
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

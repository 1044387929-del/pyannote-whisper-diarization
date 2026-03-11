"""Live 实时转写：WebSocket + 前端页面"""
import asyncio
import base64
import json
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse
from fastapi.websockets import WebSocket, WebSocketDisconnect

from core.audio import transcribe_chunk_with_speakers, transcribe_chunk
from core.llm.refine_pipeline import RefinePipeline
from utils.common import webm_to_wav
from utils.errors import ws_error

router = APIRouter(tags=["live"])

# 项目根目录的 static（routers/audio/live.py -> 根目录）
STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"


@router.get("/live")
async def live_page():
    """Live 实时转写前端页面"""
    return FileResponse(STATIC_DIR / "live.html")


@router.websocket("/ws/transcriptions/live")
async def websocket_transcriptions_live(websocket: WebSocket):
    """
    WebSocket 实时转写：客户端持续发送音频块，服务端近似实时返回转写结果。
    init 传入 speakers 时做 diarization + 声纹匹配（分人），否则仅 Whisper 转写。
    init 可传 "refine": true，开启后每块转写结果会做增量精修再返回（纠错/标点/说话人推断，仅保留输入说话人）。

    消息协议：
    - 客户端 -> init: {"type": "init", "language": "zh", "speakers": [...], "refine": false}
    - 服务端 -> ready: {"type": "ready", "language": "zh", "has_speakers": true, "refine": false}
    - 客户端 -> audio: {"type": "audio", "data": "<base64>", "chunk_index": 1}
    - 服务端 -> transcript: {"type": "transcript", "utterances": [...], "text": "...", "chunk_index": 1}
    - 客户端 -> end: {"type": "end"}
    - 服务端 -> done: {"type": "done", "total_chunks": N}
    """
    await websocket.accept()
    lang: str | None = None
    speakers_list: list[dict] = []
    refine = False
    pipeline: RefinePipeline | None = None
    allowed_speakers_from_input: list[dict] | None = None
    refined_so_far: list[dict] = []
    loop = asyncio.get_event_loop()
    chunk_count = 0

    try:
        while True:
            # 接收消息
            raw = await websocket.receive_text()
            try:
                # 解析消息
                msg = json.loads(raw)
            except json.JSONDecodeError:
                # 发送错误消息
                await websocket.send_json(ws_error("Invalid JSON"))
                continue
            # 获取消息类型
            msg_type = msg.get("type", "")

            if msg_type == "init":
                # 获取语言
                lang = (msg.get("language") or "").strip() or None
                # 是否边录边修正
                refine = bool(msg.get("refine", False))
                # 获取说话人列表
                speakers_raw = msg.get("speakers")
                # 初始化说话人列表
                speakers_list = []
                # 如果说话人列表是列表，则遍历说话人列表
                if isinstance(speakers_raw, list):
                    # 遍历说话人列表
                    for s in speakers_raw:
                        # 如果说话人是一个字典，并且说话人中有 embedding 字段，则添加到说话人列表
                        if isinstance(s, dict) and s.get("embedding") is not None:
                            speakers_list.append(s)
                if refine:
                    pipeline = RefinePipeline(verbose=False)
                    allowed_speakers_from_input = (
                        [{"speaker": s.get("name") or s.get("student_id"), "student_id": s.get("student_id")} for s in speakers_list]
                    )
                    allowed_speakers_from_input = list(allowed_speakers_from_input) if speakers_list else None
                    refined_so_far = []
                else:
                    pipeline = None
                    allowed_speakers_from_input = None
                    refined_so_far = []
                print(f"[服务端] Live 转写连接建立 | language={lang} | speakers={len(speakers_list)} | refine={refine}")
                # 发送准备消息
                await websocket.send_json({
                    "type": "ready",
                    "language": lang or "auto",
                    "has_speakers": len(speakers_list) > 0,
                    "refine": refine,
                })

            elif msg_type == "audio":
                # 获取音频数据
                data_b64 = msg.get("data")
                # 获取块索引
                chunk_index = msg.get("chunk_index", chunk_count + 1)
                # 如果音频数据为空，则发送错误消息
                if not data_b64:
                    await websocket.send_json(ws_error("Missing audio data"))
                    continue
                try:
                    # 解码音频数据
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
                if refine and pipeline and utterances:
                    batch = await pipeline.run_incremental(
                        refined_so_far, utterances, allowed_speakers_from_input=allowed_speakers_from_input
                    )
                    refined_so_far.extend(batch)
                    utterances = batch
                    text = " ".join((u.get("text") or "").strip() for u in utterances).strip()
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

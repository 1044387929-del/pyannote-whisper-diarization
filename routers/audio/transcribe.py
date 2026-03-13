"""转写接口：批量与流式"""
import asyncio
import json
import tempfile
import threading
from pathlib import Path
from typing import AsyncGenerator, List

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from core.audio import transcribe_with_speakers, transcribe_with_speakers_stream
from core.llm.refine_pipeline import RefinePipeline
from utils.common import ALLOWED_EXT, get_audio_suffix, secs_to_hms
from utils.errors import (
    ERR_AUDIO_EMPTY,
    err_embedding_format,
    err_embedding_not_array,
    err_speakers_mismatch,
    err_transcribe,
    err_unsupported_format,
)

router = APIRouter(tags=["transcribe"])


async def _stream_transcribe_events(
    tmp_path: str, speakers_list: list, lang: str | None, refine: bool = False
) -> AsyncGenerator[str, None]:
    """SSE 流式生成：每完成一句转写就 yield 一个 data 事件；refine=True 时在流中做增量精修后推送。"""
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

    pipeline = RefinePipeline(verbose=False) if refine else None
    allowed_speakers_from_input = (
        [{"speaker": s.get("name") or s.get("student_id"), "student_id": s.get("student_id")} for s in speakers_list]
    )
    allowed_speakers_from_input = list(allowed_speakers_from_input) if speakers_list else None
    refined_so_far: list = []
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
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
            else:
                count += 1
                t0, t1 = item.get("start", 0), item.get("end", 0)
                speaker = item.get("speaker", "?")
                text = (item.get("text") or "").strip()
                preview = (text[:30] + "..") if len(text) > 30 else text
                ts = f"{secs_to_hms(t0)} - {secs_to_hms(t1)}"
                print(f"[服务端] 进度 第 {count} 句 | {ts} | {speaker}: {preview}")
                if refine and pipeline:
                    raw_event = {**item, "type": "raw"}
                    yield f"data: {json.dumps(raw_event, ensure_ascii=False)}\n\n"
                    batch = await pipeline.run_incremental(
                        refined_so_far, [item], allowed_speakers_from_input=allowed_speakers_from_input
                    )
                    refined_so_far.extend(batch)
                    for r in batch:
                        refined_event = {**r, "type": "refined"}
                        yield f"data: {json.dumps(refined_event, ensure_ascii=False)}\n\n"
                else:
                    yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
    finally:
        Path(tmp_path).unlink(missing_ok=True)
        print(f"[服务端] 流式转录结束 | 共 {count} 句")


async def _wait_client_disconnect(request: Request, cancelled: threading.Event) -> None:
    """客户端（如 Django 代理）断开连接时设置 cancelled，便于转录循环提前退出。"""
    try:
        message = await request.receive()
        if message.get("type") == "http.disconnect":
            cancelled.set()
            print("[服务端] 取消转录：收到客户端断开连接，已中止本次转录任务")
    except Exception as e:
        print(f"[服务端] 取消转录：监听连接时异常，已中止 | {e}")
        cancelled.set()


async def _parse_speakers_and_audio(
    student_id: List[str],
    name: List[str],
    embedding: List[str],
    audio: UploadFile,
) -> tuple[list, bytes, str]:
    """校验并解析 form，返回 (speakers_list, content, suffix)。"""
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
    return speakers_list, content, suffix


@router.post("/transcriptions")
async def create_transcription(
    request: Request,
    student_id: List[str] = Form(..., description="学号，可多个，按顺序与 name、embedding 对应"),
    name: List[str] = Form(..., description="姓名，可多个"),
    embedding: List[str] = Form(..., description="256 维向量 JSON 字符串，如 [0.1,-0.2,...]"),
    audio: UploadFile = File(..., description="待转写音频"),
    language: str = Form("", description="Whisper 语言，如 zh、en，空则自动检测"),
    stream: bool = Form(False, description="为 true 时以 SSE 流式返回，每完成一句推送一条"),
    refine: bool = Form(False, description="仅 stream=true 时有效：是否在流中同时做精修后推送"),
):
    """
    提交若干个（学号、姓名、向量）三元组 + 待测音频。
    stream=false：返回 JSON { "utterances": [ ... ] }。
    stream=true：返回 text/event-stream，每完成一句推送一条 data 事件；可加 refine=true 在流中精修后推送。

    curl 示例（2 个说话人，非流式）:
      curl -X POST http://127.0.0.1:8001/transcriptions \\
        -F "student_id=2021001" -F "student_id=2021002" \\
        -F "name=张三" -F "name=李四" \\
        -F "embedding=[-0.13,0.12,...]" -F "embedding=[0.2,-0.1,...]" \\
        -F "audio=@audio.wav" -F "language=zh"
    流式: 同上并加 -F "stream=true" ；带精修再加 -F "refine=true"
    """
    speakers_list, content, suffix = await _parse_speakers_and_audio(
        student_id, name, embedding, audio
    )
    n = len(speakers_list)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(content)
        tmp_path = f.name
    lang = language.strip() or None

    if stream:
        print(f"[服务端] 转录 stream=true | 说话人数: {n} | 音频: {audio.filename} | refine={refine}")
        return StreamingResponse(
            _stream_transcribe_events(tmp_path, speakers_list, lang, refine=refine),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    cancelled = threading.Event()
    disconnect_task = asyncio.create_task(_wait_client_disconnect(request, cancelled))
    try:
        print(f"[服务端] /transcriptions 转录开始 | 说话人数: {n} | 音频: {audio.filename}")
        loop = asyncio.get_event_loop()
        utterances = await loop.run_in_executor(
            None,
            lambda: transcribe_with_speakers(tmp_path, speakers_list, language=lang, cancelled=cancelled),
        )
        if cancelled.is_set():
            print("[服务端] 取消转录：/transcriptions 任务已提前结束（客户端已断开）")
        else:
            print(f"[服务端] /transcriptions 转录完成 | 共 {len(utterances)} 句")
        return {"utterances": utterances}
    except Exception as e:
        raise HTTPException(**err_transcribe(e))
    finally:
        disconnect_task.cancel()
        try:
            await disconnect_task
        except asyncio.CancelledError:
            pass
        Path(tmp_path).unlink(missing_ok=True)

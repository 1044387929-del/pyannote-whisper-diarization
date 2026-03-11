"""转写接口：批量与流式"""
import asyncio
import json
import tempfile
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from core.audio import transcribe_with_speakers, transcribe_with_speakers_stream
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


@router.post("/transcribe")
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


@router.post("/transcribe/stream")
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

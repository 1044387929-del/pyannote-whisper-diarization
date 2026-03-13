"""流水线接口：推断说话人 + 合并碎片 + 纠错（浅度流水线，可选 stream 看进度）"""
import asyncio
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from core.llm import run_pipeline, run_pipeline_chunked

router = APIRouter(tags=["refine"])


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/refinements")
async def create_refinement(body: dict):
    """
    浅度流水线：1) 推断 unknown 说话人  2) 合并同人同句碎片  3) 纠错与标点。
    请求体: {
      "utterances": [ { "start", "end", "speaker", "student_id", "text" }, ... ],
      "infer_speakers": true,
      "merge": true,
      "correct_text": true,
      "context_size": 5,
      "allowed_speakers": [ { "speaker": "姓名或显示名", "student_id": "学号" }, ... ],
      "stream": false,  // 为 true 时按块精修，修完一块即 SSE 推送该块结果（真正流式）
      "chunk_size": 15 // 可选，stream=true 时每块条数，默认 15
    }
    非流式响应: { "utterances": [ ... ] }
    流式响应 (stream=true): 按块执行，每完成一块即推送 data: {"stage": "utterance", "index": i, "utterance": {...}, "progress": 0~1}；最后 data: {"stage": "done", "progress": 1, "utterances": [...]}；错误 data: {"stage": "error", "detail": "..."}
    """

    utterances = body.get("utterances")
    if not isinstance(utterances, list):
        raise HTTPException(status_code=400, detail="缺少 utterances 数组")
    infer_speakers = body.get("infer_speakers", True)
    merge = body.get("merge", True)
    correct_text = body.get("correct_text", True)
    context_size = max(1, min(10, int(body.get("context_size", 3))))
    allowed_speakers = body.get("allowed_speakers")
    if allowed_speakers is not None and not isinstance(allowed_speakers, list):
        raise HTTPException(status_code=400, detail="allowed_speakers 须为数组")
    if allowed_speakers and not all(
        isinstance(a, dict) and (a.get("speaker") is not None or a.get("student_id") is not None)
        for a in allowed_speakers
    ):
        raise HTTPException(status_code=400, detail="allowed_speakers 每项须为含 speaker 或 student_id 的对象")
    stream = body.get("stream", False)

    if stream:
        queue: asyncio.Queue = asyncio.Queue()

        def progress_callback(stage: str, progress: float, extra=None):
            queue.put_nowait({"stage": stage, "progress": progress, "extra": extra})

        chunk_size = max(1, min(50, int(body.get("chunk_size", 15))))

        async def run_with_callback():
            try:
                await run_pipeline_chunked(
                    utterances,
                    chunk_size=chunk_size,
                    infer_speakers=infer_speakers,
                    merge=merge,
                    correct_text=correct_text,
                    context_size=context_size,
                    allowed_speakers_from_input=allowed_speakers if allowed_speakers else None,
                    progress_callback=progress_callback,
                )
            except Exception as e:
                queue.put_nowait({"stage": "error", "progress": 0, "extra": str(e)})

        async def event_gen():
            asyncio.create_task(run_with_callback())
            while True:
                ev = await queue.get()
                stage = ev.get("stage", "")
                progress = ev.get("progress", 0)
                extra = ev.get("extra")
                if stage == "error":
                    yield _sse({"stage": "error", "progress": 0, "detail": extra})
                    return
                if stage == "utterance" and extra is not None:
                    try:
                        i, u = extra[0], extra[1]
                    except (TypeError, IndexError):
                        i, u = 0, extra
                    yield _sse({"stage": "utterance", "index": i, "utterance": u, "progress": progress})
                    continue
                if stage == "done" and extra is not None:
                    yield _sse({"stage": "done", "progress": 1, "utterances": extra})
                    break
                payload = {"stage": stage, "progress": progress}
                yield _sse(payload)

        return StreamingResponse(
            event_gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        result = await run_pipeline(
            utterances,
            infer_speakers=infer_speakers,
            merge=merge,
            correct_text=correct_text,
            context_size=context_size,
            allowed_speakers_from_input=allowed_speakers if allowed_speakers else None,
        )
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"流水线失败: {e}")
    return {"utterances": result}

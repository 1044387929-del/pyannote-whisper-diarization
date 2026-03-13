"""转录文本 T-SEDA 标记接口，使用项目内 rag_tseda RAG 流程"""
import asyncio
import json
import re

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from rag_tseda.recall import label_by_rag

router = APIRouter(tags=["label"])

# 默认上下各 3 条作为上下文
CONTEXT_WINDOW = 3
# 并发数，避免打满 LLM/embedding
MAX_CONCURRENT = 8


def get_context_from_utterances(
    utterances: list[dict],
    current_index: int,
    *,
    text_key: str = "text",
    speaker_key: str = "speaker",
    window_size: int = CONTEXT_WINDOW,
) -> str:
    """
    从 utterance 列表中取当前句的前后窗口，拼成上下文字符串。
    格式与 text_annotation/label.py 的 get_context_info 一致。
    """
    n = len(utterances)
    start = max(0, current_index - window_size)
    end = min(n, current_index + window_size + 1)
    lines = []
    for i in range(start, end):
        u = utterances[i]
        text = (u.get(text_key) or "").strip()
        if not text:
            continue
        speaker = (u.get(speaker_key) or "").strip() or "未知"
        if i == current_index:
            lines.append(f"[待标记行] {speaker}：{text}")
        else:
            lines.append(f"{speaker}：{text}")
    return "\n".join(lines) if lines else "无上下文信息"


def _parse_label_result(result_text: str) -> tuple[str | None, str | None]:
    """从 RAG 返回的字符串中解析 JSON，取出 match_label 和 match_reason。"""
    if not (result_text or result_text.strip()):
        return None, None
    json_str = result_text.strip()
    if "```json" in json_str:
        m = re.search(r"```json\s*(.*?)\s*```", json_str, re.DOTALL)
        if m:
            json_str = m.group(1).strip()
    elif "```" in json_str:
        m = re.search(r"```\s*(.*?)\s*```", json_str, re.DOTALL)
        if m:
            json_str = m.group(1).strip()
    if json_str.startswith('"') and json_str.endswith('"'):
        json_str = json_str[1:-1]
    try:
        data = json.loads(json_str)
        return (
            data.get("match_label") or None,
            data.get("match_reason") or None,
        )
    except json.JSONDecodeError:
        return None, None


async def _label_one(
    u: dict,
    utterances: list[dict],
    index: int,
    context_window: int,
) -> dict:
    """单条标注，返回 { text, speaker, label, reason }。"""
    text = (u.get("text") or "").strip()
    speaker = (u.get("speaker") or "").strip()
    out = {"text": text or "", "speaker": speaker, "label": None, "reason": None}
    if not text:
        return out
    context_info = get_context_from_utterances(
        utterances, index, window_size=context_window
    )
    try:
        res = await label_by_rag(text, context_info=context_info)
        raw = (res or {}).get("result", "")
        label, reason = _parse_label_result(raw)
        out["label"] = label
        out["reason"] = reason
    except ValueError:
        raise
    except Exception as e:
        out["reason"] = f"标注异常: {e}"
    return out


@router.post("/labels")
async def label_utterances(body: dict):
    """
    为转录好的文本做 T-SEDA 标记（使用项目内 rag_tseda RAG）。

    请求体:
      {
        "utterances": [
          { "text": "句子内容", "speaker": "说话人（可选）" },
          ...
        ],
        "context_window": 3,   // 可选，前后各 N 句作为上下文，默认 3
        "stream": true,        // 可选，为 true 时使用 SSE 按条流式返回结果
        "max_concurrent": 8    // 可选，最大并发数，默认 8
      }

    非流式响应:
      { "utterances": [ { "text", "speaker", "label", "reason" }, ... ] }

    流式响应 (stream=true):
      text/event-stream，每条事件为: data: {"index": i, "utterance": {...}}
      结束时: data: {"done": true}
    """
    utterances = body.get("utterances")
    if not isinstance(utterances, list):
        raise HTTPException(status_code=400, detail="缺少 utterances 数组")
    context_window = max(0, min(10, int(body.get("context_window", CONTEXT_WINDOW))))
    stream = body.get("stream", False)
    max_concurrent = max(1, min(20, int(body.get("max_concurrent", MAX_CONCURRENT))))
    sem = asyncio.Semaphore(max_concurrent)

    async def run_one(i: int, u: dict) -> tuple[int, dict]:
        async with sem:
            out = await _label_one(u, utterances, i, context_window)
            return (i, out)

    tasks = [run_one(i, u) for i, u in enumerate(utterances)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    ordered: list[tuple[int, dict]] = []
    for r in results:
        if isinstance(r, ValueError):
            err_msg = str(r)
            if stream:
                async def event_gen_err(msg: str):
                    yield f"data: {json.dumps({'error': msg}, ensure_ascii=False)}\n\n"
                return StreamingResponse(
                    event_gen_err(err_msg),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
                )
            raise HTTPException(status_code=503, detail=err_msg)
        if isinstance(r, Exception):
            err_msg = str(r)
            if stream:
                async def event_gen_err(msg: str):
                    yield f"data: {json.dumps({'error': msg}, ensure_ascii=False)}\n\n"
                return StreamingResponse(
                    event_gen_err(err_msg),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
                )
            raise HTTPException(status_code=500, detail=err_msg)
        ordered.append(r)
    ordered.sort(key=lambda x: x[0])
    result_list = [out for _, out in ordered]

    if stream:
        async def event_gen():
            for i, out in ordered:
                yield f"data: {json.dumps({'index': i, 'utterance': out}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'done': True}, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return {"utterances": result_list}

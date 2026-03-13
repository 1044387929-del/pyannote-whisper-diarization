"""根据讨论指标（POST /metrics 返回的 JSON）生成小组与参与者的评价与建议；context 来自 FAISS RAG。"""
import asyncio
import json
import re

from fastapi import APIRouter, HTTPException

from core.llm import get_llm
from prompts.eval_and_sug import REVIEW_PROMPT
from rag_tseda.recall import get_eval_rag_context

router = APIRouter(tags=["eval_suggestion"])

# RAG 未命中或未提供额外 context 时的补充说明
FALLBACK_CONTEXT = "请依据本提示中的「字段含义说明」与 T-SEDA 标签含义进行评价，不编造文档外内容。"


def _extract_json_from_llm(text: str) -> dict:
    """从 LLM 返回文本中解析 JSON（允许被 markdown 代码块包裹）。"""
    if not (text or text.strip()):
        raise ValueError("模型返回为空")
    raw = text.strip()
    if "```json" in raw:
        m = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
        if m:
            raw = m.group(1).strip()
    elif "```" in raw:
        m = re.search(r"```\s*(.*?)\s*```", raw, re.DOTALL)
        if m:
            raw = m.group(1).strip()
    return json.loads(raw)


@router.post("/evaluation")
async def create_evaluation(body: dict):
    """
    根据讨论指标生成小组整体评价与每位参与者的评价、建议。

    请求体:
      - metrics: 必填，POST /metrics 的返回结果（summary、participants、group）
      - context: 可选，与 RAG 召回内容拼接的额外参考文档

    评价时的参考 context 来自 FAISS RAG（与 T-SEDA 标注共用向量库），再拼接 body.context（若有）。

    返回: 与 prompts/eval_and_sug 中 output 结构一致
      - group_evaluation: 小组整体评价
      - participants_evaluation: 列表，每项含 speaker、student_id、evaluation、suggestion
    """
    metrics_data = body.get("metrics")
    if metrics_data is None:
        raise HTTPException(status_code=400, detail="缺少 metrics 字段")
    if not isinstance(metrics_data, dict):
        raise HTTPException(status_code=400, detail="metrics 须为对象（指标 JSON）")

    try:
        input_text = json.dumps(metrics_data, ensure_ascii=False, indent=2)
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"metrics 无法序列化为 JSON: {e}")

    # RAG：用 FAISS 根据指标内容召回相关文档，避免阻塞事件循环
    rag_context = await asyncio.to_thread(get_eval_rag_context, metrics_data, 5)
    extra = (body.get("context") or "").strip()
    context = (rag_context.strip() + "\n\n" + extra).strip() if (rag_context or extra) else FALLBACK_CONTEXT

    llm = get_llm(streaming=False)
    messages = REVIEW_PROMPT.invoke({"input_text": input_text, "context": context})
    response = await llm.ainvoke(messages)

    content = (response.content or "").strip()
    if not content:
        raise HTTPException(status_code=502, detail="模型未返回内容")

    try:
        result = _extract_json_from_llm(content)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=502,
            detail=f"模型返回无法解析为 JSON: {e}. 原始片段: {content[:500]}",
        )
    except ValueError as e:
        raise HTTPException(status_code=502, detail=str(e))

    if not isinstance(result, dict):
        raise HTTPException(status_code=502, detail="解析结果不是对象")
    return result

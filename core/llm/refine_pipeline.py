"""
遍历式流水线：解决语句破碎、纠错词汇、识别未知说话人
"""
import copy
import re
from typing import List

# 句末标点
SENTENCE_END = re.compile(r"[。！？.!?]\s*$")


def _is_sentence_end(text: str) -> bool:
    if not (text or "").strip():
        return False
    return bool(SENTENCE_END.search((text or "").strip()))


def _normalize_speaker(s: str) -> str:
    if s is None:
        return "unknown"
    s = (s or "").strip().lower()
    if s in ("unknown", ""):
        return "unknown"
    return s


def infer_unknown_speakers(
    utterances: List[dict],
    llm,
    context_size: int = 3,
) -> List[dict]:
    from prompts.refine import infer_speaker_prompt

    n = len(utterances)
    for i in range(n):
        u = utterances[i]
        if _normalize_speaker(u.get("speaker")) != "unknown":
            continue
        text = (u.get("text") or "").strip()
        if not text:
            continue
        start = max(0, i - context_size)
        end = min(n, i + context_size + 1)
        context_parts = []
        for j in range(start, end):
            t = (utterances[j].get("text") or "").strip()
            if not t:
                continue
            if j == i:
                context_parts.append("[当前句] " + t)
            else:
                context_parts.append(utterances[j].get("speaker", "?") + ": " + t)
        context = "\n".join(context_parts)
        if not context.strip():
            continue
        messages = infer_speaker_prompt.invoke({"context": context, "current_text": text})
        try:
            resp = llm.invoke(messages)
            content = (getattr(resp, "content", None) or str(resp)).strip()
            name = content.split("\n")[0].strip().split()[0] if content else "unknown"
            if name.lower() == "unknown" or not name:
                name = "unknown"
            u["speaker"] = name
            u["student_id"] = name
        except Exception:
            pass
    return utterances


def merge_fragments(utterances: List[dict]) -> List[dict]:
    if not utterances:
        return []
    result = []
    cur = None
    for u in utterances:
        speaker = _normalize_speaker(u.get("speaker"))
        text = (u.get("text") or "").strip()
        start = u.get("start", 0)
        end = u.get("end", 0)
        student_id = u.get("student_id") or speaker
        if cur is None:
            cur = {"start": start, "end": end, "speaker": speaker, "student_id": student_id, "text": text}
            continue
        if speaker == _normalize_speaker(cur.get("speaker")) and not _is_sentence_end(cur["text"]):
            cur["end"] = end
            cur["text"] = (cur["text"] + " " + text).strip() if cur["text"] else text
            cur["student_id"] = student_id
        else:
            result.append(cur)
            cur = {"start": start, "end": end, "speaker": speaker, "student_id": student_id, "text": text}
    if cur is not None:
        result.append(cur)
    return result


def correct_utterance_texts(utterances: List[dict], llm) -> List[dict]:
    from prompts.refine import correct_text_prompt

    for u in utterances:
        text = (u.get("text") or "").strip()
        if not text:
            continue
        try:
            messages = correct_text_prompt.invoke({"text": text})
            resp = llm.invoke(messages)
            content = (getattr(resp, "content", None) or str(resp)).strip()
            if content:
                u["text"] = content
        except Exception:
            pass
    return utterances


def run_pipeline(
    utterances: List[dict],
    llm=None,
    infer_speakers: bool = True,
    merge: bool = True,
    correct_text: bool = True,
    context_size: int = 3,
) -> List[dict]:
    if not utterances:
        return []
    if llm is None:
        from .llm_client import get_llm
        llm = get_llm()
    out = copy.deepcopy(utterances)
    if infer_speakers:
        out = infer_unknown_speakers(out, llm, context_size=context_size)
    if merge:
        out = merge_fragments(out)
    if correct_text:
        out = correct_utterance_texts(out, llm)
    return out

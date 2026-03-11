"""
遍历式流水线：解决语句破碎、纠错词汇、识别未知说话人
异步实现：推断说话人顺序 ainvoke，纠错步骤并发请求，减少阻塞。
"""
import asyncio
import copy
import re
from typing import List, Optional

# 句末标点（判断是否已是一句、拆分多句）
SENTENCE_END = re.compile(r"[。！？.!?]\s*$")
# 按句末标点拆分：在 。！？.!? 之后切分，保留标点在该句末尾
SENTENCE_SPLIT = re.compile(r"(?<=[。！？.!?])\s*")


def _is_sentence_end(text: str) -> bool:
    if not (text or "").strip():
        return False
    return bool(SENTENCE_END.search((text or "").strip()))


def _split_text_into_sentences(text: str) -> List[str]:
    """按句末标点（。！？.!?）拆成多句，保留标点；无句末标点则整体视为一句。"""
    if not (text or "").strip():
        return []
    text = text.strip()
    parts = SENTENCE_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def _normalize_speaker(s: str) -> str:
    if s is None:
        return "unknown"
    s = (s or "").strip().lower()
    if s in ("unknown", ""):
        return "unknown"
    return s


# 当整段没有任何已标注说话人时，用于兜底的默认说话人（保证每条发言都有归属）
DEFAULT_SPEAKER = "speaker_0"


def _get_allowed_speakers(utterances: List[dict]) -> List[dict]:
    """从原始列表中收集所有非 unknown 的 (speaker, student_id)，去重并保持顺序。若全为 unknown 则返回默认说话人。"""
    seen = set()
    out = []
    for u in utterances:
        s = _normalize_speaker(u.get("speaker"))
        if s == "unknown":
            continue
        key = (s, (u.get("student_id") or s))
        if key in seen:
            continue
        seen.add(key)
        out.append({"speaker": u.get("speaker"), "student_id": u.get("student_id") or u.get("speaker")})
    if not out:
        out = [{"speaker": DEFAULT_SPEAKER, "student_id": DEFAULT_SPEAKER}]
    return out


def _has_unknown_speaker(utterances: List[dict]) -> bool:
    """是否存在说话人为 unknown 的条目（用于判断是否需要调用 LLM 做说话人推断）。"""
    return any(_normalize_speaker(u.get("speaker")) == "unknown" for u in utterances)


async def infer_unknown_speakers(
    out: List[dict],
    original: List[dict],
    llm,
    allowed_speakers: List[dict],
    context_size: int = 3,
    verbose: bool = True,
) -> List[dict]:
    """推断 unknown 说话人。仅对 speaker 为 unknown 的条目调用 LLM，已有归属的不猜。"""
    from prompts.refine import infer_speaker_prompt

    if not allowed_speakers:
        return out
    if not _has_unknown_speaker(out):
        if verbose:
            print("[1/4] 推断未知说话人：无 unknown，跳过（省 token）")
        return out
    allowed_names = {_normalize_speaker(a["speaker"]) for a in allowed_speakers}
    allowed_by_name = {_normalize_speaker(a["speaker"]): a for a in allowed_speakers}
    allowed_str = ", ".join(f"{a['speaker']}({a['student_id']})" for a in allowed_speakers)

    n = len(out)
    done = 0
    for i in range(n):
        u = out[i]
        if _normalize_speaker(u.get("speaker")) != "unknown":
            continue
        text = (u.get("text") or "").strip()
        if not text:
            # 无文本也必须有归属
            u["speaker"] = allowed_speakers[0]["speaker"]
            u["student_id"] = allowed_speakers[0]["student_id"]
            continue
        start = max(0, i - context_size)
        end = min(n, i + context_size + 1)
        context_parts = []
        for j in range(start, end):
            if j < i:
                row = out[j]
                t = (row.get("text") or "").strip()
                label = row.get("speaker", "?")
            elif j > i:
                row = original[j] if j < len(original) else out[j]
                t = (row.get("text") or "").strip()
                label = row.get("speaker", "?")
            else:
                t = text
                label = "[当前句]"
            if not t:
                continue
            context_parts.append(f"{label}: {t}" if label != "[当前句]" else f"[当前句] {t}")
        context = "\n".join(context_parts)
        if not context.strip():
            continue
        done += 1
        if verbose:
            if done == 1:
                print("[1/4] 推断未知说话人（仅从已有说话人中选择）")
            short = text[:20] + "..." if len(text) > 20 else text
            print(f"  {done}. {short} -> ", end="", flush=True)
        messages = infer_speaker_prompt.invoke({
            "context": context,
            "current_text": text,
            "allowed_speakers": allowed_str,
        })
        try:
            resp = await llm.ainvoke(messages)
            content = (getattr(resp, "content", None) or str(resp)).strip()
            name_raw = content.split("\n")[0].strip().split()[0] if content else ""
            name_lower = name_raw.lower() if name_raw else ""
            # 必须归属：不允许 unknown，无效或 unknown 时用第一个允许的说话人兜底
            fallback = allowed_speakers[0]
            if name_lower and name_lower in allowed_names:
                entry = allowed_by_name.get(name_lower)
                if entry:
                    u["speaker"] = entry["speaker"]
                    u["student_id"] = entry["student_id"]
                else:
                    u["speaker"] = fallback["speaker"]
                    u["student_id"] = fallback["student_id"]
            else:
                u["speaker"] = fallback["speaker"]
                u["student_id"] = fallback["student_id"]
                if verbose and (name_lower == "unknown" or not name_lower or name_lower not in allowed_names):
                    print(f"{u['speaker']}(兜底)", flush=True)
                    continue
            if verbose:
                print(u["speaker"], flush=True)
        except Exception as e:
            if verbose:
                print(f"失败: {e}", flush=True)
            # 异常时也必须有归属
            u["speaker"] = allowed_speakers[0]["speaker"]
            u["student_id"] = allowed_speakers[0]["student_id"]
    return out


def merge_fragments(utterances: List[dict], verbose: bool = True) -> List[dict]:
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
    if verbose:
        print(f"[2/4] 合并片段: {len(utterances)} -> {len(result)} 条")
    return result


async def correct_utterance_texts(
    utterances: List[dict],
    llm,
    verbose: bool = True,
    max_concurrent: int = 10,
) -> List[dict]:
    """纠错与标点。多条并发请求，用 semaphore 限制并发数，减少阻塞与总耗时。"""
    from prompts.refine import correct_text_prompt

    to_correct = [(i, u) for i, u in enumerate(utterances) if (u.get("text") or "").strip()]
    total = len(to_correct)
    if not total:
        return utterances

    if verbose:
        print(f"[3/4] 纠错文本，共 {total} 条（异步并发，max_concurrent={max_concurrent}）")

    sem = asyncio.Semaphore(max_concurrent)

    async def correct_one(i: int, u: dict) -> tuple[int, dict, Optional[str]]:
        text = (u.get("text") or "").strip()
        if not text:
            return i, u, None
        messages = correct_text_prompt.invoke({"text": text})
        async with sem:
            resp = await llm.ainvoke(messages)
        content = (getattr(resp, "content", None) or str(resp)).strip()
        return i, u, content if content else None

    tasks = [correct_one(i, u) for i, u in to_correct]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for idx, r in enumerate(results):
        if isinstance(r, BaseException):
            if verbose:
                print(f"  {idx + 1}/{total} 失败: {r}", flush=True)
            continue
        _i, u, content = r
        if content is not None:
            u["text"] = content
        if verbose:
            short_in = (u.get("text") or "")[:20] + ("..." if len(u.get("text") or "") > 20 else "")
            print(f"  {idx + 1}/{total} -> {short_in}", flush=True)

    return utterances


def split_utterances_by_sentence(utterances: List[dict], verbose: bool = True) -> List[dict]:
    """按句末标点（。！？.!?）拆分：一句对应一个 utterance，多句拆成多条并按字符比例分配时间。"""
    if not utterances:
        return []
    result = []
    for u in utterances:
        text = (u.get("text") or "").strip()
        if not text:
            result.append(u)
            continue
        sentences = _split_text_into_sentences(text)
        if len(sentences) <= 1:
            result.append(u)
            continue
        start = float(u.get("start", 0))
        end = float(u.get("end", 0))
        total_chars = len(text)
        cumulative = 0
        for sent in sentences:
            seg_start = start + (end - start) * (cumulative / total_chars) if total_chars else start
            cumulative += len(sent)
            seg_end = start + (end - start) * (cumulative / total_chars) if total_chars else end
            result.append({
                "start": seg_start,
                "end": seg_end,
                "speaker": u.get("speaker"),
                "student_id": u.get("student_id"),
                "text": sent,
            })
    if verbose:
        print(f"[按句拆分] {len(utterances)} -> {len(result)} 条（一句一 utterance）")
    return result


# 视为无意义并删除：空、纯空白、或仅语气词/极短
_MEANINGLESS = frozenset({"嗯", "啊", "哦", "呃", "唉", "咳", "呵", "哈", "诶", "噢", "唔", "欸", "..."})


def filter_empty_and_meaningless(utterances: List[dict], verbose: bool = True) -> List[dict]:
    """删除空文本及无意义短句（如纯语气词）。"""
    out = []
    dropped = 0
    for u in utterances:
        text = (u.get("text") or "").strip()
        if not text:
            dropped += 1
            continue
        if text in _MEANINGLESS or (len(text) <= 1 and not text.isalnum()):
            dropped += 1
            continue
        out.append(u)
    if verbose and dropped:
        print(f"[4/4] 删除空/无意义发言: {dropped} 条，剩余 {len(out)} 条")
    return out


class RefinePipeline:
    """
    转写精修流水线（面向对象封装）。
    配置集中在实例上，便于维护与复用；入口为 async run(utterances)。
    """

    def __init__(
        self,
        llm=None,
        *,
        infer_speakers: bool = True,
        merge: bool = True,
        correct_text: bool = True,
        filter_meaningless: bool = True,
        context_size: int = 3,
        verbose: bool = False,
        correct_max_concurrent: int = 10,
    ):
        if llm is None:
            from .llm_client import get_llm
            llm = get_llm()
        self.llm = llm
        self.infer_speakers = infer_speakers
        self.merge = merge
        self.correct_text = correct_text
        self.filter_meaningless = filter_meaningless
        self.context_size = context_size
        self.verbose = verbose
        self.correct_max_concurrent = correct_max_concurrent

    async def run(self, utterances: List[dict]) -> List[dict]:
        """执行精修流水线，返回精修后的 utterance 列表。"""
        if not utterances:
            return []
        # 深拷贝原始列表
        original = copy.deepcopy(utterances)
        # 深拷贝原始列表
        out = copy.deepcopy(utterances)
        # 获取允许的说话人列表
        allowed_speakers = _get_allowed_speakers(original)

        # 如果需要推断说话人，并且有 unknown 说话人，则推断说话人
        if self.infer_speakers and _has_unknown_speaker(out):
            # 推断说话人
            out = await infer_unknown_speakers(
                out, original, self.llm, allowed_speakers,
                context_size=self.context_size, verbose=self.verbose,
            )
        # 如果需要合并片段，则合并片段
        if self.merge:
            # 合并片段
            out = merge_fragments(out, verbose=self.verbose)
        # 如果需要纠错文本，则纠错文本
        if self.correct_text:
            # 纠错文本
            out = await correct_utterance_texts(
                out, self.llm, verbose=self.verbose,
                max_concurrent=self.correct_max_concurrent,
            )
        out = split_utterances_by_sentence(out, verbose=self.verbose)
        if self.filter_meaningless:
            out = filter_empty_and_meaningless(out, verbose=self.verbose)
        out = _force_no_unknown(out, verbose=self.verbose)
        return out


async def run_pipeline(
    utterances: List[dict],
    llm=None,
    infer_speakers: bool = True,
    merge: bool = True,
    correct_text: bool = True,
    filter_meaningless: bool = True,
    context_size: int = 3,
    verbose: bool = False,
    correct_max_concurrent: int = 10,
) -> List[dict]:
    """兼容入口：构造 RefinePipeline 并执行 run。保留原有函数式调用方式。"""
    pipeline = RefinePipeline(
        llm=llm,
        infer_speakers=infer_speakers,
        merge=merge,
        correct_text=correct_text,
        filter_meaningless=filter_meaningless,
        context_size=context_size,
        verbose=verbose,
        correct_max_concurrent=correct_max_concurrent,
    )
    return await pipeline.run(utterances)


def _force_no_unknown(utterances: List[dict], verbose: bool = True) -> List[dict]:
    """将仍为 unknown 的发言归为 DEFAULT_SPEAKER，保证每条都有归属。"""
    replaced = 0
    for u in utterances:
        if _normalize_speaker(u.get("speaker")) == "unknown":
            u["speaker"] = DEFAULT_SPEAKER
            u["student_id"] = DEFAULT_SPEAKER
            replaced += 1
    if verbose and replaced:
        print(f"[兜底] 将 {replaced} 条 unknown 归为 {DEFAULT_SPEAKER}")
    return utterances

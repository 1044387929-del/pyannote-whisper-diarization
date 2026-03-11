"""流水线接口：推断说话人 + 合并碎片 + 纠错"""
from fastapi import APIRouter, HTTPException

from core.llm import run_pipeline

router = APIRouter(tags=["refine"])


@router.post("/refinements")
async def create_refinement(body: dict):
    """
    遍历式流水线：1) 推断 unknown 说话人  2) 合并同人同句碎片  3) 纠错与标点。
    请求体: {
      "utterances": [ { "start", "end", "speaker", "student_id", "text" }, ... ],
      "infer_speakers": true,
      "merge": true,
      "correct_text": true,
      "context_size": 5,
      "allowed_speakers": [ { "speaker": "姓名或显示名", "student_id": "学号" }, ... ]  // 可选，传则只保留这些说话人，无 speaker_0/unknown
    }
    响应: { "utterances": [ ... ] }
    """

    # 1. 验证请求体
    utterances = body.get("utterances")
    # 2. 验证 utterances 是否为数组
    if not isinstance(utterances, list):
        raise HTTPException(status_code=400, detail="缺少 utterances 数组")
    # 3. 验证 infer_speakers 是否为布尔值
    infer_speakers = body.get("infer_speakers", True)
    # 4. 验证 merge 是否为布尔值
    merge = body.get("merge", True)
    # 5. 验证 correct_text 是否为布尔值
    correct_text = body.get("correct_text", True)
    # 6. 验证 context_size 是否为整数，且在 1-10 之间
    context_size = max(1, min(10, int(body.get("context_size", 3))))
    allowed_speakers = body.get("allowed_speakers")
    if allowed_speakers is not None and not isinstance(allowed_speakers, list):
        raise HTTPException(status_code=400, detail="allowed_speakers 须为数组")
    if allowed_speakers and not all(isinstance(a, dict) and (a.get("speaker") is not None or a.get("student_id") is not None) for a in allowed_speakers):
        raise HTTPException(status_code=400, detail="allowed_speakers 每项须为含 speaker 或 student_id 的对象")
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

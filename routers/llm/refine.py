"""流水线接口：推断说话人 + 合并碎片 + 纠错"""
from fastapi import APIRouter, HTTPException

from core.llm import run_pipeline

router = APIRouter(tags=["refine"])


@router.post("/refine")
async def refine(body: dict):
    """
    遍历式流水线：1) 推断 unknown 说话人  2) 合并同人同句碎片  3) 纠错与标点。
    请求体: {
      "utterances": [ { "start", "end", "speaker", "student_id", "text" }, ... ],
      "infer_speakers": true,
      "merge": true,
      "correct_text": true,
      "context_size": 3
    }
    响应: { "utterances": [ ... ] }
    """
    utterances = body.get("utterances")
    if not isinstance(utterances, list):
        raise HTTPException(status_code=400, detail="缺少 utterances 数组")
    infer_speakers = body.get("infer_speakers", True)
    merge = body.get("merge", True)
    correct_text = body.get("correct_text", True)
    context_size = max(1, min(10, int(body.get("context_size", 3))))
    try:
        result = await run_pipeline(
            utterances,
            infer_speakers=infer_speakers,
            merge=merge,
            correct_text=correct_text,
            context_size=context_size,
        )
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"流水线失败: {e}")
    return {"utterances": result}

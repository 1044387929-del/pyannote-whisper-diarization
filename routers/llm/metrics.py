"""根据 T-SEDA 标记结果计算讨论评价指标（表层参与、认知指标、熵）。"""
from fastapi import APIRouter, HTTPException

from core.metrics import compute_tseda_metrics

router = APIRouter(tags=["metrics"])


@router.post("/metrics")
def create_metrics(body: dict):
    """
    根据标记好的发言列表计算 T-SEDA 指标。

    请求体: { "utterances": [ { "start", "end", "speaker", "student_id", "text", "label" }, ... ] }
    与 shallow_pipeline 输出格式一致（含 label、reason 亦可）。

    返回:
      - summary: 总发言条数、参与人数
      - participants: 每人表层参与（发言次数、频率比、总时长、平均时长、语速）、认知指标（BE/CDI/IDI/MDI/KCI/CCI）、熵
      - group: 小组总标注数、各类编码次数、小组熵
    """
    utterances = body.get("utterances")
    if not isinstance(utterances, list):
        raise HTTPException(status_code=400, detail="缺少 utterances 数组")
    return compute_tseda_metrics(utterances)

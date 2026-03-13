"""大模型相关接口：文本纠错、未知说话人猜测、T-SEDA 标记、讨论指标、评价与建议"""
from fastapi import APIRouter

from . import eval_suggestion, label, metrics, refine

router = APIRouter(tags=["llm"])
router.include_router(refine.router)
router.include_router(label.router)
router.include_router(metrics.router)
router.include_router(eval_suggestion.router)

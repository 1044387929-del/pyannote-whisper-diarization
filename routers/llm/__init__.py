"""大模型相关接口：文本纠错、未知说话人猜测、T-SEDA 标记等"""
from fastapi import APIRouter

from . import label, refine

router = APIRouter(tags=["llm"])
router.include_router(refine.router)
router.include_router(label.router)

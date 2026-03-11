"""大模型相关接口：文本纠错、未知说话人猜测等"""
from fastapi import APIRouter

from . import refine

router = APIRouter(tags=["llm"])
router.include_router(refine.router)

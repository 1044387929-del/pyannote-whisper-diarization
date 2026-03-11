"""大模型相关接口：文本纠错、未知说话人猜测等"""
from fastapi import APIRouter

from . import correct, guess_speakers

router = APIRouter(tags=["llm"])
router.include_router(correct.router)
router.include_router(guess_speakers.router)

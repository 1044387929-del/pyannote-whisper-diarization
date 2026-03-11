"""音频转录相关接口：embedding、转写、Live"""
from fastapi import APIRouter

from . import embedding, live, transcribe

router = APIRouter(tags=["audio"])
router.include_router(embedding.router)
router.include_router(transcribe.router)
router.include_router(live.router)

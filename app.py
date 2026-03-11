"""
FastAPI 应用入口

- 音频转录：embedding、转写、Live（routers/audio）
- 大模型纠正：纠错、未知说话人猜测（routers/llm）

启动: uvicorn app:app --host 0.0.0.0 --port 8001 --reload
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from routers import audio, health, llm

app = FastAPI(
    title="声纹 API",
    description="embedding 提取、转录、Live 转写；大模型纠错与说话人猜测",
)

# 音频转录相关（无 prefix，路径保持 /embedding、/transcribe、/live、/ws/live_transcribe）
app.include_router(audio.router)
# 大模型纠正相关（无 prefix，路径为 /correct、/guess_speakers）
app.include_router(llm.router)
# 健康检查
app.include_router(health.router)

# 静态文件
STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)

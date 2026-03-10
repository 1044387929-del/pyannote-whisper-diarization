"""
FastAPI 应用入口：声纹 embedding、转写、Live 实时转写

启动: uvicorn app:app --host 0.0.0.0 --port 8001 --reload
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from routers import embedding, health, live, transcribe

app = FastAPI(
    title="声纹 API",
    description="embedding 提取、转录（学号+姓名+向量 -> 转写结果）",
)

# 路由挂载（无 prefix，路径保持 /embedding、/transcribe 等）
app.include_router(embedding.router)
app.include_router(transcribe.router)
app.include_router(live.router)
app.include_router(health.router)

# 静态文件与 /live 页面（/live 在 live.router 中定义，此处仅挂载 static 目录）
STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)

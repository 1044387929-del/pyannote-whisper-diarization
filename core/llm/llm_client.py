"""LLM 客户端：读配置、创建 ChatOpenAI（DashScope 兼容）"""
import os
from pathlib import Path

from dotenv import load_dotenv

# 项目根（core/llm/llm_client.py -> 向上 2 层）
_BASE_DIR = Path(__file__).resolve().parent.parent
_ENV_PATH = _BASE_DIR / "config" / "llm_model.env"
if _ENV_PATH.exists():
    load_dotenv(dotenv_path=_ENV_PATH)


def get_llm(
    model: str = "qwen3.5-plus",
    api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    streaming: bool = False,
):
    from langchain_openai import ChatOpenAI
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("未设置 DASHSCOPE_API_KEY，请在 config/llm_model.env 中配置")
    return ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base=api_base,
        model=model,
        streaming=streaming,
    )

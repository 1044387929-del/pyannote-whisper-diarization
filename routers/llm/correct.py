"""大模型文本纠错：对转写结果中的 text 进行纠错"""
from fastapi import APIRouter

router = APIRouter(tags=["correct"])


@router.post("/correct")
async def correct_text():
    """
    对转写结果中的文本进行纠错（错别字、同音字、标点等）。
    请求体: { "utterances": [ { "start", "end", "speaker", "student_id", "text" }, ... ], "language": "zh" }
    响应: { "utterances": [ ... ] }，仅 text 为纠错后内容。
    TODO: 接入大模型实现。
    """
    return {"message": "TODO: 接入大模型实现纠错", "utterances": []}

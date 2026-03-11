"""大模型未知说话人猜测：根据对话内容推断 unknown 说话人"""
from fastapi import APIRouter

router = APIRouter(tags=["guess_speakers"])


@router.post("/guess_speakers")
async def guess_speakers():
    """
    根据对话内容为 speaker/student_id 为 unknown 的片段推断最可能的说话人。
    请求体: { "utterances": [ ... ], "candidate_speakers": [ "张三", "李四", ... ] }（候选可选）
    响应: { "utterances": [ ... ] }，对 unknown 项补全或增加 suggested_speaker / suggested_student_id。
    TODO: 接入大模型实现。
    """
    return {"message": "TODO: 接入大模型实现未知说话人猜测", "utterances": []}

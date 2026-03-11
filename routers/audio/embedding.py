"""声纹 embedding 提取接口"""
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from core.audio import get_embedding
from utils.common import ALLOWED_EXT, get_audio_suffix
from utils.errors import ERR_AUDIO_EMPTY, err_embedding_extract, err_unsupported_format

router = APIRouter(tags=["embedding"])


@router.post("/embeddings")
async def create_embedding(
    student_id: str = Form(..., description="学号"),
    name: str = Form("", description="姓名（可选）"),
    audio: UploadFile = File(..., description="音频文件"),
):
    """
    上传学号 + 音频，返回 256 维 embedding 向量。

    curl 示例:
      curl -X POST http://127.0.0.1:8001/embeddings \\
        -F "student_id=2021001" -F "name=张三" -F "audio=@peppa.wav"
    """
    suf = Path(audio.filename or "").suffix.lower()
    if suf and suf not in ALLOWED_EXT:
        raise HTTPException(**err_unsupported_format(suf, ", ".join(ALLOWED_EXT)))

    content = await audio.read()
    if not content:
        raise HTTPException(**ERR_AUDIO_EMPTY)

    suffix = get_audio_suffix(audio.filename)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(content)
        tmp_path = f.name

    try:
        emb = get_embedding(tmp_path)
        emb_list = emb.cpu().numpy().tolist() if hasattr(emb, "cpu") else emb.tolist()
        dim = len(emb_list)
    except Exception as e:
        raise HTTPException(**err_embedding_extract(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {
        "student_id": student_id,
        "name": name or student_id,
        "embedding": emb_list,
        "embedding_dim": dim,
    }

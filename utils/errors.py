"""统一错误码与 HTTP 异常参数，供 raise HTTPException(**err_xxx(...)) 解包使用"""

# 纯静态错误：可直接解包
ERR_AUDIO_EMPTY = {"status_code": 400, "detail": "音频文件为空"}


def err_unsupported_format(ext: str, allowed: str) -> dict:
    """不支持的音频格式"""
    return {"status_code": 400, "detail": f"不支持的音频格式 {ext}，支持: {allowed}"}


def err_speakers_mismatch(n_sid: int, n_name: int, n_emb: int) -> dict:
    """student_id/name/embedding 数量不一致"""
    return {
        "status_code": 400,
        "detail": f"student_id、name、embedding 数量须相同且非空，当前 {n_sid}/{n_name}/{n_emb}",
    }


def err_embedding_format(index: int, e: Exception | None = None) -> dict:
    """embedding JSON 解析错误"""
    msg = f"第 {index} 个 embedding 格式错误"
    if e is not None:
        msg += f": {e}"
    return {"status_code": 400, "detail": msg}


def err_embedding_not_array(index: int) -> dict:
    """embedding 非数组"""
    return {"status_code": 400, "detail": f"第 {index} 个 embedding 须为数组"}


def err_embedding_extract(e: Exception) -> dict:
    """音频提取 embedding 失败"""
    return {"status_code": 422, "detail": f"音频提取 embedding 失败: {e}"}


def err_transcribe(e: Exception) -> dict:
    """转录失败"""
    return {"status_code": 422, "detail": f"转录失败: {e}"}


def ws_error(message: str, chunk_index: int | None = None) -> dict:
    """WebSocket 错误响应，可解包给 send_json"""
    out: dict = {"type": "error", "message": message}
    if chunk_index is not None:
        out["chunk_index"] = chunk_index
    return out

"""测试 T-SEDA 标注接口（流式）：只发送服务需要的 text/speaker，结果仅保存标注，由客户端按 index 整合。"""
import json
import sys
from pathlib import Path

import requests

# 项目根（scripts/test/test_label_api.py -> 向上 3 层）
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_JSON = ROOT / "data" / "json" / "transcribe_output_refined_api.json"
LABELS_URL = "http://127.0.0.1:8001/labels"
# 仅保存标注结果，不含 start/end/student_id 等，客户端自行按 index 与本地数据合并
OUT_LABELS_JSON = ROOT / "data" / "json" / "transcribe_output_labels_only.json"

# 只测前 N 条；设为 None 表示全部
MAX_UTTERANCES = None


def main():
    if not DATA_JSON.exists():
        print(f"数据文件不存在: {DATA_JSON}")
        sys.exit(1)

    with open(DATA_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    utterances_raw = data.get("utterances", [])
    if not utterances_raw:
        print("utterances 为空")
        sys.exit(1)

    # 仅发送服务端需要的字段：text、speaker（不发送 student_id / start / end 等）
    utterances = [
        {"text": u.get("text", ""), "speaker": u.get("speaker", "")}
        for u in (utterances_raw[:MAX_UTTERANCES] if MAX_UTTERANCES else utterances_raw)
    ]

    total = len(utterances)
    print(f"请求条数: {total}（流式），仅发送 text+speaker")
    print(f"POST {LABELS_URL} stream=true ...\n")

    try:
        r = requests.post(
            LABELS_URL,
            json={"utterances": utterances, "context_window": 3, "stream": True},
            timeout=120,
            stream=True,
        )
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"请求失败: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(e.response.text[:500])
        sys.exit(1)

    result_list = []
    for line in r.iter_lines(decode_unicode=True):
        # 收到一条就处理一条，配合服务端按完成顺序流式推送
        if not line or not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if not payload:
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if obj.get("done"):
            break
        if "error" in obj:
            print(f"服务端错误: {obj['error']}")
            sys.exit(1)
        idx = obj.get("index", len(result_list))
        utt = obj.get("utterance", {})
        result_list.append(utt)
        done = len(result_list)
        pct = int(100 * done / total) if total else 0
        text_preview = (utt.get("text") or "")[:50]
        if len(utt.get("text") or "") > 50:
            text_preview += "..."
        reason_preview = (utt.get("reason") or "")[:60]
        if len(utt.get("reason") or "") > 60:
            reason_preview += "..."
        print(f"[{idx+1}/{total}] {utt.get('speaker', '')}: {text_preview}", flush=True)
        print(f"    label={utt.get('label')}  reason={reason_preview}", flush=True)
        print(f"    当前进度: {done}/{total} ({pct}%)", flush=True)
        print(flush=True)

    print(f"返回条数: {len(result_list)}")

    # 仅保存标注结果（text, speaker, label, reason），客户端按 index 与本地数据合并
    out = {"utterances": result_list}
    OUT_LABELS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_LABELS_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"标注结果已写入: {OUT_LABELS_JSON}（由客户端按 index 整合）")


if __name__ == "__main__":
    main()

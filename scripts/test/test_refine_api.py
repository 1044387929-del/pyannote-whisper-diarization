"""测试 POST /refinements 接口。需先启动服务: uvicorn app:app --host 0.0.0.0 --port 8001
在项目根目录运行: python scripts/test/test_refine_api.py
"""
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE))

import requests

INPUT_JSON = BASE / "data" / "json" / "transcribe_output.json"
OUTPUT_JSON = BASE / "data" / "json" / "transcribe_output_refined_api.json"
URL = "http://127.0.0.1:8001/refinements"


def main():
    if not INPUT_JSON.exists():
        print(f"未找到输入文件: {INPUT_JSON}")
        sys.exit(1)

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    utterances = data.get("utterances", [])
    print(f"输入 utterance 数: {len(utterances)}")

    body = {
        "utterances": utterances,
        "infer_speakers": True,
        "merge": True,
        "correct_text": True,
        "context_size": 5,
    }

    print(f"请求 POST /refinements ...")
    try:
        r = requests.post(URL, json=body, timeout=600)
    except requests.exceptions.ConnectionError as e:
        print(f"连接失败，请先启动服务: uvicorn app:app --host 0.0.0.0 --port 8001\n{e}")
        sys.exit(1)

    print(f"状态码: {r.status_code}")
    if r.status_code != 200:
        print(r.text)
        sys.exit(1)

    result = r.json()
    out_utterances = result.get("utterances", [])
    print(f"输出 utterance 数: {len(out_utterances)}")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"utterances": out_utterances}, f, ensure_ascii=False, indent=2)
    print(f"已保存: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

"""本地测试流水线：读 transcribe_output.json → 推断说话人 + 合并 + 纠错 → 保存"""
import asyncio
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE))

INPUT_JSON = BASE / "data" / "json" / "transcribe_output.json"
OUTPUT_JSON = BASE / "data" / "json" / "transcribe_output_refined.json"


def main():
    if not INPUT_JSON.exists():
        print(f"未找到: {INPUT_JSON}")
        return
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    utterances = data.get("utterances", [])
    print(f"输入片段数: {len(utterances)}")

    from core.llm import run_pipeline

    result = asyncio.run(
        run_pipeline(
            utterances,
            infer_speakers=True,
            merge=True,
            correct_text=True,
            context_size=3,
            verbose=True,
        )
    )
    print(f"输出条数: {len(result)}")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"utterances": result}, f, ensure_ascii=False, indent=2)
    print(f"已保存: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

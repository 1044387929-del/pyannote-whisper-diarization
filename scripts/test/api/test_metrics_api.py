"""测试 POST /metrics 接口。需先启动服务: uvicorn app:app --host 0.0.0.0 --port 8001
在项目根目录运行: python scripts/test/api/test_metrics_api.py

返回指标与计算公式（T-SEDA 参考文档）：
---
summary（汇总）
  total_utterances: T = 整场发言总条数
  total_participants: N = 按 (speaker, student_id) 去重后的参与人数

participants[].surface（表层参与）
  utterance_count: T_i = 该参与者发言次数
  frequency_ratio: F_i = T_i / T  发言频率比例
  total_duration_sec: D_i = Σ_j (end_j - start_j)  该参与者发言总时长（秒）
  avg_duration_sec: d̄_i = D_i / T_i  平均发言时长（秒）
  total_chars: W_i = Σ_j len(text_j)  该参与者发言总字符数
  speech_rate_chars_per_sec: SR_i = W_i / D_i  语速（字/秒）

participants[].cognitive（认知指标，仅统计 label 属于 E/B/IB/CH/IRE/R/CA/C/RD/G 的条数）
  labeled_count: N_i = 该参与者被标为上述 10 类编码的发言数（不含 NULL）
  BE: 基础表达比例 = n_E^(i) / N_i
  CDI: 认知深度指数 = (n_R + n_CH + n_B + n_CA) / N_i
  IDI: 探究驱动指数 = (n_IRE + n_IB) / N_i
  MDI: 元认知指数 = n_RD / N_i
  KCI: 知识联系指数 = n_C / N_i
  CCI: 协作建构指数 = (n_B + n_CA + n_IB) / N_i
  label_counts: 各类编码条数 n_k^(i)

participants[].entropy（信息熵）
  H_raw: H_i = -Σ_k p_k^(i) · ln(p_k^(i))，其中 p_k^(i) = n_k^(i) / N_i
  H_normalized: H_i* = H_i / ln(10)，取值 0～1

group（小组层面）
  total_labeled: N_G = 全场被标为 10 类编码的发言总数
  label_counts: 全场各类编码条数 n_k^(G)
  entropy_raw: H_G = -Σ_k p_k^(G) · ln(p_k^(G))，p_k^(G) = n_k^(G) / N_G
  entropy_normalized: H_G* = H_G / ln(10)
"""
import json
import sys
from pathlib import Path

# 项目根（scripts/test/api/ -> 上溯 4 层）
BASE = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(BASE))
TEST_DATA = Path(__file__).resolve().parent.parent / "data"

import requests

INPUT_JSON = TEST_DATA / "json" / "shallow_pipeline_output.json"
OUTPUT_JSON = TEST_DATA / "json" / "metrics_output.json"
URL = "http://127.0.0.1:8001/metrics"


def main():
    if not INPUT_JSON.exists():
        print(f"未找到输入文件: {INPUT_JSON}")
        sys.exit(1)

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    utterances = data.get("utterances", [])
    print(f"输入 utterance 数: {len(utterances)}")

    print("请求 POST /metrics ...")
    try:
        r = requests.post(URL, json={"utterances": utterances}, timeout=30)
    except requests.exceptions.ConnectionError as e:
        print(f"连接失败，请先启动服务: uvicorn app:app --host 0.0.0.0 --port 8001\n{e}")
        sys.exit(1)

    print(f"状态码: {r.status_code}")
    if r.status_code != 200:
        print(r.text)
        sys.exit(1)

    result = r.json()
    summary = result.get("summary", {})
    participants = result.get("participants", [])
    group = result.get("group", {})

    print(f"\n--- 汇总 ---")
    print(f"总发言条数: {summary.get('total_utterances')}, 参与人数: {summary.get('total_participants')}")
    print(f"小组总标注数: {group.get('total_labeled')}, 小组熵(归一化): {group.get('entropy_normalized')}")

    print(f"\n--- 各参与者 ---")
    for p in participants:
        s = p.get("surface", {})
        c = p.get("cognitive", {})
        e = p.get("entropy", {})
        print(f"  {p.get('speaker')} ({p.get('student_id')}): "
              f"发言{s['utterance_count']}次 频率{s['frequency_ratio']} 时长{s['total_duration_sec']}s "
              f"BE={c.get('BE')} CDI={c.get('CDI')} H*={e.get('H_normalized')}")

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

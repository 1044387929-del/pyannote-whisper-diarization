"""
评价与建议提示词：输入为 POST /metrics 返回的指标 JSON（见 metrics_output_schema.json），
输出小组整体评价与每位参与者的评价、建议。字段含义以 schema 与业务数据为准。
"""
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# 与 metrics 输出结构一致的输入格式说明（供模型理解各字段含义）
INPUT_STRUCTURE_DESC = """
- summary：讨论整体汇总
  - total_utterances：整场讨论的发言总条数 T
  - total_participants：参与人数 N（按 speaker + student_id 去重）
- participants：每位参与者的指标数组
  - speaker：说话人显示名（如旁白 aside、角色名）
  - student_id：学号或角色标识
  - surface（表层参与）：utterance_count 发言次数，frequency_ratio 发言频率比例，total_duration_sec 发言总时长（秒），avg_duration_sec 平均发言时长，total_chars 发言总字符数，speech_rate_chars_per_sec 语速（字/秒）
  - cognitive（认知指标，基于 T-SEDA 编码）：labeled_count 被标注的发言数，BE 基础表达比例，CDI 认知深度指数，IDI 探究驱动指数，MDI 元认知指数，KCI 知识联系指数，CCI 协作建构指数，label_counts 各类编码条数（E/B/IB/CH/IRE/R/CA/C/RD/G）
  - entropy：H_raw 个体熵，H_normalized 归一化熵（0～1）
- group：小组层面
  - total_utterances、total_labeled 全场标注数，label_counts 全场各类编码条数，entropy_raw、entropy_normalized 小组熵
- T-SEDA 标签含义：E 表达想法，B 补充建构，IB 邀请建构，CH 质疑，IRE 邀请推理，R 推理论证，CA 协调整合，C 知识联系，RD 反思，G 引导
"""

# 与 metrics_output.json 结构一致的输入示例（便于模型解析）；花括号已转义避免被当作模板变量
INPUT_STRUCTURE_EXAMPLE = """
{{
  "summary": {{
    "total_utterances": 69,
    "total_participants": 4
  }},
  "participants": [
    {{
      "speaker": "aside",
      "student_id": "T1",
      "surface": {{
        "utterance_count": 25,
        "frequency_ratio": 0.3623,
        "total_duration_sec": 124.17,
        "avg_duration_sec": 4.97,
        "total_chars": 375,
        "speech_rate_chars_per_sec": 3.02
      }},
      "cognitive": {{
        "labeled_count": 10,
        "BE": 0.1,
        "CDI": 0.4,
        "IDI": 0.0,
        "MDI": 0.0,
        "KCI": 0.3,
        "CCI": 0.0,
        "label_counts": {{ "R": 4, "C": 3, "G": 2, "E": 1, "CH": 0, "B": 0, "CA": 0, "IRE": 0, "IB": 0, "RD": 0 }}
      }},
      "entropy": {{ "H_raw": 1.2799, "H_normalized": 0.5558 }}
    }},
    {{
      "speaker": "peppa",
      "student_id": "T2",
      "surface": {{ "utterance_count": 22, "frequency_ratio": 0.3188, "total_duration_sec": 55.82, "avg_duration_sec": 2.54, "total_chars": 205, "speech_rate_chars_per_sec": 3.67 }},
      "cognitive": {{ "labeled_count": 9, "BE": 0.3333, "CDI": 0.1111, "IDI": 0.2222, "MDI": 0.0, "KCI": 0.0, "CCI": 0.2222, "label_counts": {{ "E": 3, "CA": 1, "G": 3, "IB": 1, "IRE": 1, "R": 0, "CH": 0, "B": 0, "RD": 0, "C": 0 }} }},
      "entropy": {{ "H_raw": 1.4648, "H_normalized": 0.6362 }}
    }}
  ],
  "group": {{
    "total_utterances": 69,
    "total_labeled": 36,
    "label_counts": {{ "R": 8, "C": 5, "G": 6, "E": 7, "CA": 2, "IB": 2, "IRE": 2, "RD": 1, "B": 1, "CH": 2 }},
    "entropy_raw": 2.0669,
    "entropy_normalized": 0.8976
  }}
}}
"""

# 输出结构：与 participants 一一对应，用 speaker、student_id 标识；花括号已转义避免被当作模板变量
OUTPUT_STRUCTURE_EXAMPLE = """
{{
  "group_evaluation": "<小组整体评价>结合 summary、group 的发言数、参与人数、标注数、各类编码分布与小组熵，给出小组整体表现评价（等级或类型、协作与认知特点）。</小组整体评价>",
  "participants_evaluation": [
    {{
      "speaker": "aside",
      "student_id": "T1",
      "evaluation": "<该参与者评价>结合 surface、cognitive、entropy 简要解读并给出等级或类型判断。</该参与者评价>",
      "suggestion": "<该参与者建议>针对其指标特点给出可操作的改进建议。</该参与者建议>"
    }},
    {{
      "speaker": "peppa",
      "student_id": "T2",
      "evaluation": "<该参与者评价>...</该参与者评价>",
      "suggestion": "<该参与者建议>...</该参与者建议>"
    }}
  ]
}}
"""

system_template = """
# 角色定义
你是一个课堂讨论表现评价专家，专门基于 T-SEDA 编码与多维指标对参与者或小组的讨论表现进行诊断与评价。

# 输入内容
你会收到一段 JSON，为「讨论指标」接口（POST /metrics）的返回结果，包含：
- summary：整场发言总数、参与人数
- participants：每位参与者的表层参与（发言次数、时长、语速等）、认知指标（BE/CDI/IDI/MDI/KCI/CCI、label_counts）、熵
- group：小组总标注数、各类 T-SEDA 编码条数、小组熵

## 字段含义说明（与 metrics_output_schema.json 一致）
""" + INPUT_STRUCTURE_DESC + """

## 输入格式示例（结构需按此理解，具体数值以用户输入为准）
<input_structure_example>
""" + INPUT_STRUCTURE_EXAMPLE + """
</input_structure_example>

# 背景知识
请严格依据以下参考文档进行评价。参考文档包含：字段含义说明、评价法则、T-SEDA 标签定义。
若调用方未提供参考文档，则仅依据本提示中的「字段含义说明」与 T-SEDA 标签含义进行评价，不得编造文档中未出现的内容。

## 知识库信息
{context}

# 工作流
请你对「待评价内容」中的 summary、group 与每位 participant 进行解读与评价。输出需包含：
1）小组整体评价：结合发言数、参与人数、标注数、编码分布与小组熵，给出小组类型或等级判断；
2）每位参与者评价与建议：与 participants 数组一一对应，用 speaker、student_id 标识，给出 evaluation（解读+等级/类型）与 suggestion（可操作建议）；
3）输出必须为合法 JSON，且仅包含 JSON，不要输出 markdown 代码块或其它前后缀。

# 输出格式
请严格按以下结构输出 JSON，participants_evaluation 与输入 participants 顺序、数量、speaker/student_id 一致。
<output_structure_example>
""" + OUTPUT_STRUCTURE_EXAMPLE + """
</output_structure_example>
"""

human_template = """
待评价内容（讨论指标 JSON）：
<content>
{input_text}
</content>
"""

REVIEW_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(human_template),
])

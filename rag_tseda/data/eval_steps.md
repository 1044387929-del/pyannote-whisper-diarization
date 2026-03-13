# 基于教学智能体的课堂小组讨论评价文案生成规范

## 1. 评价文案生成逻辑框架

本套文案生成遵循以下核心逻辑，确保评价的客观性与教育学意义：

1.  **数据驱动**：所有定性描述必须基于量化指标（如发言次数 `utterance_count`、发言频率比例 `frequency_ratio`、认知深度指数 CDI、探究驱动指数 IDI、归一化熵 `entropy.H_normalized` 等，均来自 POST /metrics 返回的 summary、participants、group）。
2.  **相对定位**：不仅看绝对值，更看该生在小组内的相对位置（高于均值/低于均值）。
3.  **三层递进**：
    *   **第一层（行为）**：判断参与状态（主导/均衡/边缘），依据 surface（utterance_count、frequency_ratio、total_duration_sec、avg_duration_sec）。
    *   **第二层（认知）**：判断思维质量（高阶推理/浅层表达/元认知），依据 cognitive（CDI、IDI、BE、CCI、MDI、KCI 等）。
    *   **第三层（结构）**：判断角色多样性与协作复杂度（多功能/单一角色），依据 entropy（H_raw、H_normalized）。
4.  **建设性反馈**：每个评价后必须跟随具体的改进建议（Suggestion）。

---

## 2. 小组整体诊断文案模板

**适用场景**：教师端首页概览、讨论详情页顶部总结。  
**输入数据**：来自 `group` 与 `summary`。小组层面有 `total_utterances`、`total_labeled`、`label_counts`（各类 T-SEDA 编码条数）、`entropy_raw`、`entropy_normalized`。小组的认知维度（如“高 CDI/高 IDI”效果）可由 `group.label_counts` 与 `group.total_labeled` 推算：CDI=(R+CH+B+CA)/N，IDI=(IRE+IB)/N，BE=E/N，CCI=(B+CA+IB)/N 等。

### 2.1 总体定性评价（基于核心指标组合）

| 小组类型 | 触发条件 | 评价文案示例 |
| :--- | :--- | :--- |
| **探究深化型** | 小组 CDI 高 且 IDI 高（由 group.label_counts 与 total_labeled 推算） | “本小组展现出卓越的**探究深化**特征。成员间不仅频繁发起高质量的质疑与挑战（高 IDI），更能进行深度的逻辑推演与观点建构（高 CDI）。讨论未停留在表面信息的交换，而是进入了实质性的知识共创阶段，思维碰撞激烈且富有成效。” |
| **协作共建型** | 小组 CCI 高 且 group.entropy_normalized 高 | “本小组呈现出理想的**协作共建**生态。成员间互动紧密，善于在他人观点基础上进行补充与协调（高 CCI），且角色分工灵活多样（高熵值），无明显的话语垄断现象。讨论结构均衡，每位成员都能找到适合自己的贡献方式，团队凝聚力强。” |
| **浅层表达型** | 小组 BE 过高 且 CDI 低（由 label_counts/total_labeled 推算） | “本小组讨论目前主要停留在**浅层表达**阶段。虽然发言频次尚可（参考 summary.total_utterances），但内容多以简单陈述、事实罗列为主（高 BE），缺乏深度的推理、质疑与证据支撑（低 CDI）。讨论氛围较为平和，但思维挑战性不足，需警惕‘伪讨论’现象。” |
| **结构失衡型** | group.entropy_normalized 低 且 各参与者 surface.utterance_count 方差大 | “本小组讨论结构呈现明显的**失衡**状态。话语权高度集中在个别成员手中，互动模式单一（低熵值），部分成员处于‘沉默’或‘被动跟随’状态。这种固化的角色分工限制了观点的多元性，可能阻碍了深层知识的生成。” |

### 2.2 改进建议（针对小组）

*   **针对浅层表达**：“建议引入‘强制质疑’或‘证据链’规则，要求每次发言必须包含‘因为……所以……’或‘我反对……理由是……’的结构，推动对话向深层认知迁移。”
*   **针对结构失衡**：“建议实施‘轮流主持制’或‘角色分配卡’（如记录员、质疑者、总结者），强制轮换发言权，激活沉默成员的参与度，提升互动的多样性。”

---

## 3. 个人画像诊断文案模板

**适用场景**：学生端个人中心、讨论详情页个人卡片、教师端个体诊断。  
**输入数据**：每位参与者的 `surface`（utterance_count、frequency_ratio、total_duration_sec、avg_duration_sec）、`cognitive`（CDI、IDI、MDI、CCI、BE、KCI、label_counts）、`entropy`（H_raw、H_normalized），与小组均值或 group 层面指标对比。

### 3.1 角色定位与行为评价（第一层：表层参与）

| 角色类型 | 触发条件 | 评价文案示例 |
| :--- | :--- | :--- |
| **主导型参与者** | surface.utterance_count 或 frequency_ratio 高于小组均值 且 total_duration_sec 长 | “你是本次讨论的**核心推动者**。你的发言频次与时长均显著高于小组平均水平，承担了主要的信息输出与流程引导任务。你的存在保证了讨论的连续性，是小组不可或缺的‘引擎’。” |
| **高频回应者** | surface.utterance_count 高 但 avg_duration_sec 短 | “你是小组活跃的**高频回应者**。你反应敏捷，频繁介入对话，但单次发言时长较短。这种‘短平快’的互动风格有助于维持讨论热度，但可能在深度阐述上略显不足。” |
| **边缘/沉默参与者** | surface.utterance_count 或 frequency_ratio 低于小组均值约 50% | “你在本次讨论中表现为**谨慎的倾听者**。你的发言频次明显低于小组平均水平，更多时候在吸收他人观点。虽然倾听是学习的重要部分，但过度的沉默可能让你错失表达独特见解的机会。” |

### 3.2 认知质量深度剖析（第二层：认知互动）

| 角色类型 | 触发条件 | 评价文案示例 |
| :--- | :--- | :--- |
| **高阶思维者** | cognitive.CDI 高 + cognitive.IDI 高 | “你的思维具有极高的**深度与批判性**。你不仅善于提出引发深思的问题（高 IDI），更能运用严密的逻辑进行推理论证（高 CDI）。你的发言往往是讨论转折点的关键，有效提升了全组的认知水位。” |
| **协作整合者** | cognitive.CCI 高 + cognitive.CDI 中等 | “你是优秀的**协作整合者**。你擅长捕捉他人的观点并进行补充、完善或协调（高 CCI），是小组共识达成的‘粘合剂’。虽然独立提出的颠覆性观点不多，但你对团队知识建构的贡献巨大。” |
| **反思调适者** | cognitive.MDI 高 | “你展现了出色的**元认知监控能力**。你能够跳出讨论内容本身，对讨论进程、策略及团队协作状态进行反思与调整（高 MDI）。这种‘关于思考的思考’对于优化小组合作效率至关重要。” |
| **浅层表达者** | cognitive.BE 高 + cognitive.CDI 低 | “你的发言多以**基础表达**为主。虽然参与度不错，但内容多集中于事实陈述或简单同意，缺乏深入的推理与质疑。建议尝试从‘是什么’转向‘为什么’和‘怎么做’，提升思维的颗粒度。” |

### 3.3 角色多样性与结构诊断（第三层：结构复杂性）

| 角色类型 | 触发条件 | 评价文案示例 |
| :--- | :--- | :--- |
| **多功能认知参与者** | entropy.H_normalized 高 + cognitive.label_counts 中高阶编码多 | “你的讨论角色**丰富多变**。你既能发起挑战，又能进行推理，还能协调共识，在不同情境下灵活切换身份。这种高多样性的参与模式表明你具备全面的协作素养。” |
| **功能单一者** | entropy.H_normalized 低 | “你的参与模式相对**固定**。你倾向于使用某一种特定的对话策略（如仅做记录或仅做附和）。尝试突破舒适区，练习使用不同类型的对话行为（如尝试一次质疑或一次总结），将使你的协作能力更加全面。” |

### 3.4 个性化改进建议（Actionable Suggestions）

*   **给主导者**：“尝试‘留白’艺术。在发表完观点后，刻意停顿，主动邀请沉默的组员发言（使用‘邀请’类编码 IB/IRE），将话语权分享出去。”
*   **给沉默者**：“设定一个小目标：在下一次讨论中至少提出一个‘为什么’或补充一个‘具体例子’。不必追求长篇大论，简短有力的观点同样有价值。”
*   **给浅层表达者**：“使用‘因为……所以……’句式强迫自己进行因果推理；或在同意他人前，先尝试找一个反例进行‘温和挑战’。”
*   **给单一角色者**：“参考 T-SEDA 编码表，刻意练习一种你从未用过的对话行为（如‘联系外部知识’或‘反思讨论过程’）。”

---

## 4. 系统提示词（Prompt）配置示例

> **注**：此部分供开发人员参考，用于配置 LLM 生成上述文案。输入结构以 POST /metrics 返回为准（summary、participants、group）。

```markdown
# Role
你是一位基于 T-SEDA 框架的课堂讨论分析专家。你的任务是根据输入的 JSON 数据（即 POST /metrics 返回的 summary、participants、group），生成专业、客观且具有建设性的评价文案。

# Constraints
1. 严禁编造数据，所有评价必须基于输入指标与小组均值的相对关系。
2. 语气需温暖、鼓励且具有指导性，避免使用指责性语言。
3. 必须涵盖“行为参与”、“认知质量”、“结构角色”三个维度。
4. 建议必须具体可操作，关联到 T-SEDA 的具体编码行为（如“邀请”、“质疑”、“推理”）。

# Input Data Structure（与 metrics_output_schema 一致）
{
  "summary": { "total_utterances": 69, "total_participants": 4 },
  "participants": [
    {
      "speaker": "aside",
      "student_id": "T1",
      "surface": {
        "utterance_count": 25,
        "frequency_ratio": 0.36,
        "total_duration_sec": 124.17,
        "avg_duration_sec": 4.97,
        "total_chars": 375,
        "speech_rate_chars_per_sec": 3.02
      },
      "cognitive": {
        "labeled_count": 10,
        "BE": 0.1,
        "CDI": 0.4,
        "IDI": 0.0,
        "MDI": 0.0,
        "KCI": 0.3,
        "CCI": 0.0,
        "label_counts": { "R": 4, "C": 3, "G": 2, "E": 1, ... }
      },
      "entropy": { "H_raw": 1.28, "H_normalized": 0.56 }
    }
  ],
  "group": {
    "total_utterances": 69,
    "total_labeled": 36,
    "label_counts": { "R": 8, "C": 5, "G": 6, "E": 7, ... },
    "entropy_raw": 2.07,
    "entropy_normalized": 0.90
  }
}

# Generation Steps
1. **判定小组类型**：根据 group.label_counts、group.total_labeled、group.entropy_normalized 推算小组 CDI/IDI/BE 等，判定小组是“探究型”、“协作型”还是“浅层型”。
2. **判定个人角色**：
   - 行为层：根据 surface.utterance_count、frequency_ratio、total_duration_sec 判定是“主导”、“均衡”还是“边缘”。
   - 认知层：根据 cognitive.CDI、IDI、CCI、MDI、BE 判定思维特征（如“高阶思维”、“协作整合”）。
   - 结构层：根据 entropy.H_normalized 判定角色多样性。
3. **生成评价**：组合上述判定，形成流畅的自然语言段落。
4. **生成建议**：针对短板提出一条具体的 T-SEDA 行为改进建议。

# Output Format (JSON)
{
  "group_evaluation": "小组整体评价文本...",
  "participants_evaluation": [
    {
      "speaker": "aside",
      "student_id": "T1",
      "evaluation": "该参与者评价文本...",
      "suggestion": "该参与者建议文本..."
    }
  ]
}
```

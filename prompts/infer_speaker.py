"""推断说话人提示词：根据上下文推断未标注句子的说话人归属（Few-shot 为 HumanMessage/AIMessage）。"""
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    FewShotChatMessagePromptTemplate,
)

# 推断说话人系统提示词
INFER_SPEAKER_SYSTEM = """
# 角色定义
你是一个对话转写与说话人标注助手，专门根据对话上下文推断未标注句子的说话人归属。

# 任务要求
1. 仅从「允许的说话人」列表中选择一个标识，不得编造新 ID。
2. 不得返回 unknown：必须为当前句指定一个说话人归属；难以判断时选上下文最可能的那一个。
3. 只输出一个说话人标识，不要解释、不要换行。

# 注意
优先依据称呼、人称（我们/我）和话题连续性；画外音/旁白选 aside 或 narrator。禁止输出 unknown。
"""

# 推断说话人人类提示词
INFER_SPEAKER_HUMAN = """允许的说话人（只能从下列中选择一个回复）：
{allowed_speakers}

上下文（上文为已标注的发言，下文为未标注的）：
{context}

[当前句] {current_text}

说话人："""

# 推断说话人示例
INFER_SPEAKER_FEW_SHOT_EXAMPLES = [
    # 四个字段：allowed_speakers, context, current_text, speaker
    # allowed_speakers: 允许的说话人列表
    # context: 上下文
    # current_text: 当前句
    # speaker: 说话人
    {
        "allowed_speakers": "peppa, aside, peppa_mom, peppa_dad",
        "context": "peppa：我是佩奇。\npeppa：这是我的弟弟乔治。\n[当前句] 这是我的妈妈。\npeppa：这是我的爸爸。",
        "current_text": "这是我的妈妈。",
        "speaker": "peppa",
    },
    {
        "allowed_speakers": "peppa, aside, peppa_mom, peppa_dad",
        "context": "aside：猪爸爸戴上眼镜之后，一切看得很清楚；但是当猪爸爸摘掉眼镜之后，就什么都看不清了。\naside：所以对猪爸爸来说，知道自己的眼镜在哪儿是很重要的。\n[当前句] 有的时候，猪爸爸会找不到他的眼镜。\npeppa_mom：佩奇、乔治。",
        "current_text": "有的时候，猪爸爸会找不到他的眼镜。",
        "speaker": "aside",
    },
    {
        "allowed_speakers": "peppa, aside, peppa_mom, peppa_dad",
        "context": "peppa：爸爸的眼镜不见了！\n[当前句] 哈哈哈！\naside：有的时候，猪爸爸会找不到他的眼镜。",
        "current_text": "哈哈哈！",
        "speaker": "peppa",
    },
    {
        "allowed_speakers": "peppa, aside, peppa_mom, peppa_dad",
        "context": "peppa_mom：佩奇、乔治。\n[当前句] 你们见到过爸爸的眼镜吗？\npeppa_mom：他到处找，都找不到。",
        "current_text": "你们见到过爸爸的眼镜吗？",
        "speaker": "peppa_mom",
    },
]

# 推断说话人示例提示词
_infer_speaker_example_prompt = ChatPromptTemplate.from_messages([
    ("human", INFER_SPEAKER_HUMAN),
    ("ai", "{speaker}"),
])

# 推断说话人示例提示词
infer_speaker_few_shot = FewShotChatMessagePromptTemplate(
    example_prompt=_infer_speaker_example_prompt,
    examples=INFER_SPEAKER_FEW_SHOT_EXAMPLES,
)

# 推断说话人提示词
infer_speaker_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(INFER_SPEAKER_SYSTEM),
    infer_speaker_few_shot,
    HumanMessagePromptTemplate.from_template(INFER_SPEAKER_HUMAN),
])

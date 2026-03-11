"""纠错与标点提示词：ASR 转写纠错与标点补全（Few-shot 为 HumanMessage/AIMessage）。"""
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    FewShotChatMessagePromptTemplate,
)

CORRECT_TEXT_SYSTEM = """
# 角色定义
你是一个中文转写文本纠错与标点补全助手，专门处理 ASR 产出的原始文本。

# 任务要求
1. 纠错：修正错别字、同音词误识别（如眼睛/眼镜），明显漏字，不改变原意。
2. 标点：补全句末标点（。？！等）和句中必要标点，使断句合理。
3. 只输出纠正后的单句文本，不要解释、不要分段。
4. 专有名词尽量保持原样；同音词结合语境修正。
"""

CORRECT_TEXT_HUMAN = """请对以下 ASR 转写句子进行纠错与标点补全，只输出纠正后的一句话：

{text}"""

CORRECT_TEXT_FEW_SHOT_EXAMPLES = [
    {"text": "所以对猪爸爸来说知道自己的眼镜在哪儿是很重要的", "corrected": "所以对猪爸爸来说，知道自己的眼镜在哪儿是很重要的。"},
    {"text": "有的时候猪爸爸会找不到他的眼睛", "corrected": "有的时候，猪爸爸会找不到他的眼镜。"},
    {"text": "佩奇和乔治不知道爸爸的眼睛在哪里", "corrected": "佩奇和乔治不知道爸爸的眼镜在哪里。"},
    {"text": "你们见到过爸爸的眼镜吗?", "corrected": "你们见到过爸爸的眼镜吗？"},
    {"text": "他到处找都找不到。", "corrected": "他到处找，都找不到。"},
    {"text": "噢 糟了", "corrected": "噢，糟了！"},
]

_correct_text_example_prompt = ChatPromptTemplate.from_messages([
    ("human", CORRECT_TEXT_HUMAN),
    ("ai", "{corrected}"),
])
correct_text_few_shot = FewShotChatMessagePromptTemplate(
    example_prompt=_correct_text_example_prompt,
    examples=CORRECT_TEXT_FEW_SHOT_EXAMPLES,
)

correct_text_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(CORRECT_TEXT_SYSTEM),
    correct_text_few_shot,
    HumanMessagePromptTemplate.from_template(CORRECT_TEXT_HUMAN),
])

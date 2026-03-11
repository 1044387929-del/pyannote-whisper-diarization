"""流水线用 LangChain 提示词：推断说话人、纠错与标点"""
from langchain_core.prompts import ChatPromptTemplate

# 推断未知说话人：给一段上下文 + 当前句，只返回说话人称呼
INFER_SPEAKER_SYSTEM = """根据对话上下文，判断标记为 [当前句] 的发言最可能来自谁。已知说话人可能包括：佩奇、乔治、猪妈妈、猪爸爸、旁白、画外音等。只回复一个称呼，无法推断时回复 unknown。不要解释。"""

INFER_SPEAKER_HUMAN = """上下文：
{context}

[当前句] {current_text}

说话人："""

infer_speaker_prompt = ChatPromptTemplate.from_messages([
    ("system", INFER_SPEAKER_SYSTEM),
    ("human", INFER_SPEAKER_HUMAN),
])

# 纠错与标点：输入一句 ASR 文本，只输出纠正后的句子
CORRECT_TEXT_SYSTEM = """你是转写文本纠错助手。输入为一句 ASR 转写（可能有错别字、同音词、缺标点）。请只做纠错并补全中文标点，保持原意。只输出纠正后的一句话，不要其他内容。"""

CORRECT_TEXT_HUMAN = """{text}"""

correct_text_prompt = ChatPromptTemplate.from_messages([
    ("system", CORRECT_TEXT_SYSTEM),
    ("human", CORRECT_TEXT_HUMAN),
])

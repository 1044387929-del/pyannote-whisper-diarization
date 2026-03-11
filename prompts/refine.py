"""流水线用提示词统一入口：推断说话人、纠错与标点（实现拆到 infer_speaker / correct_text）。"""
from .infer_speaker import infer_speaker_prompt
from .correct_text import correct_text_prompt

__all__ = ["infer_speaker_prompt", "correct_text_prompt"]

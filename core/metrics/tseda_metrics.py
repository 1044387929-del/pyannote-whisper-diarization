"""
基于 T-SEDA 的课堂讨论指标计算。

指标定义见参考文档：表层参与（发言频率、时长、语速）、认知指标（BE、CDI、IDI、MDI、KCI、CCI）、信息熵。
"""
import math
from collections import defaultdict
from typing import Any

# T-SEDA 编码集合（不含 NULL）
TSEDA_LABELS = ("E", "B", "IB", "CH", "IRE", "R", "CA", "C", "RD", "G")


def _participant_key(u: dict) -> tuple:
    """发言归属的参与者键：(speaker, student_id)。"""
    return (u.get("speaker") or "", u.get("student_id") or "")


def _duration(u: dict) -> float:
    """单条发言时长（秒）。"""
    return max(0.0, float(u.get("end", 0) or 0) - float(u.get("start", 0) or 0))


def _char_count(u: dict) -> int:
    """单条发言字符数。"""
    return len((u.get("text") or "").strip())


def _label(u: dict) -> str | None:
    """标准化标签：在 TSEDA_LABELS 内则返回，否则返回 None（不参与 N_i）。"""
    lab = (u.get("label") or "").strip().upper()
    if lab in TSEDA_LABELS:
        return lab
    return None


def compute_tseda_metrics(utterances: list[dict]) -> dict[str, Any]:
    """
    根据带 T-SEDA 标记的 utterances 计算指标。

    输入每条需含：start, end, speaker, student_id, text, label（可选）。
    返回结构：
      - summary: { total_utterances, total_participants }
      - participants: [ { id, speaker, student_id, surface, cognitive, entropy }, ... ]
      - group: { total_utterances, total_labeled, entropy_raw, entropy_normalized }
    """
    if not utterances:
        return {
            "summary": {"total_utterances": 0, "total_participants": 0},
            "participants": [],
            "group": {"total_utterances": 0, "total_labeled": 0, "entropy_raw": 0.0, "entropy_normalized": 0.0},
        }

    # 按参与者聚合
    by_participant: dict[tuple, list[dict]] = defaultdict(list)
    for u in utterances:
        by_participant[_participant_key(u)].append(u)

    total_T = len(utterances)
    participants_out = []
    group_label_counts = defaultdict(int)
    group_N = 0

    for (speaker, student_id), u_list in by_participant.items():
        T_i = len(u_list)
        D_i = sum(_duration(u) for u in u_list)
        W_i = sum(_char_count(u) for u in u_list)

        # 表层参与
        F_i = T_i / total_T if total_T else 0.0
        d_bar_i = D_i / T_i if T_i else 0.0
        SR_i = W_i / D_i if D_i else 0.0

        surface = {
            "utterance_count": T_i,
            "frequency_ratio": round(F_i, 4),
            "total_duration_sec": round(D_i, 2),
            "avg_duration_sec": round(d_bar_i, 2),
            "total_chars": W_i,
            "speech_rate_chars_per_sec": round(SR_i, 2),
        }

        # 认知：仅统计 label 在 TSEDA_LABELS 的条数
        n_k = defaultdict(int)
        for u in u_list:
            lab = _label(u)
            if lab:
                n_k[lab] += 1
        N_i = sum(n_k.values())
        for k, v in n_k.items():
            group_label_counts[k] += v
        group_N += N_i

        if N_i == 0:
            BE_i = CDI_i = IDI_i = MDI_i = KCI_i = CCI_i = 0.0
            H_i = 0.0
            H_i_star = 0.0
        else:
            BE_i = n_k["E"] / N_i
            CDI_i = (n_k["R"] + n_k["CH"] + n_k["B"] + n_k["CA"]) / N_i
            IDI_i = (n_k["IRE"] + n_k["IB"]) / N_i
            MDI_i = n_k["RD"] / N_i
            KCI_i = n_k["C"] / N_i
            CCI_i = (n_k["B"] + n_k["CA"] + n_k["IB"]) / N_i

            probs = [n_k[l] / N_i for l in TSEDA_LABELS]
            H_i = -sum(p * math.log(p) for p in probs if p > 0)
            H_i_star = H_i / math.log(10) if math.log(10) else 0.0

        cognitive = {
            "labeled_count": N_i,
            "BE": round(BE_i, 4),
            "CDI": round(CDI_i, 4),
            "IDI": round(IDI_i, 4),
            "MDI": round(MDI_i, 4),
            "KCI": round(KCI_i, 4),
            "CCI": round(CCI_i, 4),
            "label_counts": dict(n_k),
        }
        entropy = {"H_raw": round(H_i, 4), "H_normalized": round(H_i_star, 4)}

        participants_out.append({
            "speaker": speaker,
            "student_id": student_id,
            "surface": surface,
            "cognitive": cognitive,
            "entropy": entropy,
        })

    # 小组熵
    if group_N == 0:
        H_G = 0.0
        H_G_star = 0.0
    else:
        probs_G = [group_label_counts[l] / group_N for l in TSEDA_LABELS]
        H_G = -sum(p * math.log(p) for p in probs_G if p > 0)
        H_G_star = H_G / math.log(10) if math.log(10) else 0.0

    return {
        "summary": {
            "total_utterances": total_T,
            "total_participants": len(participants_out),
        },
        "participants": participants_out,
        "group": {
            "total_utterances": total_T,
            "total_labeled": group_N,
            "label_counts": dict(group_label_counts),
            "entropy_raw": round(H_G, 4),
            "entropy_normalized": round(H_G_star, 4),
        },
    }

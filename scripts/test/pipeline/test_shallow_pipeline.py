"""
浅度流水线端到端测试：转录 → 精修 → 标注 → 指标 → 评价；每环节写出 JSON 供查看。

面向对象实现：PipelineContext 承载状态，PipelineStep 子类实现各环节，ShallowPipeline 顺序执行。
配置：同目录下 shallow_pipeline_config.yaml（路径相对 scripts/test/data）。
用法（需先启动服务: uvicorn app:app --host 0.0.0.0 --port 8001）:
  python scripts/test/pipeline/test_shallow_pipeline.py
"""
import json
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import requests

# 项目根（scripts/test/pipeline/ -> 上溯 4 层）
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

# 测试数据目录：scripts/test/data（配置中的路径相对此目录解析）
TEST_DATA = Path(__file__).resolve().parent.parent / "data"
CONFIG_PATH = Path(__file__).resolve().parent / "shallow_pipeline_config.yaml"


# ---------------------------------------------------------------------------
# 配置与路径
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        raise RuntimeError("请安装 PyYAML: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def resolve_path(p: Any, base: Path | None = None) -> Path | None:
    if base is None:
        base = TEST_DATA
    if p is None:
        return None
    path = Path(p)
    return path if path.is_absolute() else (base / path).resolve()


def load_speakers(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = data.get("speakers", data.get("utterances", []))
    return data


# ---------------------------------------------------------------------------
# API 调用（保持原有函数，供 Step 调用）
# ---------------------------------------------------------------------------

def transcribe(
    audio_path: Path, speakers_path: Path, language: str, base_url: str, stream: bool = False
) -> list:
    speakers = load_speakers(speakers_path)
    data = [("language", language)]
    for s in speakers:
        data.append(("student_id", s.get("student_id", "")))
        data.append(("name", s.get("name", "")))
        data.append(("embedding", json.dumps(s["embedding"], ensure_ascii=False)))
    if not stream:
        url = f"{base_url.rstrip('/')}/transcriptions"
        with open(audio_path, "rb") as f:
            r = requests.post(url, data=data, files={"audio": (audio_path.name, f, "audio/wav")}, timeout=600)
        r.raise_for_status()
        return r.json().get("utterances", [])
    data.append(("stream", "true"))
    url = f"{base_url.rstrip('/')}/transcriptions"
    with open(audio_path, "rb") as f:
        r = requests.post(
            url, data=data, files={"audio": (audio_path.name, f, "audio/wav")}, timeout=600, stream=True
        )
    r.raise_for_status()
    out_utterances = []
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        try:
            obj = json.loads(line[6:].strip())
        except json.JSONDecodeError:
            continue
        if obj.get("status") == "done":
            continue
        if obj.get("type") == "raw":
            continue
        if obj.get("type") == "refined":
            out_utterances.append({k: v for k, v in obj.items() if k != "type"})
            idx = len(out_utterances)
            speaker = obj.get("speaker", "") or "?"
            text = (obj.get("text", "") or "").strip()
            snippet = (text[:50] + "…") if len(text) > 50 else text or "(空)"
            print(f"      [转录] 第 {idx} 条: {speaker}: {snippet}")
            continue
        out_utterances.append(obj)
        idx = obj.get("index", len(out_utterances))
        total = obj.get("total") or 0
        progress = obj.get("progress", 0)
        speaker = obj.get("speaker", "") or "?"
        text = (obj.get("text", "") or "").strip()
        snippet = (text[:50] + "…") if len(text) > 50 else text or "(空)"
        pct = f" ({progress:.0f}%)" if total else ""
        print(f"      [转录] 第 {idx} 条{pct}: {speaker}: {snippet}")
    return out_utterances


def refine(utterances: list, stream: bool, allowed_speakers: list | None, base_url: str) -> list:
    url = f"{base_url.rstrip('/')}/refinements"
    body = {
        "utterances": utterances,
        "infer_speakers": True,
        "merge": True,
        "correct_text": True,
        "context_size": 5,
        "stream": stream,
    }
    if allowed_speakers:
        body["allowed_speakers"] = allowed_speakers
    if not stream:
        r = requests.post(url, json=body, timeout=600)
        r.raise_for_status()
        return r.json().get("utterances", [])
    r = requests.post(url, json=body, timeout=600, stream=True)
    r.raise_for_status()
    out_utterances = []
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        try:
            obj = json.loads(line[6:].strip())
        except json.JSONDecodeError:
            continue
        if obj.get("stage") == "error":
            raise RuntimeError(obj.get("detail", "refine error"))
        stage = obj.get("stage")
        progress = obj.get("progress")
        if stage == "utterance":
            u = obj.get("utterance")
            idx = obj.get("index", len(out_utterances))
            progress = obj.get("progress")
            if u is not None:
                out_utterances.append(u)
                speaker = u.get("speaker", "") or "?"
                text = (u.get("text", "") or "").strip()
                snippet = (text[:50] + "…") if len(text) > 50 else text or "(空)"
                pct = f" ({progress:.0%})" if progress is not None else ""
                print(f"      [精修] 第 {idx + 1} 条{pct}: {speaker}: {snippet}")
        if stage == "done" and "utterances" in obj:
            out_utterances = obj["utterances"]
            break
    return out_utterances


def label(utterances: list, stream: bool, max_items: int | None, base_url: str) -> list:
    url = f"{base_url.rstrip('/')}/labels"
    payload = [{"text": u.get("text", ""), "speaker": u.get("speaker", "")} for u in utterances]
    if max_items is not None:
        payload = payload[:max_items]
    total = len(payload)
    body = {"utterances": payload, "context_window": 3, "stream": stream}
    if not stream:
        r = requests.post(url, json=body, timeout=300)
        r.raise_for_status()
        return r.json().get("utterances", [])
    r = requests.post(url, json=body, timeout=300, stream=True)
    r.raise_for_status()
    result_list = []
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        try:
            obj = json.loads(line[6:].strip())
        except json.JSONDecodeError:
            continue
        if obj.get("done"):
            break
        if "error" in obj:
            raise RuntimeError(obj["error"])
        if "utterance" in obj:
            u = obj["utterance"]
            result_list.append(u)
            idx = obj.get("index", len(result_list) - 1)
            current = idx + 1
            pct = int(100 * current / total) if total else 0
            lab = u.get("label", "")
            speaker = u.get("speaker", "") or "?"
            text = (u.get("text") or "").strip()
            snippet = (text[:50] + "…") if len(text) > 50 else text or "(空)"
            print(f"      [标注] {current}/{total} ({pct}%) {speaker}: {snippet} -> {lab}")
    return result_list


def metrics(utterances: list, base_url: str) -> dict:
    url = f"{base_url.rstrip('/')}/metrics"
    r = requests.post(url, json={"utterances": utterances}, timeout=30)
    r.raise_for_status()
    return r.json()


def evaluation(metrics_result: dict, base_url: str) -> dict:
    url = f"{base_url.rstrip('/')}/evaluation"
    r = requests.post(url, json={"metrics": metrics_result}, timeout=120)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# 流水线上下文与步骤抽象
# ---------------------------------------------------------------------------

class PipelineContext:
    """流水线运行时的配置与中间结果，各 Step 读写此对象。"""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.base_url = cfg.get("base_url") or "http://127.0.0.1:8001"
        self.stream_transcribe = bool(cfg.get("stream_transcribe", False))
        self.stream_refine = bool(cfg.get("stream_refine", False))
        self.stream_labels = bool(cfg.get("stream_labels", False))
        self.max_labels = cfg.get("max_labels")
        self.language = cfg.get("language") or "zh"
        self.from_json = cfg.get("from_json")
        self.audio = cfg.get("audio")
        speakers = cfg.get("speakers") or "json/speakers_embedding.json"
        out = cfg.get("out") or "json/shallow_pipeline_output.json"

        self.speakers_path = resolve_path(speakers)
        self.audio_path = resolve_path(self.audio) if self.audio else None
        self.out_path = resolve_path(out)
        self.path_transcribe = resolve_path(cfg.get("out_transcribe"))
        self.path_refined = resolve_path(cfg.get("out_refined"))
        self.path_metrics = resolve_path(cfg.get("out_metrics"))
        self.path_eval = resolve_path(cfg.get("out_eval"))

        self.utterances: list = []
        self.refined: list = []
        self.metrics_result: dict | None = None
        self.eval_result: dict | None = None

        self.out_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def allowed_speakers(self) -> list | None:
        if not self.speakers_path or not self.speakers_path.exists():
            return None
        speakers_list = load_speakers(self.speakers_path)
        return [
            {"speaker": s.get("name") or s.get("student_id"), "student_id": s.get("student_id")}
            for s in speakers_list
        ]


class PipelineStep(ABC):
    """流水线单步抽象：从 ctx 读输入、写输出，可选写 JSON。"""

    name: str = ""

    @abstractmethod
    def run(self, ctx: PipelineContext, index: int, total: int) -> None:
        pass

    def _write_json(self, path: Path | None, data: dict, desc: str = "已保存") -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"      {desc}: {path}")


class TranscribeStep(PipelineStep):
    name = "转录"

    def run(self, ctx: PipelineContext, index: int, total: int) -> None:
        if ctx.from_json:
            path = resolve_path(ctx.from_json)
            if not path or not path.exists():
                raise FileNotFoundError(f"文件不存在: {path}")
            with open(path, "r", encoding="utf-8") as f:
                ctx.utterances = json.load(f).get("utterances", [])
            print(f"[{index}/{total}] 跳过转录，从 {path.name} 加载，共 {len(ctx.utterances)} 条")
            self._write_json(ctx.path_transcribe, {"utterances": ctx.utterances})
            return
        if not ctx.audio_path or not ctx.audio_path.exists():
            raise FileNotFoundError(f"音频不存在: {ctx.audio_path}")
        if not ctx.speakers_path or not ctx.speakers_path.exists():
            raise FileNotFoundError(f"声纹不存在: {ctx.speakers_path}")
        print(f"[{index}/{total}] 转录 (stream={ctx.stream_transcribe}): {ctx.audio_path.name} + {ctx.speakers_path.name} ...")
        ctx.utterances = transcribe(
            ctx.audio_path, ctx.speakers_path, ctx.language, ctx.base_url, stream=ctx.stream_transcribe
        )
        print(f"      转录结果: {len(ctx.utterances)} 条")
        self._write_json(ctx.path_transcribe, {"utterances": ctx.utterances})


class RefineStep(PipelineStep):
    name = "精修"

    def run(self, ctx: PipelineContext, index: int, total: int) -> None:
        print(f"[{index}/{total}] 精修 (stream={ctx.stream_refine}) ...")
        ctx.refined = refine(
            ctx.utterances, ctx.stream_refine, ctx.allowed_speakers, ctx.base_url
        )
        print(f"      精修结果: {len(ctx.refined)} 条")
        self._write_json(ctx.path_refined, {"utterances": ctx.refined})
        if not ctx.stream_refine:
            for i, u in enumerate(ctx.refined):
                speaker = u.get("speaker", "") or "?"
                text = (u.get("text", "") or "").strip()
                snippet = (text[:50] + "…") if len(text) > 50 else text or "(空)"
                print(f"      [{i}] {speaker}: {snippet}")


class LabelStep(PipelineStep):
    name = "标注"

    def run(self, ctx: PipelineContext, index: int, total: int) -> None:
        print(f"[{index}/{total}] 标注 (stream={ctx.stream_labels}, max={ctx.max_labels}) ...")
        labels_only = label(ctx.refined, ctx.stream_labels, ctx.max_labels, ctx.base_url)
        print(f"      标注结果: {len(labels_only)} 条")
        for i, lab in enumerate(labels_only):
            if i < len(ctx.refined):
                ctx.refined[i]["label"] = lab.get("label")
                ctx.refined[i]["reason"] = lab.get("reason")
        ctx.out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ctx.out_path, "w", encoding="utf-8") as f:
            json.dump({"utterances": ctx.refined}, f, ensure_ascii=False, indent=2)
        print(f"      已保存: {ctx.out_path}")


class MetricsStep(PipelineStep):
    name = "计算指标"

    def run(self, ctx: PipelineContext, index: int, total: int) -> None:
        print(f"[{index}/{total}] 计算指标 ...")
        try:
            ctx.metrics_result = metrics(ctx.refined, ctx.base_url)
            self._write_json(ctx.path_metrics, ctx.metrics_result)
            s = ctx.metrics_result.get("summary", {})
            g = ctx.metrics_result.get("group", {})
            print(f"      参与人数: {s.get('total_participants')}, 小组标注数: {g.get('total_labeled')}, 小组熵: {g.get('entropy_normalized')}")
        except requests.exceptions.RequestException as e:
            print(f"      指标请求失败（可稍后单独调用 POST /metrics）: {e}")


class EvaluationStep(PipelineStep):
    name = "评价"

    def run(self, ctx: PipelineContext, index: int, total: int) -> None:
        print(f"[{index}/{total}] 评价（小组整体 + 个人）...")
        if ctx.metrics_result is None:
            print("      跳过（无指标结果）")
            return
        try:
            ctx.eval_result = evaluation(ctx.metrics_result, ctx.base_url)
            self._write_json(ctx.path_eval, ctx.eval_result)
            ge = ctx.eval_result.get("group_evaluation", "")
            pe = ctx.eval_result.get("participants_evaluation", [])
            print(f"      小组评价: {(ge[:80] + '…') if len(ge) > 80 else ge}")
            print(f"      个人评价: {len(pe)} 人")
        except requests.exceptions.RequestException as e:
            print(f"      评价请求失败（可稍后单独调用 POST /evaluation）: {e}")


class ShallowPipeline:
    """按顺序执行若干 PipelineStep，共享同一 PipelineContext。"""

    def __init__(self, steps: list[PipelineStep]):
        self.steps = steps

    def run(self, ctx: PipelineContext) -> None:
        total = len(self.steps)
        if not total:
            return
        self.steps[0].run(ctx, 1, total)
        if not ctx.utterances:
            return
        for i, step in enumerate(self.steps[1:], start=2):
            step.run(ctx, i, total)


def build_pipeline() -> ShallowPipeline:
    """根据默认环节构建流水线，便于后续按配置裁剪步骤。"""
    return ShallowPipeline([
        TranscribeStep(),
        RefineStep(),
        LabelStep(),
        MetricsStep(),
        EvaluationStep(),
    ])


def main() -> None:
    if not CONFIG_PATH.exists():
        print(f"配置文件不存在: {CONFIG_PATH}")
        sys.exit(1)
    cfg = load_config(CONFIG_PATH)
    ctx = PipelineContext(cfg)

    if not ctx.from_json and (not ctx.audio or not ctx.speakers_path):
        print("请在 shallow_pipeline_config.yaml 中设置 from_json，或同时设置 audio 与 speakers")
        sys.exit(1)

    pipeline = build_pipeline()
    try:
        pipeline.run(ctx)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    if not ctx.utterances:
        print("无 utterance，退出")
        sys.exit(0)


if __name__ == "__main__":
    main()

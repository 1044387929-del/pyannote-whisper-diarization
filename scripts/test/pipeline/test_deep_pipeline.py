"""
深层流水线端到端测试（方案 A）：WebSocket 实时转录 → HTTP 精修 → HTTP 标注。
不做指标与评价（需等录制结束后单独跑）。
面向对象实现：DeepPipelineContext 承载状态，PipelineStep 子类实现各环节，DeepPipeline 顺序执行。
用法（需先启动服务: uvicorn app:app --host 0.0.0.0 --port 8001）:
  python scripts/test/pipeline/test_deep_pipeline.py
"""
import asyncio
import base64
import json
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import requests

# 项目根（scripts/test/pipeline/ -> 上溯 4 层）
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

TEST_DATA = Path(__file__).resolve().parent.parent / "data"
CONFIG_PATH = Path(__file__).resolve().parent / "deep_pipeline_config.yaml"

try:
    import websockets
except ImportError:
    websockets = None

try:
    import torchaudio
except ImportError:
    torchaudio = None


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
# WebSocket 实时转录：发送音频块，收集所有 transcript 的 utterances
# ---------------------------------------------------------------------------

def split_audio_into_chunks(audio_path: Path, chunk_seconds: float) -> list[bytes]:
    """将 wav 按时间切分成块。无 torchaudio 时整段作为一块。"""
    if not torchaudio:
        data = audio_path.read_bytes()
        return [data] if len(data) > 100 else []
    import tempfile
    waveform, sample_rate = torchaudio.load(str(audio_path))
    chunk_samples = int(sample_rate * chunk_seconds)
    total_samples = waveform.shape[1]
    chunks = []
    for start in range(0, total_samples, chunk_samples):
        end = min(start + chunk_samples, total_samples)
        seg = waveform[:, start:end]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            torchaudio.save(f.name, seg, sample_rate)
            chunks.append(Path(f.name).read_bytes())
    return chunks


async def live_transcribe(
    audio_path: Path,
    speakers_path: Path | None,
    language: str,
    base_url: str,
    chunk_seconds: float,
    refine_on_ws: bool,
) -> list[dict]:
    """
    通过 WebSocket /ws/transcriptions/live 发送音频块，收集所有 transcript 的 utterances。
    返回合并后的 utterance 列表。
    """
    if not websockets:
        raise RuntimeError("请安装 websockets: pip install websockets")
    chunks = split_audio_into_chunks(audio_path, chunk_seconds)
    if not chunks:
        raise RuntimeError("无法切分音频或音频过短")
    speakers = []
    if speakers_path and speakers_path.exists():
        speakers = load_speakers(speakers_path)
        for s in speakers:
            s.setdefault("embedding", [])
    ws_url = base_url.rstrip("/").replace("http://", "ws://").replace("https://", "wss://")
    ws_url = f"{ws_url}/ws/transcriptions/live"
    all_utterances: list[dict] = []
    chunk_offset = 0.0
    async with websockets.connect(ws_url) as ws:
        init_msg = {"type": "init", "language": language or "zh", "refine": refine_on_ws}
        if speakers:
            init_msg["speakers"] = speakers
        await ws.send(json.dumps(init_msg))
        resp = json.loads(await ws.recv())
        if resp.get("type") != "ready":
            raise RuntimeError(f"WebSocket init 失败: {resp}")
        for i, chunk in enumerate(chunks):
            data_b64 = base64.b64encode(chunk).decode()
            await ws.send(json.dumps({"type": "audio", "data": data_b64, "chunk_index": i + 1}))
            while True:
                msg = json.loads(await ws.recv())
                t = msg.get("type")
                if t == "error":
                    raise RuntimeError(msg.get("message", "未知错误"))
                if t == "transcript_raw":
                    continue
                if t == "transcript":
                    utts = msg.get("utterances") or []
                    for u in list(utts):
                        u = dict(u)
                        start = u.get("start", 0)
                        end = u.get("end", 0)
                        u["start"] = chunk_offset + start
                        u["end"] = chunk_offset + end
                        all_utterances.append(u)
                    chunk_offset += chunk_seconds
                    break
        await ws.send(json.dumps({"type": "end"}))
        msg = json.loads(await ws.recv())
        if msg.get("type") != "done":
            pass
    return all_utterances


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
        if obj.get("stage") == "utterance":
            u = obj.get("utterance")
            if u is not None:
                out_utterances.append(u)
        if obj.get("stage") == "done" and "utterances" in obj:
            out_utterances = obj["utterances"]
            break
    return out_utterances


def label(utterances: list, stream: bool, max_items: int | None, base_url: str) -> list:
    url = f"{base_url.rstrip('/')}/labels"
    payload = [{"text": u.get("text", ""), "speaker": u.get("speaker", "")} for u in utterances]
    if max_items is not None:
        payload = payload[:max_items]
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
            result_list.append(obj["utterance"])
    return result_list


# ---------------------------------------------------------------------------
# 流水线上下文与步骤抽象
# ---------------------------------------------------------------------------

class DeepPipelineContext:
    """深层流水线运行时的配置与中间结果，各 Step 读写此对象。"""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.base_url = cfg.get("base_url") or "http://127.0.0.1:8001"
        self.language = cfg.get("language") or "zh"
        self.chunk_seconds = max(0.5, float(cfg.get("chunk_seconds", 10)))
        self.refine_on_ws = bool(cfg.get("refine_on_ws", False))
        self.stream_refine = bool(cfg.get("stream_refine", True))
        self.stream_labels = bool(cfg.get("stream_labels", True))
        self.max_labels = cfg.get("max_labels")
        audio = cfg.get("audio")
        speakers = cfg.get("speakers") or "json/speakers_embedding.json"
        out = cfg.get("out") or "json/deep_pipeline_output.json"

        self.audio_path = resolve_path(audio)
        self.speakers_path = resolve_path(speakers)
        self.out_path = resolve_path(out)
        self.path_transcribe = resolve_path(cfg.get("out_transcribe"))
        self.path_refined = resolve_path(cfg.get("out_refined"))

        self.utterances: list = []
        self.refined: list = []

        if self.out_path:
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
    def run(self, ctx: DeepPipelineContext, index: int, total: int) -> None:
        pass

    def _write_json(self, path: Path | None, data: dict, desc: str = "已保存") -> None:
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"      {desc}: {path}")


class LiveTranscribeStep(PipelineStep):
    name = "WebSocket 实时转录"

    def run(self, ctx: DeepPipelineContext, index: int, total: int) -> None:
        if not ctx.audio_path or not ctx.audio_path.exists():
            raise FileNotFoundError(f"音频不存在: {ctx.audio_path}")
        print(
            f"[{index}/{total}] {self.name} (chunk_seconds={ctx.chunk_seconds}, refine_on_ws={ctx.refine_on_ws}): "
            f"{ctx.audio_path.name} ..."
        )
        ctx.utterances = asyncio.run(
            live_transcribe(
                ctx.audio_path,
                ctx.speakers_path,
                ctx.language,
                ctx.base_url,
                ctx.chunk_seconds,
                ctx.refine_on_ws,
            )
        )
        print(f"      转录结果: {len(ctx.utterances)} 条")
        self._write_json(ctx.path_transcribe, {"utterances": ctx.utterances})


class RefineStep(PipelineStep):
    name = "精修"

    def run(self, ctx: DeepPipelineContext, index: int, total: int) -> None:
        print(f"[{index}/{total}] {self.name} (stream={ctx.stream_refine}) ...")
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

    def run(self, ctx: DeepPipelineContext, index: int, total: int) -> None:
        print(f"[{index}/{total}] {self.name} (stream={ctx.stream_labels}, max={ctx.max_labels}) ...")
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


class DeepPipeline:
    """按顺序执行若干 PipelineStep，共享同一 DeepPipelineContext。"""

    def __init__(self, steps: list[PipelineStep]):
        self.steps = steps

    def run(self, ctx: DeepPipelineContext) -> None:
        total = len(self.steps)
        if not total:
            return
        self.steps[0].run(ctx, 1, total)
        if not ctx.utterances:
            return
        for i, step in enumerate(self.steps[1:], start=2):
            step.run(ctx, i, total)


def build_pipeline() -> DeepPipeline:
    """构建深层流水线：实时转录 → 精修 → 标注（无指标与评价）。"""
    return DeepPipeline([
        LiveTranscribeStep(),
        RefineStep(),
        LabelStep(),
    ])


def main() -> None:
    if not CONFIG_PATH.exists():
        print(f"配置文件不存在: {CONFIG_PATH}")
        sys.exit(1)
    cfg = load_config(CONFIG_PATH)
    ctx = DeepPipelineContext(cfg)

    if not ctx.audio_path:
        print("请在 deep_pipeline_config.yaml 中设置 audio")
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

    print("深层流水线完成（未做指标与评价）。")


if __name__ == "__main__":
    main()

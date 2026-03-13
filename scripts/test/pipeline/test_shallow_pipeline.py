"""
浅度流水线端到端测试：从音频+声纹（或已有转录 JSON）→ 转录 → 精修 → 标注，保存最终结果。

配置：同目录下 shallow_pipeline_config.yaml（路径相对 scripts/test/data）。
用法（需先启动服务: uvicorn app:app --host 0.0.0.0 --port 8001）:
  python scripts/test/pipeline/test_shallow_pipeline.py
"""
import json
import sys
from pathlib import Path

import requests

# 项目根（scripts/test/pipeline/ -> 上溯 4 层）
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

# 测试数据目录：scripts/test/data（配置中的路径相对此目录解析）
TEST_DATA = Path(__file__).resolve().parent.parent / "data"
CONFIG_PATH = Path(__file__).resolve().parent / "shallow_pipeline_config.yaml"

DEFAULT_SPEAKERS = TEST_DATA / "json" / "speakers_embedding.json"
DEFAULT_AUDIO = TEST_DATA / "audio" / "audio_all.wav"
OUT_JSON = TEST_DATA / "json" / "shallow_pipeline_output.json"


def load_config(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        raise RuntimeError("请安装 PyYAML: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def resolve_path(p, base: Path = None) -> Path:
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


def transcribe(audio_path: Path, speakers_path: Path, language: str, base_url: str) -> list:
    url = f"{base_url.rstrip('/')}/transcriptions"
    speakers = load_speakers(speakers_path)
    data = [("language", language)]
    for s in speakers:
        data.append(("student_id", s.get("student_id", "")))
        data.append(("name", s.get("name", "")))
        data.append(("embedding", json.dumps(s["embedding"], ensure_ascii=False)))
    with open(audio_path, "rb") as f:
        r = requests.post(url, data=data, files={"audio": (audio_path.name, f, "audio/wav")}, timeout=600)
    r.raise_for_status()
    return r.json().get("utterances", [])


def refine(utterances: list, stream: bool, allowed_speakers: list, base_url: str) -> list:
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


def label(utterances: list, stream: bool, max_items: int, base_url: str) -> list:
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


def main():
    config_path = CONFIG_PATH
    if not config_path.exists():
        print(f"配置文件不存在: {config_path}")
        sys.exit(1)
    cfg = load_config(config_path)
    base_url = cfg.get("base_url") or "http://127.0.0.1:8001"
    audio = cfg.get("audio")
    speakers = cfg.get("speakers") or "json/speakers_embedding.json"
    from_json = cfg.get("from_json")
    stream_refine = bool(cfg.get("stream_refine", False))
    stream_labels = bool(cfg.get("stream_labels", False))
    max_labels = cfg.get("max_labels")
    language = cfg.get("language") or "zh"
    out = cfg.get("out") or "json/shallow_pipeline_output.json"


    speakers_path = resolve_path(speakers)
    out_path = resolve_path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if from_json:
        path = resolve_path(from_json)
        if not path or not path.exists():
            print(f"文件不存在: {path}")
            sys.exit(1)
        with open(path, "r", encoding="utf-8") as f:
            utterances = json.load(f).get("utterances", [])
        print(f"[跳过转录] 从 {path.name} 加载，共 {len(utterances)} 条")
    elif audio and speakers_path:
        audio_path = resolve_path(audio)
        if not audio_path or not audio_path.exists():
            print(f"音频不存在: {audio_path}")
            sys.exit(1)
        if not speakers_path.exists():
            print(f"声纹不存在: {speakers_path}")
            sys.exit(1)
        print(f"[1/3] 转录: {audio_path.name} + {speakers_path.name} ...")
        utterances = transcribe(audio_path, speakers_path, language, base_url)
        print(f"      转录结果: {len(utterances)} 条")
    else:
        print("请在 shallow_pipeline_config.yaml 中设置 from_json 或同时设置 audio 与 speakers")
        sys.exit(1)

    if not utterances:
        print("无 utterance，退出")
        sys.exit(0)

    print(f"[2/3] 精修 (stream={stream_refine}) ...")
    refined = refine(utterances, stream_refine, None, base_url)
    print(f"      精修结果: {len(refined)} 条")
    if not stream_refine:
        for i, u in enumerate(refined):
            speaker = u.get("speaker", "") or "?"
            text = (u.get("text", "") or "").strip()
            snippet = (text[:50] + "…") if len(text) > 50 else text or "(空)"
            print(f"      [{i}] {speaker}: {snippet}")

    print(f"[3/3] 标注 (stream={stream_labels}, max={max_labels}) ...")
    labels_only = label(refined, stream_labels, max_labels, base_url)
    print(f"      标注结果: {len(labels_only)} 条")

    for i, lab in enumerate(labels_only):
        if i < len(refined):
            refined[i]["label"] = lab.get("label")
            refined[i]["reason"] = lab.get("reason")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"utterances": refined}, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {out_path}")


if __name__ == "__main__":
    main()

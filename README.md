# 声纹 API · Speaker Diarization & Transcription

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

基于 **pyannote.audio** 与 **Faster-Whisper** 的多说话人语音转写与声纹识别服务。提供声纹 embedding 提取、批量/流式转写、WebSocket 实时转写，支持按学号/姓名区分说话人。

---

## 目录

- [功能特性](#功能特性)
- [技术架构](#技术架构)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [API 参考](#api-参考)
- [转写流水线](#转写流水线)
- [项目结构](#项目结构)
- [开发与测试](#开发与测试)
- [参与贡献](#参与贡献)
- [许可证](#许可证)

---

## 功能特性

| 功能 | 说明 |
|------|------|
| **声纹 embedding** | 上传学号 + 姓名 + 音频，返回 256 维 embedding 向量，用于说话人注册 |
| **批量转写** | 提交说话人列表（学号、姓名、embedding）+ 音频，返回带说话人标签的完整转写 JSON |
| **流式转写** | 同上参数，通过 **Server-Sent Events (SSE)** 实时推送每句结果，附带进度与耗时统计 |
| **Live 实时转写** | **WebSocket** 接收音频块，近似实时返回转写；`init` 时传入 `speakers` 则进行 diarization + 声纹匹配（分人），否则仅 Whisper 转写 |

- 支持 WAV、MP3、WebM 等常见音频格式（依赖 pydub）。
- 声纹匹配基于余弦相似度，可配置阈值。
- 流式接口适合长音频，无需等待整段处理完成即可展示结果。

---

## 技术架构

### 整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           FastAPI (app.py)                                │
│  /embedding │ /transcribe │ /transcribe/stream │ /ws/live-transcribe      │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     core/pipeline.py (DiarizedTranscriber)                │
│  1. Diarization  2. Embedding 提取  3. 声纹匹配  4. Whisper 转写          │
└─────────────────────────────────────────────────────────────────────────┘
         │                    │                    │                    │
         ▼                    ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ core/diarize │    │core/embedding │    │  speakers   │    │core/transcribe│
│ pyannote     │    │ Wespeaker    │    │ Registry    │    │Faster-Whisper │
│ Pipeline     │    │ 256-dim      │    │ 余弦匹配     │    │ ASR          │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

### 数据流（转写流程）

1. **输入**：音频文件（或 WebSocket 音频块）+ 可选说话人列表 `[{ student_id, name, embedding }, ...]`。
2. **说话人分割 (Diarization)**：pyannote.audio 对音频做 segmentation + clustering，输出若干 `(start, end, speaker_label)` 片段（标签为匿名 SPEAKER_00 等）。
3. **声纹提取**：对每个片段的音频提取 256 维 embedding（与 pyannote 使用的 Wespeaker 模型一致）。
4. **声纹匹配**：将片段 embedding 与用户提供的说话人 embedding 做余弦相似度匹配，超过阈值则赋学号/姓名，否则为 `unknown`。
5. **语音识别 (ASR)**：Faster-Whisper 对每个片段做转写，得到文本。
6. **输出**：合并为 `{ start, end, speaker, student_id, text }` 的 utterance 列表；流式/WebSocket 下按句或按块推送。

### 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| Web 框架 | FastAPI | REST、SSE、WebSocket |
| 说话人分割 | pyannote.audio ≥3.1 | Segmentation + 聚类 + PLDA，输出谁在何时说话 |
| 声纹模型 | Wespeaker (e.g. voxceleb-resnet34-LM) | 256 维 speaker embedding |
| 语音识别 | Faster-Whisper | 流式/离线 ASR，支持多语言 |
| 声纹匹配 | 自定义 (speakers.py) | 余弦相似度 + 阈值，SpeakerRegistry |

---

## 环境要求

- **Python**：3.10 或 3.12（推荐，已在 3.12 下测试）。
- **运行环境**：建议 8GB+ 内存；GPU 可选（pyannote、Whisper 均可使用 CUDA 加速）。
- **模型文件**：
  - pyannote：segmentation 与 embedding 模型（可本地路径或 HuggingFace/ModelScope 模型 ID）。
  - PLDA：可选，用于聚类 refinement，需 `xvec_transform.npz` 与 `plda.npz`。
  - Faster-Whisper：需下载对应规模模型（如 large-v3），路径在 `core/config.py` 中配置。

---

## 快速开始

### 1. 克隆与依赖

```bash
git clone <repository-url>
cd pyannote_diarization
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置

- 在 **`core/config.py`** 中设置：
  - `WHISPER_MODEL_PATH`：Faster-Whisper 模型目录。
  - `EMBEDDING_MODEL_PATH`：可选，若与 config.yaml 不一致时可在此覆盖。
- 在 **`config.yaml`** 中设置：
  - `pipeline.params.segmentation` / `embedding`：pyannote 模型路径或 HF 模型名。
  - `pipeline.params.plda.checkpoint`：PLDA 目录（若使用）。
  - `params.clustering.threshold`、`min_cluster_size` 等按需调整。

若使用 Hugging Face 上的 pyannote 模型，需在 HF 网站接受模型条款并配置 `HF_TOKEN`（见 pyannote 文档）。

### 3. 启动服务

```bash
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

### 4. 验证

```bash
curl http://127.0.0.1:8001/health
```

浏览器访问 **http://127.0.0.1:8001/live** 可使用 Live 实时转写页面（麦克风 + WebSocket）。

---

## 配置说明

### core/config.py

| 变量 | 说明 |
|------|------|
| `BASE_DIR` | 项目根目录（自动） |
| `CONFIG_PATH` | config.yaml 路径 |
| `PLDA_DIR` | 本地 PLDA 目录（若与 config.yaml 中一致可不用） |
| `EMBEDDING_MODEL_PATH` | 声纹 embedding 模型路径（.bin 或目录） |
| `WHISPER_MODEL_PATH` | Faster-Whisper 模型目录 |
| `DATA_DIR` | 数据目录（可选） |

### config.yaml

| 配置项 | 说明 |
|--------|------|
| `pipeline.params.segmentation` | 分割模型路径或 HuggingFace 模型 ID |
| `pipeline.params.embedding` | 声纹模型路径或 HF 模型 ID |
| `pipeline.params.embedding_batch_size` | embedding 批大小，影响显存与速度 |
| `pipeline.params.segmentation_batch_size` | 分割批大小 |
| `pipeline.params.plda.checkpoint` | PLDA 目录（须含 xvec_transform.npz、plda.npz） |
| `params.segmentation.min_duration_off` | 说话人之间最小静音时长（秒） |
| `params.clustering.method` | 聚类方法（如 centroid） |
| `params.clustering.threshold` | 聚类合并的余弦距离阈值 |
| `params.clustering.min_cluster_size` | 每类最少样本数 |

---

## API 参考

### 健康检查

- **GET** `/health`  
- **响应**：`200 OK`，简单健康状态。

---

### POST /embedding

上传学号、姓名与音频，返回声纹 embedding。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| student_id | string | 是 | 学号 |
| name | string | 否 | 姓名 |
| audio | file | 是 | 音频文件（WAV/MP3 等） |

**响应** `200 OK`：

```json
{
  "student_id": "2021001",
  "name": "张三",
  "embedding": [-0.13, 0.12, ...],
  "embedding_dim": 256
}
```

---

### POST /transcribe

提交说话人列表与音频，返回完整转写结果（同步）。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| student_id | string[] | 是 | 学号列表，与 name、embedding 一一对应 |
| name | string[] | 是 | 姓名列表 |
| embedding | string[] | 是 | 各说话人 embedding 的 JSON 数组字符串，如 `["[-0.1,0.2,...]","[...]"]` |
| audio | file | 是 | 音频文件 |
| language | string | 否 | 语言代码，如 zh、en |

**响应** `200 OK`：

```json
{
  "utterances": [
    {
      "start": 3.0,
      "end": 3.6,
      "speaker": "peppa",
      "student_id": "2021001",
      "text": "我是佩奇"
    }
  ]
}
```

---

### POST /transcribe/stream

参数与 `/transcribe` 相同，响应为 **SSE 流**：每完成一句推送一条 `data:` 行（JSON 对象），最后推送一条 `status: "done"` 的汇总（含 `diarization_seconds`、`whisper_seconds` 等）。

**事件示例**：

```
data: {"start": 3.0, "end": 3.6, "speaker": "peppa", "student_id": "2021001", "text": "我是佩奇", "index": 1, "total": 111, "progress": 0.9}
data: {"status": "done", "total": 111, "progress": 100, "diarization_seconds": 12.34, "whisper_seconds": 89.56}
```

---

### WebSocket /ws/live-transcribe

实时转写：客户端发送音频块，服务端返回该块的转写结果。若在 `init` 中传入 `speakers`，则对该块做 diarization + 声纹匹配；否则仅做 Whisper 转写。

**消息类型**：

| 方向 | type | 说明 |
|------|------|------|
| 客户端 → | `init` | `{ "type": "init", "language": "zh", "speakers": [{"student_id","name","embedding"}, ...] }`，speakers 可选 |
| 服务端 ← | `ready` | `{ "type": "ready", "language": "zh", "has_speakers": true/false }` |
| 客户端 → | `audio` | `{ "type": "audio", "data": "<base64 编码的 WAV>", "chunk_index": 1 }` |
| 服务端 ← | `transcript` | `{ "type": "transcript", "utterances": [...], "text": "...", "chunk_index": 1 }` |
| 客户端 → | `end` | `{ "type": "end" }` |
| 服务端 ← | `done` | `{ "type": "done", "total_chunks": N }` |

建议音频块为 WAV，每块 5–15 秒。

---

### 通用错误

- 音频格式不支持、音频为空、embedding 提取/转写异常等均返回相应 HTTP 状态码与 JSON 错误体（见 `utils/errors.py`）。

---

## 转写流水线

1. **Diarization**（core/diarize.py）：加载 config.yaml 中的 pyannote Pipeline，对整段音频做 segmentation → embedding → clustering（+ 可选 PLDA），得到 `Annotation`（时间轴上的说话人片段）。
2. **声纹匹配**（speakers.py）：对每个片段用与 pyannote 相同的 embedding 模型提取向量，与请求中的说话人 embedding 做余弦相似度，取最大且超过阈值者赋 `speaker` / `student_id`。
3. **ASR**（core/transcribe.py）：Faster-Whisper 按片段时间戳截取音频并转写，得到文本。
4. **输出**：组装为 `Utterance(start, end, speaker, student_id, text)` 列表；流式/WebSocket 下按句或按块推送，并附带进度与耗时。

---

## 项目结构

```
pyannote_diarization/
├── app.py                  # FastAPI 应用：路由、/embedding、/transcribe、SSE、WebSocket
├── config.yaml             # pyannote 管道与聚类配置
├── speakers.py             # SpeakerRegistry：说话人列表构建与余弦匹配
├── core/
│   ├── __init__.py
│   ├── config.py           # 路径与模型路径
│   ├── pipeline.py         # DiarizedTranscriber：串联 diarize → 匹配 → Whisper
│   ├── diarize.py          # pyannote Pipeline 封装
│   ├── embedding.py        # 声纹 embedding 提取（Wespeaker）
│   └── transcribe.py       # Faster-Whisper 封装
├── utils/
│   ├── common.py           # 音频后缀、webm→wav、时长格式化等
│   └── errors.py           # HTTP 错误响应构造
├── scripts/test/
│   ├── test_transcribe.py
│   ├── test_transcribe_stream.py
│   └── test_live_transcribe.py
├── static/
│   └── live.html           # Live 转写前端
└── requirements.txt
```

---

## 开发与测试

```bash
# 激活虚拟环境后
python scripts/test/test_transcribe_stream.py   # 流式转写
python scripts/test/test_transcribe.py         # 批量转写
python scripts/test/test_live_transcribe.py    # WebSocket Live
```

开发时推荐使用 `uvicorn app:app --reload` 以便代码变更自动重载。

---

## 参与贡献

1. Fork 本仓库  
2. 创建功能分支（`git checkout -b feat/xxx`）  
3. 提交更改（`git commit -m 'feat: xxx'`）  
4. 推送到分支（`git push origin feat/xxx`）  
5. 提交 Pull Request  

---

## 许可证

[MIT](LICENSE)

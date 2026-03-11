# 声纹 API · Speaker Diarization & Transcription

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

基于 **pyannote.audio** 与 **Faster-Whisper** 的多说话人语音转写与声纹识别服务。业务分为**两大部分**：声纹与转写（音频→带说话人文本）、转写精修（LLM 推断说话人 + 纠错标点 + 按句拆分）。

---

## 目录

- [业务模块概览](#业务模块概览)
- [一、声纹与转写](#一声纹与转写)
- [二、转写精修（LLM）](#二转写精修llm)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [API 参考](#api-参考)
- [项目结构](#项目结构)
- [开发与测试](#开发与测试)
- [参与贡献](#参与贡献)
- [许可证](#许可证)

---

## 业务模块概览

| 模块 | 职责 | 典型入口 |
|------|------|----------|
| **一、声纹与转写** | 音频 → 说话人分割 → 声纹匹配 → ASR → 带说话人标签的 utterance 列表 | `/embedding`、`/transcribe`、`/transcribe/stream`、`/ws/live-transcribe` |
| **二、转写精修（LLM）** | 对已有转写结果做：推断 unknown 说话人、合并碎片、纠错标点、按句拆分、过滤无意义 | `POST /refine` |

两者可串联使用：先走声纹转写得到 `utterances`，再对结果调用 `/refine` 做精修。

---

## 一、声纹与转写

将音频转为带说话人归属的文本片段（utterance 列表）。支持声纹注册、批量/流式转写、WebSocket 实时转写。

### 1.1 功能特性

| 功能 | 说明 |
|------|------|
| **声纹 embedding** | 上传学号 + 姓名 + 音频，返回 256 维 embedding 向量，用于说话人注册 |
| **批量转写** | 提交说话人列表（学号、姓名、embedding）+ 音频，返回带说话人标签的完整转写 JSON |
| **流式转写** | 同上参数，通过 **Server-Sent Events (SSE)** 实时推送每句结果，附带进度与耗时统计 |
| **Live 实时转写** | **WebSocket** 接收音频块，近似实时返回转写；`init` 时传入 `speakers` 则进行 diarization + 声纹匹配（分人），否则仅 Whisper 转写 |

- 支持 WAV、MP3、WebM 等常见音频格式（依赖 pydub）。
- 声纹匹配基于余弦相似度，可配置阈值。
- 流式接口适合长音频，无需等待整段处理完成即可展示结果。

### 1.2 技术架构与数据流

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

1. **输入**：音频文件（或 WebSocket 音频块）+ 可选说话人列表 `[{ student_id, name, embedding }, ...]`。
2. **说话人分割 (Diarization)**：pyannote.audio 做 segmentation → clustering，输出 `(start, end, speaker_label)` 片段（匿名 SPEAKER_00 等）。
3. **声纹提取**：对每段音频提取 256 维 embedding（Wespeaker，与 pyannote 一致）。
4. **声纹匹配**：与请求中的说话人 embedding 余弦相似度匹配，超过阈值则赋学号/姓名，否则为 `unknown`。
5. **ASR**：Faster-Whisper 按片段时间戳转写，得到文本。
6. **输出**：`{ start, end, speaker, student_id, text }` 的 utterance 列表；流式/WebSocket 下按句或按块推送。

### 1.3 技术栈（声纹与转写）

| 组件 | 技术 | 说明 |
|------|------|------|
| Web 框架 | FastAPI | REST、SSE、WebSocket |
| 说话人分割 | pyannote.audio ≥3.1 | Segmentation + 聚类 + PLDA |
| 声纹模型 | Wespeaker (e.g. voxceleb-resnet34-LM) | 256 维 speaker embedding |
| 语音识别 | Faster-Whisper | 流式/离线 ASR，多语言 |
| 声纹匹配 | speakers.py | 余弦相似度 + 阈值，SpeakerRegistry |

---

## 二、转写精修（LLM）

对**已有**转写结果（utterance 列表）做后处理：推断未标注说话人、合并同人同句碎片、纠错与标点、按句拆分、过滤无意义发言。全部为异步流水线，纠错步骤可并发请求以节省时间。

### 2.1 功能与流水线顺序

| 步骤 | 说明 |
|------|------|
| **推断 unknown 说话人** | 仅对 `speaker=unknown` 的条目调用 LLM，根据上下文从已有说话人中选一个归属；已有归属的不调用，省 token |
| **合并片段** | 同一说话人且上一句无句末标点时合并为一条，减少碎句 |
| **纠错与标点** | 对每条文本调用 LLM 做错别字/同音词纠错与中文标点补全，多条并发 |
| **按句拆分** | 按句末标点（。！？.!?）拆成「一句一 utterance」 |
| **过滤无意义** | 删除空文本、纯语气词等 |
| **兜底** | 保证输出无 `unknown`，未归属的归为默认说话人 |

### 2.2 架构与数据流

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        FastAPI (routers/llm/refine.py)                   │
│                              POST /refine                                │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              core/llm/refine_pipeline.py · run_pipeline()                │
│  输入: utterances[]  输出: 精修后的 utterances[]（一句一条、无 unknown）   │
└─────────────────────────────────────────────────────────────────────────┘
         │                │                │                │
         ▼                ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ 1. 推断说话人 │  │ 2. 合并片段  │  │ 3. 纠错标点  │  │ 4. 按句拆分  │
│ infer_unknown│  │ merge_       │  │ correct_     │  │ split_       │
│ _speakers    │  │ fragments    │  │ utterance_   │  │ utterances_  │
│ (仅 unknown) │  │ (规则)       │  │ texts(并发)  │  │ by_sentence  │
└──────┬───────┘  └──────────────┘  └──────┬───────┘  └──────┬───────┘
       │                                   │                 │
       ▼                                   ▼                 │
┌──────────────┐                    ┌──────────────┐         │
│ prompts/     │                    │ prompts/     │         │
│ infer_       │                    │ correct_     │         │
│ speaker.py   │                    │ text.py      │         │
└──────┬───────┘                    └──────┬───────┘         │
       │                                   │                 │
       └───────────────┬───────────────────┘                 │
                       ▼                                      │
              ┌──────────────────┐                            │
              │ core/llm/        │                            │
              │ llm_client       │                            │
              │ (DashScope/qwen) │                            │
              └──────────────────┘                            │
                                                              │
         ┌────────────────────────────────────────────────────┘
         ▼
┌──────────────────────────────────────────────────────────────┐
│ 5. 过滤无意义 (filter_empty_and_meaningless)                  │
│ 6. 兜底 (_force_no_unknown，unknown→默认说话人)               │
└──────────────────────────────────────────────────────────────┘
```

1. **输入**：`utterances`（含 `start`、`end`、`speaker`、`student_id`、`text`），可有 `unknown`、碎句、无标点。
2. **推断说话人**：仅对 `speaker=unknown` 的条目前后文 + `prompts/infer_speaker` 调 LLM，顺序 ainvoke。
3. **合并**：同说话人且上句无句末标点则合并，纯规则，不调 LLM。
4. **纠错与标点**：每条文本 + `prompts/correct_text` 调 LLM，异步并发（Semaphore 限流）。
5. **按句拆分**：按 `。！？.!?` 拆成一句一条，时间按字符比例分配。
6. **过滤与兜底**：删空/语气词；剩余 `unknown` 归为默认说话人。
7. **输出**：精修后的 `utterances`，一句一 utterance、无 `unknown`。

### 2.3 技术栈（转写精修）

| 组件 | 说明 |
|------|------|
| LLM | LangChain ChatOpenAI，兼容 DashScope（qwen 等），需 `config/llm_model.env` 配置 `DASHSCOPE_API_KEY` |
| 提示词 | `prompts/infer_speaker.py`（推断说话人）、`prompts/correct_text.py`（纠错标点），Few-shot 为 HumanMessage/AIMessage 形式 |
| 流水线 | `core/llm/refine_pipeline.py`：`run_pipeline()` 为异步入口，支持 `infer_speakers`、`merge`、`correct_text`、`filter_meaningless` 等开关 |

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

### 一、声纹与转写

- **core/config.py**：`WHISPER_MODEL_PATH`、`EMBEDDING_MODEL_PATH`、`CONFIG_PATH`、`PLDA_DIR`、`DATA_DIR` 等。
- **config.yaml**：pyannote 管道与聚类（segmentation、embedding、plda、clustering 等），见下表。

| 配置项 | 说明 |
|--------|------|
| `pipeline.params.segmentation` | 分割模型路径或 HuggingFace 模型 ID |
| `pipeline.params.embedding` | 声纹模型路径或 HF 模型 ID |
| `pipeline.params.embedding_batch_size` | embedding 批大小 |
| `pipeline.params.plda.checkpoint` | PLDA 目录（须含 xvec_transform.npz、plda.npz） |
| `params.clustering.threshold` | 聚类合并的余弦距离阈值 |
| `params.clustering.min_cluster_size` | 每类最少样本数 |

### 二、转写精修（LLM）

- **config/llm_model.env**：配置 `DASHSCOPE_API_KEY`（或所用 LLM 服务的 API Key）。  
- **core/llm/llm_client.py**：`get_llm()` 默认使用 qwen-plus、DashScope 兼容端点，可按需改 model / api_base。

---

## API 参考

### 健康检查

- **GET** `/health`  
- **响应**：`200 OK`，简单健康状态。

---

### 一、声纹与转写

#### POST /embedding

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

#### POST /transcribe

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

#### POST /transcribe/stream

参数与 `/transcribe` 相同，响应为 **SSE 流**：每完成一句推送一条 `data:` 行（JSON 对象），最后推送一条 `status: "done"` 的汇总（含 `diarization_seconds`、`whisper_seconds` 等）。

**事件示例**：

```
data: {"start": 3.0, "end": 3.6, "speaker": "peppa", "student_id": "2021001", "text": "我是佩奇", "index": 1, "total": 111, "progress": 0.9}
data: {"status": "done", "total": 111, "progress": 100, "diarization_seconds": 12.34, "whisper_seconds": 89.56}
```

---

#### WebSocket /ws/live-transcribe

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

### 二、转写精修（LLM）

#### POST /refine

对已有转写结果做精修：推断 unknown 说话人、合并片段、纠错标点、按句拆分、过滤无意义。请求体为 JSON。

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| utterances | array | 是 | 转写结果，每项含 `start`、`end`、`speaker`、`student_id`、`text` |
| infer_speakers | bool | 否 | 是否推断 unknown 说话人，默认 true |
| merge | bool | 否 | 是否合并同人同句碎片，默认 true |
| correct_text | bool | 否 | 是否纠错与标点，默认 true |
| context_size | int | 否 | 推断说话人时的上下文句数，默认 3 |

**响应** `200 OK`：

```json
{
  "utterances": [
    { "start": 3.0, "end": 5.2, "speaker": "peppa", "student_id": "T2", "text": "我是佩奇，这是我的弟弟乔治。" }
  ]
}
```

精修后保证每条 utterance 均有说话人归属（无 `unknown`），且按句末标点拆成「一句一条」。

---

### 通用错误

- 音频格式不支持、音频为空、embedding 提取/转写异常等均返回相应 HTTP 状态码与 JSON 错误体（见 `utils/errors.py`）。

---

## 项目结构

```
pyannote_diarization/
├── app.py                     # FastAPI 应用入口，挂载 audio / llm / health 路由
├── config.yaml                # 【一、声纹与转写】pyannote 管道与聚类配置
├── speakers.py                # 【一、声纹与转写】SpeakerRegistry、余弦匹配
├── core/
│   ├── config.py              # 【一】路径与模型路径
│   ├── pipeline.py            # 【一】DiarizedTranscriber：diarize → 匹配 → Whisper
│   ├── diarize.py             # 【一】pyannote Pipeline 封装
│   ├── embedding.py           # 【一】声纹 embedding 提取（Wespeaker）
│   ├── transcribe.py          # 【一】Faster-Whisper 封装
│   └── llm/                   # 【二、转写精修】
│       ├── llm_client.py       # LLM 客户端（DashScope 兼容）
│       └── refine_pipeline.py # 精修流水线：推断说话人、合并、纠错、按句拆分
├── prompts/                   # 【二、转写精修】提示词模板
│   ├── refine.py              # 统一入口
│   ├── infer_speaker.py       # 推断说话人（Few-shot）
│   └── correct_text.py        # 纠错与标点（Few-shot）
├── routers/
│   ├── audio.py               # 【一】/embedding、/transcribe、/transcribe/stream、/ws/...
│   └── llm/                   # 【二】/refine
├── utils/
│   ├── common.py
│   └── errors.py
├── config/
│   └── llm_model.env          # 【二】LLM API Key 等
├── scripts/test/
│   ├── test_transcribe.py
│   ├── test_transcribe_stream.py
│   ├── test_live_transcribe.py
│   └── test_refine.py         # 【二】精修流水线本地测试
├── static/
│   └── live.html
└── requirements.txt
```

---

## 开发与测试

```bash
# 激活虚拟环境后
# 一、声纹与转写
python scripts/test/test_transcribe_stream.py   # 流式转写
python scripts/test/test_transcribe.py         # 批量转写
python scripts/test/test_live_transcribe.py    # WebSocket Live

# 二、转写精修（需配置 config/llm_model.env）
python scripts/test/test_refine.py              # 精修流水线（读 data/json/transcribe_output.json → 输出 refined）
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

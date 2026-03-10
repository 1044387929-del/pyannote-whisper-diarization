# 基于pyannote.audio和whisper的语音识别api仓库

#### 介绍
{**以下是 Gitee 平台说明，您可以替换此简介**
Gitee 是 OSCHINA 推出的基于 Git 的代码托管平台（同时支持 SVN）。专为开发者提供稳定、高效、安全的云端软件开发协作平台
无论是个人、团队、或是企业，都能够用 Gitee 实现代码托管、项目管理、协作开发。企业项目请看 [https://gitee.com/enterprises](https://gitee.com/enterprises)}

#### 软件架构
软件架构说明


#### 安装教程

1.  xxxx
2.  xxxx
3.  xxxx

#### 使用说明

1.  xxxx
2.  xxxx
3.  xxxx

#### 参与贡献

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request


#### 特技

1.  使用 Readme\_XXX.md 来支持不同的语言，例如 Readme\_en.md, Readme\_zh.md
2.  Gitee 官方博客 [blog.gitee.com](https://blog.gitee.com)
3.  你可以 [https://gitee.com/explore](https://gitee.com/explore) 这个地址来了解 Gitee 上的优秀开源项目
4.  [GVP](https://gitee.com/gvp) 全称是 Gitee 最有价值开源项目，是综合评定出的优秀开源项目
5.  Gitee 官方提供的使用手册 [https://gitee.com/help](https://gitee.com/help)
6.  Gitee 封面人物是一档用来展示 Gitee 会员风采的栏目 [https://gitee.com/gitee-stars/](https://gitee.com/gitee-stars/)
# 声纹 API · Speaker Diarization & Transcription

基于 **pyannote.audio** + **Faster-Whisper** 的声纹 embedding 提取与多说话人转写服务。支持流式 SSE 推送，实时展示转录进度。

---

## 功能概览

| 功能 | 说明 |
|------|------|
| **声纹 embedding** | 上传学号 + 音频，返回 256 维 embedding 向量 |
| **批量转写** | 提交说话人信息 + 音频，返回完整转写 JSON |
| **流式转写** | SSE 实时推送每句结果，附带进度与耗时统计 |
| **Live 实时转写** | WebSocket 持续接收音频块，近似实时返回转写；init 传入 speakers 时做 diarization + 声纹匹配（分人） |

---

## 技术栈

- **pyannote.audio**：说话人分割（diarization）
- **Faster-Whisper**：语音转文字
- **声纹匹配**：自定义说话人 embedding 匹配
- **FastAPI**：HTTP API + SSE 流式响应

---

## 项目结构

```
pyannote_diarization/
├── app.py                 # FastAPI 入口
├── config.yaml            # pyannote 配置
├── speakers.py            # 声纹匹配逻辑
├── core/
│   ├── pipeline.py        # 转写流程（diarization + Whisper）
│   ├── diarize.py         # 说话人分割
│   ├── embedding.py       # 声纹 embedding
│   └── config.py          # 路径配置
├── scripts/
│   └── test/
│       ├── test_transcribe.py       # 测试 /transcribe
│       └── test_transcribe_stream.py # 测试 /transcribe/stream
└── data/
    └── json/
        └── speakers_embedding.json  # 说话人声纹库
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置模型路径

在 `core/config.py` 中设置 `WHISPER_MODEL_PATH`、`CONFIG_PATH` 等。`config.yaml` 中配置 pyannote 模型与 PLDA 路径。

### 3. 启动服务

```bash
# 开发模式（自动重载）
uvicorn app:app --host 0.0.0.0 --port 8001 --reload

# 或直接运行
python app.py
```

### 4. 健康检查

```bash
curl http://127.0.0.1:8001/health
```

---

## API 接口

### 1. POST /embedding

上传学号 + 音频，返回声纹 embedding 向量。

```bash
curl -X POST http://127.0.0.1:8001/embedding \
  -F "student_id=2021001" \
  -F "name=张三" \
  -F "audio=@data/embedding_audios/peppa.wav"
```

**响应示例：**

```json
{
  "student_id": "2021001",
  "name": "张三",
  "embedding": [-0.13, 0.12, ...],
  "embedding_dim": 256
}
```

---

### 2. POST /transcribe

提交说话人信息（学号、姓名、embedding）+ 音频，返回完整转写结果。

```bash
curl -X POST http://127.0.0.1:8001/transcribe \
  -F "student_id=2021001" -F "student_id=2021002" \
  -F "name=张三" -F "name=李四" \
  -F "embedding=[-0.13,0.12,...]" -F "embedding=[0.2,-0.1,...]" \
  -F "audio=@data/audio/audio_all.wav" \
  -F "language=zh"
```

**响应示例：**

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

### 3. POST /transcribe/stream（流式）

参数与 `/transcribe` 相同，以 **SSE** 流式推送每句结果，无需等待全部完成。

```bash
curl -N -X POST http://127.0.0.1:8001/transcribe/stream \
  -F "student_id=2021001" -F "student_id=2021002" \
  -F "name=张三" -F "name=李四" \
  -F "embedding=[-0.13,0.12,...]" -F "embedding=[0.2,-0.1,...]" \
  -F "audio=@data/audio/audio_all.wav" \
  -F "language=zh"
```

**流式事件格式：**

每条 utterance：

```
data: {"start": 3.0, "end": 3.6, "speaker": "peppa", "student_id": "2021001", "text": "我是佩奇", "index": 1, "total": 111, "progress": 0.9}
```

结束时汇总事件：

```
data: {"status": "done", "total": 111, "progress": 100, "diarization_seconds": 12.34, "whisper_seconds": 89.56}
```

---

### 4. WebSocket /ws/live-transcribe（Live 实时转写）

录音界面边录边传，服务端近似实时返回转写。**init 传入 speakers 时做 diarization + 声纹匹配（分人）**，否则仅 Whisper 转写。

**消息协议：**

| 方向 | type | 说明 |
|------|------|------|
| 客户端→ | `init` | `{"type": "init", "language": "zh", "speakers": [{"student_id","name","embedding"}, ...]}`（speakers 可选） |
| ←服务端 | `ready` | `{"type": "ready", "language": "zh", "has_speakers": true/false}` |
| 客户端→ | `audio` | `{"type": "audio", "data": "<base64>", "chunk_index": 1}` |
| ←服务端 | `transcript` | `{"type": "transcript", "utterances": [{start,end,speaker,student_id,text}, ...], "text": "...", "chunk_index": 1}` |
| 客户端→ | `end` | `{"type": "end"}` |
| ←服务端 | `done` | `{"type": "done", "total_chunks": N}` |

音频格式：wav，建议每块 5–15 秒。

---

## 测试脚本

```bash
# 测试流式转写（推荐）
python pyannote_diarization/scripts/test/test_transcribe_stream.py

# 测试非流式转写
python pyannote_diarization/scripts/test/test_transcribe.py

# 测试 Live WebSocket 实时转写
python pyannote_diarization/scripts/test/test_live_transcribe.py

# Live 前端（麦克风录音 + 实时转写）
# 启动服务后访问: http://127.0.0.1:8001/live
```

---

## 转写流程说明

1. **pyannote 说话人分割**：对整段音频做 diarization，得到各说话片段及时间戳
2. **声纹匹配**：对每段提取 embedding，与已知说话人比对，得到学号/姓名
3. **Whisper 转写**：逐段转写文本
4. **流式输出**：每完成一句即通过 SSE 推送给客户端，并附带进度与耗时

---

## 配置说明

`config.yaml` 主要参数：

| 参数 | 说明 |
|------|------|
| `segmentation.min_duration_off` | 说话人之间的最小静音时长（秒） |
| `clustering.threshold` | 聚类合并的余弦距离阈值 |
| `clustering.min_cluster_size` | 每个说话人类最少样本数 |

---

## 返回 JSON 字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `utterances` | list | 转写结果列表 |
| `total` | int | 总句数 |
| `progress` | float | 完成百分比 |
| `diarization_seconds` | float | pyannote 分割耗时（秒） |
| `whisper_seconds` | float | Whisper 转写耗时（秒） |

每条 utterance 含：`start`、`end`、`speaker`、`student_id`、`text`、`index`、`total`、`progress`。

---

## License

MIT

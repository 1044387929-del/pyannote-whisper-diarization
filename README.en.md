# Speaker Diarization & Transcription API

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Multi-speaker speech transcription and speaker recognition service built on **pyannote.audio** and **Faster-Whisper**. The project is split into **two business parts**: (I) Voice & Transcription (audio → speaker-labeled utterances), and (II) Transcript Refinement (LLM-based speaker inference, correction, punctuation, and sentence splitting).

---

## Table of Contents

- [Business Overview](#business-overview)
- [Part I: Voice & Transcription](#part-i-voice--transcription)
- [Part II: Transcript Refinement (LLM)](#part-ii-transcript-refinement-llm)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Development & Testing](#development--testing)
- [Contributing](#contributing)
- [License](#license)

---

## Business Overview

| Part | Role | Entrypoints |
|------|------|-------------|
| **I. Voice & Transcription** | Audio → diarization → speaker matching → ASR → utterance list with speaker labels | `POST /embeddings`, `POST /transcriptions`, `POST /transcriptions/stream`, `/ws/transcriptions/live` |
| **II. Transcript Refinement (LLM)** | Post-process existing transcripts: infer unknown speakers, merge fragments, correct text & punctuation, split by sentence, filter noise | `POST /refinements` |

You can chain them: run voice & transcription to get `utterances`, then call `POST /refinements` on the result.

---

## Part I: Voice & Transcription

Turns audio into speaker-labeled text segments (utterance list). Supports embedding registration, batch/streaming transcription, and WebSocket live transcription.

### 1.1 Features

| Feature | Description |
|--------|-------------|
| **Voice embedding** | Upload student ID + name + audio; returns 256-dim embedding for speaker registration |
| **Batch transcription** | Submit speaker list (student_id, name, embedding) + audio; returns full transcript JSON with speaker labels |
| **Streaming transcription** | Same params; results streamed via **Server-Sent Events (SSE)** with progress and timing |
| **Live transcription** | **WebSocket**: send audio chunks, receive near real-time transcripts; pass `speakers` in `init` for diarization + speaker matching, otherwise Whisper-only |

- Supports WAV, MP3, WebM and other common formats (via pydub).
- Speaker matching uses cosine similarity with configurable threshold.
- Streaming is suitable for long audio; results can be shown incrementally.

### 1.2 Architecture & Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           FastAPI (app.py)                               │
│  /embeddings │ /transcriptions │ /transcriptions/stream │ /ws/transcriptions/live │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     core/pipeline.py (DiarizedTranscriber)                │
│  1. Diarization  2. Embedding  3. Speaker match  4. Whisper ASR            │
└─────────────────────────────────────────────────────────────────────────┘
         │                    │                    │                    │
         ▼                    ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ core/diarize │    │core/embedding│    │  speakers   │    │core/transcribe│
│ pyannote     │    │ Wespeaker    │    │ Registry    │    │Faster-Whisper │
│ Pipeline     │    │ 256-dim      │    │ cosine match│    │ ASR           │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

1. **Input**: Audio file (or WebSocket chunks) + optional speaker list `[{ student_id, name, embedding }, ...]`.
2. **Diarization**: pyannote.audio runs segmentation → clustering, outputting `(start, end, speaker_label)` segments (e.g. SPEAKER_00).
3. **Embedding**: 256-dim embedding per segment (Wespeaker, same as pyannote).
4. **Speaker matching**: Cosine similarity with request embeddings; above threshold → student_id/name, else `unknown`.
5. **ASR**: Faster-Whisper transcribes each segment.
6. **Output**: Utterances `{ start, end, speaker, student_id, text }`; streamed or sent per sentence/chunk.

### 1.3 Tech Stack (Voice & Transcription)

| Component | Technology | Notes |
|-----------|------------|--------|
| Web | FastAPI | REST, SSE, WebSocket |
| Diarization | pyannote.audio ≥3.1 | Segmentation + clustering + PLDA |
| Embedding | Wespeaker (e.g. voxceleb-resnet34-LM) | 256-dim speaker embedding |
| ASR | Faster-Whisper | Streaming/offline, multilingual |
| Matching | speakers.py | Cosine similarity + threshold, SpeakerRegistry |

---

## Part II: Transcript Refinement (LLM)

Post-processes **existing** transcript utterance lists: infer unknown speakers, merge same-speaker fragments, correct text and punctuation, split by sentence, filter meaningless segments. Fully async pipeline; correction step uses concurrent LLM calls to reduce latency.

### 2.1 Pipeline Steps

| Step | Description |
|------|-------------|
| **Infer unknown speakers** | Call LLM only for `speaker=unknown`; pick one allowed speaker from context; skip when already labeled to save tokens |
| **Merge fragments** | Merge consecutive same-speaker segments when the previous segment has no sentence-ending punctuation |
| **Correct & punctuate** | Call LLM per segment for typo/homophone correction and punctuation; requests run concurrently |
| **Split by sentence** | Split on sentence-ending punctuation (。！？.!?) so one sentence = one utterance |
| **Filter meaningless** | Drop empty text and filler-only segments |
| **Fallback** | Ensure no `unknown` in output; assign remaining unknowns to a default speaker |

### 2.2 Architecture & Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        FastAPI (routers/llm/refine.py)                   │
│                              POST /refinements                           │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              core/llm/refine_pipeline.py · run_pipeline()                │
│  Input: utterances[]  Output: refined utterances[] (one per sentence,   │
│  no unknown)                                                             │
└─────────────────────────────────────────────────────────────────────────┘
         │                │                │                │
         ▼                ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ 1. Infer     │  │ 2. Merge     │  │ 3. Correct   │  │ 4. Split     │
│ speaker      │  │ fragments   │  │ & punctuate  │  │ by sentence  │
│ (unknown only)│  │ (rules)     │  │ (concurrent) │  │ by_sentence  │
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
│ 5. Filter meaningless (filter_empty_and_meaningless)          │
│ 6. Fallback (_force_no_unknown, unknown→default speaker)     │
└──────────────────────────────────────────────────────────────┘
```

1. **Input**: `utterances` with `start`, `end`, `speaker`, `student_id`, `text`; may contain `unknown`, fragments, no punctuation.
2. **Infer speaker**: For `speaker=unknown` only, build context + `prompts/infer_speaker` and call LLM (sequential ainvoke).
3. **Merge**: Same speaker and no sentence end → merge; rules only, no LLM.
4. **Correct & punctuate**: Per-segment text + `prompts/correct_text` and LLM; async concurrent (Semaphore-limited).
5. **Split by sentence**: Split on `。！？.!?`; one sentence per utterance; time split by character ratio.
6. **Filter & fallback**: Drop empty/filler; assign remaining `unknown` to default speaker.
7. **Output**: Refined `utterances`, one sentence per utterance and no `unknown`.

### 2.3 Tech Stack (Refinement)

| Component | Description |
|-----------|-------------|
| LLM | LangChain ChatOpenAI, DashScope-compatible (e.g. qwen); set `DASHSCOPE_API_KEY` in `config/llm_model.env` |
| Prompts | `prompts/infer_speaker.py` (speaker inference), `prompts/correct_text.py` (correction & punctuation); few-shot as HumanMessage/AIMessage |
| Pipeline | `core/llm/refine_pipeline.py`: async `run_pipeline()` with flags `infer_speakers`, `merge`, `correct_text`, `filter_meaningless` |

---

## Requirements

- **Python**: 3.10 or 3.12 (recommended; tested on 3.12).
- **Runtime**: 8GB+ RAM recommended; GPU optional (pyannote and Whisper can use CUDA).
- **Models**:
  - pyannote: segmentation and embedding (local or HuggingFace/ModelScope IDs).
  - PLDA: optional, for clustering refinement; requires `xvec_transform.npz`, `plda.npz`.
  - Faster-Whisper: download model (e.g. large-v3) and set path in `core/config.py`.

---

## Quick Start

### 1. Clone and install

```bash
git clone <repository-url>
cd pyannote_diarization
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configuration

- **core/config.py**: `WHISPER_MODEL_PATH`, `EMBEDDING_MODEL_PATH`, `CONFIG_PATH`, `PLDA_DIR`, `DATA_DIR`.
- **config.yaml**: pyannote pipeline and clustering (segmentation, embedding, plda, clustering). See table below.
- **config/llm_model.env** (for Part II): Set `DASHSCOPE_API_KEY` (or your LLM API key).

If using pyannote from Hugging Face, accept model terms and set `HF_TOKEN`.

### 3. Run the server

```bash
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

### 4. Verify

```bash
curl http://127.0.0.1:8001/health
```

Open **http://127.0.0.1:8001/live** for the Live transcription page (microphone + WebSocket).

---

## Configuration

### Part I: Voice & Transcription

- **core/config.py**: `WHISPER_MODEL_PATH`, `EMBEDDING_MODEL_PATH`, `CONFIG_PATH`, `PLDA_DIR`, `DATA_DIR`.
- **config.yaml**: pyannote pipeline and clustering.

| Option | Description |
|--------|-------------|
| `pipeline.params.segmentation` | Segmentation model path or HuggingFace ID |
| `pipeline.params.embedding` | Embedding model path or HF ID |
| `pipeline.params.embedding_batch_size` | Embedding batch size |
| `pipeline.params.plda.checkpoint` | PLDA dir (xvec_transform.npz, plda.npz) |
| `params.clustering.threshold` | Cosine distance threshold for merging |
| `params.clustering.min_cluster_size` | Min samples per cluster |

### Part II: Transcript Refinement (LLM)

- **config/llm_model.env**: `DASHSCOPE_API_KEY` (or your LLM provider key).
- **core/llm/llm_client.py**: `get_llm()` defaults to qwen-plus and DashScope-compatible endpoint; override model/api_base if needed.

---

## API Reference

### Health

- **GET** `/health`  
- **Response**: `200 OK`, simple health payload.

---

### Part I: Voice & Transcription

#### POST /embeddings

Upload student ID, name and audio; returns voice embedding.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| student_id | string | yes | Student ID |
| name | string | no | Name |
| audio | file | yes | Audio file (WAV/MP3 etc.) |

**Response** `200 OK`:

```json
{
  "student_id": "2021001",
  "name": "Zhang San",
  "embedding": [-0.13, 0.12, ...],
  "embedding_dim": 256
}
```

---

#### POST /transcriptions

Submit speaker list and audio; returns full transcript (synchronous).

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| student_id | string[] | yes | List of student IDs, aligned with name and embedding |
| name | string[] | yes | List of names |
| embedding | string[] | yes | JSON-array strings per speaker, e.g. `["[-0.1,0.2,...]","[...]"]` |
| audio | file | yes | Audio file |
| language | string | no | Language code (e.g. zh, en) |

**Response** `200 OK`:

```json
{
  "utterances": [
    {
      "start": 3.0,
      "end": 3.6,
      "speaker": "peppa",
      "student_id": "2021001",
      "text": "I am Peppa"
    }
  ]
}
```

---

#### POST /transcriptions/stream

Same parameters as `POST /transcriptions`; response is an **SSE stream**: one `data:` line per completed utterance (JSON), then a final line with `status: "done"` and `diarization_seconds`, `whisper_seconds`, etc.

**Example events**:

```
data: {"start": 3.0, "end": 3.6, "speaker": "peppa", "student_id": "2021001", "text": "I am Peppa", "index": 1, "total": 111, "progress": 0.9}
data: {"status": "done", "total": 111, "progress": 100, "diarization_seconds": 12.34, "whisper_seconds": 89.56}
```

---

#### WebSocket /ws/transcriptions/live

Live transcription: client sends audio chunks; server returns transcript per chunk. If `speakers` is sent in `init`, diarization + speaker matching is applied; otherwise Whisper-only.

**Message types**:

| Direction | type | Description |
|-----------|------|-------------|
| Client → | `init` | `{ "type": "init", "language": "zh", "speakers": [{"student_id","name","embedding"}, ...] }`, speakers optional |
| Server ← | `ready` | `{ "type": "ready", "language": "zh", "has_speakers": true/false }` |
| Client → | `audio` | `{ "type": "audio", "data": "<base64 WAV>", "chunk_index": 1 }` |
| Server ← | `transcript` | `{ "type": "transcript", "utterances": [...], "text": "...", "chunk_index": 1 }` |
| Client → | `end` | `{ "type": "end" }` |
| Server ← | `done` | `{ "type": "done", "total_chunks": N }` |

Use WAV chunks; 5–15 seconds per chunk recommended.

---

### Part II: Transcript Refinement (LLM)

#### POST /refinements

Refine an existing transcript: infer unknown speakers, merge fragments, correct text and punctuation, split by sentence, filter meaningless. Request body is JSON.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| utterances | array | yes | Transcript list; each item has `start`, `end`, `speaker`, `student_id`, `text` |
| infer_speakers | bool | no | Whether to infer unknown speakers; default true |
| merge | bool | no | Whether to merge same-speaker fragments; default true |
| correct_text | bool | no | Whether to correct and punctuate; default true |
| context_size | int | no | Context size (sentences) for speaker inference; default 3 |

**Response** `200 OK`:

```json
{
  "utterances": [
    { "start": 3.0, "end": 5.2, "speaker": "peppa", "student_id": "T2", "text": "I am Peppa. This is my brother George." }
  ]
}
```

Output is guaranteed to have a speaker for every utterance (no `unknown`) and one utterance per sentence (split by sentence-ending punctuation).

---

### Errors

Unsupported format, empty audio, embedding/transcription failures, etc. return appropriate HTTP status and JSON body (see `utils/errors.py`).

---

## Project Structure

```
pyannote_diarization/
├── app.py                     # FastAPI entry; mounts audio / llm / health routers
├── config.yaml                # [Part I] pyannote pipeline and clustering
├── speakers.py                # [Part I] SpeakerRegistry, cosine matching
├── core/
│   ├── config.py              # [Part I] Paths and model paths
│   ├── pipeline.py            # [Part I] DiarizedTranscriber: diarize → match → Whisper
│   ├── diarize.py             # [Part I] pyannote Pipeline wrapper
│   ├── embedding.py           # [Part I] Voice embedding (Wespeaker)
│   ├── transcribe.py          # [Part I] Faster-Whisper wrapper
│   └── llm/                   # [Part II]
│       ├── llm_client.py      # LLM client (DashScope-compatible)
│       └── refine_pipeline.py # Refine pipeline: infer, merge, correct, split
├── prompts/                   # [Part II] Prompt templates
│   ├── refine.py              # Entry
│   ├── infer_speaker.py       # Speaker inference (few-shot)
│   └── correct_text.py        # Correction & punctuation (few-shot)
├── routers/
│   ├── audio/                 # [Part I] /embeddings, /transcriptions, /transcriptions/stream, /ws/transcriptions/live
│   └── llm/                   # [Part II] /refinements
├── utils/
│   ├── common.py
│   └── errors.py
├── config/
│   └── llm_model.env          # [Part II] LLM API key etc.
├── scripts/test/
│   ├── test_transcribe.py
│   ├── test_transcribe_stream.py
│   ├── test_live_transcribe.py
│   └── test_refine.py         # [Part II] Refine pipeline (reads data/json/transcribe_output.json)
├── static/
│   └── live.html
└── requirements.txt
```

---

## Development & Testing

```bash
# With venv activated
# Part I: Voice & Transcription
python scripts/test/test_transcribe_stream.py   # Streaming
python scripts/test/test_transcribe.py          # Batch
python scripts/test/test_live_transcribe.py     # WebSocket Live

# Part II: Refinement (requires config/llm_model.env)
python scripts/test/test_refine.py              # Refine pipeline (data/json/transcribe_output.json → refined)
```

Use `uvicorn app:app --reload` during development for auto-reload.

---

## Contributing

1. Fork the repository  
2. Create a feature branch (`git checkout -b feat/xxx`)  
3. Commit changes (`git commit -m 'feat: xxx'`)  
4. Push to the branch (`git push origin feat/xxx`)  
5. Open a Pull Request  

---

## License

[MIT](LICENSE)

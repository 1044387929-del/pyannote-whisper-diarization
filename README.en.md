# Speaker Diarization & Transcription API

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Multi-speaker speech transcription and speaker recognition service built on **pyannote.audio** and **Faster-Whisper**. Provides voice embedding extraction, batch/streaming transcription, and WebSocket live transcription, with optional speaker identification by student ID and name.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Transcription Pipeline](#transcription-pipeline)
- [Project Structure](#project-structure)
- [Development & Testing](#development--testing)
- [Contributing](#contributing)
- [License](#license)

---

## Features

| Feature | Description |
|--------|-------------|
| **Voice embedding** | Upload student ID + name + audio; returns a 256-dim embedding vector for speaker registration |
| **Batch transcription** | Submit speaker list (student_id, name, embedding) + audio; returns full transcript JSON with speaker labels |
| **Streaming transcription** | Same params as batch; results streamed via **Server-Sent Events (SSE)** with progress and timing |
| **Live transcription** | **WebSocket**: send audio chunks, receive near real-time transcripts; pass `speakers` in `init` for diarization + speaker matching, otherwise Whisper-only |

- Supports WAV, MP3, WebM and other common formats (via pydub).
- Speaker matching uses cosine similarity with configurable threshold.
- Streaming endpoint is suitable for long audio; results can be displayed incrementally.

---

## Architecture

### Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           FastAPI (app.py)                               │
│  /embedding │ /transcribe │ /transcribe/stream │ /ws/live-transcribe     │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     core/pipeline.py (DiarizedTranscriber)               │
│  1. Diarization  2. Embedding  3. Speaker match  4. Whisper ASR         │
└─────────────────────────────────────────────────────────────────────────┘
         │                    │                    │                    │
         ▼                    ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ core/diarize │    │core/embedding │    │  speakers   │    │core/transcribe│
│ pyannote     │    │ Wespeaker     │    │ Registry    │    │Faster-Whisper │
│ Pipeline     │    │ 256-dim       │    │ cosine match│    │ ASR           │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

### Data Flow (Transcription)

1. **Input**: Audio file (or WebSocket chunks) + optional speaker list `[{ student_id, name, embedding }, ...]`.
2. **Speaker diarization**: pyannote.audio runs segmentation + clustering, outputting segments `(start, end, speaker_label)` with anonymous labels (e.g. SPEAKER_00).
3. **Embedding extraction**: Each segment is passed through the same Wespeaker-style model to get a 256-dim embedding.
4. **Speaker matching**: Segment embeddings are compared to user-provided speaker embeddings via cosine similarity; above-threshold matches get student_id/name, else `unknown`.
5. **ASR**: Faster-Whisper transcribes each segment to text.
6. **Output**: Combined into utterances `{ start, end, speaker, student_id, text }`; streamed or sent over WebSocket per sentence or per chunk.

### Tech Stack

| Component | Technology | Notes |
|-----------|------------|--------|
| Web | FastAPI | REST, SSE, WebSocket |
| Diarization | pyannote.audio ≥3.1 | Segmentation + clustering + optional PLDA |
| Embedding | Wespeaker (e.g. voxceleb-resnet34-LM) | 256-dim speaker embedding |
| ASR | Faster-Whisper | Streaming/offline, multilingual |
| Matching | Custom (speakers.py) | Cosine similarity + threshold, SpeakerRegistry |

---

## Requirements

- **Python**: 3.10 or 3.12 (recommended; tested on 3.12).
- **Runtime**: 8GB+ RAM recommended; GPU optional (pyannote and Whisper can use CUDA).
- **Models**:
  - pyannote: segmentation and embedding checkpoints (local paths or HuggingFace/ModelScope IDs).
  - PLDA: optional, for clustering refinement; requires `xvec_transform.npz` and `plda.npz`.
  - Faster-Whisper: download a model (e.g. large-v3) and set path in `core/config.py`.

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

- In **`core/config.py`** set:
  - `WHISPER_MODEL_PATH`: Faster-Whisper model directory.
  - `EMBEDDING_MODEL_PATH`: optional override for embedding model.
- In **`config.yaml`** set:
  - `pipeline.params.segmentation` / `embedding`: pyannote model paths or HuggingFace IDs.
  - `pipeline.params.plda.checkpoint`: PLDA directory if used.
  - `params.clustering.threshold`, `min_cluster_size`, etc. as needed.

If using pyannote models from Hugging Face, accept the model terms and set `HF_TOKEN` (see pyannote docs).

### 3. Run the server

```bash
uvicorn app:app --host 0.0.0.0 --port 8001 --reload
```

### 4. Verify

```bash
curl http://127.0.0.1:8001/health
```

Open **http://127.0.0.1:8001/live** in a browser for the Live transcription page (microphone + WebSocket).

---

## Configuration

### core/config.py

| Variable | Description |
|----------|-------------|
| `BASE_DIR` | Project root (auto) |
| `CONFIG_PATH` | Path to config.yaml |
| `PLDA_DIR` | Local PLDA directory (optional) |
| `EMBEDDING_MODEL_PATH` | Embedding model path (.bin or dir) |
| `WHISPER_MODEL_PATH` | Faster-Whisper model directory |
| `DATA_DIR` | Data directory (optional) |

### config.yaml

| Option | Description |
|--------|-------------|
| `pipeline.params.segmentation` | Segmentation model path or HuggingFace ID |
| `pipeline.params.embedding` | Embedding model path or HF ID |
| `pipeline.params.embedding_batch_size` | Embedding batch size |
| `pipeline.params.segmentation_batch_size` | Segmentation batch size |
| `pipeline.params.plda.checkpoint` | PLDA dir (must contain xvec_transform.npz, plda.npz) |
| `params.segmentation.min_duration_off` | Min silence between speakers (seconds) |
| `params.clustering.method` | Clustering method (e.g. centroid) |
| `params.clustering.threshold` | Cosine distance threshold for merging clusters |
| `params.clustering.min_cluster_size` | Min samples per cluster |

---

## API Reference

### Health

- **GET** `/health`  
- **Response**: `200 OK`, simple health payload.

---

### POST /embedding

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

### POST /transcribe

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

### POST /transcribe/stream

Same parameters as `/transcribe`; response is an **SSE stream**: one `data:` line per completed utterance (JSON object), then a final summary line with `status: "done"` and `diarization_seconds`, `whisper_seconds`, etc.

**Example events**:

```
data: {"start": 3.0, "end": 3.6, "speaker": "peppa", "student_id": "2021001", "text": "I am Peppa", "index": 1, "total": 111, "progress": 0.9}
data: {"status": "done", "total": 111, "progress": 100, "diarization_seconds": 12.34, "whisper_seconds": 89.56}
```

---

### WebSocket /ws/live-transcribe

Live transcription: client sends audio chunks; server returns transcript for each chunk. If `speakers` is sent in `init`, diarization + speaker matching is applied to the chunk; otherwise Whisper-only.

**Message types**:

| Direction | type | Description |
|-----------|------|-------------|
| Client → | `init` | `{ "type": "init", "language": "zh", "speakers": [{"student_id","name","embedding"}, ...] }`, speakers optional |
| Server ← | `ready` | `{ "type": "ready", "language": "zh", "has_speakers": true/false }` |
| Client → | `audio` | `{ "type": "audio", "data": "<base64 WAV>", "chunk_index": 1 }` |
| Server ← | `transcript` | `{ "type": "transcript", "utterances": [...], "text": "...", "chunk_index": 1 }` |
| Client → | `end` | `{ "type": "end" }` |
| Server ← | `done` | `{ "type": "done", "total_chunks": N }` |

Use WAV chunks; 5–15 seconds per chunk is recommended.

---

### Errors

Unsupported format, empty audio, embedding/transcription failures, etc. return appropriate HTTP status and JSON body (see `utils/errors.py`).

---

## Transcription Pipeline

1. **Diarization** (core/diarize.py): Load pyannote Pipeline from config.yaml; run segmentation → embedding → clustering (+ optional PLDA) to get an `Annotation` of speaker segments.
2. **Speaker matching** (speakers.py): Extract embedding per segment with the same model; cosine similarity against request speakers; assign `speaker` / `student_id` when above threshold.
3. **ASR** (core/transcribe.py): Faster-Whisper transcribes each segment by time range.
4. **Output**: Build `Utterance(start, end, speaker, student_id, text)` list; for stream/WebSocket, emit per sentence or per chunk with progress and timing.

---

## Project Structure

```
pyannote_diarization/
├── app.py                  # FastAPI app: routes, /embedding, /transcribe, SSE, WebSocket
├── config.yaml              # pyannote pipeline and clustering config
├── speakers.py              # SpeakerRegistry: build list + cosine match
├── core/
│   ├── __init__.py
│   ├── config.py            # Paths and model paths
│   ├── pipeline.py          # DiarizedTranscriber: diarize → match → Whisper
│   ├── diarize.py           # pyannote Pipeline wrapper
│   ├── embedding.py         # Voice embedding (Wespeaker)
│   └── transcribe.py        # Faster-Whisper wrapper
├── utils/
│   ├── common.py            # Audio suffix, webm→wav, formatting
│   └── errors.py            # HTTP error helpers
├── scripts/test/
│   ├── test_transcribe.py
│   ├── test_transcribe_stream.py
│   └── test_live_transcribe.py
├── static/
│   └── live.html            # Live transcription frontend
└── requirements.txt
```

---

## Development & Testing

```bash
# With venv activated
python scripts/test/test_transcribe_stream.py   # Streaming
python scripts/test/test_transcribe.py          # Batch
python scripts/test/test_live_transcribe.py     # WebSocket Live
```

Use `uvicorn app:app --reload` during development for auto-reload on code changes.

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

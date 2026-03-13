# 测试脚本

需先启动服务：`uvicorn app:app --host 0.0.0.0 --port 8001`（在项目根目录执行）。

## 目录结构

- **data/** — 测试用数据（与项目根 `data/` 分离，避免混用）  
  - `data/json/` — 声纹、转录/精修输入输出 JSON（如 `speakers_embedding.json`、`transcribe_output.json` 等）  
  - `data/audio/` — 测试音频（如 `audio_all.wav`）  
  - 若目录为空，可从项目根 `data/json/`、`data/audio/` 拷贝所需文件到此处。

- **api/** — 单接口测试  
  - `test_transcribe.py` — POST /transcriptions  
  - `test_transcribe_stream.py` — POST /transcriptions（stream=true）  
  - `test_live_transcribe.py` — WebSocket /ws/transcriptions/live  
  - `test_client_disconnect.py` — 客户端断开时取消转录  
  - `test_refine.py` / `test_refine_api.py` — POST /refinements  
  - `test_label_api.py` — POST /labels（流式标注）

- **pipeline/** — 流水线测试  
  - `test_shallow_pipeline.py` — 浅度流水线：转录 → 精修 → 标注  
  - `shallow_pipeline_config.yaml` — 流水线配置（路径相对 `scripts/test/data`、stream、输出等）

## 运行方式（在项目根目录）

```bash
# 单接口
python scripts/test/api/test_transcribe.py
python scripts/test/api/test_label_api.py
# ...

# 流水线（按 shallow_pipeline_config.yaml 配置）
python scripts/test/pipeline/test_shallow_pipeline.py
```

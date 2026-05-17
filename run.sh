#!/usr/bin/env bash
# 使用 uv 管理依赖并启动服务（默认 :8001）
set -euo pipefail
cd "$(dirname "$0")"

export PATH="${HOME}/.local/bin:${PATH}"
if ! command -v uv >/dev/null 2>&1; then
  echo "未找到 uv，请先安装: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

uv sync
exec uv run uvicorn app:app --host 0.0.0.0 --port 8001 --reload

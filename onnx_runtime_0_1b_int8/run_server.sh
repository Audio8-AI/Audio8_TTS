#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  VENV_PYTHON="$ROOT/.venv/bin/python"
elif [[ -x "$ROOT/.venv/Scripts/python.exe" ]]; then
  VENV_PYTHON="$ROOT/.venv/Scripts/python.exe"
else
  echo "Virtual environment not found. Run setup.sh or setup.ps1 first." >&2
  exit 1
fi
export ARKTTS_MODEL_DIR="${ARKTTS_MODEL_DIR:-$ROOT/model}"
export ARKTTS_VOICES_DIR="${ARKTTS_VOICES_DIR:-$ROOT/voices}"
export ARKTTS_REGISTRATION_DIR="${ARKTTS_REGISTRATION_DIR:-$ARKTTS_MODEL_DIR/registration}"
export ARKTTS_PRECISION="${ARKTTS_PRECISION:-int8}"
export ARKTTS_CODEC_PRECISION="${ARKTTS_CODEC_PRECISION:-fp16}"
export ARKTTS_THREADS="${ARKTTS_THREADS:-5}"

exec "$VENV_PYTHON" -m uvicorn arktts_runtime.service:app \
  --app-dir "$ROOT" \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-8024}"

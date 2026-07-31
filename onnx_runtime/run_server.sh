#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
export ARKTTS_MODEL_DIR="${ARKTTS_MODEL_DIR:-$ROOT/model}"
export ARKTTS_VOICES_DIR="${ARKTTS_VOICES_DIR:-$ROOT/voices}"
export ARKTTS_REGISTRATION_DIR="${ARKTTS_REGISTRATION_DIR:-$ARKTTS_MODEL_DIR/registration}"
export ARKTTS_PRECISION="int4"
export ARKTTS_CODEC_PRECISION="fp16"
export ARKTTS_THREADS="${ARKTTS_THREADS:-5}"

exec "$ROOT/.venv/bin/uvicorn" arktts_runtime.service:app \
  --app-dir "$ROOT" \
  --host "${HOST:-127.0.0.1}" \
  --port "${PORT:-8024}"

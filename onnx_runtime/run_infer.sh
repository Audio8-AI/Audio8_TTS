#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
exec "$ROOT/.venv/bin/python" -m arktts_runtime.cli \
  --model-dir "${ARKTTS_MODEL_DIR:-$ROOT/model}" \
  --voices-dir "${ARKTTS_VOICES_DIR:-$ROOT/voices}" \
  "$@"

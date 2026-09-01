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
exec "$VENV_PYTHON" -m arktts_runtime.cli \
  --model-dir "${ARKTTS_MODEL_DIR:-$ROOT/model}" \
  --voices-dir "${ARKTTS_VOICES_DIR:-$ROOT/voices}" \
  "$@"

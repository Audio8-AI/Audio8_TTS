#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python was not found. Set PYTHON_BIN to a Python 3.11+ executable." >&2
  exit 1
fi

if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(sys.version_info < (3, 11))'; then
  echo "Python 3.11 or newer is required." >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$ROOT/.venv"
"$ROOT/.venv/bin/python" -m pip install --upgrade pip
"$ROOT/.venv/bin/python" -m pip install -r "$ROOT/requirements.txt"
# Only the local Rust tokenizer runtime is needed. Its optional Hub dependency
# tree (huggingface-hub, httpx, hf-xet, and friends) is not used here.
"$ROOT/.venv/bin/python" -m pip install --no-deps tokenizers==0.22.2
echo "Environment ready: $ROOT/.venv"

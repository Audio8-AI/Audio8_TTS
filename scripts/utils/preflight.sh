#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

"$PYTHON" - <<'PY'
import importlib.metadata
import os

for name in ("torch", "transformers", "deepspeed", "numpy", "safetensors", "soundfile"):
    print(f"{name}: {importlib.metadata.version(name)}")

import fish_speech
print(f"fish_speech: {fish_speech.__file__}")
print(f"FISH_SPEECH_ROOT: {os.environ['FISH_SPEECH_ROOT']}")
PY

if [[ -n "${HOSTFILE:-}" ]]; then
  require_file "$HOSTFILE"
  echo "hostfile: $HOSTFILE"
fi

echo "preflight: OK"

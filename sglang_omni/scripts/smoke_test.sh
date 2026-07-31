#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8010}"
OUTPUT="${OUTPUT:-/tmp/audio8_sglang_omni_smoke.wav}"

curl -sS \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "audio8/tts-0.6b",
    "input": "你好，这是 Audio8 TTS 的 SGLang Omni 推理测试。",
    "response_format": "wav",
    "max_new_tokens": 128,
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 50
  }' \
  "${BASE_URL}/v1/audio/speech" \
  -o "${OUTPUT}"

magic="$(od -An -c -N4 "${OUTPUT}" | tr -d ' ')"
if [[ "${magic}" != "RIFF" ]]; then
  echo "Unexpected response header: ${magic}" >&2
  exit 1
fi
wc -c "${OUTPUT}"

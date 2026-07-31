#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$ROOT/.service.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No managed Audio8 TTS service is running."
  exit 0
fi

PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  for _ in {1..30}; do
    if ! kill -0 "$PID" 2>/dev/null; then
      break
    fi
    sleep 1
  done
fi
rm -f "$PID_FILE"
echo "Audio8 TTS service stopped."

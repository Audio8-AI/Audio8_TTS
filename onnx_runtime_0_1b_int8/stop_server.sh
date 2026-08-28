#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$ROOT/.service.pid"
if [[ ! -f "$PID_FILE" ]]; then
  echo "Service is not running"
  exit 0
fi
PID="$(<"$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  echo "Stopped Audio8 0.1B TTS (PID $PID)"
else
  echo "Service PID $PID is not running"
fi
rm -f "$PID_FILE"

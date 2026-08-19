#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PID_FILE="$ROOT/.service.pid"
PORT="${PORT:-8024}"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Service is already running with PID $(cat "$PID_FILE")"
  exit 0
fi

PORT="$PORT" nohup "$ROOT/run_server.sh" >"$ROOT/service.log" 2>&1 &
PID=$!
echo "$PID" >"$PID_FILE"

for _ in {1..60}; do
  if curl -fsS "http://127.0.0.1:$PORT/api/health" >/dev/null 2>&1; then
    echo "Audio8 TTS is ready: http://127.0.0.1:$PORT"
    exit 0
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "Service exited during startup. See $ROOT/service.log" >&2
    exit 1
  fi
  sleep 1
done

echo "Service did not become ready in 60 seconds. See $ROOT/service.log" >&2
exit 1

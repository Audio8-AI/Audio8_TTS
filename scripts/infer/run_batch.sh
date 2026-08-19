#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"

MODEL="${MODEL:?Set MODEL to the Audio8 0.1B export path or Hugging Face model ID}"
INPUT_JSONL="${INPUT_JSONL:-$PROJECT_ROOT/examples/inference.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/inference_outputs}"
DEVICE="${DEVICE:-cuda}"
if [[ "$DEVICE" == cuda* ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU:-0}}"
fi

[[ -f "$INPUT_JSONL" ]] || { echo "Input JSONL not found: $INPUT_JSONL" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"

exec "$PYTHON" "$PROJECT_ROOT/scripts/infer/batch_infer_arktts.py" \
  --model "$MODEL" \
  --input-jsonl "$INPUT_JSONL" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --batch-size "${BATCH_SIZE:-1}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-512}" \
  --temperature "${TEMPERATURE:-0.7}" \
  --top-p "${TOP_P:-0.9}" \
  --top-k "${TOP_K:-50}" \
  --seed "${SEED:-42}" \
  "$@"

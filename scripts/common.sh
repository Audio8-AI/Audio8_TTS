#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

PYTHON="${PYTHON:-python3}"
FISH_SPEECH_ROOT="${FISH_SPEECH_ROOT:-}"
AUDIO8_ROOT="${AUDIO8_ROOT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/outputs}"
EXPORT_ROOT="${EXPORT_ROOT:-$PROJECT_ROOT/exports}"
CACHE_ROOT="${CACHE_ROOT:-${TMPDIR:-/tmp}/audio8-falcon-h1-cache}"
RUNTIME_ROOT="${RUNTIME_ROOT:-${TMPDIR:-/tmp}/audio8-falcon-h1-runtime}"

if [[ -z "$FISH_SPEECH_ROOT" || ! -d "$FISH_SPEECH_ROOT/fish_speech" ]]; then
  echo "FISH_SPEECH_ROOT must point to a fish-speech checkout" >&2
  exit 1
fi

export PROJECT_ROOT PYTHON FISH_SPEECH_ROOT AUDIO8_ROOT OUTPUT_ROOT EXPORT_ROOT CACHE_ROOT RUNTIME_ROOT
export PYTHONPATH="$PROJECT_ROOT/src:$FISH_SPEECH_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="${TMPDIR:-$RUNTIME_ROOT}"
export TEMP="${TEMP:-$TMPDIR}"
export TMP="${TMP:-$TMPDIR}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$TMPDIR/torch_extensions}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$TMPDIR/triton_cache}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$TMPDIR/hf_modules}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export USE_LIBUV="${USE_LIBUV:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0,bond0,enp,ens,eno}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export FISH_DISABLE_EMBED_SCALE="${FISH_DISABLE_EMBED_SCALE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Triton and CUDA extensions need the active Python environment's shared
# libraries on the loader path. Set PYTHON_LIB_DIR explicitly when using a
# non-standard environment layout.
PYTHON_LIB_DIR="${PYTHON_LIB_DIR:-$(cd "$(dirname "$PYTHON")/../lib" 2>/dev/null && pwd)}"
if [[ -d "$PYTHON_LIB_DIR" ]]; then
  export LD_LIBRARY_PATH="$PYTHON_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

mkdir -p "$OUTPUT_ROOT" "$EXPORT_ROOT" "$CACHE_ROOT" "$RUNTIME_ROOT" \
  "$TMPDIR" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$HF_MODULES_CACHE"

require_file() {
  [[ -f "$1" ]] || { echo "Required file not found: $1" >&2; exit 1; }
}

require_dir() {
  [[ -d "$1" ]] || { echo "Required directory not found: $1" >&2; exit 1; }
}

#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <sglang-omni source root or sglang_omni package directory>" >&2
  exit 2
fi

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="${BUNDLE_ROOT}/adapter/sglang_omni/models/audio8_tts"
TARGET="$1"

if [[ -d "${TARGET}/sglang_omni/models" ]]; then
  PACKAGE_DIR="${TARGET}/sglang_omni"
elif [[ -d "${TARGET}/models" && -f "${TARGET}/__init__.py" ]]; then
  PACKAGE_DIR="${TARGET}"
else
  echo "Cannot find sglang_omni/models under: ${TARGET}" >&2
  exit 1
fi

DEST_DIR="${PACKAGE_DIR}/models/audio8_tts"
mkdir -p "${DEST_DIR}"
cp -a "${SOURCE_DIR}/." "${DEST_DIR}/"

echo "Installed Audio8 adapter to: ${DEST_DIR}"
echo "No SGLang-Omni core files were modified."

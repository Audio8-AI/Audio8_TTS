#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path)
    args = parser.parse_args()

    from sglang_omni.models.audio8_tts.config import Audio8TTSPipelineConfig
    from sglang_omni.models.audio8_tts.factory import make_server_args
    from sglang_omni.models.audio8_tts.sglang_model import (
        Audio8SGLangModel,
        FastAttention,
    )

    assert Audio8TTSPipelineConfig.architecture == "ArkttsModel"
    assert Audio8SGLangModel.__name__ == "Audio8SGLangModel"
    assert callable(FastAttention.forward_audio8_cached)
    assert callable(make_server_args)

    if args.model_path is not None:
        required = {
            "config.json",
            "model.safetensors",
            "codec.pth",
            "tokenizer.json",
            "tokenizer_config.json",
            "modeling_arktts_codec.py",
        }
        missing = sorted(name for name in required if not (args.model_path / name).is_file())
        if missing:
            raise SystemExit(f"Model directory is missing: {missing}")
        config = json.loads((args.model_path / "config.json").read_text())
        if config.get("architectures") != ["ArkttsModel"]:
            raise SystemExit("config.json architectures must contain ArkttsModel")

    print("Audio8 SGLang-Omni adapter import check passed")


if __name__ == "__main__":
    main()

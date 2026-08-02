# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import (
    ExecutorConfig,
    PipelineConfig,
    RelayConfig,
    StageConfig,
)
from sglang_omni.config.schema import StreamTargetConfig

_PKG = "sglang_omni.models.audio8_tts.pipeline"


class Audio8TTSPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "ArkttsModel"

    model_path: str
    entry_stage: str = "preprocessing"
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            executor=ExecutorConfig(
                factory=f"{_PKG}.stages.create_preprocessing_executor",
                args={"device": "cuda:0"},
            ),
            get_next=f"{_PKG}.next_stage.preprocessing_next",
            relay=RelayConfig(device="cpu"),
        ),
        StageConfig(
            name="tts_engine",
            executor=ExecutorConfig(
                factory=f"{_PKG}.stages.create_sglang_tts_engine_executor",
                args={"device": "cuda:0", "max_new_tokens": 1024},
            ),
            get_next=f"{_PKG}.next_stage.tts_engine_next",
            relay=RelayConfig(device="cuda"),
            stream_to=[StreamTargetConfig(to_stage="vocoder")],
        ),
        StageConfig(
            name="vocoder",
            executor=ExecutorConfig(
                factory=f"{_PKG}.stages.create_vocoder_executor",
                args={"device": "cuda:0"},
            ),
            get_next=f"{_PKG}.next_stage.vocoder_next",
            relay=RelayConfig(device="cpu"),
        ),
    ]


EntryClass = Audio8TTSPipelineConfig

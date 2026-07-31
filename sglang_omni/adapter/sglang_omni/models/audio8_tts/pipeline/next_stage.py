# SPDX-License-Identifier: Apache-2.0
from typing import Any


def preprocessing_next(request_id: str, output: Any) -> str:
    del request_id, output
    return "tts_engine"


def tts_engine_next(request_id: str, output: Any) -> str:
    del request_id, output
    return "vocoder"


def vocoder_next(request_id: str, output: Any) -> None:
    del request_id, output
    return None

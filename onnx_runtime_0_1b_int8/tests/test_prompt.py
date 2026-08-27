from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("tokenizers")

from arktts_runtime.prompt import clean_text, format_reference_text


def test_text_normalization_preserves_special_tokens() -> None:
    assert clean_text("\x00 你好\n世界 ") == "你好世界"
    assert clean_text("你好\nworld") == "你好 world"
    assert format_reference_text("你好") == "<|speaker:0|>你好"
    assert format_reference_text("<|speaker:2|>hello") == "<|speaker:2|>hello"


def test_sample_state_update_handles_full_and_delta_cache() -> None:
    from arktts_runtime.runtime import ArkTtsRuntime

    cache = np.zeros((1, 2, 10, 64), dtype=np.float32)
    delta = np.ones_like(cache[:, :, :1])
    ArkTtsRuntime._update_fast_cache(cache, delta, 3)
    assert np.all(cache[:, :, 3:4] == 1)
    replacement = np.full_like(cache, 2)
    ArkTtsRuntime._update_fast_cache(cache, replacement, 0)
    assert np.all(cache == 2)

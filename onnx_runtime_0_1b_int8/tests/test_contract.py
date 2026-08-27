from __future__ import annotations

import os
from pathlib import Path

import pytest

ort = pytest.importorskip("onnxruntime")

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.getenv("ARKTTS_MODEL_DIR", str(ROOT / "model")))


def _session(name: str):
    path = MODEL_DIR / name
    if not path.is_file():
        pytest.skip(f"model file is not available: {path}")
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def test_slow_hybrid_contract() -> None:
    session = _session("slow_ar_int8.onnx")
    inputs = {item.name: item for item in session.get_inputs()}
    assert set(inputs) == {
        "codes",
        "position",
        "cache_keys",
        "cache_values",
        "conv_states",
        "ssm_states",
    }
    assert tuple(inputs["codes"].shape) == (1, 11, 1)
    assert tuple(inputs["position"].shape) == (1,)
    assert tuple(inputs["cache_keys"].shape) == (24, 1, 2, 2048, 64)
    assert tuple(inputs["cache_values"].shape) == (24, 1, 2, 2048, 64)
    assert tuple(inputs["conv_states"].shape) == (24, 1, 896, 4)
    assert tuple(inputs["ssm_states"].shape) == (24, 1, 24, 32, 64)

    outputs = {item.name: item for item in session.get_outputs()}
    assert set(outputs) == {
        "logits",
        "hidden",
        "key_delta",
        "value_delta",
        "next_conv_states",
        "next_ssm_states",
    }
    assert tuple(outputs["logits"].shape) == (1, 1, 4097)
    assert tuple(outputs["hidden"].shape) == (1, 1, 512)


def test_fast_and_decoder_contract() -> None:
    fast = _session("fast_ar_int8.onnx")
    fast_inputs = {item.name: item for item in fast.get_inputs()}
    assert {name for name in fast_inputs if name.startswith("cache_key_")} == {
        "cache_key_0",
        "cache_key_1",
        "cache_key_2",
        "cache_key_3",
    }
    assert {name for name in fast_inputs if name.startswith("cache_value_")} == {
        "cache_value_0",
        "cache_value_1",
        "cache_value_2",
        "cache_value_3",
    }
    assert tuple(fast_inputs["slow_hidden"].shape) == (1, 1, 512)
    assert tuple(fast_inputs["token_id"].shape) == (1, 1)
    assert tuple(fast_inputs["input_pos"].shape) == (1,)
    assert [item.name for item in fast.get_outputs()] == [
        "logits",
        "key_delta_0",
        "value_delta_0",
        "key_delta_1",
        "value_delta_1",
        "key_delta_2",
        "value_delta_2",
        "key_delta_3",
        "value_delta_3",
    ]
    assert tuple(fast.get_outputs()[0].shape) == (1, 1, 4096)

    decoder = _session("codec_decoder_fp16.onnx")
    assert tuple(decoder.get_inputs()[0].shape[1:2]) == (10,)

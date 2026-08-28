from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from .prompt import PromptBuilder
from .voices import VoiceStore


ORT_DTYPES: dict[str, Any] = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(bool)": np.bool_,
}


def _session(path: Path, threads: int | None = None) -> ort.InferenceSession:
    if not path.is_file():
        raise FileNotFoundError(f"ONNX model was not found: {path}")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    if threads is not None:
        options.intra_op_num_threads = max(1, int(threads))
        options.inter_op_num_threads = max(1, int(threads) // 2)
    return ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def _sample(
    logits: np.ndarray,
    temperature: float,
    top_p: float,
    top_k: int,
    rng: np.random.Generator,
) -> int:
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).any():
        raise ValueError("model returned empty or non-finite logits")
    temperature = max(float(temperature), 1e-5)
    top_p = min(max(float(top_p), 1e-5), 1.0)
    top_k = min(max(int(top_k), 1), values.size)

    order = np.argsort(values)[::-1]
    sorted_values = values[order]
    probabilities = np.exp(sorted_values - np.max(sorted_values))
    probabilities /= probabilities.sum()
    cumulative = np.cumsum(probabilities)
    remove = (cumulative > top_p) | (np.arange(values.size) >= top_k)
    remove[0] = False
    filtered = values.copy()
    filtered[order[remove]] = -np.inf
    scaled = filtered / temperature
    scaled -= np.max(scaled)
    probabilities = np.exp(scaled)
    probabilities /= probabilities.sum()
    # Gumbel-max is stable and makes the RNG sequence explicit for reproducible CLI runs.
    uniform = np.clip(rng.random(probabilities.size), 1e-12, 1.0 - 1e-12)
    gumbel = -np.log(-np.log(uniform))
    return int(np.argmax(np.log(np.clip(probabilities, 1e-300, 1.0)) + gumbel))


def _dtype(value: str) -> np.dtype:
    try:
        return np.dtype(ORT_DTYPES[value])
    except KeyError as exc:
        raise ValueError(f"unsupported ONNX tensor type: {value}") from exc


def _shape(info: ort.NodeArg, fallback: tuple[int, ...]) -> tuple[int, ...]:
    """Resolve symbolic dimensions to the dimensions used by this export."""
    result: list[int] = []
    for index, dimension in enumerate(info.shape):
        if isinstance(dimension, int) and dimension > 0:
            result.append(dimension)
        elif index < len(fallback):
            result.append(int(fallback[index]))
        else:
            raise ValueError(f"cannot resolve dynamic shape for {info.name}: {info.shape}")
    return tuple(result)


class ArkTtsRuntime:
    """CPU runtime for ``Audio8/audio8-TTS-0.1B-ONNX-INT8``.

    The 0.1B graph is a Falcon-H1 hybrid export. Its slow graph consumes one
    ``[1, 11, 1]`` column at a time and returns state deltas; it is not
    compatible with the 0.6B runtime's independent per-layer KV inputs.
    """

    _SLOW_INPUTS = {
        "codes",
        "position",
        "cache_keys",
        "cache_values",
        "conv_states",
        "ssm_states",
    }
    _FAST_INPUTS = {
        "slow_hidden",
        "token_id",
        "use_slow_hidden",
        "input_pos",
        "cache_key_0",
        "cache_key_1",
        "cache_key_2",
        "cache_key_3",
        "cache_value_0",
        "cache_value_1",
        "cache_value_2",
        "cache_value_3",
    }

    def __init__(
        self,
        model_dir: Path,
        voices_dir: Path,
        precision: str | None = None,
        codec_precision: str | None = None,
        threads: int | None = None,
    ):
        self.model_dir = Path(model_dir).resolve()
        manifest_path = self.model_dir / "runtime_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"runtime manifest was not found: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.precision = precision or self.manifest["default_precision"]
        if self.precision not in self.manifest["available_precisions"]:
            raise ValueError(f"unsupported precision: {self.precision}")
        self.codec_precision = codec_precision or self.manifest.get(
            "default_codec_precision", "fp16"
        )
        available_codec = self.manifest.get("available_codec_precisions", ["fp16"])
        if self.codec_precision not in available_codec:
            raise ValueError(f"unsupported codec precision: {self.codec_precision}")

        slow_name = self.manifest.get("slow_decode_models", {}).get(
            self.precision,
            self.manifest.get("slow_decode_model", f"slow_ar_{self.precision}.onnx"),
        )
        fast_name = self.manifest.get("fast_models", {}).get(
            self.precision,
            self.manifest.get("fast_model", f"fast_ar_{self.precision}.onnx"),
        )
        codec_name = self.manifest.get("codec_models", {}).get(
            self.codec_precision,
            f"codec_decoder_{self.codec_precision}.onnx",
        )
        self.slow = _session(self.model_dir / slow_name, threads)
        self.fast = _session(self.model_dir / fast_name, threads)
        self.decoder = _session(self.model_dir / codec_name, threads)

        self.slow_inputs = {item.name: item for item in self.slow.get_inputs()}
        self.fast_inputs = {item.name: item for item in self.fast.get_inputs()}
        self._validate_contract()
        self._slow_outputs = {item.name: index for index, item in enumerate(self.slow.get_outputs())}
        self._fast_outputs = {item.name: index for index, item in enumerate(self.fast.get_outputs())}

        self.prompt_builder = PromptBuilder(
            self.model_dir / "tokenizer",
            int(self.manifest["semantic_begin_id"]),
            int(self.manifest["num_codebooks"]),
        )
        self.voices = VoiceStore(Path(voices_dir), int(self.manifest["num_codebooks"]))

    def _validate_contract(self) -> None:
        slow_names = set(self.slow_inputs)
        missing = self._SLOW_INPUTS - slow_names
        if missing:
            raise ValueError(
                "this runtime expects the 0.1B hybrid slow graph; "
                f"missing inputs: {sorted(missing)}"
            )
        fast_missing = self._FAST_INPUTS - set(self.fast_inputs)
        if fast_missing:
            raise ValueError(f"0.1B fast graph is missing inputs: {sorted(fast_missing)}")
        if tuple(self.slow_inputs["codes"].shape[-2:]) != (11, 1):
            raise ValueError(f"unexpected slow codes shape: {self.slow_inputs['codes'].shape}")
        if tuple(self.slow_inputs["position"].shape) not in ((1,), (None,)):
            raise ValueError(f"unexpected slow position shape: {self.slow_inputs['position'].shape}")
        if int(self.manifest["num_layers"]) != int(self.slow_inputs["cache_keys"].shape[0]):
            raise ValueError("runtime_manifest num_layers does not match slow cache_keys")
        if int(self.manifest["num_fast_layers"]) != 4:
            raise ValueError("the 0.1B fast export must have four cache layers")

    def _empty_slow_state(self) -> dict[str, np.ndarray]:
        specs = {
            "cache_keys": (
                int(self.manifest["num_layers"]),
                1,
                int(self.manifest["n_local_heads"]),
                int(self.manifest["max_seq_len"]),
                int(self.manifest["head_dim"]),
            ),
            "cache_values": (
                int(self.manifest["num_layers"]),
                1,
                int(self.manifest["n_local_heads"]),
                int(self.manifest["max_seq_len"]),
                int(self.manifest["head_dim"]),
            ),
            "conv_states": (
                int(self.manifest["num_layers"]),
                1,
                896,
                int(self.manifest["mamba_d_conv"]),
            ),
            "ssm_states": (
                int(self.manifest["num_layers"]),
                1,
                int(self.manifest["mamba_n_heads"]),
                int(self.manifest["mamba_d_head"]),
                int(self.manifest["mamba_d_state"]),
            ),
        }
        return {
            name: np.zeros(
                _shape(self.slow_inputs[name], fallback),
                dtype=_dtype(self.slow_inputs[name].type),
            )
            for name, fallback in specs.items()
        }

    def _empty_fast_caches(self) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for name, info in self.fast_inputs.items():
            if not (name.startswith("cache_key_") or name.startswith("cache_value_")):
                continue
            fallback = (1, 2, int(self.manifest["num_codebooks"]), int(self.manifest["fast_head_dim"]))
            result[name] = np.zeros(_shape(info, fallback), dtype=_dtype(info.type))
        return result

    @staticmethod
    def _output(outputs: list[np.ndarray], names: dict[str, int], name: str, index: int) -> np.ndarray:
        output_index = names.get(name, index)
        if output_index >= len(outputs):
            raise ValueError(f"ONNX graph did not return expected output {name}")
        return np.asarray(outputs[output_index])

    def _slow_step(
        self,
        codes: np.ndarray,
        position: int,
        state: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        codes = np.asarray(codes, dtype=np.int64)
        if codes.shape != (1, int(self.manifest["num_codebooks"]) + 1, 1):
            raise ValueError(f"slow codes must have shape [1, 11, 1], got {codes.shape}")
        position_value = np.asarray([int(position)], dtype=np.int64)
        feeds = {
            "codes": codes,
            "position": position_value,
            "cache_keys": state["cache_keys"],
            "cache_values": state["cache_values"],
            "conv_states": state["conv_states"],
            "ssm_states": state["ssm_states"],
        }
        outputs = self.slow.run(None, feeds)
        key_delta = self._output(outputs, self._slow_outputs, "key_delta", 2)
        value_delta = self._output(outputs, self._slow_outputs, "value_delta", 3)
        next_conv = self._output(outputs, self._slow_outputs, "next_conv_states", 4)
        next_ssm = self._output(outputs, self._slow_outputs, "next_ssm_states", 5)
        if not 0 <= int(position) < state["cache_keys"].shape[3]:
            raise ValueError(f"slow position {position} exceeds cache length")
        state["cache_keys"][:, :, :, int(position), :] = key_delta
        state["cache_values"][:, :, :, int(position), :] = value_delta
        state["conv_states"] = next_conv.astype(state["conv_states"].dtype, copy=False)
        state["ssm_states"] = next_ssm.astype(state["ssm_states"].dtype, copy=False)
        logits = self._output(outputs, self._slow_outputs, "logits", 0)
        hidden = self._output(outputs, self._slow_outputs, "hidden", 1)
        return logits[0, -1], hidden[:, -1:, :]

    def _fast_step(
        self,
        hidden: np.ndarray,
        token_id: int,
        use_slow_hidden: bool,
        position: int,
        caches: dict[str, np.ndarray],
    ) -> np.ndarray:
        hidden_info = self.fast_inputs["slow_hidden"]
        use_info = self.fast_inputs["use_slow_hidden"]
        feeds: dict[str, np.ndarray] = {
            "slow_hidden": np.asarray(hidden, dtype=_dtype(hidden_info.type)),
            "token_id": np.asarray([[int(token_id)]], dtype=_dtype(self.fast_inputs["token_id"].type)),
            "use_slow_hidden": np.asarray(
                [bool(use_slow_hidden)], dtype=_dtype(use_info.type)
            ),
            "input_pos": np.asarray([int(position)], dtype=_dtype(self.fast_inputs["input_pos"].type)),
        }
        feeds.update(caches)
        outputs = self.fast.run(None, feeds)
        for index in range(4):
            key_name = f"cache_key_{index}"
            value_name = f"cache_value_{index}"
            key_delta = self._output(
                outputs,
                self._fast_outputs,
                f"key_delta_{index}",
                1 + index * 2,
            )
            value_delta = self._output(
                outputs,
                self._fast_outputs,
                f"value_delta_{index}",
                2 + index * 2,
            )
            self._update_fast_cache(caches[key_name], key_delta, position)
            self._update_fast_cache(caches[value_name], value_delta, position)
        logits = self._output(outputs, self._fast_outputs, "logits", 0)
        return logits[0, -1]

    @staticmethod
    def _update_fast_cache(cache: np.ndarray, value: np.ndarray, position: int) -> None:
        value = np.asarray(value, dtype=cache.dtype)
        if value.shape == cache.shape:
            cache[...] = value
            return
        if value.ndim == cache.ndim and value.shape[:2] == cache.shape[:2] and value.shape[3:] == cache.shape[3:]:
            length = value.shape[2]
            cache[:, :, int(position) : int(position) + length, :] = value
            return
        raise ValueError(f"unexpected fast cache output shape {value.shape}; expected {cache.shape}")

    def _sample_semantic(
        self,
        logits: np.ndarray,
        previous: list[int],
        temperature: float,
        top_p: float,
        top_k: int,
        rng: np.random.Generator,
    ) -> int:
        begin = int(self.manifest["semantic_begin_id"])
        end = int(self.manifest["semantic_end_id"])
        stop = int(self.manifest["im_end_id"])
        expected = int(self.manifest["codebook_size"]) + 1
        values = np.asarray(logits, dtype=np.float32).reshape(-1)
        if self.manifest.get("slow_logits_layout") != "relative_semantic_then_eos":
            raise ValueError("this runtime only supports relative_semantic_then_eos logits")
        if values.size != expected:
            raise ValueError(f"unexpected slow logits size: {values.size}, expected {expected}")
        normal_index = _sample(values, temperature, top_p, top_k, rng)
        high_index = _sample(values, 1.0, 0.9, top_k, rng)
        normal = stop if normal_index == expected - 1 else begin + normal_index
        high = stop if high_index == expected - 1 else begin + high_index
        if begin <= normal <= end and normal in previous:
            return high
        return normal

    def iter_codes(
        self,
        text: str,
        voice: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        seed: int = 42,
        stop_event: threading.Event | None = None,
    ) -> Iterator[np.ndarray]:
        reference_codes, meta = self.voices.load(voice)
        prompt = self.prompt_builder.build(text, meta["reference_text"], reference_codes)
        prompt_len = int(prompt.shape[2])
        max_seq_len = int(self.manifest["max_seq_len"])
        if prompt_len >= max_seq_len:
            raise ValueError(f"prompt length {prompt_len} exceeds max sequence length {max_seq_len}")
        max_new_tokens = min(max(0, int(max_new_tokens)), max_seq_len - prompt_len)

        rng = np.random.default_rng(int(seed))
        slow_state = self._empty_slow_state()
        logits: np.ndarray | None = None
        hidden: np.ndarray | None = None
        # The exported hybrid graph is a one-token graph, including prefill.
        for position in range(prompt_len):
            logits, hidden = self._slow_step(prompt[:, :, position : position + 1], position, slow_state)

        previous: list[int] = []
        begin = int(self.manifest["semantic_begin_id"])
        stop = int(self.manifest["im_end_id"])
        codebook_size = int(self.manifest["codebook_size"])
        assert logits is not None and hidden is not None
        for step in range(max_new_tokens):
            if stop_event is not None and stop_event.is_set():
                return
            semantic = self._sample_semantic(logits, previous, temperature, top_p, top_k, rng)
            if semantic == stop:
                return
            previous.append(semantic)
            previous = previous[-10:]

            fast_caches = self._empty_fast_caches()
            self._fast_step(hidden, 0, True, 0, fast_caches)
            first_code = semantic - begin
            if not 0 <= first_code < codebook_size:
                raise ValueError(f"semantic token {semantic} is outside the codebook range")
            codebooks = [first_code]
            token = first_code
            for fast_position in range(1, int(self.manifest["num_codebooks"])):
                fast_logits = self._fast_step(hidden, token, False, fast_position, fast_caches)
                token = _sample(fast_logits, temperature, top_p, top_k, rng)
                codebooks.append(token)
            frame = np.asarray(codebooks, dtype=np.int64)
            yield frame
            if step + 1 >= max_new_tokens:
                return
            column = np.concatenate([[semantic], frame]).reshape(1, -1, 1)
            logits, hidden = self._slow_step(column, prompt_len + step, slow_state)

    def decode_codes(self, codes: np.ndarray) -> np.ndarray:
        values = np.asarray(codes, dtype=np.int64)
        if values.ndim == 2:
            values = values[np.newaxis]
        expected = int(self.manifest["num_codebooks"])
        if values.ndim != 3 or values.shape[1] != expected or values.shape[2] == 0:
            raise ValueError(f"invalid generated codes shape: {values.shape}")
        decoder_input = self.decoder.get_inputs()[0]
        values = values.astype(_dtype(decoder_input.type), copy=False)
        audio = self.decoder.run(None, {decoder_input.name: values})[0]
        return np.asarray(audio, dtype=np.float32).reshape(-1)

    def synthesize(self, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        frames = list(self.iter_codes(**kwargs))
        if not frames:
            raise RuntimeError("model produced no codec frames")
        codes = np.stack(frames, axis=1)
        return self.decode_codes(codes), codes

    def stream(self, chunk_frames: int = 12, **kwargs: Any):
        all_frames: list[np.ndarray] = []
        emitted_samples = 0
        hop = int(self.manifest["codec_hop_length"])
        context = int(self.manifest.get("stream_context_frames", 128))
        guard = int(self.manifest.get("stream_guard_frames", 1)) * hop
        sequence = 0
        for frame in self.iter_codes(**kwargs):
            all_frames.append(frame)
            if len(all_frames) % max(1, int(chunk_frames)) != 0:
                continue
            start_frame = max(0, len(all_frames) - context - int(chunk_frames))
            window = np.stack(all_frames[start_frame:], axis=1)
            audio = self.decode_codes(window)
            absolute_start = start_frame * hop
            stable_end = absolute_start + max(0, audio.size - guard)
            begin = max(0, emitted_samples - absolute_start)
            end = max(begin, stable_end - absolute_start)
            if end > begin:
                chunk = np.ascontiguousarray(audio[begin:end])
                emitted_samples += chunk.size
                yield {
                    "type": "audio_chunk",
                    "seq": sequence,
                    "audio": chunk,
                    "frame_count": len(all_frames),
                }
                sequence += 1
        if not all_frames:
            raise RuntimeError("model produced no codec frames")
        start_frame = max(0, len(all_frames) - context - int(chunk_frames))
        audio = self.decode_codes(np.stack(all_frames[start_frame:], axis=1))
        absolute_start = start_frame * hop
        begin = max(0, emitted_samples - absolute_start)
        if begin < audio.size:
            yield {
                "type": "audio_chunk",
                "seq": sequence,
                "audio": np.ascontiguousarray(audio[begin:]),
                "frame_count": len(all_frames),
            }
        yield {"type": "complete", "codes": np.stack(all_frames, axis=1)}

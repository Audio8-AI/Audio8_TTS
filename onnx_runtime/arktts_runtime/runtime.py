from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .prompt import PromptBuilder
from .voices import VoiceStore

ORT_DTYPES = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(int64)": np.int64,
    "tensor(bool)": np.bool_,
}


def _session(path: Path, threads: int | None = None) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    if threads is not None:
        options.intra_op_num_threads = int(threads)
        options.inter_op_num_threads = max(1, int(threads) // 2)
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def _sample(logits: np.ndarray, temperature: float, top_p: float, top_k: int, rng) -> int:
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    order = np.argsort(values)[::-1]
    sorted_values = values[order]
    base = np.exp(sorted_values - np.max(sorted_values))
    base /= base.sum()
    cumulative = np.cumsum(base)
    remove = (cumulative > float(top_p)) | (np.arange(base.size) >= int(top_k))
    remove[0] = False
    masked = values.copy()
    masked[order[remove]] = -np.inf
    scaled = masked / max(float(temperature), 1e-5)
    scaled -= np.max(scaled)
    probs = np.exp(scaled)
    probs /= probs.sum()
    noise = -np.log(np.clip(rng.random(probs.size), 1e-12, 1.0))
    return int(np.argmax(probs / noise))


class ArkTtsRuntime:
    def __init__(
        self,
        model_dir: Path,
        voices_dir: Path,
        precision: str | None = None,
        codec_precision: str | None = None,
        threads: int | None = None,
    ):
        self.model_dir = Path(model_dir).resolve()
        self.manifest = json.loads((self.model_dir / "runtime_manifest.json").read_text())
        self.precision = precision or self.manifest["default_precision"]
        if self.precision not in self.manifest["available_precisions"]:
            raise ValueError(f"unsupported precision: {self.precision}")
        self.codec_precision = codec_precision or self.manifest.get(
            "default_codec_precision", "fp16"
        )
        available_codec = self.manifest.get("available_codec_precisions", ["fp16"])
        if self.codec_precision not in available_codec:
            raise ValueError(f"unsupported codec precision: {self.codec_precision}")
        self.slow = _session(
            self.model_dir / f"slow_ar_{self.precision}.onnx",
            threads,
        )
        self.fast = _session(
            self.model_dir / f"fast_ar_{self.precision}.onnx",
            threads,
        )
        codec_models = self.manifest.get("codec_models", {"fp16": "codec_decoder_fp16.onnx"})
        self.decoder = _session(
            self.model_dir / codec_models[self.codec_precision],
            threads,
        )
        self.prompt_builder = PromptBuilder(
            self.model_dir / "tokenizer",
            self.manifest["semantic_begin_id"],
            self.manifest["num_codebooks"],
        )
        self.voices = VoiceStore(Path(voices_dir), self.manifest["num_codebooks"])
        self.slow_inputs = {item.name: item for item in self.slow.get_inputs()}
        self.fast_inputs = {item.name: item for item in self.fast.get_inputs()}

    def _empty_slow_caches(self) -> list[np.ndarray]:
        dtype = ORT_DTYPES[self.slow_inputs["cache_key_0"].type]
        shape = (
            1,
            int(self.manifest["n_local_heads"]),
            int(self.manifest["max_seq_len"]),
            int(self.manifest["head_dim"]),
        )
        return [np.zeros(shape, dtype=dtype) for _ in range(2 * int(self.manifest["num_layers"]))]

    def _empty_fast_caches(self) -> list[np.ndarray]:
        dtype = ORT_DTYPES[self.fast_inputs["cache_key_0"].type]
        shape = (
            1,
            int(self.manifest["fast_n_local_heads"]),
            int(self.manifest["num_codebooks"]),
            int(self.manifest["fast_head_dim"]),
        )
        return [
            np.zeros(shape, dtype=dtype) for _ in range(2 * int(self.manifest["num_fast_layers"]))
        ]

    @staticmethod
    def _update_caches(
        caches: list[np.ndarray], positions: np.ndarray, deltas: list[np.ndarray]
    ) -> None:
        for index, delta in enumerate(deltas):
            caches[index][:, :, positions, :] = delta

    def _slow_step(self, codes: np.ndarray, positions: np.ndarray, caches):
        feeds = {"codes": codes.astype(np.int64), "input_pos": positions.astype(np.int64)}
        for i in range(int(self.manifest["num_layers"])):
            feeds[f"cache_key_{i}"] = caches[2 * i]
            feeds[f"cache_value_{i}"] = caches[2 * i + 1]
        outputs = self.slow.run(None, feeds)
        self._update_caches(caches, positions, outputs[2:])
        return np.asarray(outputs[0])[0, -1], np.asarray(outputs[1])[:, -1:, :]

    def _fast_step(self, hidden, token_id: int, use_hidden: bool, position: int, caches):
        hidden_dtype = ORT_DTYPES[self.fast_inputs["slow_hidden"].type]
        feeds = {
            "slow_hidden": np.asarray(hidden, dtype=hidden_dtype),
            "token_id": np.asarray([[token_id]], dtype=np.int64),
            "use_slow_hidden": np.asarray([use_hidden], dtype=np.bool_),
            "input_pos": np.asarray([position], dtype=np.int64),
        }
        for i in range(int(self.manifest["num_fast_layers"])):
            feeds[f"cache_key_{i}"] = caches[2 * i]
            feeds[f"cache_value_{i}"] = caches[2 * i + 1]
        outputs = self.fast.run(None, feeds)
        self._update_caches(caches, np.asarray([position]), outputs[1:])
        return np.asarray(outputs[0])[0, -1]

    def _sample_semantic(self, logits, previous, temperature, top_p, top_k, rng):
        begin = int(self.manifest["semantic_begin_id"])
        end = int(self.manifest["semantic_end_id"])
        stop = int(self.manifest["im_end_id"])
        allowed_ids = np.concatenate([np.arange(begin, end + 1), np.asarray([stop])])
        values = np.asarray(logits).reshape(-1)
        allowed_logits = (
            values
            if self.manifest.get("slow_logits_layout") == "semantic_then_eos"
            else values[allowed_ids]
        )
        if allowed_logits.size != allowed_ids.size:
            raise ValueError(
                f"unexpected slow logits size: {allowed_logits.size}, expected {allowed_ids.size}"
            )
        normal_index = _sample(allowed_logits, temperature, top_p, top_k, rng)
        normal = int(allowed_ids[normal_index])
        high_index = _sample(allowed_logits, 1.0, 0.9, top_k, rng)
        high = int(allowed_ids[high_index])
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
            raise ValueError(
                f"prompt length {prompt_len} exceeds max sequence length {max_seq_len}"
            )
        max_new_tokens = min(int(max_new_tokens), max_seq_len - prompt_len)
        rng = np.random.default_rng(int(seed))
        slow_caches = self._empty_slow_caches()
        positions = np.arange(prompt_len, dtype=np.int64)
        logits, hidden = self._slow_step(prompt, positions, slow_caches)
        previous: list[int] = []
        begin = int(self.manifest["semantic_begin_id"])
        stop = int(self.manifest["im_end_id"])
        codebook_size = int(self.manifest["codebook_size"])

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
            token = min(max(semantic - begin, 0), codebook_size - 1)
            codebooks = [token]
            for fast_pos in range(1, int(self.manifest["num_codebooks"])):
                fast_logits = self._fast_step(hidden, token, False, fast_pos, fast_caches)
                token = _sample(fast_logits, temperature, top_p, top_k, rng)
                codebooks.append(token)
            frame = np.asarray(codebooks, dtype=np.int64)
            yield frame
            if step + 1 >= max_new_tokens:
                return
            column = np.concatenate([[semantic], frame]).reshape(1, -1, 1)
            position = np.asarray([prompt_len + step], dtype=np.int64)
            logits, hidden = self._slow_step(column, position, slow_caches)

    def decode_codes(self, codes: np.ndarray) -> np.ndarray:
        values = np.asarray(codes, dtype=np.int64)
        if values.ndim == 2:
            values = values[np.newaxis]
        if values.ndim != 3 or values.shape[1] != int(self.manifest["num_codebooks"]):
            raise ValueError(f"invalid generated codes shape: {values.shape}")
        audio = self.decoder.run(None, {"codes": values})[0]
        return np.asarray(audio, dtype=np.float32).reshape(-1)

    def synthesize(self, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        frames = list(self.iter_codes(**kwargs))
        if not frames:
            raise RuntimeError("model produced no codec frames")
        codes = np.stack(frames, axis=1)
        return self.decode_codes(codes), codes

    def stream(self, chunk_frames: int = 12, **kwargs):
        all_frames: list[np.ndarray] = []
        emitted_samples = 0
        hop = int(self.manifest["codec_hop_length"])
        context = int(self.manifest.get("stream_context_frames", 128))
        guard = int(self.manifest.get("stream_guard_frames", 1)) * hop
        seq = 0
        for frame in self.iter_codes(**kwargs):
            all_frames.append(frame)
            if len(all_frames) % int(chunk_frames) != 0:
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
                    "seq": seq,
                    "audio": chunk,
                    "frame_count": len(all_frames),
                }
                seq += 1
        if not all_frames:
            raise RuntimeError("model produced no codec frames")
        start_frame = max(0, len(all_frames) - context - int(chunk_frames))
        window = np.stack(all_frames[start_frame:], axis=1)
        audio = self.decode_codes(window)
        absolute_start = start_frame * hop
        begin = max(0, emitted_samples - absolute_start)
        if begin < audio.size:
            chunk = np.ascontiguousarray(audio[begin:])
            yield {
                "type": "audio_chunk",
                "seq": seq,
                "audio": chunk,
                "frame_count": len(all_frames),
            }
        yield {"type": "complete", "codes": np.stack(all_frames, axis=1)}

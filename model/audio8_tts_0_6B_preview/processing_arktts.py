"""Text and reference-audio processor for audio8_tts generation."""

from __future__ import annotations

import inspect
import json
import os
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from transformers import AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin


def _clean_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def _as_list(value: Any, batch_size: int, name: str) -> list[Any]:
    if isinstance(value, (str, Path)) or value is None or np.isscalar(value):
        return [value] * batch_size
    if isinstance(value, torch.Tensor) and value.ndim <= 2:
        return [value] if batch_size == 1 else list(value)
    if isinstance(value, np.ndarray) and value.ndim <= 2:
        return [value] if batch_size == 1 else list(value)
    values = list(value)
    if len(values) != batch_size:
        raise ValueError(f"{name} must contain {batch_size} items, got {len(values)}")
    return values


def _pad_1d(rows: list[torch.Tensor], pad_value: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max((row.numel() for row in rows), default=0)
    values = torch.full((len(rows), max_len), pad_value, dtype=torch.long)
    mask = torch.zeros((len(rows), max_len), dtype=torch.long)
    for idx, row in enumerate(rows):
        length = row.numel()
        values[idx, :length] = row
        mask[idx, :length] = 1
    return values, mask


class ArkttsProcessor(ProcessorMixin):
    """Build DualAR prompts from text and optional zero-shot voice references.

    Reference conditioning accepts an audio path/array or precomputed codec
    indices. All samples in one batch must use the same conditioning mode.
    """

    attributes = ["tokenizer"]
    tokenizer_class = ("PreTrainedTokenizerFast", "PreTrainedTokenizer")
    valid_kwargs = ["num_codebooks", "semantic_begin_id", "audio_sampling_rate"]

    def __init__(
        self,
        tokenizer,
        num_codebooks: int = 10,
        semantic_begin_id: int = 151678,
        audio_sampling_rate: int = 44100,
        **kwargs,
    ):
        super().__init__(tokenizer)
        self.num_codebooks = int(num_codebooks)
        self.semantic_begin_id = int(semantic_begin_id)
        self.audio_sampling_rate = int(audio_sampling_rate)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> "ArkttsProcessor":
        trust_remote_code = bool(kwargs.pop("trust_remote_code", False))
        shared_names = {
            "cache_dir", "force_download", "local_files_only", "token", "revision", "subfolder"
        }
        shared = {key: kwargs[key] for key in list(kwargs) if key in shared_names}
        config = {}
        local_config = os.path.join(str(pretrained_model_name_or_path), "processor_config.json")
        if os.path.isfile(local_config):
            with open(local_config, "r", encoding="utf-8") as handle:
                config = json.load(handle)
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            use_fast=True,
            trust_remote_code=trust_remote_code,
            fix_mistral_regex=False,
            **shared,
        )
        return cls(
            tokenizer=tokenizer,
            num_codebooks=config.get("num_codebooks", 10),
            semantic_begin_id=config.get("semantic_begin_id", 151678),
            audio_sampling_rate=config.get("audio_sampling_rate", 44100),
        )

    def _encode(self, text: str) -> torch.Tensor:
        encode_kwargs = {"add_special_tokens": False}
        if "allowed_special" in inspect.signature(self.tokenizer.encode).parameters:
            encode_kwargs["allowed_special"] = "all"
        return torch.tensor(self.tokenizer.encode(text, **encode_kwargs), dtype=torch.long)

    @staticmethod
    def _format_reference_text(text: str) -> str:
        cleaned = _clean_text(text)
        if re.search(r"<\|speaker:\d+\|>", cleaned):
            return cleaned
        return f"<|speaker:0|>{cleaned}"

    def _prompt_segments(self, text: str, reference_text: str | None, has_reference: bool):
        """Return text before and after the optional reference codec frames."""
        target = _clean_text(text)
        if not target:
            raise ValueError("text must not be empty")
        def encode_parts(parts: list[str]) -> torch.Tensor:
            return torch.cat([self._encode(part) for part in parts])

        if not has_reference:
            full = encode_parts([
                "<|im_start|>system\n",
                "convert the provided text to speech",
                "<|im_end|>\n",
                "<|im_start|>user\n",
                target,
                "<|im_end|>\n",
                "<|im_start|>assistant\n<|voice|>",
            ])
            return full, self._encode("")
        if not reference_text:
            raise ValueError("reference_text is required when a reference voice is provided")
        prefix = encode_parts([
            "<|im_start|>system\n",
            "convert the provided text to speech reference to the following:\n\nText:\n",
            self._format_reference_text(reference_text),
            "\n\nSpeech:\n",
        ])
        suffix = encode_parts([
            "<|im_end|>\n",
            "<|im_start|>user\n",
            target,
            "<|im_end|>\n",
            "<|im_start|>assistant\n<|voice|>",
        ])
        return prefix, suffix

    def _load_audio(self, value: Any, sampling_rate: int | None) -> torch.Tensor:
        """Load, mix down, and resample one reference waveform."""
        source_rate = sampling_rate
        if isinstance(value, (str, Path)):
            try:
                import soundfile as sf
            except ImportError as exc:
                raise ImportError("soundfile is required for reference audio paths") from exc
            array, source_rate = sf.read(str(value), dtype="float32", always_2d=True)
            array = array.mean(axis=1)
            audio = torch.from_numpy(np.asarray(array, dtype=np.float32))
        else:
            if isinstance(value, dict):
                source_rate = value.get("sampling_rate", source_rate)
                value = value.get("array")
            if isinstance(value, (tuple, list)) and len(value) == 2 and np.isscalar(value[1]):
                value, source_rate = value
            audio = torch.as_tensor(value, dtype=torch.float32)
            if audio.ndim == 2:
                audio = audio.mean(dim=0)
            if audio.ndim != 1:
                raise ValueError(f"reference audio must be mono or channels-first, got {tuple(audio.shape)}")
        if audio.numel() == 0:
            raise ValueError("reference audio must not be empty")
        if source_rate is None:
            raise ValueError("sampling_rate is required for reference audio arrays")
        if int(source_rate) != self.audio_sampling_rate:
            try:
                from torchaudio.functional import resample
            except ImportError as exc:
                raise ImportError("torchaudio is required to resample reference audio") from exc
            audio = resample(audio, int(source_rate), self.audio_sampling_rate)
        return audio.contiguous()

    def __call__(
        self,
        text: str | Sequence[str],
        reference_text: str | Sequence[str] | None = None,
        reference_audio: Any = None,
        reference_codes: Any = None,
        sampling_rate: int | Sequence[int] | None = None,
        return_tensors: str = "pt",
        **kwargs,
    ) -> BatchFeature:
        """Create padded model inputs for text-to-speech generation."""
        if kwargs:
            raise TypeError(f"Unexpected processor arguments: {sorted(kwargs)}")
        if return_tensors != "pt":
            raise ValueError("ArkttsProcessor currently supports return_tensors='pt' only")
        texts = [text] if isinstance(text, str) else list(text)
        if not texts:
            raise ValueError("text batch must not be empty")
        batch_size = len(texts)
        ref_texts = _as_list(reference_text, batch_size, "reference_text")
        if reference_audio is not None and reference_codes is not None:
            raise ValueError("Provide reference_audio or reference_codes, not both")
        has_reference = reference_audio is not None or reference_codes is not None

        prefix_rows, suffix_rows = zip(*[
            self._prompt_segments(item, ref_texts[idx], has_reference)
            for idx, item in enumerate(texts)
        ])
        prefix_ids, prefix_mask = _pad_1d(list(prefix_rows), self.tokenizer.pad_token_id)
        suffix_ids, suffix_mask = _pad_1d(list(suffix_rows), self.tokenizer.pad_token_id)
        data: dict[str, torch.Tensor] = {
            "prefix_input_ids": prefix_ids,
            "prefix_attention_mask": prefix_mask,
            "suffix_input_ids": suffix_ids,
            "suffix_attention_mask": suffix_mask,
        }

        if reference_codes is not None:
            code_items = _as_list(reference_codes, batch_size, "reference_codes")
            loaded = []
            for item in code_items:
                if isinstance(item, (str, Path)):
                    item = np.load(str(item))
                codes = torch.as_tensor(item, dtype=torch.long)
                if codes.ndim != 2 or codes.shape[0] != self.num_codebooks or codes.shape[1] == 0:
                    raise ValueError(
                        f"reference codes must have shape [{self.num_codebooks}, T>0], got {tuple(codes.shape)}"
                    )
                if codes.min() < 0 or codes.max() >= 4096:
                    raise ValueError("reference codes must be in [0, 4095]")
                loaded.append(codes)
            max_frames = max(item.shape[1] for item in loaded)
            padded = torch.full((batch_size, self.num_codebooks, max_frames), -1, dtype=torch.long)
            lengths = torch.empty(batch_size, dtype=torch.long)
            for idx, codes in enumerate(loaded):
                lengths[idx] = codes.shape[1]
                padded[idx, :, : codes.shape[1]] = codes
            data["reference_codes"] = padded
            data["reference_code_lengths"] = lengths

        if reference_audio is not None:
            audio_items = _as_list(reference_audio, batch_size, "reference_audio")
            rate_items = _as_list(sampling_rate, batch_size, "sampling_rate")
            loaded_audio = [self._load_audio(item, rate_items[idx]) for idx, item in enumerate(audio_items)]
            max_samples = max(item.numel() for item in loaded_audio)
            padded_audio = torch.zeros((batch_size, 1, max_samples), dtype=torch.float32)
            lengths = torch.empty(batch_size, dtype=torch.long)
            for idx, audio in enumerate(loaded_audio):
                lengths[idx] = audio.numel()
                padded_audio[idx, 0, : audio.numel()] = audio
            data["reference_audio_values"] = padded_audio
            data["reference_audio_lengths"] = lengths

        return BatchFeature(data=data)

    @property
    def model_input_names(self) -> list[str]:
        return [
            "prefix_input_ids", "prefix_attention_mask", "suffix_input_ids",
            "suffix_attention_mask", "reference_codes", "reference_code_lengths",
            "reference_audio_values", "reference_audio_lengths",
        ]

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)


__all__ = ["ArkttsProcessor"]

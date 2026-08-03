"""Shared data and prompt utilities for audio8_tts inference and SFT."""

from __future__ import annotations

import json
import os
import re
import unicodedata
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

NUM_CODEBOOKS = 10
CODEBOOK_SIZE = 4096

_CJK_RANGES = (
    "\u1100-\u11ff\u2e80-\u2fdf\u3000-\u303f\u3040-\u30ff\u3100-\u31ff"
    "\u3400-\u4dbf\u4e00-\u9fff\ua960-\ua97f\uac00-\ud7a3\ud7b0-\ud7ff\uf900-\ufaff"
    "\ufe30-\ufe4f\uff01-\uff9f\U00020000-\U0002fa1f"
)
_CJK_CHARACTER_RE = re.compile(rf"[{_CJK_RANGES}]")
_LINE_BREAK_RE = re.compile(r"[\r\n\v\f\x1c-\x1e\x85\u2028\u2029]")


def _normalize_whitespace(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        left = text[match.start() - 1] if match.start() else ""
        right = text[match.end()] if match.end() < len(text) else ""
        if (
            _LINE_BREAK_RE.search(match.group())
            and _CJK_CHARACTER_RE.fullmatch(left)
            and _CJK_CHARACTER_RE.fullmatch(right)
        ):
            return ""
        return " "

    return re.sub(r"\s+", replace, text).strip()


def clean_text(value: Any, *, field_name: str = "text") -> str:
    """Remove non-linguistic controls and normalize whitespace for inference."""
    text = "".join(
        char if char.isspace() else "" if unicodedata.category(char).startswith("C") else char
        for char in str(value)
    )
    text = _normalize_whitespace(text)
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    return text


def validate_sample_id(value: Any, *, fallback: str | None = None) -> str:
    """Return an ID that is safe to use as one output filename component."""
    sample_id = str(value or fallback or "").strip()
    if not sample_id:
        raise ValueError("sample id must not be empty")
    if Path(sample_id).name != sample_id or sample_id in {".", ".."}:
        raise ValueError(f"unsafe sample id: {sample_id!r}")
    if not re.fullmatch(r"[\w.-]+", sample_id, flags=re.UNICODE):
        raise ValueError(f"sample id contains unsupported characters: {sample_id!r}")
    return sample_id


def read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield non-empty JSON objects together with their one-based line numbers."""
    if not path.is_file():
        raise FileNotFoundError(f"JSONL file does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: each row must be a JSON object")
            yield line_number, value


def json_line(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False) + "\n"


def resolve_manifest_path(value: Any, manifest: Path, *, field_name: str) -> Path:
    """Resolve an input path relative to the manifest that declares it."""
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field_name} must not be empty")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = manifest.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{field_name} does not exist: {path}")
    return path


def relative_to_manifest(path: Path, manifest: Path) -> str:
    """Serialize a path without leaking a machine-specific absolute prefix."""
    return Path(os.path.relpath(path.resolve(), manifest.parent.resolve())).as_posix()


def load_audio(path: Path, sample_rate: int) -> torch.Tensor:
    """Load an audio file as finite mono float32 samples at ``sample_rate``."""
    array, source_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(np.asarray(array, dtype=np.float32)).mean(dim=1)
    if waveform.numel() == 0:
        raise ValueError(f"audio is empty: {path}")
    if not torch.isfinite(waveform).all():
        raise ValueError(f"audio contains NaN or infinity: {path}")
    if int(source_rate) != int(sample_rate):
        from torchaudio.functional import resample

        waveform = resample(waveform, int(source_rate), int(sample_rate))
    return waveform.contiguous()


def pad_audio(waveforms: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad mono waveforms for the codec and return their true lengths."""
    if not waveforms:
        raise ValueError("cannot pad an empty audio batch")
    lengths = torch.tensor([item.numel() for item in waveforms], dtype=torch.long)
    values = torch.zeros((len(waveforms), 1, int(lengths.max())), dtype=torch.float32)
    for index, waveform in enumerate(waveforms):
        values[index, 0, : waveform.numel()] = waveform
    return values, lengths


def load_codes(path: Path, *, num_codebooks: int = NUM_CODEBOOKS) -> torch.Tensor:
    """Load and validate codec indices stored as a NumPy array."""
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"failed to load codec indices: {path}") from exc
    if array.ndim != 2 or array.shape[0] != num_codebooks or array.shape[1] == 0:
        raise ValueError(
            f"codec indices must have shape [{num_codebooks}, T>0], got {array.shape}: {path}"
        )
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"codec indices must use an integer dtype: {path}")
    if int(array.min()) < 0 or int(array.max()) >= CODEBOOK_SIZE:
        raise ValueError(f"codec indices must be in [0, {CODEBOOK_SIZE - 1}]: {path}")
    return torch.from_numpy(array.astype(np.int64, copy=False))


def _tokenize(tokenizer, *parts: str) -> torch.Tensor:
    rows = [tokenizer.encode(part, add_special_tokens=False) for part in parts]
    return torch.tensor([token for row in rows for token in row], dtype=torch.long)


def _format_reference_text(text: str) -> str:
    text = clean_text(text, field_name="reference_text")
    if re.search(r"<\|speaker:\d+\|>", text):
        return text
    return f"<|speaker:0|>{text}"


def build_sft_example(
    *,
    tokenizer,
    config,
    text: str,
    target_codes: torch.Tensor,
    reference_codes: torch.Tensor | None = None,
    reference_text: str | None = None,
    max_length: int | None = None,
) -> dict[str, torch.Tensor]:
    """Build shifted DualAR inputs and labels for one supervised example.

    Row zero contains text and semantic tokens. Rows 1..N contain codec
    indices only at semantic-token positions. The returned labels are already
    shifted by one position, so model output at position ``t`` predicts label
    ``t`` without an additional shift in the trainer.
    """
    text = clean_text(text)
    target_codes = target_codes.long()
    if target_codes.ndim != 2 or target_codes.shape[0] != int(config.num_codebooks):
        raise ValueError(
            f"target_codes must have shape [{config.num_codebooks}, T], "
            f"got {tuple(target_codes.shape)}"
        )

    segments: list[tuple[torch.Tensor, torch.Tensor | None, bool]] = []
    if reference_codes is None:
        segments.append(
            (
                _tokenize(
                    tokenizer,
                    "<|im_start|>system\n",
                    "convert the provided text to speech",
                    "<|im_end|>\n",
                ),
                None,
                False,
            )
        )
    else:
        if not reference_text:
            raise ValueError("reference_text is required when reference_codes are provided")
        reference_codes = reference_codes.long()
        if (
            reference_codes.ndim != 2
            or reference_codes.shape[0] != int(config.num_codebooks)
            or reference_codes.shape[1] == 0
        ):
            raise ValueError(
                f"reference_codes must have shape [{config.num_codebooks}, T>0], "
                f"got {tuple(reference_codes.shape)}"
            )
        segments.extend(
            [
                (
                    _tokenize(
                        tokenizer,
                        "<|im_start|>system\n",
                        "convert the provided text to speech reference to the following:\n\nText:\n",
                        _format_reference_text(reference_text),
                        "\n\nSpeech:\n",
                    ),
                    None,
                    False,
                ),
                (reference_codes[0] + int(config.semantic_begin_id), reference_codes, False),
                (_tokenize(tokenizer, "<|im_end|>\n"), None, False),
            ]
        )

    segments.extend(
        [
            (
                _tokenize(
                    tokenizer,
                    "<|im_start|>user\n",
                    text,
                    "<|im_end|>\n",
                    "<|im_start|>assistant\n<|voice|>",
                ),
                None,
                False,
            ),
            (target_codes[0] + int(config.semantic_begin_id), target_codes, True),
            (_tokenize(tokenizer, "<|im_end|>\n"), None, True),
        ]
    )

    full_length = sum(int(tokens.numel()) for tokens, _codes, _loss in segments)
    if full_length < 2:
        raise ValueError("encoded SFT example is too short")
    shifted_length = full_length - 1
    if max_length is not None and shifted_length > int(max_length):
        raise ValueError(
            f"encoded example has {shifted_length} tokens, exceeding max_length={max_length}"
        )

    full = torch.zeros((int(config.num_codebooks) + 1, full_length), dtype=torch.long)
    supervised = torch.zeros(full_length, dtype=torch.bool)
    codebook_supervised = torch.zeros(full_length, dtype=torch.bool)
    cursor = 0
    for tokens, codes, calculate_loss in segments:
        length = int(tokens.numel())
        full[0, cursor : cursor + length] = tokens
        if codes is not None:
            if int(codes.shape[1]) != length:
                raise ValueError("semantic-token length does not match codec frame count")
            full[1:, cursor : cursor + length] = codes
            codebook_supervised[cursor : cursor + length] = calculate_loss
        supervised[cursor : cursor + length] = calculate_loss
        cursor += length

    input_ids = full[:, :-1].contiguous()
    labels = torch.full_like(input_ids, -100)
    next_supervised = supervised[1:]
    labels[0, next_supervised] = full[0, 1:][next_supervised]
    next_codebook_supervised = codebook_supervised[1:]
    labels[1:, next_codebook_supervised] = full[1:, 1:][:, next_codebook_supervised]
    return {"input_ids": input_ids, "labels": labels}


def collate_sft_examples(
    examples: Sequence[dict[str, torch.Tensor]], *, pad_token_id: int
) -> dict[str, torch.Tensor]:
    """Right-pad shifted DualAR examples without supervising padding."""
    if not examples:
        raise ValueError("cannot collate an empty batch")
    width = max(int(example["input_ids"].shape[1]) for example in examples)
    inputs, labels, masks = [], [], []
    for example in examples:
        input_ids = example["input_ids"]
        target = example["labels"]
        length = int(input_ids.shape[1])
        padding = width - length
        if padding:
            input_ids = F.pad(input_ids, (0, padding), value=0)
            input_ids[0, length:] = int(pad_token_id)
            target = F.pad(target, (0, padding), value=-100)
        mask = torch.zeros(width, dtype=torch.long)
        mask[:length] = 1
        inputs.append(input_ids)
        labels.append(target)
        masks.append(mask)
    return {
        "input_ids": torch.stack(inputs),
        "attention_mask": torch.stack(masks),
        "labels": torch.stack(labels),
    }


def chunks(values: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]

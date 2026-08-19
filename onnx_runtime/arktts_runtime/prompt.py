from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

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


def clean_text(text: str) -> str:
    value = "".join(
        char if char.isspace() else "" if unicodedata.category(char).startswith("C") else char
        for char in str(text)
    )
    return _normalize_whitespace(value)


def format_reference_text(text: str) -> str:
    text = clean_text(text)
    return text if re.search(r"<\|speaker:\d+\|>", text) else f"<|speaker:0|>{text}"


class PromptBuilder:
    def __init__(self, tokenizer_dir: Path, semantic_begin_id: int, num_codebooks: int):
        self.tokenizer = Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))
        self.semantic_begin_id = int(semantic_begin_id)
        self.num_codebooks = int(num_codebooks)

    def encode_text(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def build(
        self, target_text: str, reference_text: str, reference_codes: np.ndarray
    ) -> np.ndarray:
        codes = np.asarray(reference_codes, dtype=np.int64)
        if codes.ndim != 2 or codes.shape[0] != self.num_codebooks or codes.shape[1] == 0:
            raise ValueError(
                f"reference codes must have shape [{self.num_codebooks}, T>0], got {codes.shape}"
            )
        prefix_parts = [
            "<|im_start|>system\n",
            "convert the provided text to speech reference to the following:\n\nText:\n",
            format_reference_text(reference_text),
            "\n\nSpeech:\n",
        ]
        suffix_parts = [
            "<|im_end|>\n",
            "<|im_start|>user\n",
            clean_text(target_text),
            "<|im_end|>\n",
            "<|im_start|>assistant\n<|voice|>",
        ]
        prefix = [token for part in prefix_parts for token in self.encode_text(part)]
        suffix = [token for part in suffix_parts for token in self.encode_text(part)]
        semantic_ids = (codes[0] + self.semantic_begin_id).tolist()
        row0 = np.asarray(prefix + semantic_ids + suffix, dtype=np.int64)
        values = np.zeros((self.num_codebooks + 1, row0.size), dtype=np.int64)
        values[0] = row0
        begin = len(prefix)
        values[1:, begin : begin + codes.shape[1]] = codes
        return values[np.newaxis]

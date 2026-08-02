from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


def clean_text(text: str) -> str:
    value = "".join(
        " " if char.isspace() else "" if unicodedata.category(char).startswith("C") else char
        for char in str(text)
    )
    return " ".join(value.split())


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

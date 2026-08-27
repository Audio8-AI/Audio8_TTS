from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class VoiceStore:
    def __init__(self, root: Path, num_codebooks: int, codebook_size: int = 4096):
        self.root = Path(root).resolve()
        self.num_codebooks = int(num_codebooks)
        self.codebook_size = int(codebook_size)

    def list(self) -> list[dict]:
        voices: list[dict] = []
        if not self.root.exists():
            return voices
        for meta_path in sorted(self.root.glob("*/meta.json")):
            try:
                voices.append(json.loads(meta_path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return voices

    def load(self, name: str) -> tuple[np.ndarray, dict]:
        if not name or Path(name).name != name:
            raise ValueError("invalid voice name")
        voice_dir = self.root / name
        meta_path = voice_dir / "meta.json"
        codes_path = voice_dir / "codes.npy"
        if not meta_path.is_file() or not codes_path.is_file():
            raise KeyError(f"voice not found: {name}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        codes = np.load(codes_path, allow_pickle=False).astype(np.int64, copy=False)
        if codes.ndim != 2 or codes.shape[0] != self.num_codebooks or codes.shape[1] == 0:
            raise ValueError(f"invalid codes for voice {name}: {codes.shape}")
        if int(codes.min()) < 0 or int(codes.max()) >= self.codebook_size:
            raise ValueError(f"voice {name} contains code outside [0, {self.codebook_size})")
        reference_text = str(meta.get("reference_text", "")).strip()
        if not reference_text:
            raise ValueError(f"voice {name} has no reference_text")
        return codes, meta

#!/usr/bin/env python3
"""Create the bundled ``default`` voice from the model's reference codes."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=root / "model")
    parser.add_argument("--voices-dir", type=Path, default=root / "voices")
    parser.add_argument("--name", default="default")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest = json.loads((args.model_dir / "runtime_manifest.json").read_text(encoding="utf-8"))
    codes_path = args.model_dir / manifest.get("reference_codes", "reference_codes.npy")
    codes = np.load(codes_path, allow_pickle=False)
    expected = int(manifest["num_codebooks"])
    codebook_size = int(manifest.get("codebook_size", 4096))
    if (
        codes.ndim != 2
        or codes.shape[0] != expected
        or codes.shape[1] == 0
        or int(codes.min()) < 0
        or int(codes.max()) >= codebook_size
    ):
        raise ValueError(f"reference codes must have shape [{expected}, T] in [0, {codebook_size})")
    name = str(args.name).strip()
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("name must be one path component")
    target = args.voices_dir / name
    if target.exists() and not args.overwrite:
        raise FileExistsError(f"voice already exists: {target}; pass --overwrite to replace it")

    args.voices_dir.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=args.voices_dir))
    try:
        np.save(temp / "codes.npy", codes.astype(np.uint16, copy=False))
        meta = {
            "name": name,
            "reference_text": str(manifest["reference_text"]),
            "shape": list(codes.shape),
            "dtype": "uint16",
            "sample_rate": int(manifest["sample_rate"]),
            "model_fingerprint": manifest["model_fingerprint"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_kind": "model_reference_codes",
        }
        (temp / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if target.exists():
            import shutil

            shutil.rmtree(target)
        os.replace(temp, target)
    finally:
        if temp.exists():
            import shutil

            shutil.rmtree(temp)
    print(f"registered {name}: shape={tuple(codes.shape)} at {target}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from .runtime import ArkTtsRuntime

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audio8 0.1B INT8 ONNX Runtime")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "model")
    parser.add_argument("--voices-dir", type=Path, default=ROOT / "voices")
    parser.add_argument("--precision", choices=["int8"], default=None)
    parser.add_argument("--codec-precision", choices=["fp16"], default=None)
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice", default="default")
    parser.add_argument("--output", type=Path, default=Path("output.wav"))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=5)
    args = parser.parse_args()

    runtime = ArkTtsRuntime(
        args.model_dir,
        args.voices_dir,
        args.precision,
        args.codec_precision,
        args.threads,
    )
    audio, codes = runtime.synthesize(
        text=args.text,
        voice=args.voice,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output), audio, int(runtime.manifest["sample_rate"]))
    np.save(args.output.with_suffix(".npy"), codes.astype(np.uint16))
    print(f"saved {args.output}")
    print(f"saved {args.output.with_suffix('.npy')}")


if __name__ == "__main__":
    main()

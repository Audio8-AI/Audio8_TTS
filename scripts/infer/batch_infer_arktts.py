#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel, AutoProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batched ArkTTS inference from JSONL")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--greedy", action="store_true")
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    records = []
    used_ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = Path(str(record.get("id", f"sample_{line_number - 1}"))).name
            if not sample_id or sample_id in used_ids:
                raise ValueError(f"Invalid or duplicate id at line {line_number}: {sample_id!r}")
            if not str(record.get("text", "")).strip():
                raise ValueError(f"Missing text at line {line_number}")
            has_codes = bool(record.get("reference_codes"))
            has_audio = bool(record.get("reference_audio"))
            if has_codes and has_audio:
                raise ValueError(f"Line {line_number} has both reference inputs")
            if (has_codes or has_audio) and not str(record.get("reference_text", "")).strip():
                raise ValueError(f"Line {line_number} needs reference_text")
            record["id"] = sample_id
            record["_mode"] = "codes" if has_codes else "audio" if has_audio else "none"
            records.append(record)
            used_ids.add(sample_id)
    if not records:
        raise ValueError(f"No samples found in {path}")
    return records


def chunks(items: list[dict], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    records = load_records(args.input_jsonl)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA inference was requested but torch.cuda.is_available() is false. "
            "Check the host GPU, NVIDIA driver, CUDA runtime, and CUDA_VISIBLE_DEVICES. "
            "Use --device cpu only for an intentional CPU run."
        )
    grouped = defaultdict(list)
    for record in records:
        grouped[record["_mode"]].append(record)

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, dtype=dtype
    ).eval().to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    completed = 0
    for mode in ("none", "codes", "audio"):
        for batch in chunks(grouped[mode], args.batch_size):
            print(
                f"[gen] mode={mode} batch={len(batch)} ids="
                f"{[item['id'] for item in batch]} generating "
                f"(max_new_tokens={args.max_new_tokens})...",
                flush=True,
            )
            processor_kwargs = {
                "text": [item["text"] for item in batch],
                "return_tensors": "pt",
            }
            if mode != "none":
                processor_kwargs["reference_text"] = [item["reference_text"] for item in batch]
            if mode == "codes":
                processor_kwargs["reference_codes"] = [item["reference_codes"] for item in batch]
            elif mode == "audio":
                processor_kwargs["reference_audio"] = [item["reference_audio"] for item in batch]

            inputs = processor(**processor_kwargs)
            inputs = {name: value.to(device) for name, value in inputs.items()}
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                do_sample=not args.greedy,
                generator=generator,
                return_dict_in_generate=True,
            )
            waveforms, waveform_lengths = model.decode_audio(output.codes)

            for index, item in enumerate(batch):
                code_length = int(output.code_lengths[index].item())
                waveform_length = int(waveform_lengths[index].item())
                np.save(
                    args.output_dir / f"{item['id']}.npy",
                    output.codes[index, :, :code_length].cpu().numpy(),
                )
                sf.write(
                    args.output_dir / f"{item['id']}.wav",
                    waveforms[index, :waveform_length].cpu().numpy(),
                    model.config.codec_sample_rate,
                )
                completed += 1
                duration = waveform_length / model.config.codec_sample_rate
                print(
                    f"[{completed}/{len(records)}] {item['id']}: "
                    f"{duration:.2f}s audio",
                    flush=True,
                )


if __name__ == "__main__":
    main()

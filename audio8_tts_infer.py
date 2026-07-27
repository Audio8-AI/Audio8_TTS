#!/usr/bin/env python3
"""Command-line inference for audio8_tts Preview."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from audio8_tts_data import chunks, clean_text, json_line, read_jsonl, resolve_manifest_path
from audio8_tts_data import validate_sample_id


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "model" / "audio8_tts_0_6B_preview"


@dataclass(frozen=True)
class InferenceItem:
    sample_id: str
    text: str
    output: Path
    reference_audio: Path | None = None
    reference_text: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthesize speech with audio8_tts Preview.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Text for one utterance.")
    source.add_argument("--input-jsonl", type=Path, help="Batch input manifest.")
    parser.add_argument("--reference-audio", type=Path)
    parser.add_argument("--reference-text")
    parser.add_argument("--output", type=Path, default=Path("output.wav"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--failures", type=Path)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--retry-max-new-tokens", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--save-codes", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def resolve_dtype(value: str, device: torch.device) -> torch.dtype:
    if value == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    dtype = getattr(torch, value)
    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError("CPU inference requires --dtype float32")
    return dtype


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if args.retry_max_new_tokens < args.max_new_tokens:
        raise ValueError("--retry-max-new-tokens must be >= --max-new-tokens")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")
    if args.top_k < 0:
        raise ValueError("--top-k must be non-negative")
    if (args.reference_audio is None) != (args.reference_text is None):
        raise ValueError("--reference-audio and --reference-text must be provided together")
    if args.input_jsonl is not None and (
        args.reference_audio is not None or args.reference_text is not None
    ):
        raise ValueError("reference options belong in each batch JSONL row")


def load_batch_items(manifest: Path, output_dir: Path) -> list[InferenceItem]:
    manifest = manifest.resolve()
    items: list[InferenceItem] = []
    seen: set[str] = set()
    for line_number, row in read_jsonl(manifest):
        try:
            sample_id = validate_sample_id(row.get("id"), fallback=f"row_{line_number:06d}")
            text = clean_text(row.get("text"), field_name="text")
            reference_value = row.get("reference_audio")
            reference_text_value = row.get("reference_text")
            if bool(reference_value) != bool(reference_text_value):
                raise ValueError("reference_audio and reference_text must be provided together")
            reference_audio = (
                resolve_manifest_path(reference_value, manifest, field_name="reference_audio")
                if reference_value
                else None
            )
            reference_text = (
                clean_text(reference_text_value, field_name="reference_text")
                if reference_text_value
                else None
            )
        except (KeyError, TypeError, ValueError, FileNotFoundError) as exc:
            raise ValueError(f"{manifest}:{line_number}: invalid inference row: {exc}") from exc
        if sample_id in seen:
            raise ValueError(f"{manifest}:{line_number}: duplicate id {sample_id!r}")
        seen.add(sample_id)
        items.append(
            InferenceItem(
                sample_id=sample_id,
                text=text,
                output=output_dir / f"{sample_id}.wav",
                reference_audio=reference_audio,
                reference_text=reference_text,
            )
        )
    if not items:
        raise ValueError(f"no inference rows found in {manifest}")
    return items


def load_items(args: argparse.Namespace) -> list[InferenceItem]:
    if args.input_jsonl is not None:
        return load_batch_items(args.input_jsonl, args.output_dir)
    reference_audio = args.reference_audio.resolve() if args.reference_audio else None
    if reference_audio is not None and not reference_audio.is_file():
        raise FileNotFoundError(f"reference audio does not exist: {reference_audio}")
    return [
        InferenceItem(
            sample_id=validate_sample_id(args.output.stem, fallback="output"),
            text=clean_text(args.text),
            output=args.output,
            reference_audio=reference_audio,
            reference_text=(
                clean_text(args.reference_text, field_name="reference_text")
                if args.reference_text
                else None
            ),
        )
    ]


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    items = load_items(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    for item in items:
        item.output.parent.mkdir(parents=True, exist_ok=True)
    batch_mode = args.input_jsonl is not None
    manifest_path = args.manifest or (args.output_dir / "manifest.jsonl")
    failures_path = args.failures or (args.output_dir / "failures.jsonl")
    if batch_mode:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        failures_path.parent.mkdir(parents=True, exist_ok=True)

    skipped = [
        item
        for item in items
        if not args.overwrite and item.output.is_file() and item.output.stat().st_size > 0
    ]
    pending = [item for item in items if item not in skipped]
    print(
        f"[audio8_tts] model={args.model} device={device} dtype={dtype} "
        f"items={len(items)} pending={len(pending)} skipped={len(skipped)}"
    )

    if not pending:
        if batch_mode:
            with manifest_path.open("w", encoding="utf-8") as handle:
                for item in skipped:
                    handle.write(
                        json_line(
                            {
                                "id": item.sample_id,
                                "status": "SKIP",
                                "output_audio": str(item.output),
                                "reference_audio": (
                                    str(item.reference_audio) if item.reference_audio else None
                                ),
                            }
                        )
                    )
            failures_path.write_text("", encoding="utf-8")
            print(f"[audio8_tts] manifest={manifest_path} failures={failures_path}")
        else:
            print(f"[audio8_tts] output={skipped[0].output} (already exists)")
        return

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, dtype=dtype
    ).eval().to(device)
    sample_rate = int(model.config.codec_sample_rate)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    def synthesize(batch: list[InferenceItem], max_new_tokens: int) -> dict[str, Any]:
        processor_kwargs: dict[str, Any] = {
            "text": [item.text for item in batch],
            "return_tensors": "pt",
        }
        if batch[0].reference_audio is not None:
            processor_kwargs.update(
                reference_audio=[item.reference_audio for item in batch],
                reference_text=[item.reference_text for item in batch],
            )
        inputs = processor(**processor_kwargs)
        inputs = {name: value.to(device) for name, value in inputs.items()}
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=not args.greedy,
            generator=generator,
            return_dict_in_generate=True,
        )
        waveforms, waveform_lengths = model.decode_audio(output.codes)
        return {
            "output": output,
            "waveforms": waveforms,
            "waveform_lengths": waveform_lengths,
        }

    def run_batch(batch: list[InferenceItem]) -> list[dict[str, Any]]:
        result = synthesize(batch, args.max_new_tokens)
        output = result["output"]
        records: list[dict[str, Any]] = []
        for index, item in enumerate(batch):
            finished = bool(output.finished[index])
            code_length = int(output.code_lengths[index])
            codes = output.codes[index, :, :code_length]
            waveform_length = int(result["waveform_lengths"][index])
            waveform = result["waveforms"][index, :waveform_length]
            if not finished and args.retry_max_new_tokens > args.max_new_tokens:
                retry = synthesize([item], args.retry_max_new_tokens)
                retry_output = retry["output"]
                finished = bool(retry_output.finished[0])
                code_length = int(retry_output.code_lengths[0])
                codes = retry_output.codes[0, :, :code_length]
                waveform_length = int(retry["waveform_lengths"][0])
                waveform = retry["waveforms"][0, :waveform_length]

            waveform_array = waveform.float().cpu().numpy()
            codes_array = codes.cpu().numpy()
            sf.write(item.output, waveform_array, sample_rate)
            if args.save_codes:
                np.save(item.output.with_suffix(".npy"), codes_array)
            records.append(
                {
                    "id": item.sample_id,
                    "status": "OK" if finished else "NO_EOS",
                    "output_audio": str(item.output),
                    "reference_audio": (
                        str(item.reference_audio) if item.reference_audio is not None else None
                    ),
                    "code_frames": int(codes_array.shape[1]),
                    "waveform_samples": int(waveform_array.shape[0]),
                    "sample_rate": sample_rate,
                }
            )
        return records

    records = [
        {
            "id": item.sample_id,
            "status": "SKIP",
            "output_audio": str(item.output),
            "reference_audio": str(item.reference_audio) if item.reference_audio else None,
        }
        for item in skipped
    ]
    failures: list[dict[str, str]] = []
    # The processor treats reference conditioning as a batch-wide mode, so the
    # two groups must not be mixed in one call.
    groups = [
        [item for item in pending if item.reference_audio is None],
        [item for item in pending if item.reference_audio is not None],
    ]
    total_batches = sum((len(group) + args.batch_size - 1) // args.batch_size for group in groups)
    progress = tqdm(total=total_batches, desc="audio8_tts")
    for group in groups:
        for batch_values in chunks(group, args.batch_size):
            batch = list(batch_values)
            try:
                records.extend(run_batch(batch))
            except Exception as batch_exc:
                print(
                    f"[audio8_tts] batch fallback after {type(batch_exc).__name__}: {batch_exc}",
                    flush=True,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                for item in batch:
                    try:
                        records.extend(run_batch([item]))
                    except Exception as exc:
                        failures.append(
                            {
                                "id": item.sample_id,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                        )
                        print(
                            f"[audio8_tts] FAILED {item.sample_id}: "
                            f"{type(exc).__name__}: {exc}",
                            flush=True,
                        )
            progress.update(1)
    progress.close()

    if batch_mode:
        with manifest_path.open("w", encoding="utf-8") as handle:
            handle.writelines(json_line(record) for record in records)
        with failures_path.open("w", encoding="utf-8") as handle:
            handle.writelines(json_line(record) for record in failures)
        print(f"[audio8_tts] manifest={manifest_path} failures={failures_path}")
    else:
        if records:
            print(f"[audio8_tts] output={records[-1]['output_audio']}")
    if failures:
        raise RuntimeError(f"inference failed for {len(failures)} sample(s)")


if __name__ == "__main__":
    main()

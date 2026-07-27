#!/usr/bin/env python3
"""Encode raw training audio into audio8_tts codec indices."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel

from audio8_tts_data import (
    chunks,
    clean_text,
    json_line,
    load_audio,
    load_codes,
    pad_audio,
    read_jsonl,
    relative_to_manifest,
    resolve_manifest_path,
    validate_sample_id,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "model" / "audio8_tts_0_6B_preview"


@dataclass(frozen=True)
class RawItem:
    sample_id: str
    text: str
    audio: Path
    target_codes: Path
    reference_audio: Path | None
    reference_text: str | None
    reference_codes: Path | None


@dataclass(frozen=True)
class EncodeJob:
    item_index: int
    kind: str
    audio: Path
    output: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare audio8_tts SFT data from a raw-audio JSONL manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--codes-dir", type=Path)
    parser.add_argument("--failures", type=Path)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument(
        "--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto"
    )
    parser.add_argument("--batch-size", type=int, default=4)
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
        raise ValueError("CPU codec preparation requires --dtype float32")
    return dtype


def load_items(input_jsonl: Path, codes_dir: Path) -> list[RawItem]:
    input_jsonl = input_jsonl.resolve()
    items: list[RawItem] = []
    seen: set[str] = set()
    for line_number, row in read_jsonl(input_jsonl):
        try:
            sample_id = validate_sample_id(row.get("id"), fallback=f"row_{line_number:06d}")
            text = clean_text(row.get("text"), field_name="text")
            audio = resolve_manifest_path(row.get("audio"), input_jsonl, field_name="audio")
            reference_value = row.get("reference_audio")
            reference_text_value = row.get("reference_text")
            if bool(reference_value) != bool(reference_text_value):
                raise ValueError("reference_audio and reference_text must be provided together")
            reference_audio = (
                resolve_manifest_path(reference_value, input_jsonl, field_name="reference_audio")
                if reference_value
                else None
            )
            reference_text = (
                clean_text(reference_text_value, field_name="reference_text")
                if reference_text_value
                else None
            )
        except (TypeError, ValueError, FileNotFoundError) as exc:
            raise ValueError(f"{input_jsonl}:{line_number}: invalid training row: {exc}") from exc
        if sample_id in seen:
            raise ValueError(f"{input_jsonl}:{line_number}: duplicate id {sample_id!r}")
        seen.add(sample_id)
        items.append(
            RawItem(
                sample_id=sample_id,
                text=text,
                audio=audio,
                target_codes=codes_dir / f"{sample_id}.target.npy",
                reference_audio=reference_audio,
                reference_text=reference_text,
                reference_codes=(
                    codes_dir / f"{sample_id}.reference.npy" if reference_audio else None
                ),
            )
        )
    if not items:
        raise ValueError(f"no training rows found in {input_jsonl}")
    return items


def valid_existing_codes(path: Path, num_codebooks: int) -> bool:
    if not path.is_file():
        return False
    try:
        load_codes(path, num_codebooks=num_codebooks)
    except ValueError:
        return False
    return True


def atomic_save_codes(path: Path, codes: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npy")
    np.save(temporary, codes)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    output_jsonl = args.output_jsonl.resolve()
    codes_dir = (args.codes_dir or output_jsonl.with_suffix("").with_name("codes")).resolve()
    failures_path = (args.failures or output_jsonl.with_suffix(".failures.jsonl")).resolve()
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    codes_dir.mkdir(parents=True, exist_ok=True)
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    items = load_items(args.input_jsonl, codes_dir)

    print(
        f"[audio8_tts.prepare] model={args.model} device={device} dtype={dtype} "
        f"items={len(items)} codes_dir={codes_dir}"
    )
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True, dtype=dtype).eval()
    model.load_codec(device=device, dtype=dtype)
    sample_rate = int(model.config.codec_sample_rate)
    num_codebooks = int(model.config.num_codebooks)

    jobs: list[EncodeJob] = []
    for item_index, item in enumerate(items):
        if args.overwrite or not valid_existing_codes(item.target_codes, num_codebooks):
            jobs.append(EncodeJob(item_index, "target", item.audio, item.target_codes))
        if item.reference_audio is not None and item.reference_codes is not None:
            if args.overwrite or not valid_existing_codes(item.reference_codes, num_codebooks):
                jobs.append(
                    EncodeJob(item_index, "reference", item.reference_audio, item.reference_codes)
                )

    failed_rows: dict[int, dict[str, str]] = {}

    def encode_batch(batch: list[EncodeJob]) -> None:
        waveforms = [load_audio(job.audio, sample_rate) for job in batch]
        audio_values, audio_lengths = pad_audio(waveforms)
        codes, code_lengths = model.encode_audio(
            audio_values.to(device), audio_lengths.to(device)
        )
        for index, job in enumerate(batch):
            length = int(code_lengths[index])
            values = codes[index, :, :length].cpu().numpy().astype(np.int32, copy=False)
            if values.shape[0] != num_codebooks or values.shape[1] == 0:
                raise RuntimeError(
                    f"codec returned invalid shape {values.shape} for {job.audio}"
                )
            atomic_save_codes(job.output, values)

    for batch_values in tqdm(
        list(chunks(jobs, args.batch_size)), desc="audio8_tts.prepare"
    ):
        batch = list(batch_values)
        try:
            encode_batch(batch)
        except Exception as batch_exc:
            print(
                f"[audio8_tts.prepare] batch fallback after "
                f"{type(batch_exc).__name__}: {batch_exc}",
                flush=True,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            for job in batch:
                try:
                    encode_batch([job])
                except Exception as exc:
                    failed_rows[job.item_index] = {
                        "id": items[job.item_index].sample_id,
                        "kind": job.kind,
                        "audio": str(job.audio),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }

    with output_jsonl.open("w", encoding="utf-8") as output_handle:
        for index, item in enumerate(items):
            if index in failed_rows:
                continue
            row = {
                "id": item.sample_id,
                "text": item.text,
                "target_codes": relative_to_manifest(item.target_codes, output_jsonl),
            }
            if item.reference_codes is not None:
                row.update(
                    reference_codes=relative_to_manifest(item.reference_codes, output_jsonl),
                    reference_text=item.reference_text,
                    reference_audio=relative_to_manifest(item.reference_audio, output_jsonl),
                )
            output_handle.write(json_line(row))
    with failures_path.open("w", encoding="utf-8") as failure_handle:
        failure_handle.writelines(json_line(row) for row in failed_rows.values())

    print(
        f"[audio8_tts.prepare] prepared={len(items) - len(failed_rows)} "
        f"failed={len(failed_rows)} output={output_jsonl}"
    )
    if failed_rows:
        raise RuntimeError(f"codec preparation failed for {len(failed_rows)} sample(s)")


if __name__ == "__main__":
    main()

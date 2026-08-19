#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor


@dataclass(frozen=True)
class Audio8Item:
    sample_id: str
    text: str
    reference_text: str
    reference_audio: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ArkTTS audio for an Audio8 web-demo bundle"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--failures", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int)
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


def load_items(bundle: Path) -> list[Audio8Item]:
    manifest = bundle / "prompts.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"Audio8 prompt manifest does not exist: {manifest}")

    items: list[Audio8Item] = []
    seen_ids: set[str] = set()
    with manifest.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                sample_id = str(record["id"])
                text = str(record["target_prompt"])
                reference_text = str(record["reference_transcript"])
                reference_relative = Path(record["audio"]["reference"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{manifest}:{line_number}: invalid Audio8 record") from exc
            if not sample_id or Path(sample_id).name != sample_id:
                raise ValueError(f"{manifest}:{line_number}: unsafe sample id {sample_id!r}")
            if sample_id in seen_ids:
                raise ValueError(f"{manifest}:{line_number}: duplicate sample id {sample_id!r}")
            reference_audio = (
                reference_relative
                if reference_relative.is_absolute()
                else bundle / reference_relative
            )
            if not reference_audio.is_file():
                raise FileNotFoundError(
                    f"{manifest}:{line_number}: reference audio does not exist: "
                    f"{reference_audio}"
                )
            seen_ids.add(sample_id)
            items.append(Audio8Item(sample_id, text, reference_text, reference_audio))
    if not items:
        raise ValueError(f"No samples found in {manifest}")
    return items


def chunks(items: list[Audio8Item], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def json_line(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False) + "\n"


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if args.retry_max_new_tokens < args.max_new_tokens:
        raise ValueError("--retry-max-new-tokens must be >= --max-new-tokens")

    items = load_items(args.bundle.resolve())
    if args.limit is not None:
        items = items[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.failures.parent.mkdir(parents=True, exist_ok=True)

    skipped: list[Audio8Item] = []
    pending: list[Audio8Item] = []
    for item in items:
        wav_path = args.output_dir / f"{item.sample_id}.wav"
        if not args.overwrite and wav_path.is_file() and wav_path.stat().st_size > 0:
            skipped.append(item)
        else:
            pending.append(item)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(
        f"[arktts-audio8] items={len(items)} pending={len(pending)} "
        f"skipped={len(skipped)} batch_size={args.batch_size} device={device}"
    )
    print(f"[arktts-audio8] model={args.model.resolve()}")
    print(f"[arktts-audio8] output_dir={args.output_dir.resolve()}")

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model, trust_remote_code=True, dtype=dtype
    ).eval().to(device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    sample_rate = int(model.config.codec_sample_rate)

    def synthesize(batch: list[Audio8Item], max_new_tokens: int):
        inputs = processor(
            text=[item.text for item in batch],
            reference_text=[item.reference_text for item in batch],
            reference_audio=[item.reference_audio for item in batch],
            return_tensors="pt",
        )
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
        return output, waveforms, waveform_lengths

    completed = 0
    failed = 0
    no_eos = 0
    with (
        args.manifest.open("w", encoding="utf-8") as manifest_handle,
        args.failures.open("w", encoding="utf-8") as failure_handle,
    ):
        for item in skipped:
            wav_path = args.output_dir / f"{item.sample_id}.wav"
            manifest_handle.write(
                json_line(
                    {
                        "id": item.sample_id,
                        "status": "SKIP",
                        "output_audio": str(wav_path.resolve()),
                        "reference_audio": str(item.reference_audio.resolve()),
                    }
                )
            )
        manifest_handle.flush()

        def run_batch(batch: list[Audio8Item]) -> None:
            nonlocal completed, no_eos
            output, waveforms, waveform_lengths = synthesize(
                batch, args.max_new_tokens
            )
            final_codes = [
                output.codes[index, :, : int(output.code_lengths[index])]
                for index in range(len(batch))
            ]
            final_waveforms = [
                waveforms[index, : int(waveform_lengths[index])]
                for index in range(len(batch))
            ]
            final_finished = [bool(value) for value in output.finished.tolist()]

            unfinished = [
                index for index, is_finished in enumerate(final_finished) if not is_finished
            ]
            if unfinished and args.retry_max_new_tokens > args.max_new_tokens:
                retry_batch = [batch[index] for index in unfinished]
                retry_output, retry_waveforms, retry_lengths = synthesize(
                    retry_batch, args.retry_max_new_tokens
                )
                for retry_index, original_index in enumerate(unfinished):
                    retry_code_length = int(retry_output.code_lengths[retry_index])
                    retry_waveform_length = int(retry_lengths[retry_index])
                    final_codes[original_index] = retry_output.codes[
                        retry_index, :, :retry_code_length
                    ]
                    final_waveforms[original_index] = retry_waveforms[
                        retry_index, :retry_waveform_length
                    ]
                    final_finished[original_index] = bool(
                        retry_output.finished[retry_index]
                    )

            for index, item in enumerate(batch):
                wav_path = args.output_dir / f"{item.sample_id}.wav"
                waveform = final_waveforms[index].float().cpu().numpy()
                codes = final_codes[index].cpu().numpy()
                sf.write(wav_path, waveform, sample_rate)
                if args.save_codes:
                    np.save(args.output_dir / f"{item.sample_id}.npy", codes)
                status = "OK" if final_finished[index] else "NO_EOS"
                completed += 1
                no_eos += int(not final_finished[index])
                manifest_handle.write(
                    json_line(
                        {
                            "id": item.sample_id,
                            "status": status,
                            "output_audio": str(wav_path.resolve()),
                            "reference_audio": str(item.reference_audio.resolve()),
                            "code_frames": int(codes.shape[1]),
                            "waveform_samples": int(waveform.shape[0]),
                            "sample_rate": sample_rate,
                        }
                    )
                )
            manifest_handle.flush()

        for batch in tqdm(
            list(chunks(pending, args.batch_size)), desc="arktts-audio8"
        ):
            try:
                run_batch(batch)
            except Exception as batch_exc:
                print(
                    f"[arktts-audio8] batch fallback after "
                    f"{type(batch_exc).__name__}: {batch_exc}",
                    flush=True,
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                for item in batch:
                    try:
                        run_batch([item])
                    except Exception as exc:
                        failed += 1
                        failure_handle.write(
                            json_line(
                                {
                                    "id": item.sample_id,
                                    "error_type": type(exc).__name__,
                                    "error": str(exc),
                                }
                            )
                        )
                        failure_handle.flush()
                        print(
                            f"[arktts-audio8] FAILED {item.sample_id}: "
                            f"{type(exc).__name__}: {exc}",
                            flush=True,
                        )

    print(
        f"[arktts-audio8] done completed={completed} skipped={len(skipped)} "
        f"no_eos={no_eos} failed={failed}"
    )
    if failed:
        raise RuntimeError(f"Audio8 inference failed for {failed} sample(s)")


if __name__ == "__main__":
    main()

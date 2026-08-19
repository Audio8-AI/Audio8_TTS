#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.request
import wave
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--output", type=Path, default=Path("/tmp/audio8_stream.wav"))
    parser.add_argument(
        "--text",
        default="Streaming speech should begin playing before the complete sentence is ready.",
    )
    args = parser.parse_args()

    body = json.dumps(
        {
            "model": "audio8/tts-0.6b",
            "input": args.text,
            "response_format": "pcm",
            "stream": True,
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 50,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{args.base_url}/v1/audio/speech",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    first_chunk_seconds: float | None = None
    sample_rate: int | None = None
    chunks: list[bytes] = []
    with urllib.request.urlopen(request, timeout=300) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            value = line[6:]
            if value == "[DONE]":
                break
            event = json.loads(value)
            audio = event.get("audio")
            if not audio:
                continue
            if audio.get("format") != "pcm":
                raise RuntimeError(f"Expected PCM stream, got {audio.get('format')}")
            if first_chunk_seconds is None:
                first_chunk_seconds = time.perf_counter() - started
            sample_rate = int(audio["sample_rate"])
            chunks.append(base64.b64decode(audio["data"]))

    total_seconds = time.perf_counter() - started
    if not chunks or sample_rate is None or first_chunk_seconds is None:
        raise RuntimeError("The server returned no streaming audio chunks")
    pcm = b"".join(chunks)
    if len(pcm) % 2:
        raise RuntimeError("PCM stream has an odd byte count")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(args.output), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm)

    print(f"chunks={len(chunks)}")
    print(f"first_chunk_seconds={first_chunk_seconds:.3f}")
    print(f"total_seconds={total_seconds:.3f}")
    print(f"sample_rate={sample_rate}")
    print(f"audio_seconds={len(pcm) / 2 / sample_rate:.3f}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()

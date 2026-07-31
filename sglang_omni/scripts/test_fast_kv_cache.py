#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang_omni.models.audio8_tts.sglang_model import FastDecoderLayer, _rope


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    config = SimpleNamespace(
        **json.loads((args.model_path / "config.json").read_text(encoding="utf-8"))
    )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the Fast AR KV-cache test")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    layers = torch.nn.ModuleList(
        [FastDecoderLayer(config) for _ in range(config.n_fast_layer)]
    ).to(device=device, dtype=torch.bfloat16).eval()
    for layer in layers:
        layer.attention.setup_audio8_cache(
            args.batch_size,
            config.num_codebooks + 1,
            device=device,
            dtype=torch.bfloat16,
        )

    inputs = torch.randn(
        args.batch_size,
        config.num_codebooks,
        config.fast_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    rope = _rope(
        config.num_codebooks,
        config.fast_head_dim,
        config.rope_base,
        device,
    )

    with torch.inference_mode():
        full = inputs
        for layer in layers:
            full = layer(full, rope)

        for layer in layers:
            layer.attention.clear_audio8_cache()
        cached_parts = []
        for position in range(config.num_codebooks):
            hidden = inputs[:, position : position + 1]
            cache_positions = torch.full(
                (args.batch_size,),
                position,
                device=device,
                dtype=torch.int32,
            )
            for layer in layers:
                hidden = layer.forward_audio8_cached(
                    hidden,
                    rope[position : position + 1],
                    cache_positions,
                )
            cached_parts.append(hidden)
        cached = torch.cat(cached_parts, dim=1)

    difference = (full.float() - cached.float()).abs()
    max_abs = float(difference.max().item())
    mean_abs = float(difference.mean().item())
    cosine = float(
        torch.nn.functional.cosine_similarity(
            full.float().flatten(),
            cached.float().flatten(),
            dim=0,
        ).item()
    )
    print(f"max_abs={max_abs:.6f}")
    print(f"mean_abs={mean_abs:.6f}")
    print(f"cosine={cosine:.8f}")
    if cosine < 0.999:
        raise SystemExit("Fast KV-cache output diverged from full causal forward")


if __name__ == "__main__":
    main()

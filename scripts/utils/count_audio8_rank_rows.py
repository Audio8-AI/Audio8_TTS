#!/usr/bin/env python3
"""Fast per-rank row counts for Audio8 code shards (reads only rows_json)."""

from __future__ import annotations

import argparse
import pathlib

import numpy as np


def rows_in_chunk(path: pathlib.Path) -> int:
    with np.load(path, mmap_mode="r") as z:
        return int(z["line_indices"].shape[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-shard-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of worker_* dirs actually consumed by the DataLoader "
        "(0 means count every worker dir).",
    )
    args = parser.parse_args()

    counts: list[tuple[str, int]] = []
    for rank_dir in sorted(args.code_shard_dir.glob("rank_*")):
        total = 0
        for chunk in sorted(rank_dir.rglob("chunk_*.npz")):
            worker_name = chunk.parent.name
            if args.num_workers and args.num_workers > 0:
                try:
                    worker_id = int(worker_name.split("_")[-1])
                except (IndexError, ValueError):
                    continue
                if worker_id >= args.num_workers:
                    continue
            total += rows_in_chunk(chunk)
        counts.append((rank_dir.name, total))
    for name, total in counts:
        print(f"{name}: {total}")
    if counts:
        print(f"min_rank_samples={min(total for _, total in counts)}")


if __name__ == "__main__":
    main()

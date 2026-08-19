#!/usr/bin/env python3
import argparse
import pathlib
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a JSONL into deterministic round-robin rank shards."
    )
    parser.add_argument("--input", type=pathlib.Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--log-every", type=int, default=1_000_000)
    return parser.parse_args()


def complete_shard_dir(path: pathlib.Path, world_size: int) -> bool:
    counts = path / "counts.txt"
    if not counts.is_file():
        return False
    return all(
        (path / f"rank_{rank:05d}.jsonl").is_file() for rank in range(world_size)
    )


def main() -> None:
    args = parse_args()
    if args.world_size <= 0:
        raise ValueError("--world-size must be positive")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    output_dir = args.input.with_name(f"{args.input.name}.shards{args.world_size}")
    if complete_shard_dir(output_dir, args.world_size):
        print(f"[shard] already complete: {output_dir}", flush=True)
        return
    if output_dir.exists():
        raise RuntimeError(
            f"incomplete output exists: {output_dir}; move it aside before retrying"
        )

    tmp_dir = output_dir.with_name(f".{output_dir.name}.tmp.{time.time_ns()}")
    tmp_dir.mkdir(parents=True, exist_ok=False)
    counts = [0] * args.world_size
    outputs = [
        (tmp_dir / f"rank_{rank:05d}.jsonl").open(
            "w", encoding="utf-8", buffering=1024 * 1024
        )
        for rank in range(args.world_size)
    ]
    start = time.monotonic()
    try:
        with args.input.open("r", encoding="utf-8", buffering=1024 * 1024) as source:
            for line_index, line in enumerate(source):
                rank = line_index % args.world_size
                outputs[rank].write(line)
                counts[rank] += 1
                completed = line_index + 1
                if args.log_every > 0 and completed % args.log_every == 0:
                    elapsed = time.monotonic() - start
                    print(
                        f"[shard] lines={completed:,} elapsed={elapsed:.1f}s "
                        f"rate={completed / max(elapsed, 1e-6):,.0f}/s",
                        flush=True,
                    )
    except BaseException:
        print(f"[shard] partial output retained for inspection: {tmp_dir}", flush=True)
        raise
    finally:
        for output in outputs:
            output.close()

    (tmp_dir / "counts.txt").write_text(
        "".join(f"{rank}\t{count}\n" for rank, count in enumerate(counts)),
        encoding="utf-8",
    )
    tmp_dir.rename(output_dir)
    elapsed = time.monotonic() - start
    print(
        f"[shard] done lines={sum(counts):,} elapsed={elapsed:.1f}s out={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Code-shard (chunk npz) dataset for the Audio8 Falcon-H1 trainer.

Reads the 20M Audio8 codec dataset layout produced by
``Audio8_TTS_nodocs/audio8_20m_build``:

    train_code_shards/rank_*/worker_*/chunk_*.npz

Each chunk contains:
    rows_json      object array of JSON strings (text / reference_text / ...)
    line_indices   int array (original manifest line)
    target_codes   object array of [10, T] int arrays (target codec frames)
    ref_codes      object array of [10, T] int arrays (reference frames)
    has_ref        bool array

Examples are built with ``audio8_tts_data.build_sft_example``, i.e. exactly the
same Falcon-H1 prompt/label format as the JSONL + NPY path, so a model can
switch between the two data sources without changing anything else.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


from audio8_tts_data import build_sft_example, clean_text


def rank0_print(*args, **kwargs):
    if int(os.environ.get("RANK", "0")) == 0:
        print(*args, **kwargs, flush=True)


class CodeShardAudio8Dataset(IterableDataset):
    """Iterable Falcon-H1 dataset backed by the 20M chunk-npz code shards."""

    def __init__(
        self,
        code_shard_dir: str,
        tokenizer,
        config,
        max_samples: Optional[int] = None,
        max_length: int = 2048,
        num_codebooks: int = 10,
        use_ref: bool = True,
        text_key: str = "text",
        ref_text_key: Optional[str] = "reference_text",
        skip_samples: int = 0,
    ):
        self.code_shard_dir = pathlib.Path(code_shard_dir)
        self.tokenizer = tokenizer
        self.config = config
        self.max_samples = max_samples
        self.max_length = int(max_length)
        self.num_codebooks = int(num_codebooks)
        self.use_ref = bool(use_ref)
        self.text_key = text_key
        self.ref_text_key = ref_text_key
        self.skip_samples = max(0, int(skip_samples or 0))
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self._chunk_files_by_worker = self._discover_chunks()
        self._length = self._resolve_length()
        rank0_print(
            f"[code-shard] dir={self.code_shard_dir} rank={self.rank}/{self.world_size} "
            f"len={self._length} max_samples={self.max_samples} "
            f"skip_samples={self.skip_samples} workers={len(self._chunk_files_by_worker)}"
        )

    def _discover_chunks(self) -> dict[int, list[pathlib.Path]]:
        rank_dir = self.code_shard_dir / f"rank_{self.rank:05d}"
        if not rank_dir.is_dir():
            raise FileNotFoundError(f"code shard rank dir not found: {rank_dir}")
        chunks: dict[int, list[pathlib.Path]] = {}
        for worker_dir in sorted(rank_dir.glob("worker_*")):
            if not worker_dir.is_dir():
                continue
            try:
                worker_id = int(worker_dir.name.split("_")[-1])
            except ValueError:
                continue
            chunks[worker_id] = sorted(worker_dir.glob("chunk_*.npz"))
        if not chunks:
            raise FileNotFoundError(f"no code shard chunks found under {rank_dir}")
        return chunks

    def _resolve_length(self) -> int:
        total = 0
        for files in self._chunk_files_by_worker.values():
            for path in files:
                with np.load(path, allow_pickle=True) as shard:
                    total += len(shard["rows_json"])
                    if self.max_samples is not None and total >= self.max_samples:
                        return self.max_samples
        return total

    def __len__(self) -> int:
        return max(0, self._length - self.skip_samples)

    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            yield from self._iter_all_workers_in_line_order()
            return
        worker_id = worker.id
        if worker_id not in self._chunk_files_by_worker:
            return
        yield from self._iter_chunk_files(self._chunk_files_by_worker[worker_id])

    def _iter_all_workers_in_line_order(self):
        records = []
        for files in self._chunk_files_by_worker.values():
            for chunk_path in files:
                with np.load(chunk_path, allow_pickle=True) as shard:
                    line_indices = (
                        shard["line_indices"]
                        if "line_indices" in shard.files
                        else np.arange(len(shard["rows_json"]))
                    )
                    for idx, line_idx in enumerate(line_indices):
                        records.append((int(line_idx), chunk_path, idx))
        records.sort(key=lambda x: x[0])
        yielded = 0
        current_path = None
        shard = None
        try:
            for _, chunk_path, idx in records:
                if self.max_samples is not None and yielded >= self.max_samples:
                    return
                if yielded < self.skip_samples:
                    yielded += 1
                    continue
                if current_path != chunk_path:
                    if shard is not None:
                        shard.close()
                    shard = np.load(chunk_path, allow_pickle=True)
                    current_path = chunk_path
                yield self._example_from_loaded_shard(shard, idx)
                yielded += 1
        finally:
            if shard is not None:
                shard.close()

    def _iter_chunk_files(self, chunk_files: list[pathlib.Path]):
        yielded = 0
        for chunk_path in chunk_files:
            with np.load(chunk_path, allow_pickle=True) as shard:
                for idx in range(len(shard["rows_json"])):
                    if self.max_samples is not None and yielded >= self.max_samples:
                        return
                    if yielded < self.skip_samples:
                        yielded += 1
                        continue
                    try:
                        example = self._example_from_loaded_shard(shard, idx)
                    except (OSError, ValueError) as exc:
                        print(
                            f"[data-skip] rank={self.rank} worker={getattr(get_worker_info(), 'id', None)} "
                            f"chunk={chunk_path.name} idx={idx} error={type(exc).__name__}: {exc}",
                            flush=True,
                        )
                        example = None
                    yield example
                    yielded += 1

    def _example_from_loaded_shard(self, shard, idx: int):
        row = json.loads(str(shard["rows_json"][idx]))
        target = torch.from_numpy(np.asarray(shard["target_codes"][idx], dtype=np.int64))
        if target.ndim != 2 or target.shape[0] != self.num_codebooks or target.shape[1] == 0:
            raise ValueError(
                f"invalid target_codes shape {tuple(target.shape)} in {shard.fid} idx={idx}"
            )
        ref = None
        if (
            self.use_ref
            and "ref_codes" in shard.files
            and "has_ref" in shard.files
            and bool(shard["has_ref"][idx])
        ):
            ref = torch.from_numpy(np.asarray(shard["ref_codes"][idx], dtype=np.int64))
            if ref.ndim != 2 or ref.shape[0] != self.num_codebooks or ref.shape[1] == 0:
                raise ValueError(
                    f"invalid ref_codes shape {tuple(ref.shape)} in {shard.fid} idx={idx}"
                )
        text = clean_text(row.get(self.text_key, ""))
        if not text:
            raise ValueError(f"missing/empty {self.text_key!r} in chunk row idx={idx}")
        ref_text = None
        if self.ref_text_key and row.get(self.ref_text_key):
            ref_text = clean_text(row[self.ref_text_key])
        elif ref is not None:
            raise ValueError(f"reference codes present but missing {self.ref_text_key!r}")
        return build_sft_example(
            tokenizer=self.tokenizer,
            config=self.config,
            text=text,
            target_codes=target,
            reference_codes=ref,
            reference_text=ref_text,
            max_length=self.max_length,
        )

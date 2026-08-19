#!/usr/bin/env python3
"""DeepSpeed/transformers training for the audio8_tts Falcon-H1-Tiny init model.

Data source: JSONL manifest with ``text``, ``fish_audio_ids_path``,
``pair_fish_audio_ids_path`` and ``pair_text`` fields. Codec files are
``[10, T]`` integer arrays with values in ``[0, 4095]`` (row 0 is the semantic
code index). The semantic row is offset by ``config.semantic_begin_id`` and
the remaining nine codebook rows are fed to the fast AR branch as-is.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import pathlib
import re
import shutil
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors_file
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
from transformers import AutoModel, AutoProcessor, HfArgumentParser, Trainer, TrainingArguments


from audio8_tts_data import build_sft_example, clean_text
from audio8_code_shard_dataset import CodeShardAudio8Dataset


CODEBOOK_PAD_TOKEN_ID = 0
CODEBOOK_SIZE = 4096
FAST_AR_PREFIXES = (
    "fast_embeddings.",
    "fast_layers.",
    "fast_norm.",
    "fast_output.",
)
FAST_AR_TRAINABLE_PREFIXES = (
    "fast_project_in.",
    *FAST_AR_PREFIXES,
)

logger = logging.getLogger(__name__)


def configure_tmpdir():
    tmpdir = os.environ.setdefault("TMPDIR", "/dev/shm/fish_s2pro_tmp")
    os.environ.setdefault("TEMP", tmpdir)
    os.environ.setdefault("TMP", tmpdir)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", os.path.join(tmpdir, "torch_extensions"))
    pathlib.Path(tmpdir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.environ["TORCH_EXTENSIONS_DIR"]).mkdir(parents=True, exist_ok=True)
    local_rank = os.environ.get("LOCAL_RANK", "0")
    triton_cache = os.environ.get("TRITON_CACHE_DIR") or os.path.join(
        tmpdir, "triton", f"rank_{local_rank}"
    )
    os.environ["TRITON_CACHE_DIR"] = triton_cache
    pathlib.Path(triton_cache).mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = tmpdir


@dataclass
class ModelArguments:
    pretrained_ckpt_path: str = field(
        default=os.environ.get("AUDIO8_INIT_MODEL", "audio8_tts_falcon_h1_tiny_init")
    )
    max_length: int = field(default=2048)
    freeze_fast_ar: bool = field(default=True)
    freeze_slow_ar: bool = field(default=False)


@dataclass
class DataArguments:
    train_jsonl: str = field(
        default=os.environ.get("TRAIN_JSONL", "train.jsonl")
    )
    eval_jsonl: Optional[str] = field(default=None)
    code_shard_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Optional 20M chunk-npz code shard dir; when set, training reads "
            "rank_*/worker_*/chunk_*.npz instead of the JSONL+NPY path."
        },
    )
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=64)
    num_codebooks: int = field(default=10)
    use_ref: bool = field(default=True)
    text_key: str = field(default="text")
    audio_ids_key: str = field(default="fish_audio_ids_path")
    ref_audio_ids_key: str = field(default="pair_fish_audio_ids_path")
    ref_text_key: Optional[str] = field(default="pair_text")
    shard_train_by_rank: bool = field(default=True)
    shard_eval_by_rank: bool = field(default=True)
    local_npy_cache_dir: Optional[str] = field(default=None)
    local_npy_cache_source_prefix: str = field(default="/")
    local_npy_cache_log_every: int = field(default=1000)
    local_npy_cache_read_only: bool = field(default=False)
    local_npy_cache_rank_subdir: bool = field(default=False)
    local_npy_cache_max_files: int = field(
        default=1000000,
        metadata={"help": "Keep at most this many cached .npy files per node; oldest files are removed."},
    )
    local_npy_cache_max_gb: float = field(
        default=20.0,
        metadata={"help": "Keep at most this many GB of cached .npy files per node."},
    )
    local_npy_cache_delete_on_exit: bool = field(
        default=True,
        metadata={"help": "Delete the whole local npy cache after training finishes."},
    )
    stream_train_jsonl: bool = field(default=True)
    skip_train_samples: int = field(
        default=0,
        metadata={"help": "Number of already-consumed examples to skip per rank shard for training."},
    )


@dataclass
class Audio8FalconTrainingArguments(TrainingArguments):
    export_dir: Optional[str] = field(default=None)
    resume_mode: str = field(
        default="none",
        metadata={"help": "auto/latest, model_only/model-only, none, or explicit checkpoint path"},
    )
    base_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Initial weight applied to the slow AR (semantic token) loss."},
    )
    base_loss_weight_final: Optional[float] = field(
        default=None,
        metadata={"help": "Optional final slow AR loss weight for linear decay over training."},
    )
    slow_ar_only: bool = field(
        default=False,
        metadata={"help": "Train only the slow AR and skip the Fast AR forward path."},
    )


def rank0_print(*args, **kwargs):
    if int(os.environ.get("RANK", "0")) == 0:
        print(*args, **kwargs)


def format_fish_reference_text(ref_text: Optional[str]) -> str:
    ref_text = clean_text(ref_text) if ref_text else ""
    if re.search(r"<\|speaker:\d+\|>", ref_text):
        return ref_text
    return f"<|speaker:0|>{ref_text}"


def _npy_cache_entries(cache_dir: pathlib.Path) -> list[tuple[float, pathlib.Path, int]]:
    entries: list[tuple[float, pathlib.Path, int]] = []
    for dirpath, dirnames, filenames in os.walk(cache_dir):
        for name in filenames:
            if not name.endswith(".npy"):
                continue
            path = pathlib.Path(dirpath) / name
            try:
                stat = path.stat()
                entries.append((stat.st_mtime, path, stat.st_size))
            except OSError:
                pass
    return entries


def trim_local_npy_cache(
    cache_dir: pathlib.Path,
    max_files: int = 0,
    max_gb: float = 0.0,
) -> int:
    """Trim the npy cache to ~80% of the limits, removing the oldest files first.

    This scans the cache only when called (from a background cleanup thread).
    """
    max_bytes = int(max_gb * 1024 ** 3) if max_gb > 0 else 0
    if (max_files <= 0 and max_bytes <= 0) or not cache_dir.is_dir():
        return 0
    target_files = max(1, int(max_files * 0.8)) if max_files > 0 else 0
    target_bytes = int(max_bytes * 0.8) if max_bytes > 0 else 0
    lock_path = cache_dir / ".cleanup.lock"
    with open(lock_path, "a+b") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        entries = _npy_cache_entries(cache_dir)
        total_files = len(entries)
        total_bytes = sum(size for _, _, size in entries)
        over_files = max_files > 0 and total_files > max_files
        over_bytes = max_bytes > 0 and total_bytes > max_bytes
        if not over_files and not over_bytes:
            return 0
        entries.sort(key=lambda item: item[0])
        removed = 0
        removed_bytes = 0
        for _, path, size in entries:
            if (
                (max_files <= 0 or total_files - removed <= target_files)
                and (max_bytes <= 0 or total_bytes - removed_bytes <= target_bytes)
            ):
                break
            try:
                path.unlink()
                removed += 1
                removed_bytes += size
            except FileNotFoundError:
                pass
            except OSError:
                pass
            lock = path.with_name(f".{path.name}.lock")
            try:
                lock.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass
        return removed


def cache_cleanup_loop(
    cache_dir: pathlib.Path,
    max_files: int,
    max_gb: float,
    interval: float = 300.0,
) -> None:
    """Periodically trim the cache from a daemon thread; never blocks workers."""
    while True:
        time.sleep(interval)
        try:
            removed = trim_local_npy_cache(cache_dir, max_files, max_gb)
            if removed > 0:
                print(
                    f"[data-cache] background cleanup removed={removed} "
                    f"max_files={max_files} max_gb={max_gb} dir={cache_dir}",
                    flush=True,
                )
        except OSError as exc:
            print(f"[data-cache] background cleanup failed: {exc}", flush=True)


def cleanup_local_npy_cache_dir(cache_dir: Optional[str]) -> int:
    """Delete every file under the local npy cache directory."""
    if not cache_dir:
        return 0
    root = pathlib.Path(cache_dir)
    if not root.is_dir():
        return 0
    removed = 0
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        for name in filenames:
            try:
                (pathlib.Path(dirpath) / name).unlink()
                removed += 1
            except OSError:
                pass
        try:
            os.rmdir(dirpath)
        except OSError:
            pass
    try:
        root.rmdir()
    except OSError:
        pass
    return removed


class Audio8NpyCacheMixin:
    """Local npy cache helpers shared by the map and streaming datasets."""

    def _localize_npy(self, path: str) -> str:
        if self.local_npy_cache_dir is None or self._local_npy_cache_disabled:
            return path

        src = pathlib.Path(path)
        rel = self._source_cache_rel(src)
        cache_dir = self._local_npy_cache_dir_for_worker()
        dst = cache_dir / rel

        if self._cached_npy_exists(dst):
            self._cache_hits += 1
            return str(dst)

        self._cache_misses += 1
        if self.local_npy_cache_read_only:
            legacy_dst = self._find_readonly_legacy_cached_npy(rel)
            if legacy_dst is not None:
                self._cache_legacy_hits += 1
                return str(legacy_dst)
            return path

        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if self.local_npy_cache_rank_subdir:
                if not self._cached_npy_exists(dst):
                    self._copy_npy_to_cache(src, dst)
            else:
                lock_path = dst.with_name(f".{dst.name}.lock")
                with open(lock_path, "a+b") as lock_file:
                    fcntl.flock(lock_file, fcntl.LOCK_EX)
                    if not self._cached_npy_exists(dst):
                        self._copy_npy_to_cache(src, dst)
        except OSError as exc:
            self._disable_local_npy_cache(exc, src)
            return path

        if self.local_npy_cache_log_every > 0 and (
            self._cache_hits + self._cache_misses
        ) % self.local_npy_cache_log_every == 0:
            rank0_print(
                f"[data-cache] rank={self.rank} hits={self._cache_hits} "
                f"misses={self._cache_misses} dir={cache_dir}"
            )
        return str(dst)

    def _local_npy_cache_dir_for_worker(self) -> pathlib.Path:
        if self.local_npy_cache_dir is None:
            raise RuntimeError("local_npy_cache_dir is not configured")
        if self.local_npy_cache_read_only:
            return pathlib.Path(self.local_npy_cache_dir)
        if not self.local_npy_cache_rank_subdir:
            return pathlib.Path(self.local_npy_cache_dir)
        worker = get_worker_info()
        if worker is None:
            return pathlib.Path(self.local_npy_cache_dir)
        return pathlib.Path(self.local_npy_cache_dir) / f"worker_{worker.id:02d}"

    def _find_readonly_legacy_cached_npy(self, rel: pathlib.Path) -> Optional[pathlib.Path]:
        cache_dir = self.local_npy_cache_dir
        if cache_dir is None:
            return None
        worker = get_worker_info()
        worker_ids = [worker.id] if worker is not None else [0, 1, 2, 3]
        for worker_id in worker_ids:
            legacy_path = (
                pathlib.Path(cache_dir)
                / f"rank_{self.rank:05d}"
                / f"worker_{worker_id:02d}"
                / rel
            )
            if self._cached_npy_exists(legacy_path):
                return legacy_path
        return None

    def _disable_local_npy_cache(self, exc: OSError, src: pathlib.Path) -> None:
        self._local_npy_cache_disabled = True
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        print(
            f"[data-cache] rank={self.rank} worker={worker_id} disabling local npy cache "
            f"after {exc.__class__.__name__}: {exc}; fallback_source={src}",
            flush=True,
        )

    def _cached_npy_exists(self, path: pathlib.Path) -> bool:
        try:
            return path.is_file()
        except OSError:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass
            return False

    def _source_cache_rel(self, path: pathlib.Path) -> pathlib.Path:
        try:
            return path.relative_to(self.local_npy_cache_source_prefix)
        except ValueError:
            return pathlib.Path(str(path).lstrip("/"))

    def _copy_npy_to_cache(self, src: pathlib.Path, dst: pathlib.Path) -> None:
        tmp = dst.with_name(f".{dst.name}.{os.getpid()}.{time.time_ns()}.tmp")
        try:
            try:
                if dst.exists():
                    dst.unlink()
            except OSError:
                pass
            shutil.copyfile(src, tmp)
            np.load(tmp, mmap_mode="r")
            os.replace(tmp, dst)
        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass


class Audio8JsonlDataset(Dataset, Audio8NpyCacheMixin):
    def __init__(
        self,
        jsonl_file: str,
        tokenizer,
        config,
        max_samples: Optional[int],
        max_length: int,
        num_codebooks: int,
        use_ref: bool,
        text_key: str,
        audio_ids_key: str,
        ref_audio_ids_key: str,
        ref_text_key: Optional[str],
        shard_by_rank: bool = False,
        local_npy_cache_dir: Optional[str] = None,
        local_npy_cache_source_prefix: str = "/",
        local_npy_cache_log_every: int = 1000,
        local_npy_cache_read_only: bool = False,
        local_npy_cache_rank_subdir: bool = False,
        local_npy_cache_max_files: int = 1000000,
        local_npy_cache_max_gb: float = 20.0,
        local_npy_cache_delete_on_exit: bool = True,
        skip_samples: int = 0,
    ):
        self.jsonl_file = pathlib.Path(jsonl_file)
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = int(max_length)
        self.num_codebooks = int(num_codebooks)
        self.use_ref = bool(use_ref)
        self.text_key = text_key
        self.audio_ids_key = audio_ids_key
        self.ref_audio_ids_key = ref_audio_ids_key
        self.ref_text_key = ref_text_key
        self.shard_by_rank = shard_by_rank
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_npy_cache_dir = (
            pathlib.Path(local_npy_cache_dir).expanduser() if local_npy_cache_dir else None
        )
        self.local_npy_cache_source_prefix = pathlib.Path(local_npy_cache_source_prefix)
        self.local_npy_cache_log_every = local_npy_cache_log_every
        self.local_npy_cache_read_only = local_npy_cache_read_only
        self.local_npy_cache_rank_subdir = local_npy_cache_rank_subdir
        self.local_npy_cache_max_files = int(local_npy_cache_max_files or 0)
        self.local_npy_cache_max_gb = float(local_npy_cache_max_gb or 0.0)
        self.local_npy_cache_delete_on_exit = bool(local_npy_cache_delete_on_exit)
        self.skip_samples = max(0, int(skip_samples or 0))
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_legacy_hits = 0
        self._local_npy_cache_disabled = False
        self.resolved_jsonl_file = self.jsonl_file
        self.using_prebuilt_shard = False
        self.rows = self._load_rows(max_samples)

    def _resolve_rank_shard(self) -> pathlib.Path:
        if not self.shard_by_rank or self.world_size <= 1:
            return self.jsonl_file
        shard_dir = self.jsonl_file.with_name(f"{self.jsonl_file.name}.shards{self.world_size}")
        shard_file = shard_dir / f"rank_{self.rank:05d}.jsonl"
        if shard_file.is_file():
            return shard_file
        return self.jsonl_file

    def _load_rows(self, max_samples: Optional[int]) -> list[dict]:
        jsonl_file = self._resolve_rank_shard()
        if not jsonl_file.is_file():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_file}")
        if (
            self.local_npy_cache_dir
            and self.local_npy_cache_rank_subdir
            and not self.local_npy_cache_read_only
        ):
            self.local_npy_cache_dir = self.local_npy_cache_dir / f"rank_{self.rank:05d}"
        rows = []
        ref_count = 0
        self.resolved_jsonl_file = jsonl_file
        self.using_prebuilt_shard = jsonl_file != self.jsonl_file
        with jsonl_file.open("r", encoding="utf-8") as f:
            selected_idx = 0
            for line_idx, line in enumerate(f):
                if (
                    self.shard_by_rank
                    and self.world_size > 1
                    and not self.using_prebuilt_shard
                    and line_idx % self.world_size != self.rank
                ):
                    continue
                if selected_idx < self.skip_samples:
                    selected_idx += 1
                    continue
                selected_idx += 1
                row = json.loads(line)
                if self.text_key not in row:
                    raise KeyError(f"Missing {self.text_key!r} at line {line_idx}")
                if not row.get(self.audio_ids_key):
                    raise KeyError(f"Missing {self.audio_ids_key!r} at line {line_idx}")
                if row.get(self.ref_audio_ids_key):
                    ref_count += 1
                rows.append(row)
                if max_samples is not None and len(rows) >= max_samples:
                    break
        rank0_print(
            f"[data] {jsonl_file}: {len(rows)} rows loaded "
            f"on rank {self.rank}/{self.world_size}, {ref_count} with ref, "
            f"shard_by_rank={self.shard_by_rank}, prebuilt_shard={self.using_prebuilt_shard}"
        )
        return rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self._row_to_example(self.rows[idx])

    def _row_to_example(self, row: dict):
        text = clean_text(row[self.text_key])
        target_codes = self._load_codes(row[self.audio_ids_key])
        ref_codes = None
        ref_text = None
        ref_path = row.get(self.ref_audio_ids_key)
        if self.use_ref and ref_path:
            ref_codes = self._load_codes(ref_path)
            if self.ref_text_key and row.get(self.ref_text_key):
                ref_text = clean_text(row[self.ref_text_key])
        return build_sft_example(
            tokenizer=self.tokenizer,
            config=self.config,
            text=text,
            target_codes=target_codes,
            reference_codes=ref_codes,
            reference_text=ref_text,
            max_length=self.max_length,
        )

    def _load_codes(self, path: str) -> torch.Tensor:
        last_exc = None
        for _ in range(3):
            load_path = self._localize_npy(path)
            try:
                arr = np.load(load_path)
                if arr.ndim != 2 or arr.shape[0] != self.num_codebooks or arr.shape[1] <= 0:
                    raise ValueError(
                        f"Invalid codec ids at {load_path}: shape={arr.shape}, "
                        f"expected [{self.num_codebooks}, T>0]"
                    )
                if not np.issubdtype(arr.dtype, np.integer):
                    raise ValueError(
                        f"Invalid codec ids at {load_path}: dtype={arr.dtype}, expected integer"
                    )
                value_min = int(arr.min())
                value_max = int(arr.max())
                if value_min < 0 or value_max >= CODEBOOK_SIZE:
                    raise ValueError(
                        f"Invalid codec ids at {load_path}: value_range=[{value_min}, {value_max}], "
                        f"expected [0, {CODEBOOK_SIZE - 1}]"
                    )
                return torch.from_numpy(arr.astype(np.int64, copy=False))
            except (OSError, ValueError) as exc:
                last_exc = exc
                if self._is_local_cached_path(load_path):
                    self._invalidate_cached_npy(load_path, source_path=path)
                    continue
                raise ValueError(f"Failed to load codec ids at {load_path}; source={path}") from exc
        raise ValueError(f"Failed to load codec ids after cache retries; source={path}") from last_exc

    def _is_local_cached_path(self, path: str) -> bool:
        if self.local_npy_cache_dir is None:
            return False
        try:
            pathlib.Path(path).relative_to(self.local_npy_cache_dir)
            return True
        except ValueError:
            return False

    def _invalidate_cached_npy(self, path: str, source_path: Optional[str] = None) -> int:
        removed = 0
        candidates = [pathlib.Path(path)]
        if source_path is not None:
            try:
                candidates.append(self.local_npy_cache_dir / self._source_cache_rel(pathlib.Path(source_path)))
            except (TypeError, ValueError):
                pass
        for cached in candidates:
            try:
                cached.unlink()
                removed += 1
            except FileNotFoundError:
                pass
            except OSError:
                pass
        return removed


class Audio8StreamingJsonlDataset(IterableDataset):
    def __init__(
        self,
        jsonl_file: str,
        tokenizer,
        config,
        max_samples: Optional[int],
        max_length: int,
        num_codebooks: int,
        use_ref: bool,
        text_key: str,
        audio_ids_key: str,
        ref_audio_ids_key: str,
        ref_text_key: Optional[str],
        shard_by_rank: bool = False,
        local_npy_cache_dir: Optional[str] = None,
        local_npy_cache_source_prefix: str = "/",
        local_npy_cache_log_every: int = 1000,
        local_npy_cache_read_only: bool = False,
        local_npy_cache_rank_subdir: bool = False,
        local_npy_cache_max_files: int = 1000000,
        local_npy_cache_max_gb: float = 20.0,
        local_npy_cache_delete_on_exit: bool = True,
        skip_samples: int = 0,
    ):
        self.map_dataset = Audio8JsonlDataset.__new__(Audio8JsonlDataset)
        self.map_dataset.jsonl_file = pathlib.Path(jsonl_file)
        self.map_dataset.tokenizer = tokenizer
        self.map_dataset.config = config
        self.map_dataset.max_length = int(max_length)
        self.map_dataset.num_codebooks = int(num_codebooks)
        self.map_dataset.use_ref = bool(use_ref)
        self.map_dataset.text_key = text_key
        self.map_dataset.audio_ids_key = audio_ids_key
        self.map_dataset.ref_audio_ids_key = ref_audio_ids_key
        self.map_dataset.ref_text_key = ref_text_key
        self.map_dataset.shard_by_rank = shard_by_rank
        self.map_dataset.rank = int(os.environ.get("RANK", "0"))
        self.map_dataset.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.map_dataset.local_npy_cache_dir = (
            pathlib.Path(local_npy_cache_dir).expanduser() if local_npy_cache_dir else None
        )
        self.map_dataset.local_npy_cache_source_prefix = pathlib.Path(local_npy_cache_source_prefix)
        self.map_dataset.local_npy_cache_log_every = local_npy_cache_log_every
        self.map_dataset.local_npy_cache_read_only = local_npy_cache_read_only
        self.map_dataset.local_npy_cache_rank_subdir = local_npy_cache_rank_subdir
        self.map_dataset.local_npy_cache_max_files = int(local_npy_cache_max_files or 0)
        self.map_dataset.local_npy_cache_max_gb = float(local_npy_cache_max_gb or 0.0)
        self.map_dataset.local_npy_cache_delete_on_exit = bool(local_npy_cache_delete_on_exit)
        self.map_dataset._cache_hits = 0
        self.map_dataset._cache_misses = 0
        self.map_dataset._cache_legacy_hits = 0
        self.map_dataset._local_npy_cache_disabled = False
        self.map_dataset.resolved_jsonl_file = self.map_dataset._resolve_rank_shard()
        self.map_dataset.using_prebuilt_shard = (
            self.map_dataset.resolved_jsonl_file != self.map_dataset.jsonl_file
        )
        self.rank = self.map_dataset.rank
        self.world_size = self.map_dataset.world_size
        self.max_samples = max_samples
        self.skip_samples = max(0, int(skip_samples or 0))
        self._length = self._resolve_length()
        if (
            self.map_dataset.local_npy_cache_dir
            and self.map_dataset.local_npy_cache_rank_subdir
            and not self.map_dataset.local_npy_cache_read_only
        ):
            self.map_dataset.local_npy_cache_dir = (
                self.map_dataset.local_npy_cache_dir / f"rank_{self.rank:05d}"
            )
        rank0_print(
            f"[data-stream] {self.map_dataset.resolved_jsonl_file}: len={self._length} "
            f"rank {self.rank}/{self.world_size}, "
            f"prebuilt_shard={self.map_dataset.using_prebuilt_shard}, "
            f"skip_samples={self.skip_samples}"
        )

    def _resolve_length(self) -> int:
        if self.max_samples is not None:
            return self.max_samples
        shard_path = self.map_dataset.resolved_jsonl_file
        counts = shard_path.with_name("counts.txt")
        if counts.is_file():
            with counts.open("r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2 and int(parts[0]) == self.rank:
                        return int(parts[1])
        with shard_path.open("r", encoding="utf-8") as f:
            total_rows = sum(1 for _ in f)
        if (
            self.map_dataset.shard_by_rank
            and self.world_size > 1
            and not self.map_dataset.using_prebuilt_shard
        ):
            rows_per_rank, remainder = divmod(total_rows, self.world_size)
            return rows_per_rank + int(self.rank < remainder)
        return total_rows

    def __len__(self):
        return max(0, self._length - self.skip_samples)

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        num_workers = worker.num_workers if worker else 1
        using_prebuilt_shard = bool(self.map_dataset.using_prebuilt_shard)
        selected_idx = 0
        with self.map_dataset.resolved_jsonl_file.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if (
                    self.map_dataset.shard_by_rank
                    and self.world_size > 1
                    and not using_prebuilt_shard
                    and line_idx % self.world_size != self.rank
                ):
                    continue
                if selected_idx < self.skip_samples:
                    selected_idx += 1
                    continue
                if self.max_samples is not None and selected_idx >= self.max_samples:
                    break
                if selected_idx % num_workers != worker_id:
                    selected_idx += 1
                    continue
                selected_idx += 1
                row = json.loads(line)
                try:
                    example = self.map_dataset._row_to_example(row)
                except (OSError, ValueError) as exc:
                    print(
                        f"[data-skip] rank={self.rank} worker={worker_id} "
                        f"error={type(exc).__name__}: {exc}; "
                        f"target={row.get(self.map_dataset.audio_ids_key)}; "
                        f"ref={row.get(self.map_dataset.ref_audio_ids_key)}",
                        flush=True,
                    )
                    example = None
                yield example


class Audio8Collator:
    def __init__(self, tokenizer, config, max_length: int):
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = int(max_length)

    def __call__(self, examples):
        valid = [example for example in examples if example is not None]
        if not valid:
            raise RuntimeError("Entire batch contains invalid codec samples")
        examples = [example if example is not None else valid[-1] for example in examples]
        max_tokens_length = min(
            max(example["input_ids"].size(1) for example in examples),
            self.max_length,
        )
        pad_id = int(self.config.pad_token_id)
        inputs, attention_masks, labels = [], [], []
        for example in examples:
            tokens = example["input_ids"][:, :max_tokens_length]
            label = example["labels"][:, :max_tokens_length]
            seq_len = tokens.size(1)
            attention = torch.ones((max_tokens_length,), dtype=torch.long)
            if seq_len < max_tokens_length:
                tokens = F.pad(tokens, (0, max_tokens_length - seq_len), value=pad_id)
                tokens[1:, seq_len:] = CODEBOOK_PAD_TOKEN_ID
                label = F.pad(label, (0, max_tokens_length - seq_len), value=-100)
                attention[seq_len:] = 0
            inputs.append(tokens)
            attention_masks.append(attention)
            labels.append(label)
        return {
            "inputs": torch.stack(inputs, dim=0),
            "attention_masks": torch.stack(attention_masks, dim=0),
            "labels": torch.stack(labels, dim=0),
        }


def fast_codebook_logits(model, slow_hidden: torch.Tensor, codebooks: torch.Tensor):
    """Teacher-force the fast AR branch for all ten codebooks in parallel."""
    prefix = codebooks[:, :-1].long()
    fast_dtype = model.fast_embeddings.weight.dtype
    hidden = model.fast_project_in(slow_hidden.to(dtype=fast_dtype))
    hidden = torch.cat((hidden[:, None, :], model.fast_embeddings(prefix)), dim=1)
    length = int(hidden.shape[1])
    positions = torch.arange(length, device=hidden.device)
    attention_mask = torch.ones((hidden.shape[0], length), dtype=torch.long, device=hidden.device)
    mask = model._causal_mask(attention_mask, positions, length)
    rope = model.fast_freqs_cis[:length]
    for layer in model.fast_layers:
        hidden = layer(hidden, rope, mask)
    return model.fast_output(model.fast_norm(hidden))


class Audio8FalconTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loss_metric_sums: dict[str, float] = {}
        self._loss_metric_count = 0

    def _base_loss_weight(self) -> float:
        start = float(getattr(self.args, "base_loss_weight", 1.0))
        final = getattr(self.args, "base_loss_weight_final", None)
        if final is None:
            return start
        max_steps = max(1, int(getattr(self.state, "max_steps", 0) or 0))
        step = min(max(0, int(getattr(self.state, "global_step", 0) or 0)), max_steps)
        progress = step / max_steps
        return start + (float(final) - start) * progress

    def _record_loss_metrics(self, metrics: dict[str, torch.Tensor]) -> None:
        self._loss_metric_count += 1
        for key, value in metrics.items():
            self._loss_metric_sums[key] = self._loss_metric_sums.get(key, 0.0) + float(
                value.detach().float().cpu().item()
            )

    def log(self, logs, start_time=None):
        if self._loss_metric_count > 0:
            for key, value in self._loss_metric_sums.items():
                logs.setdefault(key, value / self._loss_metric_count)
            self._loss_metric_sums.clear()
            self._loss_metric_count = 0
        return super().log(logs, start_time=start_time)

    def _get_local_dataloader(self, dataset, batch_size: int, sampler_fn, is_training=False):
        params = {
            "batch_size": batch_size,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "drop_last": self.args.dataloader_drop_last,
        }
        if self.args.dataloader_num_workers > 0:
            params["prefetch_factor"] = self.args.dataloader_prefetch_factor
        return DataLoader(dataset, **params)

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if isinstance(self.train_dataset, IterableDataset):
            return self._get_local_dataloader(
                dataset=self.train_dataset,
                batch_size=self._train_batch_size,
                sampler_fn=None,
                is_training=True,
            )
        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset=None):
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        if isinstance(dataset, IterableDataset):
            return self._get_local_dataloader(
                dataset=dataset,
                batch_size=self.args.eval_batch_size,
                sampler_fn=None,
            )
        return super().get_eval_dataloader(dataset)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        actual_model = model.module if hasattr(model, "module") else model
        config = actual_model.config
        labels = inputs["labels"]
        slow_ar_only = bool(getattr(self.args, "slow_ar_only", False))

        outputs = actual_model(
            input_ids=inputs["inputs"],
            attention_mask=inputs["attention_masks"],
            return_dict=True,
        )
        token_logits = outputs.logits
        raw_labels = labels[:, 0]
        compact = token_logits.size(-1) == config.codebook_size + 1
        if compact:
            # Map full-vocab semantic/EOS ids onto the compact 4097-way head:
            # 0..4095 = semantic, 4096 = EOS.
            slow_labels = torch.full_like(raw_labels, -100)
            semantic_label_pos = (raw_labels >= config.semantic_begin_id) & (
                raw_labels <= config.semantic_end_id
            )
            eos_label_pos = raw_labels == int(config.eos_token_id)
            slow_labels[semantic_label_pos] = (
                raw_labels[semantic_label_pos] - config.semantic_begin_id
            )
            slow_labels[eos_label_pos] = int(config.codebook_size)
        else:
            slow_labels = raw_labels
        base_loss = F.cross_entropy(
            token_logits.float().reshape(-1, token_logits.size(-1)),
            slow_labels.reshape(-1),
            ignore_index=-100,
        )

        semantic_mask = (labels[:, 0] >= config.semantic_begin_id) & (
            labels[:, 0] <= config.semantic_end_id
        )
        if slow_ar_only:
            loss = base_loss
            codebook_logits = None
        else:
            if not semantic_mask.any():
                raise ValueError("batch contains no supervised semantic frames")
            slow_hidden = outputs.hidden_states[semantic_mask]
            codebooks = labels[:, 1 : 1 + config.num_codebooks].permute(0, 2, 1)[semantic_mask]
            bad = (codebooks < 0) | (codebooks >= config.codebook_size)
            if bad.any().item():
                raise RuntimeError("Invalid Fast AR codebook id before model forward")
            codebook_logits = fast_codebook_logits(actual_model, slow_hidden, codebooks)
            semantic_loss = F.cross_entropy(
                codebook_logits.float().reshape(-1, codebook_logits.size(-1)),
                codebooks.reshape(-1),
                ignore_index=-100,
            )
            base_loss_weight = self._base_loss_weight()
            loss = base_loss * base_loss_weight + semantic_loss

        with torch.no_grad():
            token_ids = labels[:, 0]
            eos_mask = token_ids == int(config.eos_token_id)
            eos_count = eos_mask.sum()
            if eos_count.item() > 0:
                eos_logits = token_logits[eos_mask]
                eos_target = int(config.codebook_size) if compact else int(config.eos_token_id)
                eos_top1 = (eos_logits.argmax(dim=-1) == eos_target).float().mean()
                eos_prob = torch.softmax(eos_logits.float(), dim=-1)[:, eos_target].mean()
            else:
                eos_top1 = base_loss.new_tensor(0.0)
                eos_prob = base_loss.new_tensor(0.0)
            semantic_count = semantic_mask.sum().float()

        metrics = {
            "base_loss": base_loss,
            "eos_top1": eos_top1,
            "eos_prob": eos_prob,
            "eos_count": eos_count.float(),
            "semantic_tokens": semantic_count,
        }
        if not slow_ar_only:
            metrics["semantic_loss"] = semantic_loss
            with torch.no_grad():
                pred = codebook_logits.argmax(dim=-1)
                valid = codebooks != -100
                metrics["fast_top1"] = (
                    (pred == codebooks) & valid
                ).sum().float() / valid.sum().float().clamp_min(1.0)
        self._record_loss_metrics(metrics)

        if return_outputs:
            return loss, {"token_logits": token_logits, "codebook_logits": codebook_logits}
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, return_outputs=False)
        return loss.detach(), None, None


def freeze_fast_ar(model: torch.nn.Module, include_project_in: bool = False):
    named = dict(model.named_parameters())
    frozen = 0
    trainable = 0
    for name, param in named.items():
        is_fast_ar = name.startswith(FAST_AR_PREFIXES) or (
            include_project_in and name.startswith("fast_project_in.")
        )
        if is_fast_ar:
            param.requires_grad = False
            frozen += param.numel()
        else:
            param.requires_grad = True
            trainable += param.numel()
    rank0_print(
        f"[freeze] frozen Fast AR: {frozen / 1e6:.2f}M params; "
        f"trainable: {trainable / 1e6:.2f}M params"
    )


def freeze_slow_ar(model: torch.nn.Module):
    frozen = 0
    trainable = 0
    for name, param in model.named_parameters():
        if name.startswith(FAST_AR_TRAINABLE_PREFIXES):
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
            frozen += param.numel()
    if trainable == 0:
        raise RuntimeError("freeze_slow_ar left no trainable Fast AR parameters")
    rank0_print(
        f"[freeze] frozen Slow AR: {frozen / 1e6:.2f}M params; "
        f"trainable Fast AR: {trainable / 1e6:.2f}M params"
    )


def resolve_resume_checkpoint(output_dir: str, resume_mode: str) -> Optional[str]:
    mode = str(resume_mode or "auto").strip()
    mode_lower = mode.lower()
    if mode_lower in {"model_only", "model-only"}:
        mode = "auto"
        mode_lower = "auto"
    if mode_lower in {"", "auto", "latest", "true"}:
        checkpoint_dirs = list(pathlib.Path(output_dir).glob("checkpoint-*"))
        if not checkpoint_dirs:
            return None
        checkpoint_dirs.sort(key=lambda x: int(x.name.split("-")[-1]))
        return str(checkpoint_dirs[-1])
    if mode_lower in {"none", "false", "no", "0"}:
        return None
    return mode


def is_model_only_resume(resume_mode: str) -> bool:
    return str(resume_mode or "").strip().lower() in {"model_only", "model-only"}


def load_model_only_checkpoint(model: torch.nn.Module, checkpoint_dir: str):
    checkpoint_path = pathlib.Path(checkpoint_dir)
    safetensors_path = checkpoint_path / "model.safetensors"
    if not safetensors_path.exists():
        raise FileNotFoundError(f"model-only resume requires {safetensors_path}")
    state_dict = load_safetensors_file(str(safetensors_path), device="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    rank0_print(
        f"[resume:model_only] loaded {safetensors_path}; "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )


def read_checkpoint_global_step(checkpoint_dir: Optional[str]) -> int:
    if not checkpoint_dir:
        return 0
    trainer_state = pathlib.Path(checkpoint_dir) / "trainer_state.json"
    if not trainer_state.exists():
        return 0
    with trainer_state.open("r", encoding="utf-8") as f:
        state = json.load(f)
    return int(state.get("global_step") or 0)


def export_audio8_pretrained(trainer: Trainer, export_dir: Optional[str], source_dir: str):
    if not export_dir or int(os.environ.get("RANK", "0")) != 0:
        return
    export_path = pathlib.Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    model.save_pretrained(str(export_path))
    for name in [
        "configuration_arktts.py",
        "modeling_arktts.py",
        "modeling_arktts_codec.py",
        "processing_arktts.py",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "README.md",
    ]:
        src = pathlib.Path(source_dir) / name
        if src.exists():
            shutil.copy2(src, export_path / name)
    rank0_print(f"[export] audio8 Falcon model saved to {export_path}")


def main():
    configure_tmpdir()
    parser = HfArgumentParser((ModelArguments, DataArguments, Audio8FalconTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    training_args.remove_unused_columns = False

    if model_args.freeze_fast_ar and model_args.freeze_slow_ar:
        raise ValueError("freeze_fast_ar and freeze_slow_ar cannot both be true")
    if training_args.slow_ar_only and not model_args.freeze_fast_ar:
        raise ValueError("slow_ar_only requires freeze_fast_ar=true")

    rank0_print("[load] processor/tokenizer")
    processor = AutoProcessor.from_pretrained(model_args.pretrained_ckpt_path, trust_remote_code=True)
    tokenizer = processor.tokenizer

    rank0_print("[load] model")
    torch_dtype = torch.bfloat16 if training_args.bf16 else None
    model = AutoModel.from_pretrained(
        model_args.pretrained_ckpt_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )

    resume = resolve_resume_checkpoint(training_args.output_dir, training_args.resume_mode)
    model_only_resume = is_model_only_resume(training_args.resume_mode)
    effective_skip_train_samples = int(data_args.skip_train_samples or 0)
    if model_only_resume and effective_skip_train_samples <= 0:
        effective_skip_train_samples = read_checkpoint_global_step(resume) * int(
            training_args.per_device_train_batch_size
        )
    if effective_skip_train_samples > 0:
        rank0_print(
            f"[data-skip] skip_train_samples={effective_skip_train_samples} "
            f"per rank shard (resume={resume})"
        )
    if model_only_resume and resume:
        load_model_only_checkpoint(model, resume)

    if model_args.freeze_fast_ar:
        freeze_fast_ar(model, include_project_in=training_args.slow_ar_only)
    elif model_args.freeze_slow_ar:
        freeze_slow_ar(model)

    dataset_kwargs = dict(
        jsonl_file=data_args.train_jsonl,
        tokenizer=tokenizer,
        config=model.config,
        max_samples=data_args.max_train_samples,
        max_length=model_args.max_length,
        num_codebooks=data_args.num_codebooks,
        use_ref=data_args.use_ref,
        text_key=data_args.text_key,
        audio_ids_key=data_args.audio_ids_key,
        ref_audio_ids_key=data_args.ref_audio_ids_key,
        ref_text_key=data_args.ref_text_key,
        shard_by_rank=data_args.shard_train_by_rank,
        local_npy_cache_dir=data_args.local_npy_cache_dir,
        local_npy_cache_source_prefix=data_args.local_npy_cache_source_prefix,
        local_npy_cache_log_every=data_args.local_npy_cache_log_every,
        local_npy_cache_read_only=data_args.local_npy_cache_read_only,
        local_npy_cache_rank_subdir=data_args.local_npy_cache_rank_subdir,
        local_npy_cache_max_files=data_args.local_npy_cache_max_files,
        local_npy_cache_max_gb=data_args.local_npy_cache_max_gb,
        local_npy_cache_delete_on_exit=data_args.local_npy_cache_delete_on_exit,
        skip_samples=effective_skip_train_samples,
    )
    rank0_print("[load] datasets")
    if data_args.code_shard_dir:
        train_dataset = CodeShardAudio8Dataset(
            code_shard_dir=data_args.code_shard_dir,
            tokenizer=tokenizer,
            config=model.config,
            max_samples=data_args.max_train_samples,
            max_length=model_args.max_length,
            num_codebooks=data_args.num_codebooks,
            use_ref=data_args.use_ref,
            text_key=data_args.text_key,
            ref_text_key=data_args.ref_text_key,
            skip_samples=effective_skip_train_samples,
        )
    elif data_args.stream_train_jsonl:
        train_dataset = Audio8StreamingJsonlDataset(**dataset_kwargs)
    else:
        train_dataset = Audio8JsonlDataset(**dataset_kwargs)

    eval_dataset = None
    if data_args.eval_jsonl:
        eval_dataset = Audio8JsonlDataset(
            jsonl_file=data_args.eval_jsonl,
            tokenizer=tokenizer,
            config=model.config,
            max_samples=data_args.max_eval_samples,
            max_length=model_args.max_length,
            num_codebooks=data_args.num_codebooks,
            use_ref=data_args.use_ref,
            text_key=data_args.text_key,
            audio_ids_key=data_args.audio_ids_key,
            ref_audio_ids_key=data_args.ref_audio_ids_key,
            ref_text_key=data_args.ref_text_key,
            shard_by_rank=data_args.shard_eval_by_rank,
            local_npy_cache_dir=data_args.local_npy_cache_dir,
            local_npy_cache_source_prefix=data_args.local_npy_cache_source_prefix,
            local_npy_cache_log_every=data_args.local_npy_cache_log_every,
            local_npy_cache_read_only=data_args.local_npy_cache_read_only,
            local_npy_cache_rank_subdir=data_args.local_npy_cache_rank_subdir,
            local_npy_cache_max_files=data_args.local_npy_cache_max_files,
            local_npy_cache_max_gb=data_args.local_npy_cache_max_gb,
            local_npy_cache_delete_on_exit=data_args.local_npy_cache_delete_on_exit,
        )

    collator = Audio8Collator(tokenizer=tokenizer, config=model.config, max_length=model_args.max_length)
    trainer = Audio8FalconTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    if (
        data_args.local_npy_cache_dir
        and not data_args.code_shard_dir
        and not data_args.local_npy_cache_read_only
        and int(os.environ.get("LOCAL_RANK", "0")) == 0
        and (
            data_args.local_npy_cache_max_files > 0
            or data_args.local_npy_cache_max_gb > 0
        )
    ):
        threading.Thread(
            target=cache_cleanup_loop,
            args=(
                pathlib.Path(data_args.local_npy_cache_dir),
                int(data_args.local_npy_cache_max_files),
                float(data_args.local_npy_cache_max_gb),
            ),
            daemon=True,
        ).start()

    if training_args.do_train:
        rank0_print(f"[resume] mode={training_args.resume_mode} checkpoint={resume}")
        train_result = trainer.train(resume_from_checkpoint=None if model_only_resume else resume)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    trainer.save_model(training_args.output_dir)
    if training_args.export_dir:
        export_audio8_pretrained(trainer, training_args.export_dir, model_args.pretrained_ckpt_path)

    if (
        data_args.local_npy_cache_dir
        and not data_args.code_shard_dir
        and not data_args.local_npy_cache_read_only
        and data_args.local_npy_cache_delete_on_exit
        and int(os.environ.get("LOCAL_RANK", "0")) == 0
    ):
        removed = cleanup_local_npy_cache_dir(data_args.local_npy_cache_dir)
        rank0_print(
            f"[data-cache] cleanup_on_exit removed={removed} dir={data_args.local_npy_cache_dir}"
        )


if __name__ == "__main__":
    main()

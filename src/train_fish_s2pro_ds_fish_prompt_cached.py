#!/usr/bin/env python3
import json
import fcntl
import logging
import os
import pathlib
import re
import shutil
import sys
import tempfile
import time
import types
from dataclasses import asdict, is_dataclass
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors_file
from torch.utils.data import DataLoader, Dataset, IterableDataset, SequentialSampler, get_worker_info
from transformers import HfArgumentParser, Trainer, TrainingArguments


FISH_REPO = os.environ.get("FISH_SPEECH_ROOT", "/opt/src/fish-speech")
if FISH_REPO not in sys.path:
    sys.path.insert(0, FISH_REPO)

from fish_speech.content_sequence import TextPart, VQPart
from fish_speech.conversation import Conversation, Message
from fish_speech.models.text2semantic.llama import BaseTransformer
from fish_speech.tokenizer import FishTokenizer


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
    tempfile.tempdir = tmpdir


@dataclass
class ModelArguments:
    pretrained_ckpt_path: str = field(
        default=os.environ.get("MODEL_PATH", "model")
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
    max_train_samples: Optional[int] = field(default=None)
    max_eval_samples: Optional[int] = field(default=64)
    num_codebooks: int = field(default=10)
    use_ref: bool = field(default=True)
    text_key: str = field(default="text")
    audio_ids_key: str = field(default="fish_audio_ids_path")
    ref_audio_ids_key: str = field(default="pair_fish_audio_ids_path")
    ref_text_key: Optional[str] = field(default=None)
    shard_train_by_rank: bool = field(default=True)
    shard_eval_by_rank: bool = field(default=True)
    local_npy_cache_dir: Optional[str] = field(default=None)
    local_npy_cache_source_prefix: str = field(default="/")
    local_npy_cache_log_every: int = field(default=1000)
    local_npy_cache_read_only: bool = field(default=False)
    local_npy_cache_rank_subdir: bool = field(default=False)
    stream_train_jsonl: bool = field(default=True)
    skip_train_samples: int = field(
        default=0,
        metadata={"help": "Number of already-consumed examples to skip per rank shard for training."},
    )
    code_shard_dir: Optional[str] = field(default=None)
    code_shard_local_cache_dir: Optional[str] = field(default=None)
    code_shard_local_cache_read_only: bool = field(default=True)


@dataclass
class FishTrainingArguments(TrainingArguments):
    export_dir: Optional[str] = field(default=None)
    resume_mode: str = field(
        default="auto",
        metadata={"help": "auto/latest, model_only/model-only, none, or explicit checkpoint path"},
    )
    base_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Initial weight applied to base/token loss."},
    )
    base_loss_weight_final: Optional[float] = field(
        default=None,
        metadata={"help": "Optional final base loss weight for linear decay over training."},
    )
    slow_ar_only: bool = field(
        default=False,
        metadata={"help": "Train only Slow AR and skip the complete Fast AR forward path."},
    )


def rank0_print(*args, **kwargs):
    if int(os.environ.get("RANK", "0")) == 0:
        print(*args, **kwargs)


def clean_text_for_train(text: str) -> str:
    return " ".join(str(text).strip().split())


def format_fish_reference_text(ref_text: Optional[str]) -> str:
    ref_text = clean_text_for_train(ref_text or "")
    if re.search(r"<\|speaker:\d+\|>", ref_text):
        return ref_text
    return f"<|speaker:0|>{ref_text}"


class JsonlFishAudioDataset(Dataset):
    def __init__(
        self,
        jsonl_file: str,
        tokenizer: FishTokenizer,
        max_samples: Optional[int],
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
        skip_samples: int = 0,
    ):
        self.jsonl_file = pathlib.Path(jsonl_file)
        self.tokenizer = tokenizer
        self.num_codebooks = num_codebooks
        self.use_ref = use_ref
        self.text_key = text_key
        self.audio_ids_key = audio_ids_key
        self.ref_audio_ids_key = ref_audio_ids_key
        self.ref_text_key = ref_text_key
        self.shard_by_rank = shard_by_rank
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_npy_cache_dir = pathlib.Path(local_npy_cache_dir).expanduser() if local_npy_cache_dir else None
        self.local_npy_cache_source_prefix = pathlib.Path(local_npy_cache_source_prefix)
        self.local_npy_cache_log_every = local_npy_cache_log_every
        self.local_npy_cache_read_only = local_npy_cache_read_only
        self.local_npy_cache_rank_subdir = local_npy_cache_rank_subdir
        self.skip_samples = max(0, int(skip_samples or 0))
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_legacy_hits = 0
        self._cache_bad_files = 0
        self._cache_bad_removed = 0
        self._local_npy_cache_disabled = False
        self.resolved_jsonl_file = self.jsonl_file
        self.using_prebuilt_shard = False
        self.rows = self._load_rows(max_samples)

    def _load_rows(self, max_samples: Optional[int]) -> list[dict]:
        jsonl_file = self._resolve_rank_shard()
        if not jsonl_file.is_file():
            raise FileNotFoundError(f"JSONL file not found: {jsonl_file}")

        rows = []
        ref_count = 0
        using_prebuilt_shard = jsonl_file != self.jsonl_file
        self.resolved_jsonl_file = jsonl_file
        self.using_prebuilt_shard = using_prebuilt_shard
        with jsonl_file.open("r", encoding="utf-8") as f:
            selected_idx = 0
            for line_idx, line in enumerate(f):
                if (
                    self.shard_by_rank
                    and self.world_size > 1
                    and not using_prebuilt_shard
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
            f"on rank {self.rank}/{self.world_size}, "
            f"{ref_count} with {self.ref_audio_ids_key}, "
            f"shard_by_rank={self.shard_by_rank}, "
            f"prebuilt_shard={using_prebuilt_shard}, "
            f"skip_samples={self.skip_samples}"
        )
        if self.local_npy_cache_dir:
            if self.local_npy_cache_rank_subdir and not self.local_npy_cache_read_only:
                self.local_npy_cache_dir = self.local_npy_cache_dir / f"rank_{self.rank:05d}"
            rank0_print(
                f"[data-cache] rank={self.rank} local_npy_cache_dir={self.local_npy_cache_dir} "
                f"source_prefix={self.local_npy_cache_source_prefix} "
                f"read_only={self.local_npy_cache_read_only} "
                f"rank_subdir={self.local_npy_cache_rank_subdir}"
            )
        return rows

    def _resolve_rank_shard(self) -> pathlib.Path:
        if not self.shard_by_rank or self.world_size <= 1:
            return self.jsonl_file
        shard_dir = self.jsonl_file.with_name(f"{self.jsonl_file.name}.shards{self.world_size}")
        shard_file = shard_dir / f"rank_{self.rank:05d}.jsonl"
        if shard_file.is_file():
            return shard_file
        return self.jsonl_file

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self._row_to_example(self.rows[idx])

    def _row_to_example(self, row: dict):
        text = clean_text_for_train(row[self.text_key])
        target_codes = self._load_codes(row[self.audio_ids_key])

        ref_codes = None
        ref_text = None
        ref_path = row.get(self.ref_audio_ids_key)
        if self.use_ref and ref_path:
            ref_codes = self._load_codes(ref_path)
            if self.ref_text_key and row.get(self.ref_text_key):
                ref_text = clean_text_for_train(row[self.ref_text_key])

        tokens, labels = self._pack_sample(
            text=text,
            target_codes=target_codes,
            ref_codes=ref_codes,
            ref_text=ref_text,
        )
        return {
            "tokens": tokens,
            "labels": labels,
            "_debug_audio_path": row.get(self.audio_ids_key),
            "_debug_ref_audio_path": row.get(self.ref_audio_ids_key),
        }

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
                    removed = self._invalidate_cached_npy(load_path, source_path=path)
                    self._cache_bad_files += 1
                    self._cache_bad_removed += removed
                    total = self._cache_hits + self._cache_misses + self._cache_bad_files
                    if self.local_npy_cache_log_every > 0 and total % self.local_npy_cache_log_every == 0:
                        print(
                            f"[data-cache] rank={self.rank} bad_cached_npy={self._cache_bad_files} "
                            f"removed_aliases={self._cache_bad_removed}; using source fallback",
                            flush=True,
                        )
                    if removed > 0:
                        continue
                    self._disable_local_npy_cache(
                        OSError(f"failed to load cached npy: {load_path}: {exc}"),
                        pathlib.Path(path),
                    )
                    continue
                raise ValueError(f"Failed to load codec ids at {load_path}; source={path}") from exc
        raise ValueError(f"Failed to load codec ids after cache retries; source={path}") from last_exc

    def _localize_npy(self, path: str) -> str:
        if self.local_npy_cache_dir is None or self._local_npy_cache_disabled:
            return path

        src = pathlib.Path(path)
        rel = self._source_cache_rel(src)

        try:
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
                    total = self._cache_hits + self._cache_misses
                    if self.local_npy_cache_log_every > 0 and total % self.local_npy_cache_log_every == 0:
                        rank0_print(
                            f"[data-cache] rank={self.rank} hits={self._cache_hits} "
                            f"legacy_hits={self._cache_legacy_hits} "
                            f"shared_misses={self._cache_misses} "
                            f"source_misses={self._cache_misses - self._cache_legacy_hits} "
                            f"read_only=true dir={cache_dir}"
                        )
                    return str(legacy_dst)

                total = self._cache_hits + self._cache_misses
                if self.local_npy_cache_log_every > 0 and total % self.local_npy_cache_log_every == 0:
                    rank0_print(
                        f"[data-cache] rank={self.rank} hits={self._cache_hits} "
                        f"legacy_hits={self._cache_legacy_hits} "
                        f"shared_misses={self._cache_misses} "
                        f"source_misses={self._cache_misses - self._cache_legacy_hits} "
                        f"read_only=true dir={cache_dir}"
                    )
                return path

            dst.parent.mkdir(parents=True, exist_ok=True)
            if self.local_npy_cache_rank_subdir:
                # Rank/worker-private cache directories do not need lock files.
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

        total = self._cache_hits + self._cache_misses
        if self.local_npy_cache_log_every > 0 and total % self.local_npy_cache_log_every == 0:
            rank0_print(
                f"[data-cache] rank={self.rank} hits={self._cache_hits} "
                f"misses={self._cache_misses} dir={cache_dir}"
            )
        return str(dst)

    def _local_npy_cache_dir_for_worker(self) -> pathlib.Path:
        cache_dir = self.local_npy_cache_dir
        if cache_dir is None:
            raise RuntimeError("local_npy_cache_dir is not configured")
        if self.local_npy_cache_read_only:
            return cache_dir
        if not self.local_npy_cache_rank_subdir:
            return cache_dir
        worker = get_worker_info()
        if worker is None:
            return cache_dir
        return cache_dir / f"worker_{worker.id:02d}"

    def _find_readonly_legacy_cached_npy(self, rel: pathlib.Path) -> Optional[pathlib.Path]:
        cache_dir = self.local_npy_cache_dir
        if cache_dir is None:
            return None
        worker = get_worker_info()
        worker_ids = [worker.id] if worker is not None else [0, 1, 2, 3]
        for worker_id in worker_ids:
            legacy_path = cache_dir / f"rank_{self.rank:05d}" / f"worker_{worker_id:02d}" / rel
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

    def _cached_path_rel(self, path: pathlib.Path) -> Optional[pathlib.Path]:
        if self.local_npy_cache_dir is None:
            return None
        try:
            rel = path.relative_to(self.local_npy_cache_dir)
        except ValueError:
            return None
        parts = rel.parts
        if (
            len(parts) >= 3
            and parts[0].startswith("rank_")
            and parts[1].startswith("worker_")
        ):
            return pathlib.Path(*parts[2:])
        return rel

    def _candidate_cached_npy_paths(
        self, load_path: str, source_path: Optional[str] = None
    ) -> list[pathlib.Path]:
        if self.local_npy_cache_dir is None:
            return []

        paths: list[pathlib.Path] = []
        seen: set[str] = set()

        def add(path: pathlib.Path) -> None:
            key = str(path)
            if key not in seen:
                seen.add(key)
                paths.append(path)

        cached = pathlib.Path(load_path)
        rel = self._cached_path_rel(cached)
        if rel is not None:
            add(cached)
        if source_path is not None:
            rel = self._source_cache_rel(pathlib.Path(source_path))
        if rel is None:
            return paths

        add(self.local_npy_cache_dir / rel)
        worker = get_worker_info()
        worker_ids = range(worker.num_workers) if worker is not None else range(16)
        for worker_id in worker_ids:
            add(
                self.local_npy_cache_dir
                / f"rank_{self.rank:05d}"
                / f"worker_{worker_id:02d}"
                / rel
            )
        return paths

    def _invalidate_cached_npy(self, path: str, source_path: Optional[str] = None) -> int:
        removed = 0
        failed = []
        for cached in self._candidate_cached_npy_paths(path, source_path=source_path):
            try:
                cached.unlink()
                removed += 1
            except FileNotFoundError:
                pass
            except OSError as exc:
                failed.append(f"{cached}: {exc}")
        if failed:
            worker = get_worker_info()
            worker_id = worker.id if worker else 0
            print(
                f"[data-cache] rank={self.rank} worker={worker_id} "
                f"failed to remove bad cached npy aliases: {'; '.join(failed[:4])}",
                flush=True,
            )
        return removed

    def _is_local_cached_path(self, path: str) -> bool:
        if self.local_npy_cache_dir is None:
            return False
        try:
            pathlib.Path(path).relative_to(self.local_npy_cache_dir)
            return True
        except ValueError:
            return False

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

    def _pack_sample(
        self,
        text: str,
        target_codes: torch.Tensor,
        ref_codes: Optional[torch.Tensor],
        ref_text: Optional[str],
    ):
        conversation = Conversation()

        if ref_codes is None:
            system_parts = [
                TextPart(text="convert the provided text to speech", cal_loss=False)
            ]
        else:
            system_parts = [
                TextPart(
                    text="convert the provided text to speech reference to the following:\n\nText:\n",
                    cal_loss=False,
                ),
                TextPart(text=format_fish_reference_text(ref_text), cal_loss=False),
                TextPart(text="\n\nSpeech:\n", cal_loss=False),
                VQPart(codes=ref_codes, cal_loss=False),
            ]

        conversation.append(
            Message(
                role="system",
                parts=system_parts,
                cal_loss=False,
                add_im_start=True,
                add_im_end=True,
            )
        )

        conversation.append(
            Message(
                role="user",
                parts=[TextPart(text=text, cal_loss=False)],
                cal_loss=False,
                add_im_start=True,
                add_im_end=True,
            )
        )

        conversation.append(
            Message(
                role="assistant",
                parts=[VQPart(codes=target_codes)],
                cal_loss=True,
                modality="voice",
                add_im_start=True,
                add_im_end=True,
            )
        )

        encoded = conversation.to_content_sequence().encode(tokenizer=self.tokenizer)
        tokens_raw = encoded.tokens
        tokens = torch.zeros((self.num_codebooks + 1, len(tokens_raw)), dtype=torch.long)
        tokens[0] = tokens_raw

        vq_parts = torch.cat([part.to(tokens.device) for part in encoded.vq_parts], dim=1)
        tokens[1:, encoded.vq_mask_tokens] = vq_parts.long()

        labels = torch.full((self.num_codebooks + 1, len(encoded.labels)), -100, dtype=torch.long)
        labels[0, :] = encoded.labels
        labels[1:, encoded.vq_mask_labels] = vq_parts.long()
        labels[1:, -1:] = CODEBOOK_PAD_TOKEN_ID

        return tokens, labels


class StreamingJsonlFishAudioDataset(IterableDataset):
    def __init__(
        self,
        jsonl_file: str,
        tokenizer: FishTokenizer,
        max_samples: Optional[int],
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
        skip_samples: int = 0,
    ):
        self.map_dataset = JsonlFishAudioDataset.__new__(JsonlFishAudioDataset)
        self.map_dataset.jsonl_file = pathlib.Path(jsonl_file)
        self.map_dataset.tokenizer = tokenizer
        self.map_dataset.num_codebooks = num_codebooks
        self.map_dataset.use_ref = use_ref
        self.map_dataset.text_key = text_key
        self.map_dataset.audio_ids_key = audio_ids_key
        self.map_dataset.ref_audio_ids_key = ref_audio_ids_key
        self.map_dataset.ref_text_key = ref_text_key
        self.map_dataset.shard_by_rank = shard_by_rank
        self.map_dataset.rank = int(os.environ.get("RANK", "0"))
        self.map_dataset.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.map_dataset.local_npy_cache_dir = pathlib.Path(local_npy_cache_dir).expanduser() if local_npy_cache_dir else None
        self.map_dataset.local_npy_cache_source_prefix = pathlib.Path(local_npy_cache_source_prefix)
        self.map_dataset.local_npy_cache_log_every = local_npy_cache_log_every
        self.map_dataset.local_npy_cache_read_only = local_npy_cache_read_only
        self.map_dataset.local_npy_cache_rank_subdir = local_npy_cache_rank_subdir
        self.map_dataset._cache_hits = 0
        self.map_dataset._cache_misses = 0
        self.map_dataset._cache_legacy_hits = 0
        self.map_dataset._cache_bad_files = 0
        self.map_dataset._cache_bad_removed = 0
        self.map_dataset._local_npy_cache_disabled = False
        self.map_dataset.resolved_jsonl_file = self.map_dataset._resolve_rank_shard()
        self.map_dataset.using_prebuilt_shard = self.map_dataset.resolved_jsonl_file != self.map_dataset.jsonl_file
        self.max_samples = max_samples
        self.skip_samples = max(0, int(skip_samples or 0))
        self._length = self._resolve_length()
        if (
            self.map_dataset.local_npy_cache_dir
            and self.map_dataset.local_npy_cache_rank_subdir
            and not self.map_dataset.local_npy_cache_read_only
        ):
            self.map_dataset.local_npy_cache_dir = self.map_dataset.local_npy_cache_dir / f"rank_{self.map_dataset.rank:05d}"
        rank0_print(
            f"[data-stream] {self.map_dataset.resolved_jsonl_file}: len={self._length} "
            f"rank {self.map_dataset.rank}/{self.map_dataset.world_size}, "
            f"prebuilt_shard={self.map_dataset.using_prebuilt_shard}, "
            f"skip_samples={self.skip_samples}"
        )
        if self.map_dataset.local_npy_cache_dir:
            rank0_print(
                f"[data-cache] rank={self.map_dataset.rank} local_npy_cache_dir={self.map_dataset.local_npy_cache_dir} "
                f"source_prefix={self.map_dataset.local_npy_cache_source_prefix} "
                f"read_only={self.map_dataset.local_npy_cache_read_only} "
                f"rank_subdir={self.map_dataset.local_npy_cache_rank_subdir}"
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
                    if len(parts) >= 2 and int(parts[0]) == self.map_dataset.rank:
                        return int(parts[1])
        with shard_path.open("r", encoding="utf-8") as f:
            total_rows = sum(1 for _ in f)
        if (
            self.map_dataset.shard_by_rank
            and self.map_dataset.world_size > 1
            and not self.map_dataset.using_prebuilt_shard
        ):
            rows_per_rank, remainder = divmod(total_rows, self.map_dataset.world_size)
            return rows_per_rank + int(self.map_dataset.rank < remainder)
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
                    and self.map_dataset.world_size > 1
                    and not using_prebuilt_shard
                    and line_idx % self.map_dataset.world_size != self.map_dataset.rank
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
                        f"[data-skip] rank={self.map_dataset.rank} worker={worker_id} "
                        f"line={line_idx + 1} error={type(exc).__name__}: {exc}; "
                        f"target={row.get(self.map_dataset.audio_ids_key)}; "
                        f"ref={row.get(self.map_dataset.ref_audio_ids_key)}",
                        flush=True,
                    )
                    example = None
                else:
                    example = example
                yield example


class CodeShardFishAudioDataset(IterableDataset):
    def __init__(
        self,
        code_shard_dir: str,
        tokenizer: FishTokenizer,
        max_samples: Optional[int],
        num_codebooks: int,
        use_ref: bool,
        text_key: str,
        ref_text_key: Optional[str],
        local_cache_dir: Optional[str] = None,
        local_cache_read_only: bool = True,
    ):
        self.code_shard_dir = pathlib.Path(code_shard_dir)
        self.local_cache_dir = pathlib.Path(local_cache_dir).expanduser() if local_cache_dir else None
        self.local_cache_read_only = local_cache_read_only
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.max_samples = max_samples
        self.text_key = text_key
        self.ref_text_key = ref_text_key
        self.map_dataset = JsonlFishAudioDataset.__new__(JsonlFishAudioDataset)
        self.map_dataset.tokenizer = tokenizer
        self.map_dataset.num_codebooks = num_codebooks
        self.map_dataset.use_ref = use_ref
        self.map_dataset.text_key = text_key
        self.map_dataset.ref_text_key = ref_text_key
        self.map_dataset.rank = self.rank
        self.map_dataset.world_size = self.world_size
        self.map_dataset.local_npy_cache_dir = None
        self.map_dataset.local_npy_cache_source_prefix = pathlib.Path("/")
        self.map_dataset.local_npy_cache_log_every = 0
        self.map_dataset.local_npy_cache_read_only = True
        self.map_dataset.local_npy_cache_rank_subdir = False
        self.map_dataset._cache_hits = 0
        self.map_dataset._cache_misses = 0
        self.map_dataset._cache_legacy_hits = 0
        self.map_dataset._local_npy_cache_disabled = False
        self._chunk_files_by_worker = self._discover_chunks()
        self._length = self._resolve_length()
        rank0_print(
            f"[code-shard] dir={self.code_shard_dir} rank={self.rank}/{self.world_size} "
            f"len={self._length} local_cache_dir={self.local_cache_dir} "
            f"read_only={self.local_cache_read_only}"
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
                with np.load(self._resolve_chunk_path(path), allow_pickle=True) as shard:
                    total += len(shard["rows_json"])
                    if self.max_samples is not None and total >= self.max_samples:
                        return self.max_samples
        return total

    def __len__(self):
        return self._length

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
                load_path = self._resolve_chunk_path(chunk_path)
                with np.load(load_path, allow_pickle=True) as shard:
                    line_indices = shard["line_indices"] if "line_indices" in shard.files else np.arange(len(shard["rows_json"]))
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
                if current_path != chunk_path:
                    if shard is not None:
                        shard.close()
                    shard = np.load(self._resolve_chunk_path(chunk_path), allow_pickle=True)
                    current_path = chunk_path
                yield self._example_from_loaded_shard(shard, idx)
                yielded += 1
        finally:
            if shard is not None:
                shard.close()

    def _iter_chunk_files(self, chunk_files: list[pathlib.Path]):
        yielded = 0
        for chunk_path in chunk_files:
            load_path = self._resolve_chunk_path(chunk_path)
            with np.load(load_path, allow_pickle=True) as shard:
                rows_json = shard["rows_json"]
                for idx, row_json in enumerate(rows_json):
                    if self.max_samples is not None and yielded >= self.max_samples:
                        return
                    yield self._example_from_loaded_shard(shard, idx)
                    yielded += 1

    def _example_from_loaded_shard(self, shard, idx: int):
        row = json.loads(str(shard["rows_json"][idx]))
        target = torch.from_numpy(np.asarray(shard["target_codes"][idx], dtype=np.int64))
        ref = None
        if "ref_codes" in shard.files and "has_ref" in shard.files and bool(shard["has_ref"][idx]):
            ref = torch.from_numpy(np.asarray(shard["ref_codes"][idx], dtype=np.int64))
        text = clean_text_for_train(row[self.text_key])
        ref_text = None
        if self.ref_text_key and row.get(self.ref_text_key):
            ref_text = clean_text_for_train(row[self.ref_text_key])
        tokens, labels = self.map_dataset._pack_sample(
            text=text,
            target_codes=target,
            ref_codes=ref,
            ref_text=ref_text,
        )
        return {"tokens": tokens, "labels": labels}

    def _resolve_chunk_path(self, source_path: pathlib.Path) -> str:
        if self.local_cache_dir is None:
            return str(source_path)
        rel = source_path.relative_to(self.code_shard_dir)
        target = self.local_cache_dir / rel
        if target.is_file() and target.stat().st_size == source_path.stat().st_size:
            return str(target)
        if self.local_cache_read_only:
            return str(source_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp")
        try:
            shutil.copyfile(source_path, tmp)
            os.replace(tmp, target)
        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
        return str(target)


class FishAudioCollator:
    def __init__(self, tokenizer: FishTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples):
        valid_examples = [example for example in examples if example is not None]
        if not valid_examples:
            raise RuntimeError("Entire batch contains invalid codec samples")
        replacement = valid_examples[-1]
        examples = [example if example is not None else replacement for example in examples]
        max_tokens_length = min(
            max(example["tokens"].size(1) for example in examples),
            self.max_length,
        )
        pad_id = self.tokenizer.get_token_id("<|end_of_text|>")

        inputs, attention_masks, labels = [], [], []
        debug_meta = []
        for example in examples:
            tokens = example["tokens"][:, :max_tokens_length]
            label = example["labels"][:, :max_tokens_length]
            seq_len = tokens.size(1)

            attention_mask = torch.ones((max_tokens_length,), dtype=torch.bool)
            attention_mask[:seq_len] = False

            if seq_len < max_tokens_length:
                tokens = F.pad(tokens, (0, max_tokens_length - seq_len), value=pad_id)
                tokens[1:, seq_len:] = CODEBOOK_PAD_TOKEN_ID
                label = F.pad(label, (0, max_tokens_length - seq_len), value=-100)

            inputs.append(tokens)
            attention_masks.append(attention_mask)
            labels.append(label)
            debug_meta.append(
                {
                    "audio_path": example.get("_debug_audio_path"),
                    "ref_audio_path": example.get("_debug_ref_audio_path"),
                }
            )

        return {
            "inputs": torch.stack(inputs, dim=0),
            "attention_masks": torch.stack(attention_masks, dim=0),
            "labels": torch.stack(labels, dim=0),
            "_debug_meta": debug_meta,
        }


class FishS2ProTrainer(Trainer):
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
        metric_count = getattr(self, "_loss_metric_count", 0)
        metric_sums = getattr(self, "_loss_metric_sums", {})
        if metric_count > 0:
            for key, value in metric_sums.items():
                logs.setdefault(key, value / metric_count)
            metric_sums.clear()
            self._loss_metric_count = 0
        return super().log(logs, start_time=start_time)

    def _get_local_dataloader(
        self,
        dataset,
        batch_size: int,
        sampler_fn,
        is_training: bool = False,
    ):
        dataloader_params = {
            "batch_size": batch_size,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "drop_last": self.args.dataloader_drop_last,
        }
        if self.args.dataloader_num_workers > 0:
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
        if sampler_fn is not None and not isinstance(dataset, IterableDataset):
            dataloader_params["sampler"] = sampler_fn(dataset)
        rank0_print(
            "[dataloader] using local pre-sharded DataLoader; "
            "accelerate dataloader sharding disabled"
        )
        return DataLoader(dataset, **dataloader_params)

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
        if getattr(self.train_dataset, "using_prebuilt_shard", False):
            return self._get_local_dataloader(
                dataset=self.train_dataset,
                batch_size=self._train_batch_size,
                sampler_fn=self._get_train_sampler,
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
        if getattr(dataset, "using_prebuilt_shard", False):
            return self._get_local_dataloader(
                dataset=dataset,
                batch_size=self.args.eval_batch_size,
                sampler_fn=self._get_eval_sampler,
            )
        return super().get_eval_dataloader(eval_dataset)

    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if dataset is None:
            return None
        return SequentialSampler(dataset)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        actual_model = model.module if hasattr(model, "module") else model
        labels = inputs["labels"]
        # Catch invalid Fast AR codebook ids before the asynchronous CUDA embedding error.
        codebook_mask = (labels[:, 0] >= actual_model.config.semantic_begin_id) & (
            labels[:, 0] <= actual_model.config.semantic_end_id
        )
        selected = labels[:, 1:, :].permute(0, 2, 1)[codebook_mask][:, :-1]
        bad = (selected < 0) | (selected >= actual_model.config.codebook_size)
        if bad.any().item():
            bad_rows = bad.any(dim=1).nonzero(as_tuple=False).flatten().tolist()
            meta = inputs.get("_debug_meta", [])
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "-1")))
            for row_i in bad_rows:
                vals = selected[row_i][bad[row_i]].detach().cpu().tolist()
                print(
                    f"[BAD-CODEBOOK-ID] rank={rank} batch_row={row_i} "
                    f"values={vals[:16]} meta={meta[row_i] if row_i < len(meta) else None}",
                    flush=True,
                )
            raise RuntimeError("Invalid Fast AR codebook id before model forward")
        slow_ar_only = bool(getattr(self.args, "slow_ar_only", False))
        outputs = model(
            inp=inputs["inputs"],
            key_padding_mask=inputs["attention_masks"],
            labels=labels,
            skip_fast_ar=slow_ar_only,
        )
        token_logits = outputs.token_logits
        codebook_logits = outputs.codebook_logits

        base_loss = F.cross_entropy(
            token_logits.reshape(-1, token_logits.size(-1)),
            labels[:, 0].reshape(-1),
            ignore_index=-100,
        )

        if slow_ar_only:
            if codebook_logits is not None:
                raise RuntimeError("slow_ar_only expected the Fast AR forward path to be skipped")
            token_ids = labels[:, 0]
            semantic_mask = (token_ids >= actual_model.config.semantic_begin_id) & (
                token_ids <= actual_model.config.semantic_end_id
            )
            with torch.no_grad():
                im_end_id = actual_model.tokenizer.get_token_id("<|im_end|>")
                eos_mask = token_ids == im_end_id
                eos_count = eos_mask.sum()
                if eos_count.item() > 0:
                    eos_logits = token_logits[eos_mask]
                    eos_top1 = (eos_logits.argmax(dim=-1) == im_end_id).float().mean()
                    eos_prob = torch.softmax(eos_logits.float(), dim=-1)[:, im_end_id].mean()
                else:
                    eos_top1 = base_loss.new_tensor(0.0)
                    eos_prob = base_loss.new_tensor(0.0)
            self._record_loss_metrics(
                {
                    "eos_top1": eos_top1,
                    "eos_prob": eos_prob,
                    "eos_count": eos_count.float(),
                    "semantic_tokens": semantic_mask.sum().float(),
                }
            )
            if return_outputs:
                return base_loss, {
                    "token_logits": token_logits,
                }
            return base_loss

        base_loss_weight_value = self._base_loss_weight()
        base_loss_weight = base_loss.new_tensor(base_loss_weight_value)
        weighted_base_loss = base_loss * base_loss_weight

        token_ids = labels[:, 0]
        semantic_mask = (token_ids >= actual_model.config.semantic_begin_id) & (
            token_ids <= actual_model.config.semantic_end_id
        )
        all_codebook_labels = labels[:, 1 : 1 + actual_model.config.num_codebooks]
        filtered_codebook_labels = all_codebook_labels.permute(0, 2, 1)[semantic_mask]
        semantic_loss = F.cross_entropy(
            codebook_logits.reshape(-1, codebook_logits.size(-1)),
            filtered_codebook_labels.reshape(-1),
            ignore_index=-100,
        )
        loss = weighted_base_loss + semantic_loss

        with torch.no_grad():
            semantic_token_labels = token_ids[semantic_mask]
            valid_semantic_token_labels = semantic_token_labels != -100
            valid_semantic_token_count = valid_semantic_token_labels.sum()
            if valid_semantic_token_count.item() > 0:
                semantic_token_logits = token_logits[semantic_mask]
                slow_semantic_loss = F.cross_entropy(
                    semantic_token_logits,
                    semantic_token_labels,
                    ignore_index=-100,
                )
                slow_semantic_pred_top1 = semantic_token_logits.argmax(dim=-1)
                slow_semantic_top1 = (
                    (slow_semantic_pred_top1 == semantic_token_labels)
                    & valid_semantic_token_labels
                ).sum().float() / valid_semantic_token_count.float()
                slow_semantic_pred_top5 = semantic_token_logits.topk(k=5, dim=-1).indices
                slow_semantic_top5 = (
                    (slow_semantic_pred_top5 == semantic_token_labels.unsqueeze(-1)).any(dim=-1)
                    & valid_semantic_token_labels
                ).sum().float() / valid_semantic_token_count.float()
            else:
                slow_semantic_loss = loss.new_tensor(0.0)
                slow_semantic_top1 = loss.new_tensor(0.0)
                slow_semantic_top5 = loss.new_tensor(0.0)

            valid_codebook_labels = filtered_codebook_labels != -100
            valid_codebook_count = valid_codebook_labels.sum()
            codebook_metrics = {}
            if valid_codebook_count.item() > 0:
                pred_top1 = codebook_logits.argmax(dim=-1)
                semantic_top1 = (
                    (pred_top1 == filtered_codebook_labels)
                    & valid_codebook_labels
                ).sum().float() / valid_codebook_count.float()
                pred_top5 = codebook_logits.topk(k=5, dim=-1).indices
                semantic_top5 = (
                    (pred_top5 == filtered_codebook_labels.unsqueeze(-1)).any(dim=-1)
                    & valid_codebook_labels
                ).sum().float() / valid_codebook_count.float()

                for cb_idx in range(filtered_codebook_labels.size(1)):
                    cb_labels = filtered_codebook_labels[:, cb_idx]
                    cb_valid = cb_labels != -100
                    cb_count = cb_valid.sum()
                    if cb_count.item() > 0:
                        cb_logits = codebook_logits[:, cb_idx, :]
                        cb_loss = F.cross_entropy(cb_logits, cb_labels, ignore_index=-100)
                        cb_top1 = (
                            (pred_top1[:, cb_idx] == cb_labels) & cb_valid
                        ).sum().float() / cb_count.float()
                        cb_top5 = (
                            (pred_top5[:, cb_idx, :] == cb_labels.unsqueeze(-1)).any(dim=-1)
                            & cb_valid
                        ).sum().float() / cb_count.float()
                    else:
                        cb_loss = loss.new_tensor(0.0)
                        cb_top1 = loss.new_tensor(0.0)
                        cb_top5 = loss.new_tensor(0.0)
                    codebook_metrics[f"codebook{cb_idx}_loss"] = cb_loss
                    codebook_metrics[f"codebook{cb_idx}_top1"] = cb_top1
                    codebook_metrics[f"codebook{cb_idx}_top5"] = cb_top5

                if filtered_codebook_labels.size(1) > 1:
                    fast_labels = filtered_codebook_labels[:, 1:]
                    fast_valid = fast_labels != -100
                    fast_count = fast_valid.sum()
                    if fast_count.item() > 0:
                        fast_logits = codebook_logits[:, 1:, :]
                        fast_codebook1_9_loss = F.cross_entropy(
                            fast_logits.reshape(-1, fast_logits.size(-1)),
                            fast_labels.reshape(-1),
                            ignore_index=-100,
                        )
                        fast_codebook1_9_top1 = (
                            (pred_top1[:, 1:] == fast_labels) & fast_valid
                        ).sum().float() / fast_count.float()
                        fast_codebook1_9_top5 = (
                            (pred_top5[:, 1:, :] == fast_labels.unsqueeze(-1)).any(dim=-1)
                            & fast_valid
                        ).sum().float() / fast_count.float()
                    else:
                        fast_codebook1_9_loss = loss.new_tensor(0.0)
                        fast_codebook1_9_top1 = loss.new_tensor(0.0)
                        fast_codebook1_9_top5 = loss.new_tensor(0.0)
                else:
                    fast_codebook1_9_loss = loss.new_tensor(0.0)
                    fast_codebook1_9_top1 = loss.new_tensor(0.0)
                    fast_codebook1_9_top5 = loss.new_tensor(0.0)
            else:
                semantic_top1 = loss.new_tensor(0.0)
                semantic_top5 = loss.new_tensor(0.0)
                fast_codebook1_9_loss = loss.new_tensor(0.0)
                fast_codebook1_9_top1 = loss.new_tensor(0.0)
                fast_codebook1_9_top5 = loss.new_tensor(0.0)

        with torch.no_grad():
            im_end_id = actual_model.tokenizer.get_token_id("<|im_end|>")
            row0_labels = labels[:, 0]
            eos_mask = (row0_labels == im_end_id)
            eos_count = eos_mask.sum()
            if eos_count.item() > 0:
                eos_logits = token_logits[eos_mask]
                eos_top1 = (eos_logits.argmax(dim=-1) == im_end_id).float().mean()
                eos_prob = torch.softmax(eos_logits.float(), dim=-1)[:, im_end_id].mean()
            else:
                eos_top1 = loss.new_tensor(0.0)
                eos_prob = loss.new_tensor(0.0)

        metrics = {
                "base_loss": base_loss,
                "eos_top1": eos_top1,
                "eos_prob": eos_prob,
                "eos_count": eos_count.float(),
                "base_loss_weight": base_loss_weight,
                "weighted_base_loss": weighted_base_loss,
                "semantic_loss": semantic_loss,
                "semantic_top1": semantic_top1,
                "semantic_top5": semantic_top5,
                "slow_semantic_loss": slow_semantic_loss,
                "slow_semantic_top1": slow_semantic_top1,
                "slow_semantic_top5": slow_semantic_top5,
                "fast_codebook1_9_loss": fast_codebook1_9_loss,
                "fast_codebook1_9_top1": fast_codebook1_9_top1,
                "fast_codebook1_9_top5": fast_codebook1_9_top5,
                "semantic_tokens": semantic_mask.sum().float(),
            }
        metrics.update(codebook_metrics)
        self._record_loss_metrics(metrics)

        if return_outputs:
            return loss, {
                "token_logits": token_logits,
                "codebook_logits": codebook_logits,
                "base_loss": base_loss.detach(),
                "semantic_loss": semantic_loss.detach(),
                "semantic_top1": semantic_top1.detach(),
                "semantic_top5": semantic_top5.detach(),
                "slow_semantic_loss": slow_semantic_loss.detach(),
                "slow_semantic_top1": slow_semantic_top1.detach(),
                "slow_semantic_top5": slow_semantic_top5.detach(),
            }
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

    if not include_project_in:
        for key in ("fast_project_in.weight", "fast_project_in.bias"):
            if key in named and not named[key].requires_grad:
                raise RuntimeError(f"{key} must remain trainable")

    rank0_print(
        f"[freeze] frozen Fast AR: {frozen / 1e6:.2f}M params; "
        f"trainable: {trainable / 1e6:.2f}M params"
    )


def freeze_slow_ar(model: torch.nn.Module):
    frozen = 0
    trainable = 0
    trainable_prefix_counts: dict[str, int] = {}

    for name, param in model.named_parameters():
        if name.startswith(FAST_AR_TRAINABLE_PREFIXES):
            param.requires_grad = True
            trainable += param.numel()
            prefix = name.split(".", 1)[0]
            trainable_prefix_counts[prefix] = trainable_prefix_counts.get(prefix, 0) + param.numel()
        else:
            param.requires_grad = False
            frozen += param.numel()

    if trainable == 0:
        raise RuntimeError("freeze_slow_ar left no trainable Fast AR parameters")

    prefix_summary = ", ".join(
        f"{key}={value / 1e6:.2f}M" for key, value in sorted(trainable_prefix_counts.items())
    )
    rank0_print(
        f"[freeze] frozen Slow AR: {frozen / 1e6:.2f}M params; "
        f"trainable Fast AR: {trainable / 1e6:.2f}M params ({prefix_summary})"
    )


def ensure_hf_config_compat(model: torch.nn.Module):
    config = getattr(model, "config", None)
    if config is None:
        return

    def to_dict(self):
        if is_dataclass(self):
            return asdict(self)
        if hasattr(self, "__dict__"):
            return dict(self.__dict__)
        return {}

    def to_json_string(self, *args, **kwargs):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, default=str) + "\n"

    if not hasattr(config, "to_dict"):
        config.to_dict = types.MethodType(to_dict, config)
    if not hasattr(config, "to_json_string"):
        config.to_json_string = types.MethodType(to_json_string, config)


def export_fish_pretrained(trainer: Trainer, export_dir: Optional[str]):
    if not export_dir:
        return
    if int(os.environ.get("RANK", "0")) != 0:
        return

    export_path = pathlib.Path(export_dir)
    export_path.mkdir(parents=True, exist_ok=True)
    model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "__dict__"):
        for key, value in list(config.__dict__.items()):
            if callable(value):
                delattr(config, key)
    model.save_pretrained(str(export_path))
    rank0_print(f"[export] Fish pretrained model saved to {export_path}")


def copy_aux_files(src_dir: str, dst_dir: str):
    if int(os.environ.get("RANK", "0")) != 0:
        return
    src = pathlib.Path(src_dir)
    dst = pathlib.Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for name in [
        "README.md",
        "LICENSE.md",
        "configuration.json",
        "chat_template.jinja",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]:
        src_file = src / name
        if src_file.exists():
            shutil.copy2(src_file, dst / name)


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
    if missing:
        rank0_print(f"[resume:model_only] missing sample={missing[:8]}")
    if unexpected:
        rank0_print(f"[resume:model_only] unexpected sample={unexpected[:8]}")


def read_checkpoint_global_step(checkpoint_dir: Optional[str]) -> int:
    if not checkpoint_dir:
        return 0
    trainer_state = pathlib.Path(checkpoint_dir) / "trainer_state.json"
    if not trainer_state.exists():
        return 0
    with trainer_state.open("r", encoding="utf-8") as f:
        state = json.load(f)
    return int(state.get("global_step") or 0)


def main():
    configure_tmpdir()
    parser = HfArgumentParser((ModelArguments, DataArguments, FishTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )

    training_args.remove_unused_columns = False

    rank0_print("[load] tokenizer")
    tokenizer = FishTokenizer(model_args.pretrained_ckpt_path)
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

    rank0_print("[load] datasets")
    if data_args.code_shard_dir:
        train_dataset = CodeShardFishAudioDataset(
            code_shard_dir=data_args.code_shard_dir,
            tokenizer=tokenizer,
            max_samples=data_args.max_train_samples,
            num_codebooks=data_args.num_codebooks,
            use_ref=data_args.use_ref,
            text_key=data_args.text_key,
            ref_text_key=data_args.ref_text_key,
            local_cache_dir=data_args.code_shard_local_cache_dir,
            local_cache_read_only=data_args.code_shard_local_cache_read_only,
        )
    else:
        train_dataset_cls = StreamingJsonlFishAudioDataset if data_args.stream_train_jsonl else JsonlFishAudioDataset
        train_dataset = train_dataset_cls(
            jsonl_file=data_args.train_jsonl,
            tokenizer=tokenizer,
            max_samples=data_args.max_train_samples,
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
            skip_samples=effective_skip_train_samples,
        )
    eval_dataset = None
    if not data_args.code_shard_dir:
        # Code-shard training reads chunk npz directly; the manifest JSONL
        # does not contain per-sample npy paths, so the eager npy-based eval
        # dataset cannot be constructed. Evaluation is disabled (eval_strategy no).
        eval_jsonl = data_args.eval_jsonl or data_args.train_jsonl
        eval_dataset = JsonlFishAudioDataset(
            jsonl_file=eval_jsonl,
            tokenizer=tokenizer,
            max_samples=data_args.max_eval_samples,
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
        )

    rank0_print("[load] model")
    model = BaseTransformer.from_pretrained(
        model_args.pretrained_ckpt_path,
        load_weights=True,
        max_length=model_args.max_length,
    )
    ensure_hf_config_compat(model)
    if model_only_resume and resume:
        load_model_only_checkpoint(model, resume)
    if model_args.freeze_fast_ar and model_args.freeze_slow_ar:
        raise ValueError("freeze_fast_ar and freeze_slow_ar cannot both be true")
    if training_args.slow_ar_only and not model_args.freeze_fast_ar:
        raise ValueError("slow_ar_only requires freeze_fast_ar=true")
    if model_args.freeze_fast_ar:
        freeze_fast_ar(model, include_project_in=training_args.slow_ar_only)
    elif model_args.freeze_slow_ar:
        freeze_slow_ar(model)

    collator = FishAudioCollator(tokenizer=tokenizer, max_length=model_args.max_length)

    trainer = FishS2ProTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    if training_args.do_train:
        if resume:
            if model_only_resume:
                rank0_print(f"[resume] model_only {resume}; optimizer/scheduler reinitialized")
            else:
                rank0_print(f"[resume] {resume}")
        else:
            rank0_print(f"[resume] disabled mode={training_args.resume_mode}")
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
        export_fish_pretrained(trainer, training_args.export_dir)
        copy_aux_files(model_args.pretrained_ckpt_path, training_args.export_dir)


if __name__ == "__main__":
    main()

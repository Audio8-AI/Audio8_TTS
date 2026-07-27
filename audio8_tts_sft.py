#!/usr/bin/env python3
"""Supervised fine-tuning for audio8_tts Preview."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel, AutoProcessor, HfArgumentParser, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from audio8_tts_data import (
    build_sft_example,
    clean_text,
    collate_sft_examples,
    load_codes,
    read_jsonl,
    resolve_manifest_path,
    validate_sample_id,
)


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_ROOT / "model" / "audio8_tts_0_6B_preview"


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default=str(DEFAULT_MODEL), metadata={"help": "Local checkpoint or Hugging Face model ID."}
    )
    max_length: int = field(default=2048)
    freeze_slow_ar: bool = field(default=False)
    freeze_fast_ar: bool = field(default=False)


@dataclass
class DataArguments:
    train_jsonl: str | None = field(default=None)
    eval_jsonl: str | None = field(default=None)
    max_train_samples: int | None = field(default=None)
    max_eval_samples: int | None = field(default=None)


@dataclass
class Audio8TTSTrainingArguments(TrainingArguments):
    output_dir: str = field(default="outputs/audio8_tts_sft")
    export_dir: str | None = field(
        default=None,
        metadata={"help": "Optional clean Hugging Face export in addition to output_dir."},
    )
    resume_mode: str = field(
        default="none",
        metadata={"help": "none, auto/latest, or an explicit Trainer checkpoint path."},
    )
    slow_loss_weight: float = field(default=1.0)
    fast_loss_weight: float = field(default=1.0)


@dataclass(frozen=True)
class PreparedRow:
    sample_id: str
    text: str
    target_codes: Path
    reference_codes: Path | None
    reference_text: str | None


class Audio8TTSDataset(Dataset):
    """Map-style dataset backed by a prepared audio8_tts JSONL manifest."""

    def __init__(
        self,
        manifest: Path,
        *,
        tokenizer,
        config,
        max_length: int,
        max_samples: int | None,
    ) -> None:
        self.manifest = manifest.resolve()
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = int(max_length)
        self.rows = self._load_rows(max_samples)

    def _load_rows(self, max_samples: int | None) -> list[PreparedRow]:
        rows: list[PreparedRow] = []
        seen: set[str] = set()
        for line_number, value in read_jsonl(self.manifest):
            try:
                sample_id = validate_sample_id(
                    value.get("id"), fallback=f"row_{line_number:06d}"
                )
                text = clean_text(value.get("text"), field_name="text")
                target_codes = resolve_manifest_path(
                    value.get("target_codes"), self.manifest, field_name="target_codes"
                )
                reference_value = value.get("reference_codes")
                reference_text_value = value.get("reference_text")
                if bool(reference_value) != bool(reference_text_value):
                    raise ValueError(
                        "reference_codes and reference_text must be provided together"
                    )
                reference_codes = (
                    resolve_manifest_path(
                        reference_value, self.manifest, field_name="reference_codes"
                    )
                    if reference_value
                    else None
                )
                reference_text = (
                    clean_text(reference_text_value, field_name="reference_text")
                    if reference_text_value
                    else None
                )
            except (TypeError, ValueError, FileNotFoundError) as exc:
                raise ValueError(
                    f"{self.manifest}:{line_number}: invalid prepared row: {exc}"
                ) from exc
            if sample_id in seen:
                raise ValueError(
                    f"{self.manifest}:{line_number}: duplicate id {sample_id!r}"
                )
            seen.add(sample_id)
            rows.append(
                PreparedRow(sample_id, text, target_codes, reference_codes, reference_text)
            )
            if max_samples is not None and len(rows) >= int(max_samples):
                break
        if not rows:
            raise ValueError(f"no prepared rows found in {self.manifest}")
        LOGGER.info("Loaded %d examples from %s", len(rows), self.manifest)
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        target_codes = load_codes(
            row.target_codes, num_codebooks=int(self.config.num_codebooks)
        )
        reference_codes = (
            load_codes(row.reference_codes, num_codebooks=int(self.config.num_codebooks))
            if row.reference_codes is not None
            else None
        )
        try:
            return build_sft_example(
                tokenizer=self.tokenizer,
                config=self.config,
                text=row.text,
                target_codes=target_codes,
                reference_codes=reference_codes,
                reference_text=row.reference_text,
                max_length=self.max_length,
            )
        except ValueError as exc:
            raise ValueError(f"failed to encode sample {row.sample_id!r}: {exc}") from exc


def actual_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def set_trainable_modules(model: torch.nn.Module, *, freeze_slow: bool, freeze_fast: bool) -> None:
    """Freeze complete AR branches while preserving the requested branch."""
    slow_prefixes = ("embeddings.", "codebook_embeddings.", "layers.", "norm.")
    fast_prefixes = (
        "fast_project_in.",
        "fast_embeddings.",
        "fast_layers.",
        "fast_norm.",
        "fast_output.",
    )
    for name, parameter in model.named_parameters():
        if freeze_slow and name.startswith(slow_prefixes):
            parameter.requires_grad = False
        if freeze_fast and name.startswith(fast_prefixes):
            parameter.requires_grad = False
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    if trainable == 0:
        raise ValueError("freeze_slow_ar and freeze_fast_ar leave no trainable parameters")
    LOGGER.info("Trainable parameters: %s / %s (%.2f%%)", trainable, total, 100 * trainable / total)


def fast_codebook_logits(
    model: torch.nn.Module,
    slow_hidden: torch.Tensor,
    codebooks: torch.Tensor,
) -> torch.Tensor:
    """Teacher-force the fast AR branch for all codebooks in parallel."""
    prefix = codebooks[:, :-1].long()
    fast_dtype = model.fast_embeddings.weight.dtype
    hidden = model.fast_project_in(slow_hidden.to(dtype=fast_dtype))
    hidden = torch.cat((hidden[:, None, :], model.fast_embeddings(prefix)), dim=1)
    length = int(hidden.shape[1])
    positions = torch.arange(length, device=hidden.device)
    attention_mask = torch.ones(
        (hidden.shape[0], length), dtype=torch.long, device=hidden.device
    )
    mask = model._causal_mask(attention_mask, positions, length)
    rope = model.fast_freqs_cis[:length]
    for layer in model.fast_layers:
        if bool(getattr(model, "gradient_checkpointing", False)) and model.training:
            hidden = checkpoint(layer, hidden, rope, mask, use_reentrant=False)
        else:
            hidden = layer(hidden, rope, mask)
    return model.fast_output(model.fast_norm(hidden))


class Audio8TTSTrainer(Trainer):
    """Transformers Trainer with the two losses required by DualAR."""

    def __init__(self, *args, slow_loss_weight: float, fast_loss_weight: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.slow_loss_weight = float(slow_loss_weight)
        self.fast_loss_weight = float(fast_loss_weight)
        self._metric_sums: dict[str, float] = {}
        self._metric_count = 0

    def _record_metrics(self, values: dict[str, torch.Tensor]) -> None:
        self._metric_count += 1
        for name, value in values.items():
            self._metric_sums[name] = self._metric_sums.get(name, 0.0) + float(
                value.detach().float().cpu()
            )

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        if self._metric_count:
            for name, total in self._metric_sums.items():
                logs.setdefault(name, total / self._metric_count)
            self._metric_sums.clear()
            self._metric_count = 0
        super().log(logs, start_time=start_time)

    def compute_loss(
        self,
        model,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ):
        del num_items_in_batch
        labels = inputs.pop("labels")
        outputs = model(**inputs, return_dict=True)
        core = actual_model(model)

        slow_loss = F.cross_entropy(
            outputs.logits.reshape(-1, outputs.logits.shape[-1]),
            labels[:, 0].reshape(-1),
            ignore_index=-100,
        )
        semantic_mask = labels[:, 0].ge(int(core.config.semantic_begin_id)) & labels[
            :, 0
        ].le(int(core.config.semantic_end_id))
        if not semantic_mask.any():
            raise ValueError("batch contains no supervised semantic frames")

        slow_hidden = outputs.hidden_states[semantic_mask]
        if bool(core.config.norm_fastlayer_input):
            slow_hidden = core.norm(slow_hidden)
        codebooks = labels[:, 1:].permute(0, 2, 1)[semantic_mask]
        codebook_logits = fast_codebook_logits(core, slow_hidden, codebooks)
        fast_loss = F.cross_entropy(
            codebook_logits.reshape(-1, codebook_logits.shape[-1]),
            codebooks.reshape(-1),
        )
        loss = self.slow_loss_weight * slow_loss + self.fast_loss_weight * fast_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite loss: slow={float(slow_loss.detach())}, "
                f"fast={float(fast_loss.detach())}"
            )

        with torch.no_grad():
            slow_labels = labels[:, 0][labels[:, 0].ne(-100)]
            slow_predictions = outputs.logits[labels[:, 0].ne(-100)].argmax(dim=-1)
            slow_accuracy = slow_predictions.eq(slow_labels).float().mean()
            fast_accuracy = codebook_logits.argmax(dim=-1).eq(codebooks).float().mean()
        self._record_metrics(
            {
                "slow_loss": slow_loss,
                "fast_loss": fast_loss,
                "slow_accuracy": slow_accuracy,
                "fast_accuracy": fast_accuracy,
                "semantic_frames": loss.new_tensor(float(semantic_mask.sum())),
            }
        )
        if return_outputs:
            return loss, {
                "logits": outputs.logits,
                "codebook_logits": codebook_logits,
            }
        return loss


def resolve_resume_checkpoint(output_dir: str, mode: str) -> str | None:
    value = str(mode or "none").strip()
    if value.lower() in {"", "none", "false", "no"}:
        return None
    if value.lower() in {"auto", "latest"}:
        checkpoint_path = get_last_checkpoint(output_dir)
        if checkpoint_path is None:
            LOGGER.info("No checkpoint found in %s; starting a new run", output_dir)
        return checkpoint_path
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"resume checkpoint does not exist: {path}")
    return str(path)


def sanitize_config_for_save(config) -> None:
    """Remove runtime-only config values that cannot be serialized to JSON."""
    for key, value in list(vars(config).items()):
        try:
            json.dumps(value)
        except TypeError:
            LOGGER.warning("Dropping non-JSON config field %s (%s)", key, type(value).__name__)
            delattr(config, key)


def main() -> None:
    parser = HfArgumentParser((ModelArguments, DataArguments, Audio8TTSTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=logging.INFO,
    )
    if not data_args.train_jsonl:
        raise ValueError("--train_jsonl is required")
    if model_args.max_length < 2:
        raise ValueError("--max_length must be at least 2")
    if training_args.slow_loss_weight < 0 or training_args.fast_loss_weight < 0:
        raise ValueError("loss weights must be non-negative")
    if math.isclose(training_args.slow_loss_weight + training_args.fast_loss_weight, 0.0):
        raise ValueError("at least one loss weight must be positive")

    training_args.remove_unused_columns = False
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )
    dtype = torch.bfloat16 if training_args.bf16 else (
        torch.float16 if training_args.fp16 else torch.float32
    )
    model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        dtype=dtype,
    )
    if int(model_args.max_length) > int(model.config.max_seq_len):
        raise ValueError(
            f"--max_length={model_args.max_length} exceeds the model context "
            f"limit {model.config.max_seq_len}"
        )
    model.config.use_gradient_checkpointing = bool(training_args.gradient_checkpointing)
    set_trainable_modules(
        model,
        freeze_slow=model_args.freeze_slow_ar,
        freeze_fast=model_args.freeze_fast_ar,
    )

    train_dataset = Audio8TTSDataset(
        Path(data_args.train_jsonl),
        tokenizer=processor.tokenizer,
        config=model.config,
        max_length=model_args.max_length,
        max_samples=data_args.max_train_samples,
    )
    eval_dataset = (
        Audio8TTSDataset(
            Path(data_args.eval_jsonl),
            tokenizer=processor.tokenizer,
            config=model.config,
            max_length=model_args.max_length,
            max_samples=data_args.max_eval_samples,
        )
        if data_args.eval_jsonl
        else None
    )
    collator = partial(
        collate_sft_examples, pad_token_id=int(model.config.pad_token_id)
    )
    trainer = Audio8TTSTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        slow_loss_weight=training_args.slow_loss_weight,
        fast_loss_weight=training_args.fast_loss_weight,
    )

    resume = resolve_resume_checkpoint(training_args.output_dir, training_args.resume_mode)
    if training_args.do_train:
        result = trainer.train(resume_from_checkpoint=resume)
        trainer.log_metrics("train", result.metrics)
        trainer.save_metrics("train", result.metrics)
        trainer.save_state()
    if training_args.do_eval:
        if eval_dataset is None:
            raise ValueError("--do_eval requires --eval_jsonl")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    final_model = actual_model(trainer.model)
    final_model.gradient_checkpointing_disable()
    final_model.config.use_gradient_checkpointing = False
    sanitize_config_for_save(final_model.config)
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
    if training_args.export_dir:
        trainer.save_model(training_args.export_dir)
        processor.save_pretrained(training_args.export_dir)
    LOGGER.info("Saved final checkpoint to %s", training_args.output_dir)


if __name__ == "__main__":
    main()

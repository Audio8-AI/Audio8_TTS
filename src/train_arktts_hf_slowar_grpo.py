#!/usr/bin/env python3
import argparse
import hashlib
import json
import logging
import math
import os
import pathlib
import shutil
import select
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel, HfArgumentParser

TRAIN_DIR = pathlib.Path(__file__).resolve().parent
FISH_REPO = pathlib.Path(os.environ.get("FISH_SPEECH_ROOT", "/opt/src/fish-speech"))
for path in (TRAIN_DIR, FISH_REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_fish_s2pro_ds_fish_prompt_cached as base
import train_fish_s2pro_opd_ds as opd
from fish_speech.content_sequence import TextPart, VQPart
from fish_speech.conversation import Conversation, Message
from fish_speech.models.text2semantic.inference import decode_to_audio
from fish_speech.tokenizer import IM_END_TOKEN, FishTokenizer
from torch.utils.data import Dataset, Sampler

try:
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
except Exception as exc:  # pragma: no cover - fail loudly in remote env
    compute_grpo_outcome_advantage = None
    _VERL_IMPORT_ERROR = exc
else:
    _VERL_IMPORT_ERROR = None


logger = logging.getLogger(__name__)

SUPPORTED_ASR_LANGS = {
    "cs", "da", "de", "en", "es", "et", "fi", "fr", "hr", "hu", "it",
    "ja", "ko", "lt", "nl", "no", "pl", "pt", "ro", "sk", "sl", "sv", "zh",
}
ASR_LANG_ALIASES = {
    "chinese": "zh",
    "cn": "zh",
    "english": "en",
    "german": "de",
    "spanish": "es",
    "french": "fr",
    "italian": "it",
    "japanese": "ja",
    "korean": "ko",
}


@dataclass
class SlowARGRPOTrainingArguments(base.FishTrainingArguments):
    grpo_reward_type: str = field(default="asr", metadata={"help": "asr, sim, or asr_sim"})
    grpo_rollout_n: int = field(default=2)
    grpo_max_new_tokens: int = field(default=128)
    grpo_temperature: float = field(default=0.8)
    grpo_top_k: int = field(default=50)
    grpo_top_p: float = field(default=0.95)
    grpo_length_bucket: bool = field(
        default=True,
        metadata={"help": "Bucket training samples by prompt length to reduce padding waste."},
    )
    grpo_norm_adv_by_std: bool = field(default=True)
    grpo_kl_weight: float = field(default=0.0)
    grpo_entropy_weight: float = field(default=0.0)
    grpo_reward_alpha: float = field(default=3.0)
    grpo_asr_weight: float = field(default=0.65)
    grpo_sim_weight: float = field(default=0.35)
    grpo_sim_floor: float = field(default=0.35)
    grpo_sim_ceil: float = field(default=0.85)
    grpo_sim_reward_shape: str = field(default="logistic")
    grpo_sim_reward_beta: float = field(default=5.0)
    grpo_sim_backend: str = field(
        default="seedtts_wavlm",
        metadata={"help": "Speaker similarity backend: omnivoice, seedtts_wavlm, or cv3_eres2net."},
    )
    grpo_decode_dir: str = field(default="/dev/shm/fish_slowar_grpo_rewards")
    grpo_seedtts_root: str = field(default=os.environ.get("SEEDTTS_ROOT", "seed-tts-eval"))
    grpo_omnivoice_repo: str = field(default=os.environ.get("OMNIVOICE_REPO", "OmniVoice"))
    grpo_omnivoice_model_dir: str = field(
        default=os.environ.get("OMNIVOICE_MODEL_DIR", "OmniVoice/download/tts_eval_models")
    )
    grpo_cv3_root: str = field(default=os.environ.get("CV3_ROOT", "CV3-Eval-main"))
    grpo_cv3_speakerlab_root: str = field(
        default=os.environ.get("CV3_SPEAKERLAB_ROOT", "CV3-Eval-main/utils/3D-Speaker")
    )
    grpo_cv3_sim_checkpoint: str = field(
        default=os.environ.get("CV3_SIM_CHECKPOINT", "pretrained_eres2net.ckpt")
    )
    grpo_seedtts_python: str = field(
        default="",
        metadata={"help": "Deprecated alias. Empty means use the current training Python."},
    )
    grpo_reward_python: str = field(default="", metadata={"help": "Python used for the persistent reward worker."})
    grpo_reward_extra_pythonpath: str = field(
        default="",
        metadata={"help": "Optional extra source/dependency path prepended by reward worker."},
    )
    grpo_whisper_path: str = field(default=os.environ.get("WHISPER_PATH", "whisper-large-v3"))
    grpo_ark_asr_path: str = field(default=os.environ.get("ARK_ASR_PATH", "ark_asr_v1.1"))
    grpo_ark_asr_max_new_tokens: int = field(default=256)
    grpo_ark_asr_audio_max_seconds: float = field(default=30.0)
    grpo_ark_asr_batch_size: int = field(default=32)
    grpo_hf_modules_cache: str = field(default="/dev/shm/fish_grpo_hf_modules")
    grpo_wavlm_checkpoint: str = field(
        default=os.environ.get("WAVLM_CHECKPOINT", "wavlm_large_finetune.pth")
    )
    grpo_reward_timeout: float = field(default=300.0)
    grpo_reward_keep_wavs: bool = field(default=False)
    grpo_codec_batch_size: int = field(default=4)
    grpo_scale_codebook_embeddings: bool = field(default=False)
    sft_loss_weight: float = field(default=0.0)
    skip_final_save: bool = field(default=False)


class MetaJsonlFishAudioDataset(base.JsonlFishAudioDataset):
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
                if not row.get(self.text_key):
                    raise KeyError(f"Missing {self.text_key!r} at line {line_idx}")
                if self.use_ref and not row.get(self.ref_audio_ids_key):
                    raise KeyError(f"Missing {self.ref_audio_ids_key!r} at line {line_idx}")
                if row.get(self.ref_audio_ids_key):
                    ref_count += 1
                rows.append(row)
                if max_samples is not None and len(rows) >= max_samples:
                    break

        base.rank0_print(
            f"[grpo-data] {jsonl_file}: {len(rows)} text-only rows loaded "
            f"on rank {self.rank}/{self.world_size}, "
            f"{ref_count} with {self.ref_audio_ids_key}, "
            f"shard_by_rank={self.shard_by_rank}, "
            f"prebuilt_shard={using_prebuilt_shard}, "
            f"skip_samples={self.skip_samples}"
        )
        return rows

    def _row_to_example(self, row: dict):
        text = base.clean_text_for_train(row[self.text_key])

        target_codes = None
        target_path = row.get(self.audio_ids_key) if self.audio_ids_key else None
        if target_path:
            target_codes = self._load_codes(target_path)

        ref_codes = None
        ref_text = None
        ref_path = row.get(self.ref_audio_ids_key)
        if self.use_ref and ref_path:
            ref_codes = self._load_codes(ref_path)
            if self.ref_text_key and row.get(self.ref_text_key):
                ref_text = base.clean_text_for_train(row[self.ref_text_key])

        if ref_codes is None:
            system_parts = [TextPart(text="convert the provided text to speech", cal_loss=False)]
        else:
            system_parts = [
                TextPart(
                    text="convert the provided text to speech reference to the following:\n\nText:\n",
                    cal_loss=False,
                ),
                TextPart(text=base.format_fish_reference_text(ref_text), cal_loss=False),
                TextPart(text="\n\nSpeech:\n", cal_loss=False),
                VQPart(codes=ref_codes, cal_loss=False),
            ]
        conversation = Conversation(
            messages=[
                Message(role="system", parts=system_parts, cal_loss=False),
                Message(
                    role="user",
                    parts=[TextPart(text=text, cal_loss=False)],
                    cal_loss=False,
                ),
                Message(
                    role="assistant",
                    parts=[],
                    cal_loss=False,
                    modality="voice",
                    add_im_end=False,
                ),
            ]
        )
        tokens, _audio_masks, _audio_parts = conversation.encode_for_inference(
            self.tokenizer,
            num_codebooks=self.num_codebooks,
        )
        labels = torch.full_like(tokens, -100)
        sft_tokens = None
        sft_labels = None
        if target_codes is not None:
            sft_tokens, sft_labels = self._pack_sample(
                text=text,
                target_codes=target_codes,
                ref_codes=ref_codes,
                ref_text=ref_text,
            )
        lang = infer_lang(text, row)
        return {
            "tokens": tokens,
            "labels": labels,
            "sft_tokens": sft_tokens,
            "sft_labels": sft_labels,
            "target_text": text,
            "ref_text": ref_text or "",
            "ref_codes_path": str(ref_path or ""),
            "prompt_wav": str(row.get("pair_audio") or row.get("prompt_wav") or row.get("ref_audio") or ""),
            "lang": lang,
            "sample_id": str(row.get("sample_id") or row.get("audio") or row.get("utt") or ""),
            "_debug_audio_path": target_path,
            "_debug_ref_audio_path": ref_path,
        }


def infer_lang(text: str, row: dict) -> str:
    for key in ("lang", "language"):
        value = str(row.get(key) or "").strip().lower().replace("_", "-")
        value = ASR_LANG_ALIASES.get(value, value.split("-", 1)[0])
        if value in SUPPORTED_ASR_LANGS:
            return value
        if value:
            raise ValueError(f"unsupported ASR reward language: {value}")
    source = str(row.get("source") or "").lower().replace("_", "/")
    for part in source.split("/"):
        part = part.strip().split("-", 1)[0]
        if part in SUPPORTED_ASR_LANGS:
            return part
    text = str(text)
    if any("\u3040" <= ch <= "\u30ff" for ch in text):
        return "ja"
    if any("\uac00" <= ch <= "\ud7af" for ch in text):
        return "ko"
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return "zh"
    return "en"


class MetaFishAudioCollator(base.FishAudioCollator):
    def __call__(self, examples):
        batch = super().__call__(examples)
        for key in ("target_text", "ref_text", "ref_codes_path", "prompt_wav", "lang", "sample_id"):
            batch[key] = [example.get(key, "") for example in examples]
        sft_examples = [example for example in examples if example.get("sft_tokens") is not None]
        if len(sft_examples) != len(examples):
            batch["sft_inputs"] = None
            return batch
        pad_id = self.tokenizer.get_token_id("<|end_of_text|>")
        max_sft_length = min(
            max(example["sft_tokens"].size(1) for example in sft_examples),
            self.max_length,
        )
        sft_inputs, sft_attention_masks, sft_labels = [], [], []
        for example in examples:
            tokens = example["sft_tokens"][:, :max_sft_length]
            label = example["sft_labels"][:, :max_sft_length]
            seq_len = tokens.size(1)
            attention_mask = torch.ones((max_sft_length,), dtype=torch.bool)
            attention_mask[:seq_len] = False
            if seq_len < max_sft_length:
                tokens = F.pad(tokens, (0, max_sft_length - seq_len), value=pad_id)
                tokens[1:, seq_len:] = base.CODEBOOK_PAD_TOKEN_ID
                label = F.pad(label, (0, max_sft_length - seq_len), value=-100)
            sft_inputs.append(tokens)
            sft_attention_masks.append(attention_mask)
            sft_labels.append(label)
        batch["sft_inputs"] = torch.stack(sft_inputs, dim=0)
        batch["sft_attention_masks"] = torch.stack(sft_attention_masks, dim=0)
        batch["sft_labels"] = torch.stack(sft_labels, dim=0)
        return batch


_SANITIZED_LOGITS_FLAG = {"count": 0}


def sample_from_logits(logits: torch.Tensor, *, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    logits = logits.float() / max(float(temperature), 1e-6)
    if not torch.isfinite(logits).all():
        _SANITIZED_LOGITS_FLAG["count"] += 1
        if _SANITIZED_LOGITS_FLAG["count"] <= 3:
            print(
                f"[rollout] WARNING sanitizing non-finite logits "
                f"(count={_SANITIZED_LOGITS_FLAG['count']})",
                flush=True,
            )
        logits = torch.nan_to_num(
            logits, nan=-float("inf"), posinf=-float("inf"), neginf=-float("inf")
        )
    if top_k and top_k > 0:
        k = min(int(top_k), int(logits.size(-1)))
        vals, idx = torch.topk(logits, k=k, dim=-1)
        filtered = torch.full_like(logits, -float("inf"))
        logits = filtered.scatter(-1, idx, vals)
    if top_p and 0.0 < float(top_p) < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative > float(top_p)
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
        logits = torch.full_like(logits, -float("inf")).scatter(-1, sorted_idx, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    if not torch.isfinite(probs).all():
        # Rows whose logits are all -inf produce NaN probabilities. Fall back to
        # a uniform draw so a single diverged rollout cannot assert the CUDA
        # context and kill the whole job.
        finite_any = torch.isfinite(probs).any(dim=-1)
        if not bool(finite_any.all().item()):
            probs = torch.where(
                finite_any[:, None],
                probs,
                torch.ones_like(probs) / int(probs.size(-1)),
            )
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _uses_falcon_slow_backbone(model) -> bool:
    """True for the audio8 Falcon-H1 joint model (v3/v4 lineage)."""
    return hasattr(model, "slow") and hasattr(model.slow, "layers")


def remove_arktts_kv_caches(model) -> None:
    if _uses_falcon_slow_backbone(model):
        # Falcon-H1 keeps one shared hybrid cache object for the whole slow
        # backbone plus static per-layer fast-AR KV caches.
        model.__dict__["_slow_cache"] = None
        for layer in model.fast_layers:
            layer.attention.kv_cache = None
        return
    for layer in list(model.layers) + list(model.fast_layers):
        layer.attention.kv_cache = None


def setup_arktts_generation_caches(model, batch_size: int, max_length: int) -> None:
    model._setup_generation_caches(
        int(batch_size),
        min(int(max_length), int(model.config.max_seq_len)),
        next(model.parameters()).dtype,
    )


def _arktts_cached_step_masks(
    model,
    input_ids: torch.Tensor,
    cache_position: torch.Tensor,
    key_padding_mask: Optional[torch.Tensor],
):
    """Return (attention_mask, position_ids) for one cached slow-AR step.

    The Falcon-H1 fast Mamba path requires the mask width to match the input
    width (exactly like the model's own ``generate`` loop), while absolute
    positions still come from the full-width key-padding mask.
    """
    if _uses_falcon_slow_backbone(model):
        if key_padding_mask is None:
            position_source = torch.ones(
                (input_ids.shape[0], int(model.config.max_seq_len)),
                dtype=torch.long,
                device=input_ids.device,
            )
        else:
            position_source = key_padding_mask.logical_not().long()
        # Falcon attention masks cover the whole sequence so far (like the
        # model's own generate loop); the Mamba path ignores the mask once
        # cache_position > 0, so a full-width mask is safe there too.
        width = int(cache_position[-1].item()) + 1
        attention_mask = position_source[:, :width].contiguous()
        position_ids = (
            position_source.cumsum(-1).sub(1).clamp_min(0).index_select(1, cache_position)
        )
        return attention_mask, position_ids
    if key_padding_mask is None:
        attention_mask = torch.ones(
            (input_ids.shape[0], int(model.config.max_seq_len)),
            dtype=torch.long,
            device=input_ids.device,
        )
    else:
        attention_mask = key_padding_mask.logical_not().long()
    position_ids = attention_mask.cumsum(-1).sub(1).clamp_min(0).index_select(1, cache_position)
    return attention_mask, position_ids


def arktts_slow_forward_cached(
    model,
    input_ids: torch.Tensor,
    cache_position: torch.Tensor,
    key_padding_mask: Optional[torch.Tensor],
):
    attention_mask, position_ids = _arktts_cached_step_masks(
        model, input_ids, cache_position, key_padding_mask
    )
    logits, hidden = model._slow_step(
        input_ids,
        cache_position,
        position_ids,
        attention_mask,
    )
    return type("ArkttsSlowOutput", (), {"logits": logits[:, None], "hidden_states": hidden})()


def arktts_slow_hidden_step_cached(
    model,
    input_ids: torch.Tensor,
    cache_position: torch.Tensor,
    key_padding_mask: Optional[torch.Tensor],
):
    attention_mask, position_ids = _arktts_cached_step_masks(
        model, input_ids, cache_position, key_padding_mask
    )
    hidden = model._slow_hidden_step(
        input_ids, cache_position, position_ids, attention_mask
    )
    return type("ArkttsSlowHiddenStep", (), {"hidden_states": hidden})()


def arktts_slow_semantic_step_cached(
    model,
    input_ids: torch.Tensor,
    cache_position: torch.Tensor,
    key_padding_mask: Optional[torch.Tensor],
    im_end_id: int,
):
    attention_mask, position_ids = _arktts_cached_step_masks(
        model, input_ids, cache_position, key_padding_mask
    )
    logits, hidden = model._slow_semantic_step(
        input_ids,
        cache_position,
        position_ids,
        attention_mask,
        int(model.config.semantic_begin_id),
        int(model.config.semantic_end_id),
        int(im_end_id),
    )
    return type(
        "ArkttsSlowSemanticStep",
        (),
        {"logits": logits[:, None], "hidden_states": hidden},
    )()


def arktts_fast_logits(model, projected_hidden: torch.Tensor, prefix: torch.Tensor) -> torch.Tensor:
    hidden = projected_hidden[:, None, :]
    if prefix.numel():
        hidden = torch.cat((hidden, model.fast_embeddings(prefix.long())), dim=1)
    length = int(hidden.shape[1])
    positions = torch.arange(length, device=hidden.device)
    attention_mask = torch.ones((hidden.shape[0], length), dtype=torch.long, device=hidden.device)
    mask = model._causal_mask(attention_mask, positions, length)
    rope = model.fast_freqs_cis[:length]
    for layer in model.fast_layers:
        hidden = layer(hidden, rope, mask)
    return model.fast_output(model.fast_norm(hidden))[:, -1]


def arktts_greedy_fast_codebooks(
    model,
    projected_hidden: torch.Tensor,
    semantic_token: torch.Tensor,
) -> torch.Tensor:
    codebooks = [
        (semantic_token - int(model.config.semantic_begin_id)).clamp(
            min=0,
            max=int(model.config.codebook_size) - 1,
        )
    ]
    for _ in range(1, int(model.config.num_codebooks)):
        prefix = torch.stack(codebooks, dim=1)
        codebooks.append(arktts_fast_logits(model, projected_hidden, prefix).argmax(dim=-1))
    return torch.stack(codebooks, dim=1)


def arktts_greedy_fast_codebooks_cached(
    model,
    projected_hidden: torch.Tensor,
    semantic_token: torch.Tensor,
) -> torch.Tensor:
    fast_step = getattr(model, "_fast_step", None)
    if not callable(fast_step):
        raise RuntimeError("ArkTTS model does not expose the cached _fast_step API")
    if any(layer.attention.kv_cache is None for layer in model.fast_layers):
        raise RuntimeError("Fast AR KV caches must be initialized before cached sampling")

    fast_step(projected_hidden[:, None, :], 0)
    current = (semantic_token - int(model.config.semantic_begin_id)).clamp(
        min=0,
        max=int(model.config.codebook_size) - 1,
    )
    codebooks = [current]
    for position in range(1, int(model.config.num_codebooks)):
        hidden = model.fast_embeddings(current.long())[:, None, :]
        current = fast_step(hidden, position).argmax(dim=-1)
        codebooks.append(current)
    return torch.stack(codebooks, dim=1)


def arktts_slow_hidden_forward(
    model,
    input_ids: torch.Tensor,
    key_padding_mask: Optional[torch.Tensor],
):
    length = int(input_ids.shape[-1])
    if key_padding_mask is None:
        attention_mask = torch.ones(
            (input_ids.shape[0], length), dtype=torch.long, device=input_ids.device
        )
    else:
        attention_mask = key_padding_mask.logical_not().long()
    if _uses_falcon_slow_backbone(model):
        # Falcon-H1 teacher forcing goes through the model's packed dual-AR
        # forward; it returns the already-normalized slow hidden states.
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        return type("ArkttsSlowHidden", (), {"hidden_states": outputs.hidden_states})()
    position_ids = attention_mask.cumsum(-1).sub(1).clamp_min(0)
    rope = model.freqs_cis[position_ids]
    positions = torch.arange(length, device=input_ids.device)
    mask = model._causal_mask(attention_mask, positions, length)
    hidden = model._embed(input_ids)
    for layer in model.layers:
        if model.config.use_gradient_checkpointing and model.training:
            hidden = checkpoint(layer, hidden, rope, mask, use_reentrant=False)
        else:
            hidden = layer(hidden, rope, mask)
    normalized = model.norm(hidden)
    if model.config.norm_fastlayer_input:
        hidden = normalized
    return type("ArkttsSlowHidden", (), {"hidden_states": hidden})()


def compute_arktts_sft_loss(
    model,
    input_ids: torch.Tensor,
    key_padding_mask: torch.Tensor,
    labels: torch.Tensor,
):
    """Teacher-forced SFT loss for the Arktts dual-AR model.

    Mirrors the Fish dual-AR SFT loss in
    train_fish_s2pro_ds_fish_prompt_cached.py: slow-AR token CE plus
    teacher-forced Fast-AR codebook CE. Arktts conditions Fast AR on the
    hidden state *before* the semantic token and on codebook0 (semantic
    offset), so the Fast AR targets are codebooks 1..num_codebooks-1.
    """
    student = opd.actual_model(model)
    attention_mask = key_padding_mask.logical_not().long()
    outputs = student(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    token_logits = outputs.logits
    slow_loss = F.cross_entropy(
        token_logits.reshape(-1, token_logits.size(-1)),
        labels[:, 0].reshape(-1),
        ignore_index=-100,
    )

    num_codebooks = int(student.config.num_codebooks)
    semantic_mask = (labels[:, 0] >= int(student.config.semantic_begin_id)) & (
        labels[:, 0] <= int(student.config.semantic_end_id)
    )
    hidden = outputs.hidden_states
    normalized = student.norm(hidden)
    cond_hidden = normalized if bool(student.config.norm_fastlayer_input) else hidden
    codebook_labels = labels[:, 1 : 1 + num_codebooks]

    frame_hidden = []
    frame_codes = []
    for batch_i in range(int(labels.size(0))):
        positions = semantic_mask[batch_i].nonzero(as_tuple=False).flatten().tolist()
        for pos in positions:
            if pos <= 0:
                continue
            frame_hidden.append(cond_hidden[batch_i, pos - 1])
            frame_codes.append(codebook_labels[batch_i, :, pos])

    if frame_hidden:
        slow_hidden = torch.stack(frame_hidden, dim=0)
        codes = torch.stack(frame_codes, dim=0)
        projected = student.fast_project_in(slow_hidden)
        prefix = codes[:, :-1]
        fast_in = torch.cat(
            (projected[:, None, :], student.fast_embeddings(prefix.long())),
            dim=1,
        )
        length = int(fast_in.size(1))
        positions = torch.arange(length, device=fast_in.device)
        fast_attention = torch.ones(
            (fast_in.size(0), length), dtype=torch.long, device=fast_in.device
        )
        fast_mask = student._causal_mask(fast_attention, positions, length)
        fast_rope = student.fast_freqs_cis[:length]
        for layer in student.fast_layers:
            fast_in = layer(fast_in, fast_rope, fast_mask)
        fast_logits = student.fast_output(student.fast_norm(fast_in))[:, 1:, :]
        fast_targets = codes[:, 1:]
        fast_loss = F.cross_entropy(
            fast_logits.reshape(-1, fast_logits.size(-1)),
            fast_targets.reshape(-1),
        )
    else:
        fast_loss = slow_loss.new_tensor(0.0)

    return slow_loss + fast_loss, slow_loss, fast_loss


@torch.no_grad()
def sample_rollout_student_cached(
    model,
    inputs: torch.Tensor,
    prompt_lens: list[int],
    rollout_lens: list[int],
    max_new_tokens: int,
    pad_id: int,
    temperature: float,
    top_k: int,
    top_p: float,
    use_fast_kv_cache: bool = False,
    compact_semantic_logits: bool = False,
):
    was_training = bool(model.training)
    model.eval()
    device = inputs.device
    batch_size = int(inputs.size(0))
    codebook_dim = int(inputs.size(1))
    generated = [torch.empty((codebook_dim, 0), dtype=inputs.dtype, device=device) for _ in range(batch_size)]

    if not rollout_lens or max(rollout_lens) <= 0:
        remove_arktts_kv_caches(model)
        model.train(was_training)
        return generated

    semantic_begin = int(model.config.semantic_begin_id)
    semantic_end = int(model.config.semantic_end_id)
    im_end_id = int(model.tokenizer.get_token_id(IM_END_TOKEN))
    sample_indices = []
    capped_rollout_lens = []
    for sample_i in range(batch_size):
        capped = min(int(max_new_tokens), int(rollout_lens[sample_i]))
        if int(prompt_lens[sample_i]) > 0 and capped > 0:
            sample_indices.append(sample_i)
            capped_rollout_lens.append(capped)
    if not sample_indices:
        remove_arktts_kv_caches(model)
        model.train(was_training)
        return generated

    group_size = len(sample_indices)
    max_prompt_len = max(int(prompt_lens[i]) for i in sample_indices)
    max_group_steps = min(int(model.config.max_seq_len) - max_prompt_len, max(capped_rollout_lens))
    if max_group_steps <= 0:
        remove_arktts_kv_caches(model)
        model.train(was_training)
        return generated

    remove_arktts_kv_caches(model)
    setup_arktts_generation_caches(
        model,
        batch_size=group_size,
        max_length=max_prompt_len + max_group_steps,
    )
    if not bool(use_fast_kv_cache):
        for layer in model.fast_layers:
            layer.attention.kv_cache = None

    prompt = torch.zeros((group_size, codebook_dim, max_prompt_len), dtype=inputs.dtype, device=device)
    prompt[:, 0, :] = int(pad_id)
    key_padding_mask = torch.ones(
        (group_size, int(model.config.max_seq_len)), dtype=torch.bool, device=device
    )
    for row_i, sample_i in enumerate(sample_indices):
        prompt_len = int(prompt_lens[sample_i])
        start = max_prompt_len - prompt_len
        prompt[row_i, :, start:max_prompt_len] = inputs[sample_i, :, :prompt_len]
        key_padding_mask[row_i, start:max_prompt_len] = False

    input_pos = torch.arange(0, max_prompt_len, dtype=torch.long, device=device)
    if bool(compact_semantic_logits):
        result = arktts_slow_semantic_step_cached(
            model, prompt, input_pos, key_padding_mask, im_end_id
        )
    else:
        result = arktts_slow_forward_cached(
            model, prompt, input_pos, key_padding_mask
        )
    finished = torch.zeros((group_size,), dtype=torch.bool, device=device)
    group_rollout_lens = torch.tensor(capped_rollout_lens, dtype=torch.long, device=device)
    generated_tensor = torch.zeros((group_size, codebook_dim, max_group_steps), dtype=inputs.dtype, device=device)
    generated_counts = torch.zeros((group_size,), dtype=torch.long, device=device)

    for step in range(max_group_steps):
        active = (~finished) & group_rollout_lens.gt(step)
        if not bool(active.any().item()):
            break
        if bool(compact_semantic_logits):
            compact_token = sample_from_logits(
                result.logits[:, -1, :],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            semantic_count = semantic_end - semantic_begin + 1
            next_token = compact_token + semantic_begin
            next_token = torch.where(
                compact_token.eq(semantic_count),
                torch.full_like(compact_token, im_end_id),
                next_token,
            )
        else:
            slow_logits = opd.mask_slow_logits(result.logits[:, -1, :], model)
            next_token = sample_from_logits(
                slow_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        hidden = result.hidden_states[:, -1, :]
        col = torch.zeros((group_size, codebook_dim, 1), dtype=inputs.dtype, device=device)
        col[:, 0, 0] = int(pad_id)
        col[active, 0, 0] = next_token[active].to(inputs.dtype)

        semantic_next = active & next_token.ge(semantic_begin) & next_token.le(semantic_end)
        if bool(semantic_next.any().item()):
            semantic_idx = torch.nonzero(semantic_next, as_tuple=False).flatten()
            if bool(use_fast_kv_cache):
                # The cache batch dimension is fixed. Fill it for all rows and
                # retain codebooks only for active semantic rows.
                codebooks = arktts_greedy_fast_codebooks_cached(
                    model,
                    model.fast_project_in(hidden),
                    next_token,
                ).index_select(0, semantic_idx)
            else:
                codebooks = arktts_greedy_fast_codebooks(
                    model,
                    model.fast_project_in(hidden.index_select(0, semantic_idx)),
                    next_token.index_select(0, semantic_idx),
                )
            col[semantic_idx, 1:, 0] = codebooks.to(inputs.dtype)

        generated_tensor[active, :, step : step + 1] = col[active]
        generated_counts[active] += 1
        finished = finished | (active & next_token.eq(im_end_id))
        if step + 1 >= max_group_steps:
            break
        cache_pos = max_prompt_len + step
        key_padding_mask[:, cache_pos] = True
        key_padding_mask[active, cache_pos] = False
        cache_position = torch.tensor(
            [cache_pos], dtype=torch.long, device=device
        )
        if bool(compact_semantic_logits):
            result = arktts_slow_semantic_step_cached(
                model, col, cache_position, key_padding_mask, im_end_id
            )
        else:
            result = arktts_slow_forward_cached(
                model, col, cache_position, key_padding_mask
            )

    remove_arktts_kv_caches(model)
    for row_i, sample_i in enumerate(sample_indices):
        generated[sample_i] = generated_tensor[row_i, :, : int(generated_counts[row_i].item())]
    model.train(was_training)
    return generated


def collect_slow_log_probs(token_logits: torch.Tensor, forced_labels: torch.Tensor, generated_positions: list[list[int]], model):
    begin = int(model.config.semantic_begin_id)
    end = int(model.config.semantic_end_id)
    im_end_id = int(model.tokenizer.get_token_id(IM_END_TOKEN))
    max_len = max((len(p) for p in generated_positions), default=0)
    if max_len <= 0:
        return token_logits.new_zeros((len(generated_positions), 1)), token_logits.new_zeros((len(generated_positions), 1))
    out = token_logits.new_zeros((len(generated_positions), max_len))
    mask = token_logits.new_zeros((len(generated_positions), max_len))
    row_indices = []
    col_indices = []
    score_positions = []
    token_ids = []
    for batch_i, positions in enumerate(generated_positions):
        for j, pos in enumerate(positions):
            score_pos = int(pos) - 1
            token_id = int(forced_labels[batch_i, 0, pos].item())
            if score_pos < 0 or token_id < 0:
                continue
            if not (begin <= token_id <= end or token_id == im_end_id):
                continue
            row_indices.append(batch_i)
            col_indices.append(j)
            score_positions.append(score_pos)
            token_ids.append(token_id)
    if row_indices:
        rows = torch.tensor(row_indices, dtype=torch.long, device=token_logits.device)
        cols = torch.tensor(col_indices, dtype=torch.long, device=token_logits.device)
        score_pos = torch.tensor(score_positions, dtype=torch.long, device=token_logits.device)
        tids = torch.tensor(token_ids, dtype=torch.long, device=token_logits.device)
        selected = token_logits[rows, score_pos, :]
        selected = opd.mask_slow_logits(selected, model).float()
        logp = F.log_softmax(selected, dim=-1)
        values = logp[torch.arange(logp.size(0), device=logp.device), tids]
        out[rows, cols] = values.to(out.dtype)
        mask[rows, cols] = 1.0
    return out, mask


def collect_slow_entropy(token_logits: torch.Tensor, forced_labels: torch.Tensor, generated_positions: list[list[int]], model):
    begin = int(model.config.semantic_begin_id)
    end = int(model.config.semantic_end_id)
    im_end_id = int(model.tokenizer.get_token_id(IM_END_TOKEN))
    max_len = max((len(p) for p in generated_positions), default=0)
    if max_len <= 0:
        return token_logits.new_zeros((len(generated_positions), 1)), token_logits.new_zeros((len(generated_positions), 1))
    out = token_logits.new_zeros((len(generated_positions), max_len))
    mask = token_logits.new_zeros((len(generated_positions), max_len))
    row_indices = []
    col_indices = []
    score_positions = []
    token_ids = []
    for batch_i, positions in enumerate(generated_positions):
        for j, pos in enumerate(positions):
            score_pos = int(pos) - 1
            token_id = int(forced_labels[batch_i, 0, pos].item())
            if score_pos < 0 or token_id < 0:
                continue
            if not (begin <= token_id <= end or token_id == im_end_id):
                continue
            row_indices.append(batch_i)
            col_indices.append(j)
            score_positions.append(score_pos)
            token_ids.append(token_id)
    if row_indices:
        rows = torch.tensor(row_indices, dtype=torch.long, device=token_logits.device)
        cols = torch.tensor(col_indices, dtype=torch.long, device=token_logits.device)
        score_pos = torch.tensor(score_positions, dtype=torch.long, device=token_logits.device)
        selected = token_logits[rows, score_pos, :]
        selected = opd.mask_slow_logits(selected, model).float()
        logp = F.log_softmax(selected, dim=-1)
        probs = logp.exp()
        logp_safe = torch.where(probs > 0, logp, torch.zeros_like(logp))
        entropy = -(probs * logp_safe).sum(dim=-1)
        out[rows, cols] = entropy.to(out.dtype)
        mask[rows, cols] = 1.0
    return out, mask


class GlobalLengthBucketSampler(Sampler[int]):
    def __init__(
        self,
        lengths: list[int],
        global_batch_size: int,
        seed: int,
    ):
        self.lengths = [int(length) for length in lengths]
        self.global_batch_size = max(1, int(global_batch_size))
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.lengths)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        ordered = sorted(range(len(self.lengths)), key=self.lengths.__getitem__)
        buckets = [
            ordered[start : start + self.global_batch_size]
            for start in range(0, len(ordered), self.global_batch_size)
        ]
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        bucket_order = torch.randperm(len(buckets), generator=generator).tolist()
        for bucket_i in bucket_order:
            bucket = buckets[bucket_i]
            within = torch.randperm(len(bucket), generator=generator).tolist()
            for item_i in within:
                yield bucket[item_i]


class RewardHelper:
    def __init__(self, args: SlowARGRPOTrainingArguments, codec, device: torch.device):
        self.args = args
        self.codec = codec
        self.device = device
        self.decode_dir = pathlib.Path(args.grpo_decode_dir) / f"rank_{int(os.environ.get('RANK', '0')):05d}"
        self.decode_dir.mkdir(parents=True, exist_ok=True)
        self.worker = pathlib.Path(__file__).resolve().with_name("fish_grpo_reward_worker.py")
        self.proc = None
        self.stderr_handle = None

    def decode_codes_batch(
        self,
        codes_batch: list[torch.Tensor],
        sample_ids: list[str],
        rollout_indices: list[int],
    ) -> list[pathlib.Path | Exception]:
        import soundfile as sf

        if not (len(codes_batch) == len(sample_ids) == len(rollout_indices)):
            raise ValueError("codec batch metadata size mismatch")
        results: list[pathlib.Path | Exception | None] = [None] * len(codes_batch)
        valid_items = []
        for idx, (codes, sample_id, rollout_i) in enumerate(
            zip(codes_batch, sample_ids, rollout_indices)
        ):
            valid = codes
            if int(valid.size(1)) > 0:
                valid = valid[:, valid[1:].abs().sum(dim=0).ne(0)]
            if int(valid.size(1)) <= 0:
                results[idx] = ValueError("empty generated code sequence")
                continue
            digest = hashlib.sha1(
                f"{sample_id}-{time.time_ns()}-{rollout_i}".encode()
            ).hexdigest()[:16]
            wav_path = self.decode_dir / f"{digest}.wav"
            valid_items.append((idx, valid[1:].contiguous(), wav_path))

        # Similar lengths share a micro-batch, minimizing padded codec work.
        valid_items.sort(key=lambda item: int(item[1].size(1)))
        batch_size = max(1, int(self.args.grpo_codec_batch_size))
        for start in range(0, len(valid_items), batch_size):
            chunk = valid_items[start : start + batch_size]
            max_code_len = max(int(codes.size(1)) for _idx, codes, _path in chunk)
            num_codebooks = int(chunk[0][1].size(0))
            padded = torch.zeros(
                (len(chunk), num_codebooks, max_code_len),
                dtype=chunk[0][1].dtype,
                device=self.device,
            )
            for row_i, (_idx, codes, _path) in enumerate(chunk):
                padded[row_i, :, : int(codes.size(1))] = codes.to(self.device)
            try:
                with torch.inference_mode():
                    audio_batch = self.codec.from_indices(padded)
                if int(audio_batch.size(-1)) % max_code_len != 0:
                    raise RuntimeError(
                        f"codec output length {audio_batch.size(-1)} is not divisible "
                        f"by code length {max_code_len}"
                    )
                samples_per_code = int(audio_batch.size(-1)) // max_code_len
                for row_i, (idx, codes, wav_path) in enumerate(chunk):
                    wav_len = int(codes.size(1)) * samples_per_code
                    wav = audio_batch[row_i, 0, :wav_len].detach().cpu().float().numpy()
                    try:
                        sf.write(str(wav_path), wav, self.codec.sample_rate)
                        results[idx] = wav_path
                    except Exception as exc:
                        results[idx] = exc
                del audio_batch, padded
            except Exception as exc:
                for idx, _codes, _wav_path in chunk:
                    results[idx] = exc

        return [result if result is not None else RuntimeError("missing codec result") for result in results]

    def decode_codes(self, codes: torch.Tensor, sample_id: str, rollout_i: int) -> pathlib.Path:
        result = self.decode_codes_batch([codes], [sample_id], [rollout_i])[0]
        if isinstance(result, Exception):
            raise result
        return result

    def make_request(self, wav_path: pathlib.Path, *, text: str, lang: str, prompt_wav: str) -> dict:
        return {
            "reward_type": str(self.args.grpo_reward_type),
            "wav": str(wav_path),
            "text": str(text),
            "lang": str(lang),
            "prompt_wav": str(prompt_wav or ""),
            "device": str(self.device),
            "alpha": float(self.args.grpo_reward_alpha),
            "asr_weight": float(self.args.grpo_asr_weight),
            "sim_weight": float(self.args.grpo_sim_weight),
            "sim_floor": float(self.args.grpo_sim_floor),
            "sim_ceil": float(self.args.grpo_sim_ceil),
            "sim_reward_shape": str(self.args.grpo_sim_reward_shape),
            "sim_reward_beta": float(self.args.grpo_sim_reward_beta),
        }

    def _start_server(self):
        reward_python = str(self.args.grpo_reward_python or self.args.grpo_seedtts_python or sys.executable)
        cmd = [
            reward_python,
            str(self.worker),
            "--seedtts-root", str(self.args.grpo_seedtts_root),
            "--reward-type", str(self.args.grpo_reward_type),
            "--device", str(self.device),
            "--whisper-path", str(self.args.grpo_whisper_path),
            "--ark-asr-path", str(self.args.grpo_ark_asr_path),
            "--ark-asr-max-new-tokens", str(self.args.grpo_ark_asr_max_new_tokens),
            "--ark-asr-audio-max-seconds", str(self.args.grpo_ark_asr_audio_max_seconds),
            "--ark-asr-batch-size", str(self.args.grpo_ark_asr_batch_size),
            "--hf-modules-cache", str(self.args.grpo_hf_modules_cache),
            "--wavlm-checkpoint", str(self.args.grpo_wavlm_checkpoint),
            "--sim-backend", str(self.args.grpo_sim_backend),
            "--omnivoice-repo", str(self.args.grpo_omnivoice_repo),
            "--omnivoice-model-dir", str(self.args.grpo_omnivoice_model_dir),
            "--cv3-root", str(self.args.grpo_cv3_root),
            "--cv3-speakerlab-root", str(self.args.grpo_cv3_speakerlab_root),
            "--cv3-sim-checkpoint", str(self.args.grpo_cv3_sim_checkpoint),
            "--alpha", str(self.args.grpo_reward_alpha),
            "--asr-weight", str(self.args.grpo_asr_weight),
            "--sim-weight", str(self.args.grpo_sim_weight),
            "--sim-floor", str(self.args.grpo_sim_floor),
            "--sim-ceil", str(self.args.grpo_sim_ceil),
            "--sim-reward-shape", str(self.args.grpo_sim_reward_shape),
            "--sim-reward-beta", str(self.args.grpo_sim_reward_beta),
            "--extra-pythonpath", str(self.args.grpo_reward_extra_pythonpath),
            "--server",
        ]
        stderr_path = self.decode_dir / "reward_worker.stderr.log"
        self.stderr_handle = open(stderr_path, "a", encoding="utf-8")
        self.proc = subprocess.Popen(
            cmd,
            text=True,
            stderr=self.stderr_handle,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=1,
        )
        assert self.proc.stdout is not None
        ready = self._read_json_line(timeout=float(self.args.grpo_reward_timeout))
        if not ready:
            raise RuntimeError("reward server failed to start")
        obj = ready
        if not obj.get("ready"):
            raise RuntimeError(f"unexpected reward server ready line: {ready}")

    def _read_json_line(self, timeout: float) -> dict:
        assert self.proc is not None and self.proc.stdout is not None
        fd = self.proc.stdout.fileno()
        deadline = time.monotonic() + max(float(timeout), 1e-6)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"reward server timed out after {timeout}s")
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                raise TimeoutError(f"reward server timed out after {timeout}s")
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("reward server closed stdout")
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                # Third-party libraries should write logs to stderr, but keep
                # this guard so a stray stdout line cannot poison the protocol.
                continue

    def score_batch(self, reqs: list[dict]) -> list[dict]:
        last_error = None
        for attempt in range(2):
            if self.proc is None or self.proc.poll() is not None:
                self._start_server()
            try:
                assert self.proc is not None and self.proc.stdin is not None and self.proc.stdout is not None
                payload = {"cmd": "score_batch", "items": reqs}
                self.proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
                self.proc.stdin.flush()
                obj = self._read_json_line(timeout=float(self.args.grpo_reward_timeout))
                if not obj.get("ok"):
                    raise RuntimeError(obj.get("error", "unknown reward server error"))
                out = []
                for item in obj.get("results", []):
                    if item.get("ok"):
                        out.append(item["result"])
                    else:
                        out.append({"_failed": True, "_error": item.get("error", "unknown reward error")})
                if len(out) != len(reqs):
                    raise RuntimeError(f"reward batch size mismatch: got {len(out)} expected {len(reqs)}")
                return out
            except Exception as exc:
                last_error = exc
                self._stop_server()
        raise RuntimeError(f"reward server failed: {last_error}") from last_error

    def cleanup_wavs(self, paths: list[pathlib.Path]):
        if bool(self.args.grpo_reward_keep_wavs):
            return
        for path in paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass

    def _stop_server(self):
        proc = self.proc
        self.proc = None
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin is not None:
                proc.stdin.write(json.dumps({"cmd": "stop"}) + "\n")
                proc.stdin.flush()
        except Exception:
            pass
        try:
            proc.terminate()
        except Exception:
            pass
        if self.stderr_handle is not None:
            try:
                self.stderr_handle.close()
            except Exception:
                pass
            self.stderr_handle = None


class ArkttsHFSlowARGRPOTrainer(base.FishS2ProTrainer):
    def __init__(self, *args, pad_id: int, codec=None, ref_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        if compute_grpo_outcome_advantage is None:
            raise RuntimeError(f"failed to import verl GRPO advantage: {_VERL_IMPORT_ERROR}")
        self.pad_id = int(pad_id)
        self.codec = codec
        self.ref_model = ref_model
        self.reward_helper = None

    def _get_train_sampler(self, train_dataset=None):
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        if dataset is None or not bool(getattr(self.args, "grpo_length_bucket", True)):
            return super()._get_train_sampler(train_dataset)
        rows = getattr(dataset, "rows", None)
        tokenizer = getattr(dataset, "tokenizer", None)
        text_key = getattr(dataset, "text_key", None)
        if rows is None or tokenizer is None or not text_key:
            base.rank0_print(
                "[length-bucket] dataset has no rows/tokenizer/text_key; using base sampler"
            )
            return super()._get_train_sampler(train_dataset)

        lengths = [
            len(tokenizer.encode(base.clean_text_for_train(row[text_key])))
            for row in rows
        ]
        world_size = max(1, int(getattr(self.args, "world_size", 1)))
        global_batch_size = (
            int(self.args.per_device_train_batch_size) * world_size
        )
        base.rank0_print(
            f"[length-bucket] enabled samples={len(lengths)} "
            f"global_batch_size={global_batch_size} world_size={world_size}"
        )
        return GlobalLengthBucketSampler(
            lengths=lengths,
            global_batch_size=global_batch_size,
            seed=int(self.args.seed),
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        student = opd.actual_model(model)
        labels = inputs["labels"]
        batch_size = int(labels.size(0))
        prompt_lens = inputs["attention_masks"].logical_not().sum(dim=-1).tolist()
        rollout_n = int(self.args.grpo_rollout_n)
        max_new_tokens = int(self.args.grpo_max_new_tokens)
        rollout_lens = [max_new_tokens] * batch_size
        unique_inputs = inputs["inputs"]
        repeated_inputs = unique_inputs.repeat_interleave(rollout_n, dim=0)
        repeated_prompt_lens = [x for x in prompt_lens for _ in range(rollout_n)]
        repeated_rollout_lens = [x for x in rollout_lens for _ in range(rollout_n)]
        generated = sample_rollout_student_cached(
            model=student,
            inputs=repeated_inputs,
            prompt_lens=repeated_prompt_lens,
            rollout_lens=repeated_rollout_lens,
            max_new_tokens=max_new_tokens,
            pad_id=self.pad_id,
            temperature=float(self.args.grpo_temperature),
            top_k=int(self.args.grpo_top_k),
            top_p=float(self.args.grpo_top_p),
            compact_semantic_logits=_uses_falcon_slow_backbone(student),
        )
        forced_inputs, forced_mask, forced_labels, generated_positions = opd.build_forced_batch(
            prompt_inputs=repeated_inputs,
            prompt_lens=repeated_prompt_lens,
            generated=generated,
            pad_id=self.pad_id,
        )
        student_outputs = arktts_slow_hidden_forward(student, forced_inputs, forced_mask)
        token_logits = F.linear(student_outputs.hidden_states, student.embeddings.weight)
        log_probs, response_mask = collect_slow_log_probs(token_logits, forced_labels, generated_positions, student)

        kl_weight = float(getattr(self.args, "grpo_kl_weight", 0.0))
        entropy_weight = float(getattr(self.args, "grpo_entropy_weight", 0.0))
        kl_loss = log_probs.new_tensor(0.0, dtype=torch.float32)
        entropy_mean = log_probs.new_tensor(0.0, dtype=torch.float32)
        if kl_weight != 0.0:
            if self.ref_model is None:
                raise RuntimeError("grpo_kl_weight > 0 requires a reference model")
            with torch.no_grad():
                ref_student = opd.actual_model(self.ref_model)
                ref_outputs = arktts_slow_hidden_forward(ref_student, forced_inputs, forced_mask)
                ref_token_logits = F.linear(ref_outputs.hidden_states, ref_student.embeddings.weight)
                ref_log_probs, _ = collect_slow_log_probs(
                    ref_token_logits, forced_labels, generated_positions, ref_student
                )
            kl_per_token = torch.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1.0
            kl_loss = (kl_per_token * response_mask).sum() / response_mask.sum().clamp(min=1.0)
        if entropy_weight != 0.0:
            entropy, entropy_mask = collect_slow_entropy(
                token_logits, forced_labels, generated_positions, student
            )
            entropy_mean = (entropy * response_mask).sum() / response_mask.sum().clamp(min=1.0)

        if self.reward_helper is None:
            self.reward_helper = RewardHelper(self.args, self.codec, next(student.parameters()).device)
        scores = []
        asr_scores = []
        sim_scores = []
        extra = {"reward": [], "asr_reward": [], "sim_reward": [], "error_rate": [], "wer": [], "cer": [], "sim": [], "reward_failed": []}
        base_ids = [str(i // int(self.args.grpo_rollout_n)) for i in range(len(generated))]
        decoded_wavs: list[Optional[pathlib.Path]] = []
        reward_reqs: list[dict] = []
        reward_req_indices: list[int] = []
        per_rollout_results: list[dict] = [{} for _ in generated]
        per_rollout_failed: list[float] = [1.0 for _ in generated]
        source_indices = [i // int(self.args.grpo_rollout_n) for i in range(len(generated))]
        decode_results = self.reward_helper.decode_codes_batch(
            [codes.detach().cpu() for codes in generated],
            [inputs["sample_id"][source_i] for source_i in source_indices],
            list(range(len(generated))),
        )
        for i, decoded in enumerate(decode_results):
            source_i = i // int(self.args.grpo_rollout_n)
            if isinstance(decoded, Exception):
                print(
                    f"[reward] decode failed sample={source_i} rollout={i}: "
                    f"{type(decoded).__name__}: {decoded}",
                    flush=True,
                )
                decoded_wavs.append(None)
            else:
                wav = decoded
                decoded_wavs.append(wav)
                reward_reqs.append(self.reward_helper.make_request(
                    wav,
                    text=inputs["target_text"][source_i],
                    lang=inputs["lang"][source_i],
                    prompt_wav=inputs["prompt_wav"][source_i],
                ))
                reward_req_indices.append(i)

        if reward_reqs:
            try:
                batch_results = self.reward_helper.score_batch(reward_reqs)
            except Exception as exc:
                print(f"[reward] batch failed: {type(exc).__name__}: {exc}", flush=True)
                batch_results = [{"_failed": True, "_error": str(exc)} for _ in reward_reqs]
            for req_i, result in zip(reward_req_indices, batch_results):
                if result.get("_failed"):
                    source_i = req_i // int(self.args.grpo_rollout_n)
                    print(f"[reward] score failed sample={source_i} rollout={req_i}: {result.get('_error', 'unknown')}", flush=True)
                    continue
                per_rollout_results[req_i] = result
                per_rollout_failed[req_i] = 0.0
        self.reward_helper.cleanup_wavs([p for p in decoded_wavs if p is not None])

        for i, result in enumerate(per_rollout_results):
            score = float(result.get("score", 0.0))
            failed = float(per_rollout_failed[i])
            asr_score = result.get("asr_score")
            sim_score = result.get("sim_score")
            if asr_score is None and str(self.args.grpo_reward_type) == "asr":
                asr_score = score
            if sim_score is None and str(self.args.grpo_reward_type) == "sim":
                sim_score = score
            scores.append(score)
            asr_scores.append(float(asr_score) if asr_score is not None else float("nan"))
            sim_scores.append(float(sim_score) if sim_score is not None else float("nan"))
            extra["reward"].append(score)
            extra["asr_reward"].append(float(asr_score) if asr_score is not None else float("nan"))
            extra["sim_reward"].append(float(sim_score) if sim_score is not None else float("nan"))
            extra["error_rate"].append(float(result.get("error_rate", float("nan"))))
            extra["wer"].append(float(result.get("wer", float("nan")) if result.get("wer") is not None else float("nan")))
            extra["cer"].append(float(result.get("cer", float("nan")) if result.get("cer") is not None else float("nan")))
            extra["sim"].append(float(result.get("sim", float("nan")) if result.get("sim") is not None else float("nan")))
            extra["reward_failed"].append(failed)

        denom = response_mask.sum().clamp(min=1.0)

        def pg_loss_from_scores(values: list[float]) -> torch.Tensor:
            reward_tensor = log_probs.new_zeros(log_probs.shape)
            for row_i, value in enumerate(values):
                valid_pos = torch.nonzero(response_mask[row_i].gt(0), as_tuple=False).flatten()
                if int(valid_pos.numel()) > 0 and math.isfinite(float(value)):
                    reward_tensor[row_i, int(valid_pos[-1].item())] = float(value)
            advantages, _returns = compute_grpo_outcome_advantage(
                token_level_rewards=reward_tensor,
                response_mask=response_mask,
                index=np.array(base_ids, dtype=object),
                norm_adv_by_std_in_grpo=bool(self.args.grpo_norm_adv_by_std),
            )
            return -((log_probs * advantages.detach()) * response_mask).sum() / denom

        asr_pg_loss = log_probs.new_tensor(0.0, dtype=torch.float32)
        sim_pg_loss = log_probs.new_tensor(0.0, dtype=torch.float32)
        if str(self.args.grpo_reward_type) == "asr_sim":
            asr_pg_loss = pg_loss_from_scores(asr_scores)
            sim_pg_loss = pg_loss_from_scores(sim_scores)
            # Use a single advantage computed from the combined weighted score.
            # Separate per-reward advantages can point in opposite directions
            # and cancel each other out, leaving the policy with no net signal.
            pg_loss = pg_loss_from_scores(scores)
        else:
            pg_loss = pg_loss_from_scores(scores)
            if str(self.args.grpo_reward_type) == "asr":
                asr_pg_loss = pg_loss
            if str(self.args.grpo_reward_type) == "sim":
                sim_pg_loss = pg_loss

        sft_loss_weight = float(getattr(self.args, "sft_loss_weight", 0.0))
        sft_loss = labels.new_tensor(0.0, dtype=torch.float32)
        sft_slow_loss = labels.new_tensor(0.0, dtype=torch.float32)
        sft_fast_loss = labels.new_tensor(0.0, dtype=torch.float32)
        if sft_loss_weight != 0.0:
            if inputs.get("sft_inputs") is None:
                raise RuntimeError(
                    "sft_loss_weight > 0 but the batch has no sft_inputs; "
                    "check --audio_ids_key and that the training JSONL contains target codes"
                )
            sft_loss, sft_slow_loss, sft_fast_loss = compute_arktts_sft_loss(
                model,
                inputs["sft_inputs"],
                inputs["sft_attention_masks"],
                inputs["sft_labels"],
            )
        loss = (
            pg_loss
            + sft_loss_weight * sft_loss
            + kl_weight * kl_loss
            - entropy_weight * entropy_mean
        )

        reward_values = torch.tensor(scores, dtype=torch.float32, device=loss.device)
        metrics = {
            "grpo_loss": loss.detach(),
            "grpo_pg_loss": pg_loss.detach(),
            "grpo_asr_pg_loss": asr_pg_loss.detach(),
            "grpo_sim_pg_loss": sim_pg_loss.detach(),
            "grpo_sft_loss": sft_loss.detach(),
            "grpo_sft_slow_loss": sft_slow_loss.detach(),
            "grpo_sft_fast_loss": sft_fast_loss.detach(),
            "grpo_kl_loss": kl_loss.detach(),
            "grpo_entropy": entropy_mean.detach(),
            "grpo_reward_mean": reward_values.mean() if reward_values.numel() else loss.new_tensor(0.0),
            "grpo_reward_std": reward_values.std(unbiased=False) if reward_values.numel() else loss.new_tensor(0.0),
            "grpo_asr_weight": loss.new_tensor(float(self.args.grpo_asr_weight)),
            "grpo_sim_weight": loss.new_tensor(float(self.args.grpo_sim_weight)),
            "grpo_response_tokens_mean": response_mask.sum(dim=-1).float().mean(),
            "grpo_rollout_n": loss.new_tensor(float(self.args.grpo_rollout_n)),
            "grpo_reward_failed": loss.new_tensor(float(np.nanmean(extra["reward_failed"]))),
        }
        for metric_name, values in (("asr_reward", asr_scores), ("sim_reward", sim_scores)):
            vals = np.array(values, dtype=np.float32)
            finite = vals[np.isfinite(vals)]
            if finite.size:
                metrics[f"grpo_{metric_name}_mean"] = loss.new_tensor(float(finite.mean()))
                metrics[f"grpo_{metric_name}_std"] = loss.new_tensor(float(finite.std()))
        for name in ("error_rate", "wer", "cer", "sim"):
            vals = np.array(extra[name], dtype=np.float32)
            finite = vals[np.isfinite(vals)]
            if finite.size:
                metrics[f"grpo_{name}_mean"] = loss.new_tensor(float(finite.mean()))
        self._record_loss_metrics(metrics)
        if return_outputs:
            return loss, {"token_logits": token_logits, **{k: v.detach() for k, v in metrics.items()}}
        return loss


def make_dataset(data_args, tokenizer):
    return MetaJsonlFishAudioDataset(
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
        skip_samples=data_args.skip_train_samples,
    )


def main():
    base.configure_tmpdir()
    parser = HfArgumentParser((base.ModelArguments, base.DataArguments, SlowARGRPOTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
    training_args.remove_unused_columns = False
    device = opd.resolve_device()

    tokenizer = FishTokenizer(model_args.pretrained_ckpt_path)
    train_dataset = make_dataset(data_args, tokenizer)
    model = AutoModel.from_pretrained(
        model_args.pretrained_ckpt_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    model.tokenizer = tokenizer
    model.config.max_seq_len = int(model_args.max_length)
    model.config.use_gradient_checkpointing = True
    base.ensure_hf_config_compat(model)
    model.config.scale_codebook_embeddings = bool(training_args.grpo_scale_codebook_embeddings)
    if model_args.freeze_fast_ar and model_args.freeze_slow_ar:
        raise ValueError("freeze_fast_ar and freeze_slow_ar cannot both be true")
    if model_args.freeze_fast_ar:
        base.freeze_fast_ar(model)
    elif model_args.freeze_slow_ar:
        base.freeze_slow_ar(model)
    else:
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)
        base.rank0_print(f"[freeze] no freeze: trainable {total / 1e6:.2f}M params")

    ref_model = None
    if float(getattr(training_args, "grpo_kl_weight", 0.0)) != 0.0:
        ref_model = AutoModel.from_pretrained(
            model_args.pretrained_ckpt_path,
            trust_remote_code=True,
            dtype=torch.bfloat16,
        )
        ref_model.tokenizer = tokenizer
        ref_model.config.max_seq_len = int(model_args.max_length)
        ref_model.config.use_gradient_checkpointing = False
        base.ensure_hf_config_compat(ref_model)
        ref_model.config.scale_codebook_embeddings = bool(training_args.grpo_scale_codebook_embeddings)
        ref_model = ref_model.to(device)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        base.rank0_print("[ref] reference model loaded for KL constraint")

    dtype = torch.bfloat16 if bool(training_args.bf16) else torch.float16
    # Load the codec bundled with the ARK-TTS checkpoint. The fish-speech DAC
    # loader uses a fixed post-module size (3072) that does not match the
    # audio8_tts codec.pth (1216), so the model's own loader must be used.
    codec = model.load_codec(device=str(device), dtype=dtype)
    # RewardHelper calls codec.from_indices(); ArkttsCodec exposes the same
    # operation as decode().
    codec.from_indices = codec.decode
    codec.eval()
    for p in codec.parameters():
        p.requires_grad = False

    collator = MetaFishAudioCollator(tokenizer=tokenizer, max_length=model_args.max_length)
    pad_id = tokenizer.get_token_id("<|end_of_text|>") or tokenizer.get_token_id(IM_END_TOKEN) or 0
    trainer = ArkttsHFSlowARGRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=collator,
        pad_id=pad_id,
        codec=codec,
        ref_model=ref_model,
    )
    if training_args.do_train:
        resume = base.resolve_resume_checkpoint(training_args.output_dir, training_args.resume_mode)
        result = trainer.train(resume_from_checkpoint=resume)
        trainer.log_metrics("train", result.metrics)
        trainer.save_metrics("train", result.metrics)
        trainer.save_state()
    if training_args.skip_final_save:
        base.rank0_print("[save] skip_final_save=true")
    else:
        trainer.save_model(training_args.output_dir)
    if training_args.export_dir and not training_args.skip_final_save:
        base.export_fish_pretrained(trainer, training_args.export_dir)
        base.copy_aux_files(model_args.pretrained_ckpt_path, training_args.export_dir)


if __name__ == "__main__":
    main()

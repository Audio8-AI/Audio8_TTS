#!/usr/bin/env python3
import logging
import math
import os
import pathlib
import sys
import time
import types
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, SequentialSampler
from torch.utils.checkpoint import checkpoint
from transformers import HfArgumentParser

TRAIN_DIR = pathlib.Path(__file__).resolve().parent
FISH_REPO = pathlib.Path(os.environ.get("FISH_SPEECH_ROOT", "/opt/src/fish-speech"))
for path in (TRAIN_DIR, FISH_REPO):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_fish_s2pro_ds_fish_prompt_cached as base
from fish_speech.models.text2semantic.llama import BaseTransformer
from fish_speech.tokenizer import IM_END_TOKEN, FishTokenizer


logger = logging.getLogger(__name__)


@dataclass
class OpdModelArguments(base.ModelArguments):
    teacher_ckpt_path: str = field(default=os.environ.get("TEACHER_MODEL_PATH", "teacher_model"))
    teacher_dtype: str = field(default="bfloat16", metadata={"help": "bfloat16 or float32"})
    teacher_load_stagger_seconds: float = field(
        default=2.0,
        metadata={"help": "Sleep rank * seconds before teacher load to reduce shared-storage pressure."},
    )


@dataclass
class OpdTrainingArguments(base.FishTrainingArguments):
    opd_max_new_tokens: int = field(default=128)
    opd_top_k: int = field(default=32)
    opd_temperature: float = field(default=1.0)
    opd_slow_loss_weight: float = field(default=1.0)
    opd_fast_loss_weight: float = field(default=1.0)
    opd_fast_full_vocab_kl: bool = field(
        default=False,
        metadata={"help": "Use full codebook-vocab KL for fast AR. This avoids per-row CPU top-k union overhead."},
    )
    opd_slow_gpu_union_kl: bool = field(
        default=True,
        metadata={"help": "Build slow AR top-k union support on GPU instead of with per-row CPU Python sets."},
    )
    opd_cached_rollout: bool = field(
        default=True,
        metadata={"help": "Use per-sample KV-cache decoding for student rollout. Falls back to full-prefix rollout when false."},
    )
    opd_rollout_group_size: int = field(
        default=0,
        metadata={"help": "Maximum samples per cached rollout group. 0 means no extra split beyond prompt/length buckets."},
    )
    opd_rollout_length_bucket: int = field(
        default=0,
        metadata={"help": "Bucket cached rollout samples by target length. 0 means exact rollout length."},
    )
    sft_loss_weight: float = field(default=0.0)
    opd_include_codebook0: bool = field(
        default=False,
        metadata={"help": "Include codebook0 in fast OPD. Default false because generation derives codebook0 from the slow semantic token."},
    )
    opd_debug_steps: int = field(default=0)
    opd_disable_student_checkpointing: bool = field(
        default=True,
        metadata={"help": "Disable slow-AR gradient checkpointing during OPD forced forward to trade memory for speed."},
    )
    skip_final_save: bool = field(
        default=False,
        metadata={"help": "Skip final save/export. Intended for short throughput benchmarks."},
    )


def rank0_print(*args, **kwargs):
    base.rank0_print(*args, **kwargs)


def distributed_rank() -> int:
    return int(os.environ.get("RANK", "0") or 0)


def distributed_world_size() -> int:
    return max(int(os.environ.get("WORLD_SIZE", "1") or 1), 1)


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0") or 0)


def debug_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def all_rank_debug_print(message: str) -> None:
    prefix = f"[opd-debug-rank] rank={distributed_rank()} local_rank={local_rank()} "
    print(prefix + message, flush=True)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        lr = local_rank()
        torch.cuda.set_device(lr)
        return torch.device("cuda", lr)
    return torch.device("cpu")


def dtype_from_name(name: str) -> torch.dtype:
    value = str(name).strip().lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def actual_model(model):
    return model.module if hasattr(model, "module") else model


def label_rollout_bounds(labels: torch.Tensor, max_new_tokens: int, model) -> tuple[list[int], list[int]]:
    prompt_lens = []
    rollout_lens = []
    seq_len = int(labels.size(-1))
    slow_label_mask = labels[:, 0].ne(-100)
    semantic_begin = int(model.config.semantic_begin_id)
    semantic_end = int(model.config.semantic_end_id)
    im_end_id = int(model.tokenizer.get_token_id(IM_END_TOKEN))
    for batch_i, row in enumerate(slow_label_mask):
        idx = torch.nonzero(row, as_tuple=False).flatten()
        if idx.numel() == 0:
            prompt_lens.append(seq_len)
            rollout_lens.append(0)
            continue

        first_label_pos = int(idx[0].item())
        # Labels are next-token targets shifted left by one. To predict the first
        # supervised target at position `first_label_pos`, the prompt must include
        # input position `first_label_pos` and gather logits from that position.
        prompt_lens.append(min(first_label_pos + 1, seq_len))
        allowed = 0
        for pos in idx.tolist():
            token_id = int(labels[batch_i, 0, int(pos)].item())
            if semantic_begin <= token_id <= semantic_end or token_id == im_end_id:
                allowed += 1
                if token_id == im_end_id:
                    break
            else:
                break
        rollout_lens.append(min(int(max_new_tokens), allowed))
    return prompt_lens, rollout_lens


def pad_token_rows(rows: list[torch.Tensor], pad_id: int) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    if not rows:
        raise ValueError("Cannot pad an empty row list")
    device = rows[0].device
    codebook_dim = int(rows[0].size(0))
    max_len = max(int(row.size(1)) for row in rows)
    out = torch.zeros((len(rows), codebook_dim, max_len), dtype=rows[0].dtype, device=device)
    mask = torch.ones((len(rows), max_len), dtype=torch.bool, device=device)
    lengths = []
    for i, row in enumerate(rows):
        n = int(row.size(1))
        lengths.append(n)
        out[i, 0, :n] = row[0]
        out[i, 1:, :n] = row[1:]
        out[i, 0, n:] = int(pad_id)
        mask[i, :n] = False
    return out, mask, lengths


def build_semantic_logit_bias(model, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    vocab_size = int(model.config.vocab_size)
    bias = torch.full((vocab_size,), -float("inf"), device=device, dtype=dtype)
    bias[int(model.config.semantic_begin_id) : int(model.config.semantic_end_id) + 1] = 0
    bias[int(model.tokenizer.get_token_id(IM_END_TOKEN))] = 0
    return bias


def mask_slow_logits(logits: torch.Tensor, model) -> torch.Tensor:
    out = torch.full_like(logits, -float("inf"))
    begin = int(model.config.semantic_begin_id)
    end = int(model.config.semantic_end_id)
    out[..., begin : end + 1] = logits[..., begin : end + 1]
    im_end_id = int(model.tokenizer.get_token_id(IM_END_TOKEN))
    out[..., im_end_id] = logits[..., im_end_id]
    return out


def slow_hidden_forward(
    model,
    inp: torch.Tensor,
    key_padding_mask: Optional[torch.Tensor] = None,
):
    seq_len = int(inp.size(2))
    x = model.embed(inp)
    freqs_cis = model.freqs_cis[:seq_len]

    mask = None
    if key_padding_mask is not None:
        causal = model.causal_mask[:seq_len, :seq_len]
        mask = causal[None, None, :, :] & key_padding_mask[:, None, None, :].logical_not()

    for layer in model.layers:
        if model.config.use_gradient_checkpointing and model.training:
            x = checkpoint(layer, x, freqs_cis, mask, use_reentrant=True)
        else:
            x = layer(x, freqs_cis, mask)

    slow_out = model.norm(x)
    hidden_out = slow_out if getattr(model.config, "norm_fastlayer_input", False) else x
    return types.SimpleNamespace(hidden_states=hidden_out)


def slow_logits_from_hidden(model, hidden: torch.Tensor) -> torch.Tensor:
    if model.config.tie_word_embeddings:
        return F.linear(hidden, model.embeddings.weight)
    return model.output(hidden)


def gather_slow_logits_from_hidden(
    model,
    hidden: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    if int(hidden.numel()) == 0:
        return hidden.new_zeros((0, int(token_ids.size(-1))))
    token_ids = token_ids.long()
    if model.config.tie_word_embeddings:
        weight = model.embeddings.weight.index_select(0, token_ids.reshape(-1))
        weight = weight.reshape(*token_ids.shape, int(hidden.size(-1)))
        return torch.einsum("rd,rkd->rk", hidden, weight)
    weight = model.output.weight.index_select(0, token_ids.reshape(-1))
    weight = weight.reshape(*token_ids.shape, int(hidden.size(-1)))
    logits = torch.einsum("rd,rkd->rk", hidden, weight)
    bias = getattr(model.output, "bias", None)
    if bias is not None:
        logits = logits + bias.index_select(0, token_ids.reshape(-1)).reshape_as(token_ids)
    return logits


def fast_logits_from_hidden(model, hidden: torch.Tensor, prefix_codebooks: Optional[torch.Tensor]) -> torch.Tensor:
    """Return fast-AR logits for the next codebook.

    `hidden` is the slow hidden state for one generated semantic position after
    `fast_project_in`. `prefix_codebooks` is shaped [B, K] and contains
    generated codebooks 0..K-1. The returned logits predict codebook K.
    """
    x = hidden[:, None, :]
    if prefix_codebooks is not None and int(prefix_codebooks.numel()) > 0:
        x = torch.cat([x, model.fast_embeddings(prefix_codebooks.long())], dim=1)
    seq_len = int(x.size(1))
    fast_mask = model.causal_mask[None, None, :seq_len, :seq_len]
    fast_freqs_cis = model.fast_freqs_cis[:seq_len]
    for layer in model.fast_layers:
        x = layer(x, fast_freqs_cis, fast_mask)
    out = model.fast_norm(x)
    return model.fast_output(out[:, -1:, :])


def clear_kv_caches(model) -> None:
    for layer in getattr(model, "layers", []):
        cache = getattr(getattr(layer, "attention", None), "kv_cache", None)
        if cache is not None:
            cache.k_cache.zero_()
            cache.v_cache.zero_()
    for layer in getattr(model, "fast_layers", []):
        cache = getattr(getattr(layer, "attention", None), "kv_cache", None)
        if cache is not None:
            cache.k_cache.zero_()
            cache.v_cache.zero_()


def remove_kv_caches(model) -> None:
    for layer in getattr(model, "layers", []):
        attention = getattr(layer, "attention", None)
        if attention is not None:
            attention.kv_cache = None
    for layer in getattr(model, "fast_layers", []):
        attention = getattr(layer, "attention", None)
        if attention is not None:
            attention.kv_cache = None
    model.max_batch_size = -1
    model.max_seq_len = -1


def remove_fast_kv_caches(model) -> None:
    for layer in getattr(model, "fast_layers", []):
        attention = getattr(layer, "attention", None)
        if attention is not None:
            attention.kv_cache = None


def ensure_generation_caches(model, max_batch_size: int, max_seq_len: int) -> None:
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device
    with torch.device(device):
        model.setup_caches(
            max_batch_size=int(max_batch_size),
            max_seq_len=min(int(max_seq_len), int(model.config.max_seq_len)),
            dtype=dtype,
        )


def freeze_fast_ar_strict(model: torch.nn.Module) -> None:
    fast_prefixes = (
        "fast_project_in.",
        "fast_embeddings.",
        "fast_layers.",
        "fast_norm.",
        "fast_output.",
    )
    frozen = 0
    for name, param in model.named_parameters():
        if name.startswith(fast_prefixes):
            param.requires_grad = False
            frozen += param.numel()
    rank0_print(f"[freeze] strict Fast AR freeze, including fast_project_in: {frozen / 1e6:.2f}M params")


def greedy_fast_codebooks_from_hidden(model, hidden: torch.Tensor, semantic_token: torch.Tensor) -> torch.Tensor:
    codebooks = []
    codebook0 = (semantic_token - int(model.config.semantic_begin_id)).clamp(
        min=0,
        max=int(model.config.codebook_size) - 1,
    )
    codebooks.append(codebook0)
    for cb_idx in range(1, int(model.config.num_codebooks)):
        prefix = torch.stack(codebooks, dim=1)
        cb_logits = fast_logits_from_hidden(model, hidden, prefix).squeeze(1)
        cb_next = torch.argmax(cb_logits, dim=-1)
        codebooks.append(cb_next)
    return torch.stack(codebooks, dim=1)


@torch.no_grad()
def greedy_rollout_student(
    model,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    prompt_lens: list[int],
    rollout_lens: list[int],
    max_new_tokens: int,
    pad_id: int,
    generate_fast_codebooks: bool = True,
) -> list[torch.Tensor]:
    was_training = bool(model.training)
    model.eval()
    device = inputs.device
    rows = [inputs[i, :, : int(prompt_lens[i])].clone() for i in range(int(inputs.size(0)))]
    generated = [
        torch.empty((int(inputs.size(1)), 0), dtype=inputs.dtype, device=device)
        for _ in range(int(inputs.size(0)))
    ]
    finished = [False for _ in rows]
    im_end_id = int(model.tokenizer.get_token_id(IM_END_TOKEN))
    semantic_begin = int(model.config.semantic_begin_id)
    semantic_end = int(model.config.semantic_end_id)

    max_steps = min(int(max_new_tokens), max(rollout_lens) if rollout_lens else 0)
    for step in range(max_steps):
        active = [
            i
            for i, done in enumerate(finished)
            if not done and step < int(rollout_lens[i])
        ]
        if not active:
            break
        batch_rows = [rows[i] for i in active]
        inp, key_padding_mask, lengths = pad_token_rows(batch_rows, pad_id)
        parent = BaseTransformer.forward(model, inp=inp, key_padding_mask=key_padding_mask)
        gather_pos = torch.tensor([n - 1 for n in lengths], dtype=torch.long, device=device)
        b = torch.arange(len(active), device=device)
        slow_logits = parent.logits[b, gather_pos, :]
        next_token = torch.argmax(mask_slow_logits(slow_logits, model), dim=-1)
        slow_hidden = parent.hidden_states[b, gather_pos, :]

        next_cols = []
        active_finished = []
        semantic_next = (next_token >= semantic_begin) & (next_token <= semantic_end)
        stacked_codebooks = torch.zeros(
            (len(active), int(model.config.num_codebooks)),
            dtype=torch.long,
            device=device,
        )
        if bool(semantic_next.any()):
            semantic_idx = torch.nonzero(semantic_next, as_tuple=False).flatten()
            semantic_token = next_token[semantic_idx]
            codebook0 = (semantic_token - semantic_begin).clamp(min=0, max=int(model.config.codebook_size) - 1)
            stacked_codebooks[semantic_idx, 0] = codebook0
            if bool(generate_fast_codebooks):
                semantic_hidden = model.fast_project_in(slow_hidden[semantic_idx])
                codebooks = [codebook0]
                for cb_idx in range(1, int(model.config.num_codebooks)):
                    prefix = torch.stack(codebooks, dim=1)
                    cb_logits = fast_logits_from_hidden(model, semantic_hidden, prefix).squeeze(1)
                    cb_next = torch.argmax(cb_logits, dim=-1)
                    codebooks.append(cb_next)
                stacked_codebooks[semantic_idx] = torch.stack(codebooks, dim=1)
        for row_i, token_id in enumerate(next_token):
            col = torch.zeros((int(model.config.num_codebooks) + 1, 1), dtype=inputs.dtype, device=device)
            token_int = int(token_id.item())
            col[0, 0] = token_int
            if semantic_begin <= token_int <= semantic_end:
                col[1:, 0] = stacked_codebooks[row_i].to(inputs.dtype)
            next_cols.append(col)
            active_finished.append(token_int == im_end_id)
        for source_i, col, is_done in zip(active, next_cols, active_finished):
            rows[source_i] = torch.cat([rows[source_i], col], dim=1)
            generated[source_i] = torch.cat([generated[source_i], col], dim=1)
            finished[source_i] = bool(is_done)
    model.train(was_training)
    return generated


@torch.no_grad()
def slow_forward_generate_cached(
    model,
    inp: torch.Tensor,
    input_pos: torch.Tensor,
    key_padding_mask: Optional[torch.Tensor] = None,
):
    """Slow-AR cached generate with batch padding-mask support.

    Fish's built-in forward_generate assumes every row has the same valid prompt
    length. This variant keeps one batched rollout even when prompt lengths differ.
    """
    embeds = []
    for i in range(int(model.config.num_codebooks)):
        emb = model.codebook_embeddings(inp[:, i + 1] + i * int(model.config.codebook_size))
        embeds.append(emb)

    vq_embeds_sum = torch.stack(embeds, dim=1).sum(dim=1)
    vq_masks = (inp[:, 0] >= int(model.config.semantic_begin_id)) & (
        inp[:, 0] <= int(model.config.semantic_end_id)
    )
    vq_embeds_sum[~vq_masks] = 0
    x = model.embeddings(inp[:, 0]) + vq_embeds_sum

    if model.config.scale_codebook_embeddings:
        vq_masks_expanded = vq_masks.unsqueeze(-1).expand_as(x)
        x = torch.where(
            vq_masks_expanded,
            x / math.sqrt(int(model.config.num_codebooks) + 1),
            x,
        )

    max_seq_len = int(model.max_seq_len)
    mask = model.causal_mask[None, None, input_pos, :max_seq_len]
    if key_padding_mask is not None:
        valid_keys = key_padding_mask[:, :max_seq_len].logical_not()
        mask = mask & valid_keys[:, None, None, :]
    freqs_cis = model.freqs_cis[input_pos]

    for layer in model.layers:
        x = layer(x, freqs_cis, mask, input_pos=input_pos)

    if x.size(1) > 1:
        x = x[:, -1:]

    slow_out = model.norm(x)
    if model.config.is_reward_model:
        token_logits = model.score_output(slow_out)
    elif model.config.tie_word_embeddings:
        token_logits = F.linear(slow_out, model.embeddings.weight)
    else:
        token_logits = model.output(slow_out)

    hidden_out = (
        slow_out if getattr(model.config, "norm_fastlayer_input", False) else x
    )
    return types.SimpleNamespace(logits=token_logits, hidden_states=hidden_out)


@torch.no_grad()
def greedy_rollout_student_cached(
    model,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    prompt_lens: list[int],
    rollout_lens: list[int],
    max_new_tokens: int,
    pad_id: int,
    generate_fast_codebooks: bool = True,
    rollout_group_size: int = 0,
    rollout_length_bucket: int = 0,
) -> list[torch.Tensor]:
    was_training = bool(model.training)
    model.eval()
    device = inputs.device
    batch_size = int(inputs.size(0))
    codebook_dim = int(inputs.size(1))
    generated = [
        torch.empty((codebook_dim, 0), dtype=inputs.dtype, device=device)
        for _ in range(batch_size)
    ]
    if not rollout_lens or max(rollout_lens) <= 0:
        remove_kv_caches(model)
        model.train(was_training)
        return generated

    semantic_begin = int(model.config.semantic_begin_id)
    semantic_end = int(model.config.semantic_end_id)
    im_end_id = int(model.tokenizer.get_token_id(IM_END_TOKEN))
    sample_indices = []
    capped_rollout_lens = []
    for sample_i in range(batch_size):
        prompt_len = int(prompt_lens[sample_i])
        if prompt_len <= 0 or prompt_len >= int(model.config.max_seq_len):
            continue
        capped_rollout_len = min(int(max_new_tokens), int(rollout_lens[sample_i]))
        if capped_rollout_len <= 0:
            continue
        sample_indices.append(sample_i)
        capped_rollout_lens.append(capped_rollout_len)

    if not sample_indices:
        remove_kv_caches(model)
        model.train(was_training)
        return generated

    group_size = len(sample_indices)
    max_prompt_len = max(int(prompt_lens[i]) for i in sample_indices)
    max_group_steps = min(
        int(model.config.max_seq_len) - max_prompt_len,
        max(capped_rollout_lens),
    )
    if max_group_steps <= 0:
        remove_kv_caches(model)
        model.train(was_training)
        return generated

    remove_kv_caches(model)
    ensure_generation_caches(
        model,
        max_batch_size=group_size,
        max_seq_len=max_prompt_len + max_group_steps,
    )
    remove_fast_kv_caches(model)

    prompt = torch.zeros(
        (group_size, codebook_dim, max_prompt_len),
        dtype=inputs.dtype,
        device=device,
    )
    prompt[:, 0, :] = int(pad_id)
    key_padding_mask = torch.ones(
        (group_size, int(model.max_seq_len)),
        dtype=torch.bool,
        device=device,
    )
    for row_i, sample_i in enumerate(sample_indices):
        prompt_len = int(prompt_lens[sample_i])
        start = max_prompt_len - prompt_len
        prompt[row_i, :, start:max_prompt_len] = inputs[sample_i, :, :prompt_len]
        key_padding_mask[row_i, start:max_prompt_len] = False

    input_pos = torch.arange(0, max_prompt_len, dtype=torch.long, device=device)
    result = slow_forward_generate_cached(
        model,
        prompt,
        input_pos,
        key_padding_mask=key_padding_mask,
    )
    finished = torch.zeros((group_size,), dtype=torch.bool, device=device)
    group_rollout_lens = torch.tensor(capped_rollout_lens, dtype=torch.long, device=device)
    generated_tensor = torch.zeros(
        (group_size, codebook_dim, max_group_steps),
        dtype=inputs.dtype,
        device=device,
    )
    generated_tensor[:, 0, :] = int(pad_id)
    generated_counts = torch.zeros((group_size,), dtype=torch.long, device=device)

    for step in range(max_group_steps):
        active = (~finished) & group_rollout_lens.gt(step)
        if not bool(active.any().item()):
            break
        slow_logits = result.logits[:, -1, :]
        next_token = torch.argmax(mask_slow_logits(slow_logits, model), dim=-1)
        hidden = result.hidden_states[:, -1, :]

        col = torch.zeros((group_size, codebook_dim, 1), dtype=inputs.dtype, device=device)
        col[:, 0, 0] = int(pad_id)
        col[active, 0, 0] = next_token[active].to(inputs.dtype)
        semantic_next = (
            active
            & next_token.ge(semantic_begin)
            & next_token.le(semantic_end)
        )
        if bool(semantic_next.any().item()):
            semantic_idx = torch.nonzero(semantic_next, as_tuple=False).flatten()
            semantic_token = next_token.index_select(0, semantic_idx)
            codebook0 = (semantic_token - semantic_begin).clamp(min=0, max=int(model.config.codebook_size) - 1)
            col[semantic_idx, 1, 0] = codebook0.to(inputs.dtype)
            if bool(generate_fast_codebooks):
                codebooks = greedy_fast_codebooks_from_hidden(
                    model,
                    model.fast_project_in(hidden.index_select(0, semantic_idx)),
                    semantic_token,
                )
                col[semantic_idx, 1:, 0] = codebooks.to(inputs.dtype)

        generated_tensor[active, :, step : step + 1] = col[active]
        generated_counts[active] = generated_counts[active] + 1
        finished = finished | (active & next_token.eq(im_end_id))

        if step + 1 >= max_group_steps:
            break
        cache_pos = max_prompt_len + step
        key_padding_mask[:, cache_pos] = True
        key_padding_mask[active, cache_pos] = False
        input_pos = torch.tensor([cache_pos], dtype=torch.long, device=device)
        result = slow_forward_generate_cached(
            model,
            col,
            input_pos,
            key_padding_mask=key_padding_mask,
        )

    remove_kv_caches(model)
    for row_i, sample_i in enumerate(sample_indices):
        generated[sample_i] = generated_tensor[row_i, :, : int(generated_counts[row_i].item())]
    model.train(was_training)
    return generated


def build_forced_batch(
    prompt_inputs: torch.Tensor,
    prompt_lens: list[int],
    generated: list[torch.Tensor],
    pad_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[int]]]:
    device = prompt_inputs.device
    codebook_dim = int(prompt_inputs.size(1))
    rows = []
    label_rows = []
    generated_positions: list[list[int]] = []
    for i, gen in enumerate(generated):
        prompt = prompt_inputs[i, :, : int(prompt_lens[i])]
        row = torch.cat([prompt, gen], dim=1)
        labels = torch.full_like(row, -100)
        gen_len = int(gen.size(1))
        if gen_len > 0:
            start = int(prompt.size(1))
            labels[:, start : start + gen_len] = gen
            generated_positions.append(list(range(start, start + gen_len)))
        else:
            generated_positions.append([])
        rows.append(row)
        label_rows.append(labels)

    max_len = max(int(row.size(1)) for row in rows)
    batch = torch.zeros((len(rows), codebook_dim, max_len), dtype=prompt_inputs.dtype, device=device)
    labels = torch.full((len(rows), codebook_dim, max_len), -100, dtype=prompt_inputs.dtype, device=device)
    key_padding_mask = torch.ones((len(rows), max_len), dtype=torch.bool, device=device)
    for i, (row, label) in enumerate(zip(rows, label_rows)):
        n = int(row.size(1))
        batch[i, 0, :n] = row[0]
        batch[i, 1:, :n] = row[1:]
        batch[i, 0, n:] = int(pad_id)
        labels[i, :, :n] = label
        key_padding_mask[i, :n] = False
    return batch, key_padding_mask, labels, generated_positions


def topk_union_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    top_k: int,
    temperature: float,
    gpu_union: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    if int(student_logits.numel()) == 0 or int(student_logits.size(0)) == 0:
        zero = student_logits.sum() * 0.0
        return zero, {"rows": 0.0, "support_mean": 0.0}
    k = min(int(top_k), int(student_logits.size(-1)))
    _, student_ids = torch.topk(student_logits.detach(), k=k, dim=-1)
    _, teacher_ids = torch.topk(teacher_logits.detach(), k=k, dim=-1)
    if gpu_union:
        merged = torch.cat([student_ids, teacher_ids], dim=-1)
        sorted_ids, _ = torch.sort(merged, dim=-1)
        valid = torch.ones_like(sorted_ids, dtype=torch.bool)
        valid[:, 1:] = sorted_ids[:, 1:].ne(sorted_ids[:, :-1])
        valid_rows = valid.sum(dim=-1).ge(2)
        if not bool(valid_rows.any()):
            zero = student_logits.sum() * 0.0
            return zero, {"rows": 0.0, "support_mean": 0.0}
        gather = sorted_ids[valid_rows]
        valid = valid[valid_rows]
        row_index = torch.nonzero(valid_rows, as_tuple=False).flatten()
        student_selected = torch.gather(student_logits[row_index], dim=-1, index=gather)
        teacher_selected = torch.gather(teacher_logits[row_index], dim=-1, index=gather)
        temp = float(temperature)
        student_scores = (student_selected / temp).masked_fill(~valid, -float("inf"))
        teacher_scores = (teacher_selected / temp).masked_fill(~valid, -float("inf"))
        teacher_log_probs = F.log_softmax(teacher_scores.float(), dim=-1)
        student_log_probs = F.log_softmax(student_scores.float(), dim=-1)
        teacher_probs = teacher_log_probs.exp()
        loss = (teacher_probs * (teacher_log_probs - student_log_probs)).masked_fill(~valid, 0.0).sum(dim=-1).mean()
        loss = loss * (temp * temp)
        return loss, {
            "rows": float(row_index.numel()),
            "support_mean": float(valid.sum(dim=-1).float().mean().detach().item()),
        }

    supports = []
    max_support = 0
    for row in range(int(student_logits.size(0))):
        ids = torch.cat([student_ids[row], teacher_ids[row]], dim=0).detach().cpu().tolist()
        unique = []
        seen = set()
        for token_id in ids:
            token_id = int(token_id)
            if token_id not in seen:
                unique.append(token_id)
                seen.add(token_id)
        if len(unique) >= 2:
            supports.append(unique)
            max_support = max(max_support, len(unique))
        else:
            supports.append([])
    valid_rows = [i for i, ids in enumerate(supports) if ids]
    if not valid_rows:
        zero = student_logits.sum() * 0.0
        return zero, {"rows": 0.0, "support_mean": 0.0}
    device = student_logits.device
    gather = torch.zeros((len(valid_rows), max_support), dtype=torch.long, device=device)
    valid = torch.zeros((len(valid_rows), max_support), dtype=torch.bool, device=device)
    for out_i, row_i in enumerate(valid_rows):
        ids = supports[row_i]
        gather[out_i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        valid[out_i, : len(ids)] = True
    row_index = torch.tensor(valid_rows, dtype=torch.long, device=device)
    student_selected = torch.gather(student_logits[row_index], dim=-1, index=gather)
    teacher_selected = torch.gather(teacher_logits[row_index], dim=-1, index=gather)
    temp = float(temperature)
    student_scores = (student_selected / temp).masked_fill(~valid, -float("inf"))
    teacher_scores = (teacher_selected / temp).masked_fill(~valid, -float("inf"))
    teacher_log_probs = F.log_softmax(teacher_scores.float(), dim=-1)
    student_log_probs = F.log_softmax(student_scores.float(), dim=-1)
    teacher_probs = teacher_log_probs.exp()
    loss = (teacher_probs * (teacher_log_probs - student_log_probs)).masked_fill(~valid, 0.0).sum(dim=-1).mean()
    loss = loss * (temp * temp)
    return loss, {
        "rows": float(len(valid_rows)),
        "support_mean": float(valid.sum(dim=-1).float().mean().detach().item()),
    }


def teacher_topk_sparse_kl_from_hidden(
    student_model,
    teacher_model,
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    top_k: int,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if int(student_hidden.numel()) == 0 or int(student_hidden.size(0)) == 0:
        zero = student_hidden.sum() * 0.0
        return zero, {"rows": 0.0, "support_mean": 0.0}
    with torch.no_grad():
        teacher_logits = mask_slow_logits(slow_logits_from_hidden(teacher_model, teacher_hidden), student_model)
        k = min(int(top_k), int(teacher_logits.size(-1)))
        if k <= 1:
            zero = student_hidden.sum() * 0.0
            return zero, {"rows": 0.0, "support_mean": 0.0}
        teacher_selected, support_ids = torch.topk(teacher_logits, k=k, dim=-1)
    student_selected = gather_slow_logits_from_hidden(student_model, student_hidden, support_ids)
    temp = float(temperature)
    teacher_log_probs = F.log_softmax((teacher_selected / temp).float(), dim=-1)
    student_log_probs = F.log_softmax((student_selected / temp).float(), dim=-1)
    teacher_probs = teacher_log_probs.exp()
    loss = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1).mean()
    loss = loss * (temp * temp)
    return loss, {
        "rows": float(student_hidden.size(0)),
        "support_mean": float(k),
    }


def full_vocab_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    if int(student_logits.numel()) == 0 or int(student_logits.size(0)) == 0:
        zero = student_logits.sum() * 0.0
        return zero, {"rows": 0.0, "support_mean": 0.0}
    temp = float(temperature)
    teacher_log_probs = F.log_softmax((teacher_logits / temp).float(), dim=-1)
    student_log_probs = F.log_softmax((student_logits / temp).float(), dim=-1)
    teacher_probs = teacher_log_probs.exp()
    loss = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1).mean()
    loss = loss * (temp * temp)
    return loss, {
        "rows": float(student_logits.size(0)),
        "support_mean": float(student_logits.size(-1)),
    }


def collect_slow_logits(
    token_logits: torch.Tensor,
    generated_positions: list[list[int]],
) -> torch.Tensor:
    rows = []
    for batch_i, positions in enumerate(generated_positions):
        for pos in positions:
            score_pos = int(pos) - 1
            if score_pos >= 0:
                rows.append(token_logits[batch_i, score_pos, :])
    if not rows:
        return token_logits[:, :0, :].reshape(0, int(token_logits.size(-1)))
    return torch.stack(rows, dim=0)


def collect_slow_hidden(
    hidden_states: torch.Tensor,
    generated_positions: list[list[int]],
) -> torch.Tensor:
    rows = []
    for batch_i, positions in enumerate(generated_positions):
        for pos in positions:
            score_pos = int(pos) - 1
            if score_pos >= 0:
                rows.append(hidden_states[batch_i, score_pos, :])
    if not rows:
        return hidden_states[:, :0, :].reshape(0, int(hidden_states.size(-1)))
    return torch.stack(rows, dim=0)


def collect_semantic_hidden(
    hidden_states: torch.Tensor,
    forced_labels: torch.Tensor,
    generated_positions: list[list[int]],
    model,
) -> torch.Tensor:
    rows = []
    begin = int(model.config.semantic_begin_id)
    end = int(model.config.semantic_end_id)
    for batch_i, positions in enumerate(generated_positions):
        for pos in positions:
            token_id = int(forced_labels[batch_i, 0, pos].item())
            score_pos = int(pos) - 1
            if begin <= token_id <= end and score_pos >= 0:
                rows.append(hidden_states[batch_i, score_pos, :])
    if not rows:
        return hidden_states[:, :0, :].reshape(0, int(hidden_states.size(-1)))
    return torch.stack(rows, dim=0)


def collect_generated_codebooks(
    forced_labels: torch.Tensor,
    generated_positions: list[list[int]],
    model,
) -> torch.Tensor:
    rows = []
    begin = int(model.config.semantic_begin_id)
    end = int(model.config.semantic_end_id)
    for batch_i, positions in enumerate(generated_positions):
        for pos in positions:
            token_id = int(forced_labels[batch_i, 0, pos].item())
            if begin <= token_id <= end:
                rows.append(forced_labels[batch_i, 1 : 1 + int(model.config.num_codebooks), pos])
    if not rows:
        return forced_labels[:, 1 : 1 + int(model.config.num_codebooks), :0].permute(0, 2, 1).reshape(
            0,
            int(model.config.num_codebooks),
        )
    return torch.stack(rows, dim=0)


def fast_logits_from_hidden_all_codebooks(
    model,
    slow_hidden: torch.Tensor,
    codebooks: torch.Tensor,
) -> torch.Tensor:
    if int(slow_hidden.size(0)) == 0:
        return slow_hidden.new_zeros((0, int(model.config.num_codebooks), int(model.config.codebook_size)))
    prefix = codebooks[:, : int(model.config.num_codebooks) - 1].long()
    x = model.fast_project_in(slow_hidden)
    x = torch.cat([x[:, None, :], model.fast_embeddings(prefix)], dim=1)
    seq_len = int(x.size(1))
    fast_mask = model.causal_mask[None, None, :seq_len, :seq_len]
    fast_freqs_cis = model.fast_freqs_cis[:seq_len]
    for layer in model.fast_layers:
        x = layer(x, fast_freqs_cis, fast_mask)
    out = model.fast_norm(x)
    return model.fast_output(out)


def flatten_fast_logits(
    codebook_logits: torch.Tensor,
    include_codebook0: bool,
) -> torch.Tensor:
    start = 0 if bool(include_codebook0) else 1
    if int(codebook_logits.size(0)) == 0 or start >= int(codebook_logits.size(1)):
        return codebook_logits.reshape(0, int(codebook_logits.size(-1)))
    return codebook_logits[:, start:, :].reshape(-1, int(codebook_logits.size(-1)))


def collect_fast_logits(
    codebook_logits: torch.Tensor,
    labels: torch.Tensor,
    model,
    include_codebook0: bool,
) -> torch.Tensor:
    token_values = labels[:, 0]
    semantic_mask = (
        token_values.ne(-100)
        & token_values.ge(int(model.config.semantic_begin_id))
        & token_values.le(int(model.config.semantic_end_id))
    )
    semantic_count = int(semantic_mask.sum().item())
    if semantic_count <= 0:
        return codebook_logits.reshape(0, int(codebook_logits.size(-1)))
    if int(codebook_logits.numel()) == 0:
        return codebook_logits.reshape(0, int(codebook_logits.size(-1)))
    codebook_logits = codebook_logits[:semantic_count]
    start = 0 if bool(include_codebook0) else 1
    if start >= int(codebook_logits.size(1)):
        return codebook_logits.reshape(0, int(codebook_logits.size(-1)))
    return codebook_logits[:, start:, :].reshape(-1, int(codebook_logits.size(-1)))


class FishS2ProOpdTrainer(base.FishS2ProTrainer):
    def __init__(self, *args, teacher_model=None, pad_id: int, **kwargs):
        super().__init__(*args, **kwargs)
        if teacher_model is None:
            raise ValueError("teacher_model is required")
        self.teacher_model = teacher_model
        self.pad_id = int(pad_id)
        self._step_times = []
        self._last_step_time = None
        world_size = max(int(os.environ.get("WORLD_SIZE", "1")), 1)
        self._global_batch_size = (
            max(int(self.args.per_device_train_batch_size), 1)
            * max(int(self.args.gradient_accumulation_steps), 1)
            * world_size
        )

    def training_step(self, model, inputs, num_items_in_batch=None):
        debug_steps = int(getattr(self.args, "opd_debug_steps", 0) or 0)
        step = int(getattr(self.state, "global_step", 0) or 0)
        debug_enabled = step < debug_steps
        start = time.perf_counter()
        original_backward = self.accelerator.backward

        def timed_backward(loss, *args, **kwargs):
            if debug_enabled:
                debug_sync()
                backward_start = time.perf_counter()
                all_rank_debug_print(f"step={step} event=backward_start")
            else:
                backward_start = 0.0
            result = original_backward(loss, *args, **kwargs)
            if debug_enabled:
                debug_sync()
                all_rank_debug_print(
                    f"step={step} event=backward_end backward_s={time.perf_counter() - backward_start:.4f}"
                )
            return result

        if debug_enabled:
            self.accelerator.backward = timed_backward
            debug_sync()
            all_rank_debug_print(f"step={step} event=training_step_start")
        try:
            loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        finally:
            if debug_enabled:
                self.accelerator.backward = original_backward
        if debug_enabled:
            debug_sync()
        elapsed = time.perf_counter() - start
        if debug_enabled:
            all_rank_debug_print(f"step={step} event=training_step_end total_s={elapsed:.4f}")
        self._last_step_time = float(elapsed)
        self._step_times.append(float(elapsed))
        if len(self._step_times) > 20:
            self._step_times = self._step_times[-20:]
        return loss

    def log(self, logs, start_time=None):
        if self._last_step_time is not None and self._step_times:
            mean_step_time = sum(self._step_times) / len(self._step_times)
            logs = dict(logs)
            logs["step_time_s"] = round(float(self._last_step_time), 4)
            logs["step_time_s_avg"] = round(float(mean_step_time), 4)
            logs["global_samples_per_s"] = round(float(self._global_batch_size / mean_step_time), 4)
            logs["global_batch_size"] = float(self._global_batch_size)
        return super().log(logs, start_time=start_time)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        debug_steps = int(getattr(self.args, "opd_debug_steps", 0) or 0)
        step = int(getattr(self.state, "global_step", 0) or 0)
        debug_enabled = step < debug_steps
        if debug_enabled:
            debug_sync()
        debug_start = time.perf_counter()
        debug_last = debug_start
        debug_times = {}

        def mark_debug(name: str) -> None:
            nonlocal debug_last
            if not debug_enabled:
                return
            debug_sync()
            now = time.perf_counter()
            debug_times[name] = now - debug_last
            debug_last = now

        student = actual_model(model)
        labels = inputs["labels"]
        prompt_lens, rollout_lens = label_rollout_bounds(
            labels,
            max_new_tokens=int(self.args.opd_max_new_tokens),
            model=student,
        )
        mark_debug("bounds_s")

        sft_loss = labels.new_tensor(0.0, dtype=torch.float32)
        sft_weight = float(getattr(self.args, "sft_loss_weight", 0.0))
        if sft_weight != 0.0:
            sft_loss = super().compute_loss(model, inputs, return_outputs=False)
        mark_debug("sft_s")

        fast_weight = float(self.args.opd_fast_loss_weight)
        rollout_fn = greedy_rollout_student_cached if bool(self.args.opd_cached_rollout) else greedy_rollout_student
        rollout_kwargs = {}
        if bool(self.args.opd_cached_rollout):
            rollout_kwargs = {
                "rollout_group_size": int(self.args.opd_rollout_group_size),
                "rollout_length_bucket": int(self.args.opd_rollout_length_bucket),
            }
        generated = rollout_fn(
            model=student,
            inputs=inputs["inputs"],
            labels=labels,
            prompt_lens=prompt_lens,
            rollout_lens=rollout_lens,
            max_new_tokens=int(self.args.opd_max_new_tokens),
            pad_id=self.pad_id,
            generate_fast_codebooks=fast_weight != 0.0,
            **rollout_kwargs,
        )
        mark_debug("rollout_s")
        forced_inputs, forced_mask, forced_labels, generated_positions = build_forced_batch(
            prompt_inputs=inputs["inputs"],
            prompt_lens=prompt_lens,
            generated=generated,
            pad_id=self.pad_id,
        )
        mark_debug("forced_batch_s")
        old_student_checkpointing = bool(getattr(student.config, "use_gradient_checkpointing", False))
        if bool(getattr(self.args, "opd_disable_student_checkpointing", True)):
            student.config.use_gradient_checkpointing = False
        try:
            student_slow_outputs = slow_hidden_forward(
                student,
                inp=forced_inputs,
                key_padding_mask=forced_mask,
            )
        finally:
            student.config.use_gradient_checkpointing = old_student_checkpointing
        mark_debug("student_forward_s")
        with torch.no_grad():
            teacher_slow_outputs = slow_hidden_forward(
                self.teacher_model,
                inp=forced_inputs,
                key_padding_mask=forced_mask,
            )
        mark_debug("teacher_forward_s")

        student_slow_hidden = collect_slow_hidden(student_slow_outputs.hidden_states, generated_positions)
        teacher_slow_hidden = collect_slow_hidden(teacher_slow_outputs.hidden_states, generated_positions)
        slow_loss, slow_metrics = teacher_topk_sparse_kl_from_hidden(
            student_model=student,
            teacher_model=self.teacher_model,
            student_hidden=student_slow_hidden,
            teacher_hidden=teacher_slow_hidden,
            top_k=int(self.args.opd_top_k),
            temperature=float(self.args.opd_temperature),
        )
        mark_debug("slow_kl_s")

        if fast_weight != 0.0:
            student_codebooks = collect_generated_codebooks(forced_labels, generated_positions, student)
            student_hidden = collect_semantic_hidden(
                student_slow_outputs.hidden_states,
                forced_labels,
                generated_positions,
                student,
            )
            teacher_hidden = collect_semantic_hidden(
                teacher_slow_outputs.hidden_states,
                forced_labels,
                generated_positions,
                student,
            )
            student_fast = flatten_fast_logits(
                fast_logits_from_hidden_all_codebooks(student, student_hidden, student_codebooks),
                include_codebook0=bool(self.args.opd_include_codebook0),
            )
            teacher_fast = flatten_fast_logits(
                fast_logits_from_hidden_all_codebooks(self.teacher_model, teacher_hidden, student_codebooks),
                include_codebook0=bool(self.args.opd_include_codebook0),
            )
            if bool(self.args.opd_fast_full_vocab_kl):
                fast_loss, fast_metrics = full_vocab_kl(
                    student_logits=student_fast,
                    teacher_logits=teacher_fast,
                    temperature=float(self.args.opd_temperature),
                )
            else:
                fast_loss, fast_metrics = topk_union_kl(
                    student_logits=student_fast,
                    teacher_logits=teacher_fast,
                    top_k=int(self.args.opd_top_k),
                    temperature=float(self.args.opd_temperature),
                    gpu_union=bool(self.args.opd_slow_gpu_union_kl),
                )
        else:
            fast_loss = slow_loss.new_tensor(0.0)
            fast_metrics = {"rows": 0.0, "support_mean": 0.0}
        mark_debug("fast_loss_s")

        slow_weight = float(self.args.opd_slow_loss_weight)
        total_loss = sft_weight * sft_loss + slow_weight * slow_loss + fast_weight * fast_loss
        gen_lens = [int(x.size(1)) for x in generated]
        im_end_id = int(student.tokenizer.get_token_id(IM_END_TOKEN))
        eos_count = sum(
            1
            for x in generated
            if int(x.size(1)) > 0 and bool(x[0].eq(im_end_id).any().item())
        )
        capped_count = sum(
            1
            for x, limit in zip(generated, rollout_lens)
            if int(limit) > 0
            and int(x.size(1)) >= int(limit)
            and not (int(x.size(1)) > 0 and bool(x[0].eq(im_end_id).any().item()))
        )
        metrics = {
            "opd_loss": total_loss.detach(),
            "opd_slow_loss": slow_loss.detach(),
            "opd_fast_loss": fast_loss.detach(),
            "opd_sft_loss": sft_loss.detach() if torch.is_tensor(sft_loss) else labels.new_tensor(float(sft_loss)),
            "opd_generated_nonempty": labels.new_tensor(
                sum(1 for x in gen_lens if x > 0) / max(len(gen_lens), 1), dtype=torch.float32
            ),
            "opd_generated_tokens_mean": labels.new_tensor(
                sum(gen_lens) / max(sum(1 for x in gen_lens if x > 0), 1), dtype=torch.float32
            ),
            "opd_rollout_target_tokens_mean": labels.new_tensor(
                sum(rollout_lens) / max(sum(1 for x in rollout_lens if x > 0), 1), dtype=torch.float32
            ),
            "opd_rollout_eos_rate": labels.new_tensor(eos_count / max(len(gen_lens), 1), dtype=torch.float32),
            "opd_rollout_cap_rate": labels.new_tensor(capped_count / max(len(gen_lens), 1), dtype=torch.float32),
            "opd_slow_rows": labels.new_tensor(slow_metrics["rows"], dtype=torch.float32),
            "opd_slow_support_mean": labels.new_tensor(slow_metrics["support_mean"], dtype=torch.float32),
            "opd_fast_rows": labels.new_tensor(fast_metrics["rows"], dtype=torch.float32),
            "opd_fast_support_mean": labels.new_tensor(fast_metrics["support_mean"], dtype=torch.float32),
            "opd_fast_full_vocab_kl": labels.new_tensor(float(bool(self.args.opd_fast_full_vocab_kl)), dtype=torch.float32),
            "opd_slow_gpu_union_kl": labels.new_tensor(float(bool(self.args.opd_slow_gpu_union_kl)), dtype=torch.float32),
            "opd_sparse_slow_kl": labels.new_tensor(1.0, dtype=torch.float32),
            "opd_student_checkpointing_disabled": labels.new_tensor(
                float(bool(getattr(self.args, "opd_disable_student_checkpointing", True))),
                dtype=torch.float32,
            ),
            "opd_cached_rollout": labels.new_tensor(float(bool(self.args.opd_cached_rollout)), dtype=torch.float32),
            "opd_rollout_group_size": labels.new_tensor(float(int(self.args.opd_rollout_group_size)), dtype=torch.float32),
            "opd_rollout_length_bucket": labels.new_tensor(float(int(self.args.opd_rollout_length_bucket)), dtype=torch.float32),
        }
        if debug_enabled:
            debug_sync()
            debug_times["compute_loss_s"] = time.perf_counter() - debug_start
            debug_msg = " ".join(f"{key}={value:.4f}" for key, value in debug_times.items())
            all_rank_debug_print(
                f"step={step} event=compute_loss_end "
                f"fast_weight={fast_weight} sparse_slow_kl=1 "
                f"generated_mean={sum(gen_lens) / max(sum(1 for x in gen_lens if x > 0), 1):.2f} "
                f"forced_seq_len={int(forced_inputs.size(-1))} slow_rows={slow_metrics['rows']:.0f} "
                f"{debug_msg}"
            )
        self._record_loss_metrics(metrics)
        if return_outputs:
            student_token_logits = slow_logits_from_hidden(student, student_slow_outputs.hidden_states)
            outputs = {
                "token_logits": student_token_logits,
                **{k: v.detach() for k, v in metrics.items()},
            }
            return total_loss, outputs
        return total_loss


def make_dataset(data_args, tokenizer, is_train: bool):
    split = "train" if is_train else "eval"
    jsonl = data_args.train_jsonl if is_train else (data_args.eval_jsonl or data_args.train_jsonl)
    max_samples = data_args.max_train_samples if is_train else data_args.max_eval_samples
    all_rank_debug_print(
        f"event=dataset_init_start split={split} jsonl={jsonl} "
        f"stream_train={bool(data_args.stream_train_jsonl)} max_samples={max_samples}"
    )
    start = time.perf_counter()
    if is_train and data_args.code_shard_dir:
        dataset = base.CodeShardFishAudioDataset(
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
        shard_by_rank = data_args.shard_train_by_rank if is_train else data_args.shard_eval_by_rank
        skip_samples = data_args.skip_train_samples if is_train else 0
        cls = base.StreamingJsonlFishAudioDataset if (is_train and data_args.stream_train_jsonl) else base.JsonlFishAudioDataset
        dataset = cls(
            jsonl_file=jsonl,
            tokenizer=tokenizer,
            max_samples=max_samples,
            num_codebooks=data_args.num_codebooks,
            use_ref=data_args.use_ref,
            text_key=data_args.text_key,
            audio_ids_key=data_args.audio_ids_key,
            ref_audio_ids_key=data_args.ref_audio_ids_key,
            ref_text_key=data_args.ref_text_key,
            shard_by_rank=shard_by_rank,
            local_npy_cache_dir=data_args.local_npy_cache_dir,
            local_npy_cache_source_prefix=data_args.local_npy_cache_source_prefix,
            local_npy_cache_log_every=data_args.local_npy_cache_log_every,
            local_npy_cache_read_only=data_args.local_npy_cache_read_only,
            local_npy_cache_rank_subdir=data_args.local_npy_cache_rank_subdir,
            skip_samples=skip_samples,
        )
    all_rank_debug_print(
        f"event=dataset_init_end split={split} elapsed_s={time.perf_counter() - start:.4f} "
        f"type={dataset.__class__.__name__}"
    )
    return dataset


def validate_compat(student, teacher):
    checks = [
        ("vocab_size", int(student.config.vocab_size), int(teacher.config.vocab_size)),
        ("num_codebooks", int(student.config.num_codebooks), int(teacher.config.num_codebooks)),
        ("codebook_size", int(student.config.codebook_size), int(teacher.config.codebook_size)),
        ("semantic_begin_id", int(student.config.semantic_begin_id), int(teacher.config.semantic_begin_id)),
        ("semantic_end_id", int(student.config.semantic_end_id), int(teacher.config.semantic_end_id)),
    ]
    bad = [item for item in checks if item[1] != item[2]]
    if bad:
        raise ValueError(f"Student/teacher incompatible: {bad}")


def main():
    base.configure_tmpdir()
    parser = HfArgumentParser((OpdModelArguments, base.DataArguments, OpdTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    training_args.remove_unused_columns = False
    device = resolve_device()

    rank0_print("[load] tokenizer")
    tokenizer = FishTokenizer(model_args.pretrained_ckpt_path)
    rank0_print("[load] train dataset")
    train_dataset = make_dataset(data_args, tokenizer, is_train=True)
    eval_dataset = None
    if training_args.do_eval:
        rank0_print("[load] eval dataset")
        eval_dataset = make_dataset(data_args, tokenizer, is_train=False)
    else:
        rank0_print("[load] eval dataset skipped do_eval=false")

    rank0_print("[load] student")
    student = BaseTransformer.from_pretrained(
        model_args.pretrained_ckpt_path,
        load_weights=True,
        max_length=model_args.max_length,
    )
    base.ensure_hf_config_compat(student)
    if model_args.freeze_fast_ar:
        base.freeze_fast_ar(student)
        freeze_fast_ar_strict(student)

    rank0_print("[load] teacher")
    rank = int(os.environ.get("RANK", "0"))
    if float(model_args.teacher_load_stagger_seconds) > 0:
        time.sleep(float(model_args.teacher_load_stagger_seconds) * float(rank))
    rank0_print(
        "[teacher] placement=replicated_per_rank "
        f"path={model_args.teacher_ckpt_path} dtype={model_args.teacher_dtype} "
        "deepspeed_managed=false"
    )
    teacher_dtype = dtype_from_name(model_args.teacher_dtype)
    teacher = BaseTransformer.from_pretrained(
        model_args.teacher_ckpt_path,
        load_weights=True,
        max_length=model_args.max_length,
    )
    teacher.to(device=device, dtype=teacher_dtype)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    validate_compat(student, teacher)

    collator = base.FishAudioCollator(tokenizer=tokenizer, max_length=model_args.max_length)
    pad_id = tokenizer.get_token_id("<|end_of_text|>")
    if pad_id is None:
        pad_id = tokenizer.get_token_id(IM_END_TOKEN)
    if pad_id is None:
        pad_id = 0
    trainer = FishS2ProOpdTrainer(
        model=student,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        teacher_model=teacher,
        pad_id=pad_id,
    )

    if training_args.do_train:
        resume = base.resolve_resume_checkpoint(training_args.output_dir, training_args.resume_mode)
        rank0_print(f"[resume] {resume if resume else 'disabled'} mode={training_args.resume_mode}")
        train_result = trainer.train(resume_from_checkpoint=resume)
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    if training_args.skip_final_save:
        rank0_print("[save] skip_final_save=true, final save/export skipped")
    else:
        trainer.save_model(training_args.output_dir)
    if training_args.export_dir and not training_args.skip_final_save:
        export_model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        cfg = getattr(export_model, "config", None)
        if cfg is not None:
            for name in ("to_dict", "to_json_string"):
                if name in getattr(cfg, "__dict__", {}):
                    delattr(cfg, name)
        base.export_fish_pretrained(trainer, training_args.export_dir)
        base.copy_aux_files(model_args.pretrained_ckpt_path, training_args.export_dir)


if __name__ == "__main__":
    main()

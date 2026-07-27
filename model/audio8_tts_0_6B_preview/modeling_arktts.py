"""Hugging Face remote-code implementation of the audio8_tts DualAR model."""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint
from transformers import GenerationConfig, PreTrainedModel
from transformers.generation import (
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteriaList,
)
from transformers.modeling_outputs import ModelOutput
from transformers.utils.hub import cached_file

from .configuration_arktts import ArkttsConfig


@dataclass
class ArkttsModelOutput(ModelOutput):
    """Slow-AR logits and hidden states returned during teacher forcing."""

    logits: Optional[Tensor] = None
    hidden_states: Optional[Tensor] = None
    codebook_logits: Optional[Tensor] = None


@dataclass
class ArkttsGenerateOutput(ModelOutput):
    """Generated codec frames, per-item lengths, and EOS completion flags."""

    codes: Optional[Tensor] = None
    code_lengths: Optional[Tensor] = None
    finished: Optional[Tensor] = None


class ArkttsSemanticLogitsProcessor(LogitsProcessor):
    """Restrict slow-AR decoding to semantic tokens plus the EOS token."""

    def __init__(self, semantic_begin_id: int, semantic_end_id: int, eos_token_id: int):
        self.semantic_begin_id = int(semantic_begin_id)
        self.semantic_end_id = int(semantic_end_id)
        self.eos_token_id = int(eos_token_id)

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        filtered = torch.full_like(scores, float("-inf"))
        filtered[:, self.semantic_begin_id : self.semantic_end_id + 1] = scores[
            :, self.semantic_begin_id : self.semantic_end_id + 1
        ]
        filtered[:, self.eos_token_id] = scores[:, self.eos_token_id]
        return filtered


class ArkttsLegacyTopKTopPLogitsProcessor(LogitsProcessor):
    """Matches the candidate filtering order used by the original inference code."""

    def __init__(self, top_k: int, top_p: float):
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        self.top_k = int(top_k)
        self.top_p = float(top_p)

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        cumulative = torch.cumsum(torch.softmax(sorted_scores, dim=-1), dim=-1)
        positions = torch.arange(sorted_scores.shape[-1], device=scores.device)
        threshold = torch.tensor(self.top_p, dtype=cumulative.dtype, device=cumulative.device)
        remove_sorted = (cumulative > threshold) | (positions >= self.top_k)
        remove_sorted[..., 0] = False
        remove = torch.zeros_like(remove_sorted).scatter(1, sorted_indices, remove_sorted)
        return scores.masked_fill(remove, float("-inf"))


class ArkttsKVCache(nn.Module):
    """Preallocated attention cache shared by slow and fast AR decoding.

    Slow AR appends a prompt followed by one frame at a time and returns only
    the valid prefix. Fast AR owns a fixed ``num_codebooks`` cache and returns
    its full storage because each frame always traverses every codebook slot.
    """

    def __init__(
        self,
        batch_size: int,
        max_length: int,
        heads: int,
        head_dim: int,
        dtype,
        return_full: bool = False,
    ):
        super().__init__()
        shape = (batch_size, heads, max_length, head_dim)
        self.register_buffer("keys", torch.zeros(shape, dtype=dtype), persistent=False)
        self.register_buffer("values", torch.zeros(shape, dtype=dtype), persistent=False)
        self.return_full = bool(return_full)
        self.valid_length = 0

    def update(self, cache_position: Tensor, keys: Tensor, values: Tensor):
        """Write K/V tensors at physical cache positions and return visible storage."""
        self.keys[:, :, cache_position] = keys
        self.values[:, :, cache_position] = values
        if self.return_full:
            end = self.keys.shape[2]
        else:
            # Slow generation writes a contiguous prompt followed by one token
            # at a time. Track that prefix in Python to avoid a GPU sync in
            # every layer from cache_position[-1].item().
            self.valid_length = min(
                self.keys.shape[2], self.valid_length + int(keys.shape[-2])
            )
            end = self.valid_length
        return self.keys[:, :, :end], self.values[:, :, :end]


class ArkttsRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


def _precompute_rope(length: int, head_dim: int, base: float) -> Tensor:
    frequencies = 1.0 / (
        base ** (torch.arange(0, head_dim, 2).float()[: head_dim // 2] / head_dim)
    )
    phases = torch.outer(torch.arange(length), frequencies)
    values = torch.polar(torch.ones_like(phases), phases)
    return torch.stack((values.real, values.imag), dim=-1).to(torch.bfloat16)


def _apply_rope(x: Tensor, rope: Tensor) -> Tensor:
    shaped = x.float().reshape(*x.shape[:-1], -1, 2)
    if rope.ndim == 3:
        rope = rope[None, :, None]
    elif rope.ndim == 4:
        rope = rope[:, :, None]
    else:
        raise ValueError(f"Unexpected RoPE shape: {tuple(rope.shape)}")
    output = torch.stack(
        (
            shaped[..., 0] * rope[..., 0] - shaped[..., 1] * rope[..., 1],
            shaped[..., 1] * rope[..., 0] + shaped[..., 0] * rope[..., 1],
        ),
        dim=-1,
    )
    return output.flatten(3).to(x.dtype)


class ArkttsAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_head: int,
        n_local_heads: int,
        head_dim: int,
        qkv_bias: bool,
        output_bias: bool,
        qk_norm: bool,
        norm_eps: float,
        dropout: float,
        use_sdpa: bool,
    ):
        super().__init__()
        total = (n_head + 2 * n_local_heads) * head_dim
        self.wqkv = nn.Linear(dim, total, bias=qkv_bias)
        self.wo = nn.Linear(n_head * head_dim, dim, bias=output_bias)
        self.n_head = int(n_head)
        self.n_local_heads = int(n_local_heads)
        self.head_dim = int(head_dim)
        self.dropout = float(dropout)
        self.use_sdpa = bool(use_sdpa)
        self.qk_norm = bool(qk_norm)
        if self.qk_norm:
            self.q_norm = ArkttsRMSNorm(head_dim, norm_eps)
            self.k_norm = ArkttsRMSNorm(head_dim, norm_eps)
        self.kv_cache: Optional[ArkttsKVCache] = None

    def forward(
        self,
        x: Tensor,
        rope: Tensor,
        attention_mask: Optional[Tensor],
        cache_position: Optional[Tensor] = None,
    ) -> Tensor:
        batch, length, _ = x.shape
        query_size = self.n_head * self.head_dim
        kv_size = self.n_local_heads * self.head_dim
        query, key, value = self.wqkv(x).split((query_size, kv_size, kv_size), dim=-1)
        query = query.view(batch, length, self.n_head, self.head_dim)
        key = key.view(batch, length, self.n_local_heads, self.head_dim)
        value = value.view(batch, length, self.n_local_heads, self.head_dim)
        if self.qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)
        query = _apply_rope(query, rope).transpose(1, 2)
        key = _apply_rope(key, rope).transpose(1, 2)
        value = value.transpose(1, 2)
        if self.kv_cache is not None:
            if cache_position is None:
                raise ValueError("cache_position is required when KV cache is enabled")
            key, value = self.kv_cache.update(cache_position, key, value)
        repeats = self.n_head // self.n_local_heads
        key = key.repeat_interleave(repeats, dim=1)
        value = value.repeat_interleave(repeats, dim=1)
        if self.use_sdpa:
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
            )
        else:
            scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                scores = scores.masked_fill(~attention_mask, float("-inf"))
            probabilities = torch.softmax(scores, dim=-1)
            if self.training and self.dropout:
                probabilities = F.dropout(probabilities, p=self.dropout)
            output = probabilities @ value
        output = output.transpose(1, 2).contiguous().view(batch, length, query_size)
        return self.wo(output)


class ArkttsFeedForward(nn.Module):
    def __init__(self, dim: int, intermediate_size: int):
        super().__init__()
        self.w1 = nn.Linear(dim, intermediate_size, bias=False)
        self.w2 = nn.Linear(intermediate_size, dim, bias=False)
        self.w3 = nn.Linear(dim, intermediate_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ArkttsTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_size: int,
        n_head: int,
        n_local_heads: int,
        head_dim: int,
        qkv_bias: bool,
        output_bias: bool,
        qk_norm: bool,
        norm_eps: float,
        dropout: float,
        use_sdpa: bool,
    ):
        super().__init__()
        self.attention = ArkttsAttention(
            dim, n_head, n_local_heads, head_dim, qkv_bias, output_bias,
            qk_norm, norm_eps, dropout, use_sdpa,
        )
        self.feed_forward = ArkttsFeedForward(dim, intermediate_size)
        self.ffn_norm = ArkttsRMSNorm(dim, norm_eps)
        self.attention_norm = ArkttsRMSNorm(dim, norm_eps)

    def forward(self, x, rope, attention_mask, cache_position=None):
        hidden = x + self.attention(self.attention_norm(x), rope, attention_mask, cache_position)
        return hidden + self.feed_forward(self.ffn_norm(hidden))


class ArkttsModel(PreTrainedModel):
    """Dual autoregressive text-to-speech model used by audio8_tts Preview.

    The slow transformer predicts one semantic codec token per audio frame.
    Conditioned on its hidden state, the fast transformer predicts the ten
    codec codebooks within that frame. Generation allocates static KV caches
    for both branches, which avoids rebuilding complete prefixes per step.
    """

    config_class = ArkttsConfig
    base_model_prefix = ""
    main_input_name = "input_ids"
    _no_split_modules = ["ArkttsTransformerBlock"]
    _supports_sdpa = True
    supports_gradient_checkpointing = True

    def __init__(self, config: ArkttsConfig):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.codebook_embeddings = nn.Embedding(config.codebook_size * config.num_codebooks, config.dim)
        self.layers = nn.ModuleList([
            ArkttsTransformerBlock(
                config.dim, config.intermediate_size, config.n_head, config.n_local_heads,
                config.head_dim, config.attention_qkv_bias, config.attention_o_bias,
                config.attention_qk_norm, config.norm_eps, config.dropout, True,
            )
            for _ in range(config.n_layer)
        ])
        self.norm = ArkttsRMSNorm(config.dim, config.norm_eps)
        self.fast_project_in = (
            nn.Linear(config.dim, config.fast_dim)
            if config.fast_dim != config.dim else nn.Identity()
        )
        self.fast_embeddings = nn.Embedding(config.codebook_size, config.fast_dim)
        self.fast_layers = nn.ModuleList([
            ArkttsTransformerBlock(
                config.fast_dim, config.fast_intermediate_size, config.fast_n_head,
                config.fast_n_local_heads, config.fast_head_dim,
                config.fast_attention_qkv_bias, config.fast_attention_o_bias,
                config.fast_attention_qk_norm, config.norm_eps, config.dropout, False,
            )
            for _ in range(config.n_fast_layer)
        ])
        self.fast_norm = ArkttsRMSNorm(config.fast_dim, config.norm_eps)
        self.fast_output = nn.Linear(config.fast_dim, config.codebook_size, bias=False)
        self.register_buffer(
            "freqs_cis", _precompute_rope(config.max_seq_len, config.head_dim, config.rope_base),
            persistent=False,
        )
        self.register_buffer(
            "fast_freqs_cis",
            _precompute_rope(config.num_codebooks, config.fast_head_dim, config.rope_base),
            persistent=False,
        )
        self.__dict__["_arktts_codec"] = None
        # Transformers toggles this flag through gradient_checkpointing_enable().
        # Keeping it on the model makes training behavior explicit while leaving
        # inference and checkpoint weights unchanged.
        self.gradient_checkpointing = bool(config.use_gradient_checkpointing)
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def get_output_embeddings(self):
        return None

    def _embed(self, input_ids: Tensor) -> Tensor:
        codebook_embeds = []
        for index in range(self.config.num_codebooks):
            codebook_embeds.append(
                self.codebook_embeddings(input_ids[:, index + 1] + index * self.config.codebook_size)
            )
        codebook_sum = torch.stack(codebook_embeds, dim=1).sum(dim=1)
        semantic = (input_ids[:, 0] >= self.config.semantic_begin_id) & (
            input_ids[:, 0] <= self.config.semantic_end_id
        )
        codebook_sum = torch.where(semantic.unsqueeze(-1), codebook_sum, 0.0)
        return self.embeddings(input_ids[:, 0]) + codebook_sum

    @staticmethod
    def _causal_mask(attention_mask: Tensor, query_positions: Tensor, key_length: int) -> Tensor:
        if attention_mask.shape[1] < key_length:
            attention_mask = F.pad(
                attention_mask,
                (0, key_length - attention_mask.shape[1]),
                value=0,
            )
        key_positions = torch.arange(key_length, device=attention_mask.device)
        causal = key_positions[None, :] <= query_positions[:, None]
        return causal[None, None] & attention_mask[:, None, None, :key_length].bool()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        """Run the slow AR backbone on packed text and codec input rows.

        ``input_ids`` has shape ``[batch, num_codebooks + 1, sequence]``.
        Row zero stores text/semantic IDs; remaining rows store codec indices
        at semantic positions and zero elsewhere. SFT computes the fast-AR
        teacher-forcing loss from the returned hidden states.
        """
        del labels, output_hidden_states, kwargs
        if input_ids.ndim != 3 or input_ids.shape[1] != self.config.num_codebooks + 1:
            raise ValueError(
                f"input_ids must have shape [B, {self.config.num_codebooks + 1}, T]"
            )
        batch, _, length = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones((batch, length), dtype=torch.long, device=input_ids.device)
        position_ids = attention_mask.long().cumsum(-1).sub(1).clamp_min(0)
        rope = self.freqs_cis[position_ids]
        mask = self._causal_mask(
            attention_mask,
            torch.arange(length, device=input_ids.device),
            length,
        )
        hidden = self._embed(input_ids)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden = checkpoint(layer, hidden, rope, mask, use_reentrant=False)
            else:
                hidden = layer(hidden, rope, mask)
        normalized = self.norm(hidden)
        logits = F.linear(normalized, self.embeddings.weight)
        output = ArkttsModelOutput(logits=logits, hidden_states=hidden)
        if return_dict is False:
            return (logits, hidden)
        return output

    def _setup_generation_caches(self, batch_size: int, max_length: int, dtype):
        """Allocate per-layer static caches for one complete generation call."""
        slow_cache_length = min(max(1, int(max_length)), self.config.max_seq_len)
        for layer in self.layers:
            layer.attention.kv_cache = ArkttsKVCache(
                batch_size,
                slow_cache_length,
                self.config.n_local_heads,
                self.config.head_dim,
                dtype,
                return_full=False,
            ).to(self.device)
        for layer in self.fast_layers:
            layer.attention.kv_cache = ArkttsKVCache(
                batch_size, self.config.num_codebooks, self.config.fast_n_local_heads,
                self.config.fast_head_dim, dtype, return_full=True,
            ).to(self.device)

    def _slow_backbone_step(
        self,
        input_ids: Tensor,
        cache_position: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        hidden = self._embed(input_ids)
        rope = self.freqs_cis[position_ids]
        cache = self.layers[0].attention.kv_cache
        if cache is None:
            raise RuntimeError("Slow AR KV cache must be initialized before _slow_step")
        key_length = min(
            int(cache.keys.shape[2]),
            int(cache.valid_length) + int(input_ids.shape[-1]),
        )
        mask = self._causal_mask(
            attention_mask,
            cache_position,
            key_length,
        )
        for layer in self.layers:
            hidden = layer(hidden, rope, mask, cache_position)
        hidden = hidden[:, -1:]
        normalized = self.norm(hidden)
        fast_hidden = normalized if self.config.norm_fastlayer_input else hidden
        return normalized, fast_hidden

    def _slow_step(
        self,
        input_ids: Tensor,
        cache_position: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        normalized, fast_hidden = self._slow_backbone_step(
            input_ids, cache_position, position_ids, attention_mask
        )
        logits = F.linear(normalized, self.embeddings.weight)[:, -1]
        return logits, fast_hidden

    def _slow_hidden_step(
        self,
        input_ids: Tensor,
        cache_position: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        _normalized, fast_hidden = self._slow_backbone_step(
            input_ids, cache_position, position_ids, attention_mask
        )
        return fast_hidden

    def _slow_semantic_step(
        self,
        input_ids: Tensor,
        cache_position: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        semantic_begin_id: int,
        semantic_end_id: int,
        eos_token_id: int,
    ) -> tuple[Tensor, Tensor]:
        normalized, fast_hidden = self._slow_backbone_step(
            input_ids, cache_position, position_ids, attention_mask
        )
        begin = int(semantic_begin_id)
        end = int(semantic_end_id)
        semantic_logits = F.linear(
            normalized, self.embeddings.weight[begin : end + 1]
        )
        eos_logits = F.linear(
            normalized, self.embeddings.weight[int(eos_token_id) : int(eos_token_id) + 1]
        )
        compact_logits = torch.cat((semantic_logits, eos_logits), dim=-1)[:, -1]
        return compact_logits, fast_hidden

    def _fast_step(self, hidden: Tensor, position: int) -> Tensor:
        cache_position = torch.tensor([position], device=hidden.device, dtype=torch.long)
        rope = self.fast_freqs_cis[cache_position]
        key_mask = torch.ones(
            (hidden.shape[0], self.config.num_codebooks),
            device=hidden.device,
            dtype=torch.bool,
        )
        mask = self._causal_mask(key_mask, cache_position, self.config.num_codebooks)
        for layer in self.fast_layers:
            hidden = layer(hidden, rope, mask, cache_position)
        return self.fast_output(self.fast_norm(hidden))[:, -1]

    @staticmethod
    def _as_processor_list(value) -> LogitsProcessorList:
        if value is None:
            return LogitsProcessorList()
        if isinstance(value, LogitsProcessorList):
            return value
        return LogitsProcessorList(value)

    @staticmethod
    def _sample(scores: Tensor, generator=None) -> Tensor:
        probabilities = torch.softmax(scores, dim=-1)
        random = torch.rand(
            probabilities.shape,
            dtype=probabilities.dtype,
            device=probabilities.device,
            generator=generator,
        )
        noise = -torch.log(random)
        return torch.argmax(probabilities / noise, dim=-1)

    def _processed_scores(
        self,
        input_ids: Tensor,
        scores: Tensor,
        processors: LogitsProcessorList,
        top_k: int,
        top_p: float,
        temperature: float,
    ) -> Tensor:
        scores = processors(input_ids, scores)
        scores = ArkttsLegacyTopKTopPLogitsProcessor(top_k, top_p)(input_ids, scores)
        temperature_value = torch.tensor(
            temperature, dtype=scores.dtype, device=scores.device
        ).clamp_min(1e-5)
        return scores / temperature_value

    def _sample_semantic(
        self,
        history: Tensor,
        logits: Tensor,
        custom_processors: LogitsProcessorList,
        top_k: int,
        top_p: float,
        temperature: float,
        previous: Optional[Tensor],
        do_sample: bool,
        generator=None,
    ) -> Tensor:
        processors = LogitsProcessorList([
            ArkttsSemanticLogitsProcessor(
                self.config.semantic_begin_id,
                self.config.semantic_end_id,
                self.config.eos_token_id,
            ),
            *custom_processors,
        ])
        regular_scores = self._processed_scores(
            history, logits, processors, top_k, top_p, temperature
        )
        if not do_sample:
            return regular_scores.argmax(dim=-1)
        normal = self._sample(regular_scores, generator=generator)
        high_scores = self._processed_scores(
            history,
            logits,
            processors,
            top_k,
            self.config.ras_top_p,
            self.config.ras_temperature,
        )
        high = self._sample(high_scores, generator=generator)
        if previous is None:
            return normal
        repeated = (previous == normal[:, None]).any(dim=1)
        semantic = (normal >= self.config.semantic_begin_id) & (
            normal <= self.config.semantic_end_id
        )
        return torch.where(repeated & semantic, high, normal)

    def _generate_codebooks(
        self,
        slow_hidden: Tensor,
        semantic: Tensor,
        processors: LogitsProcessorList,
        top_k: int,
        top_p: float,
        temperature: float,
        do_sample: bool,
        generator=None,
    ) -> Tensor:
        hidden = self.fast_project_in(slow_hidden)
        self._fast_step(hidden, 0)
        current = (semantic - self.config.semantic_begin_id).clamp(0, self.config.codebook_size - 1)
        codebooks = [current]
        fast_history = current[:, None]
        hidden = self.fast_embeddings(current)[:, None]
        for position in range(1, self.config.num_codebooks):
            scores = self._fast_step(hidden, position)
            scores = self._processed_scores(
                fast_history, scores, processors, top_k, top_p, temperature
            )
            current = self._sample(scores, generator=generator) if do_sample else scores.argmax(dim=-1)
            codebooks.append(current)
            fast_history = torch.cat((fast_history, current[:, None]), dim=1)
            hidden = self.fast_embeddings(current)[:, None]
        return torch.stack(codebooks, dim=1)

    def _prepare_prompt(
        self,
        input_ids=None,
        attention_mask=None,
        prefix_input_ids=None,
        prefix_attention_mask=None,
        suffix_input_ids=None,
        suffix_attention_mask=None,
        reference_codes=None,
        reference_code_lengths=None,
        reference_audio_values=None,
        reference_audio_lengths=None,
    ):
        if input_ids is not None:
            if input_ids.ndim != 3 or input_ids.shape[1] != self.config.num_codebooks + 1:
                raise ValueError("Direct input_ids must have shape [B, num_codebooks + 1, T]")
            if attention_mask is None:
                attention_mask = torch.ones(
                    input_ids.shape[0], input_ids.shape[-1], dtype=torch.long, device=input_ids.device
                )
            return input_ids.to(self.device), attention_mask.to(self.device)
        if prefix_input_ids is None or suffix_input_ids is None:
            raise ValueError("Processor output or direct input_ids is required")
        prefix_input_ids = prefix_input_ids.to(self.device)
        suffix_input_ids = suffix_input_ids.to(self.device)
        prefix_attention_mask = prefix_attention_mask.to(self.device)
        suffix_attention_mask = suffix_attention_mask.to(self.device)
        if reference_audio_values is not None:
            if reference_codes is not None:
                raise ValueError("Provide reference audio or reference codes, not both")
            reference_codes, reference_code_lengths = self.encode_audio(
                reference_audio_values.to(self.device), reference_audio_lengths.to(self.device)
            )
        if reference_codes is not None:
            reference_codes = reference_codes.to(self.device)
            reference_code_lengths = reference_code_lengths.to(self.device)
        batch_size = prefix_input_ids.shape[0]
        rows = []
        for batch_index in range(batch_size):
            prefix = prefix_input_ids[batch_index, prefix_attention_mask[batch_index].bool()]
            suffix = suffix_input_ids[batch_index, suffix_attention_mask[batch_index].bool()]
            if reference_codes is None:
                semantic_row = torch.cat((prefix, suffix))
                values = torch.zeros(
                    (self.config.num_codebooks + 1, semantic_row.numel()),
                    dtype=torch.long,
                    device=self.device,
                )
                values[0] = semantic_row
            else:
                length = int(reference_code_lengths[batch_index])
                codes = reference_codes[batch_index, :, :length].long()
                semantic_codes = codes[0] + self.config.semantic_begin_id
                semantic_row = torch.cat((prefix, semantic_codes, suffix))
                values = torch.zeros(
                    (self.config.num_codebooks + 1, semantic_row.numel()),
                    dtype=torch.long,
                    device=self.device,
                )
                values[0] = semantic_row
                values[1:, prefix.numel() : prefix.numel() + length] = codes
            rows.append(values)
        max_length = max(row.shape[1] for row in rows)
        prompt = torch.zeros(
            (batch_size, self.config.num_codebooks + 1, max_length),
            dtype=torch.long,
            device=self.device,
        )
        prompt[:, 0] = self.config.pad_token_id
        prompt_mask = torch.zeros((batch_size, max_length), dtype=torch.long, device=self.device)
        for batch_index, row in enumerate(rows):
            start = max_length - row.shape[1]
            prompt[batch_index, :, start:] = row
            prompt_mask[batch_index, start:] = 1
        return prompt, prompt_mask

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        prefix_input_ids: Optional[Tensor] = None,
        prefix_attention_mask: Optional[Tensor] = None,
        suffix_input_ids: Optional[Tensor] = None,
        suffix_attention_mask: Optional[Tensor] = None,
        reference_codes: Optional[Tensor] = None,
        reference_code_lengths: Optional[Tensor] = None,
        reference_audio_values: Optional[Tensor] = None,
        reference_audio_lengths: Optional[Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor=None,
        codebook_logits_processor=None,
        stopping_criteria=None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        do_sample: Optional[bool] = None,
        return_dict_in_generate: bool = False,
        generator=None,
        **kwargs,
    ):
        """Generate codec frames with cached slow AR and fast AR decoding."""
        if kwargs:
            raise TypeError(f"Unexpected generation arguments: {sorted(kwargs)}")
        config = generation_config or getattr(self, "generation_config", GenerationConfig())
        config_max_new = getattr(config, "max_new_tokens", None)
        config_temperature = getattr(config, "temperature", None)
        config_top_p = getattr(config, "top_p", None)
        config_top_k = getattr(config, "top_k", None)
        max_new_tokens = int(max_new_tokens if max_new_tokens is not None else (config_max_new or 512))
        temperature = float(temperature if temperature is not None else (config_temperature or 0.7))
        top_p = float(top_p if top_p is not None else (config_top_p or 0.9))
        top_k = int(top_k if top_k is not None else (config_top_k or 50))
        do_sample = bool(do_sample if do_sample is not None else getattr(config, "do_sample", True))
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        prompt, prompt_mask = self._prepare_prompt(
            input_ids, attention_mask, prefix_input_ids, prefix_attention_mask,
            suffix_input_ids, suffix_attention_mask, reference_codes,
            reference_code_lengths, reference_audio_values, reference_audio_lengths,
        )
        batch_size, _, prompt_width = prompt.shape
        if prompt_width >= self.config.max_seq_len:
            raise ValueError(
                f"Prompt length {prompt_width} must be smaller than {self.config.max_seq_len}"
            )
        max_new_tokens = min(max_new_tokens, self.config.max_seq_len - prompt_width)
        self._setup_generation_caches(
            batch_size, prompt_width + max_new_tokens, next(self.parameters()).dtype
        )
        semantic_processors = self._as_processor_list(logits_processor)
        codebook_processors = self._as_processor_list(codebook_logits_processor)
        criteria = stopping_criteria or StoppingCriteriaList()
        if not isinstance(criteria, StoppingCriteriaList):
            criteria = StoppingCriteriaList(criteria)
        cache_position = torch.arange(prompt_width, device=self.device, dtype=torch.long)
        position_ids = prompt_mask.cumsum(-1).sub(1).clamp_min(0)
        logits, slow_hidden = self._slow_step(
            prompt, cache_position, position_ids, prompt_mask
        )
        semantic_history = prompt[:, 0]
        prompt_lengths = prompt_mask.sum(-1)
        previous = None
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        code_lengths = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        generated_frames = []

        for step in range(max_new_tokens):
            active_before = ~finished
            semantic = self._sample_semantic(
                semantic_history, logits, semantic_processors, top_k, top_p,
                temperature, previous, do_sample, generator,
            )
            codebooks = self._generate_codebooks(
                slow_hidden, semantic, codebook_processors, top_k, top_p,
                temperature, do_sample, generator,
            )
            emitted = active_before & (semantic != self.config.eos_token_id)
            frame = torch.where(emitted[:, None], codebooks, -1)
            generated_frames.append(frame)
            code_lengths += emitted.long()
            semantic_history = torch.cat((semantic_history, semantic[:, None]), dim=1)

            if previous is None:
                previous = torch.zeros(
                    (batch_size, self.config.ras_window_size),
                    dtype=torch.long,
                    device=self.device,
                )
            else:
                previous = previous.roll(-1, dims=1)
                previous[:, -1] = semantic
            finished |= semantic.eq(self.config.eos_token_id)
            if criteria:
                stopped = criteria(semantic_history, logits)
                if not isinstance(stopped, Tensor):
                    stopped = torch.full_like(finished, bool(stopped))
                finished |= stopped.to(device=self.device, dtype=torch.bool)
            if finished.all():
                break

            next_column = torch.cat((semantic[:, None], codebooks), dim=1).unsqueeze(-1)
            new_valid = active_before.long()[:, None]
            prompt_mask = torch.cat((prompt_mask, new_valid), dim=1)
            physical_position = torch.tensor([prompt_width + step], device=self.device)
            token_position = (prompt_lengths + step)[:, None]
            with sdpa_kernel(SDPBackend.MATH):
                logits, slow_hidden = self._slow_step(
                    next_column, physical_position, token_position, prompt_mask
                )

        if generated_frames:
            codes = torch.stack(generated_frames, dim=2)
            max_valid = int(code_lengths.max().item()) if code_lengths.numel() else 0
            codes = codes[:, :, :max_valid]
        else:
            codes = torch.empty(
                (batch_size, self.config.num_codebooks, 0), dtype=torch.long, device=self.device
            )
        result = ArkttsGenerateOutput(codes=codes, code_lengths=code_lengths, finished=finished)
        return result if return_dict_in_generate else codes

    def _codec_path(self) -> str:
        source = str(getattr(self.config, "_name_or_path", ""))
        local = Path(source)
        if local.is_dir() and (local / self.config.codec_filename).is_file():
            return str(local / self.config.codec_filename)
        resolved = cached_file(source, self.config.codec_filename)
        if resolved is None:
            raise FileNotFoundError(f"Could not resolve {self.config.codec_filename} from {source}")
        return resolved

    def save_pretrained(self, save_directory, *args, **kwargs):
        result = super().save_pretrained(save_directory, *args, **kwargs)
        source = Path(self._codec_path()).resolve()
        destination = Path(save_directory) / self.config.codec_filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source != destination.resolve():
            shutil.copy2(source, destination)
        return result

    def load_codec(self, device=None, dtype=None):
        """Lazily load the bundled codec and keep one reusable module instance."""
        codec = self.__dict__.get("_arktts_codec")
        target_device = torch.device(device) if device is not None else self.device
        target_dtype = dtype or self.dtype
        if target_device.type == "cpu":
            target_dtype = torch.float32
        if codec is None:
            from .modeling_arktts_codec import ArkttsCodec

            codec = ArkttsCodec(self.config)
            state = torch.load(self._codec_path(), map_location="cpu", weights_only=True)
            if "state_dict" in state:
                state = state["state_dict"]
            if any("generator." in key for key in state):
                state = {
                    key.replace("generator.", ""): value
                    for key, value in state.items() if "generator." in key
                }
            state = {
                key: value
                for key, value in state.items()
                if not key.endswith(("freqs_cis", "causal_mask"))
            }
            codec.load_state_dict(state, strict=True)
            codec.eval()
            self.__dict__["_arktts_codec"] = codec
        codec.to(device=target_device, dtype=target_dtype)
        return codec

    @torch.inference_mode()
    def encode_audio(self, audio_values: Tensor, audio_lengths: Optional[Tensor] = None):
        """Encode padded mono waveforms into ten codec-index streams."""
        codec = self.load_codec(device=audio_values.device)
        audio_values = audio_values.to(dtype=next(codec.parameters()).dtype)
        return codec.encode(audio_values, audio_lengths)

    @torch.inference_mode()
    def decode_audio(self, codes: Tensor):
        """Decode padded codec streams and return waveforms with true lengths."""
        if codes.ndim == 2:
            codes = codes.unsqueeze(0)
        if codes.ndim != 3 or codes.shape[1] != self.config.num_codebooks:
            raise ValueError(f"codes must have shape [B, {self.config.num_codebooks}, T]")
        codec = self.load_codec(device=codes.device)
        waveforms = []
        lengths = []
        for item in codes:
            valid = (item >= 0).all(dim=0)
            length = int(valid.sum().item())
            if length == 0:
                waveform = torch.empty(0, device=codes.device, dtype=torch.float32)
            else:
                waveform = codec.decode(item[:, :length].unsqueeze(0))[0, 0].float()
            waveforms.append(waveform)
            lengths.append(waveform.numel())
        max_length = max(lengths, default=0)
        padded = torch.zeros((len(waveforms), max_length), dtype=torch.float32, device=codes.device)
        for index, waveform in enumerate(waveforms):
            padded[index, : waveform.numel()] = waveform
        return padded, torch.tensor(lengths, dtype=torch.long, device=codes.device)

    @torch.inference_mode()
    def generate_audio(self, **kwargs):
        codes = self.generate(**kwargs)
        waveforms, lengths = self.decode_audio(codes)
        return waveforms, lengths, codes


__all__ = [
    "ArkttsConfig",
    "ArkttsGenerateOutput",
    "ArkttsLegacyTopKTopPLogitsProcessor",
    "ArkttsModel",
    "ArkttsModelOutput",
    "ArkttsSemanticLogitsProcessor",
]

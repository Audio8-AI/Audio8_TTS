from __future__ import annotations

from transformers import PretrainedConfig


class ArkttsConfig(PretrainedConfig):
    model_type = "arktts"

    def __init__(
        self,
        vocab_size: int = 155776,
        dim: int = 896,
        n_layer: int = 24,
        n_head: int = 14,
        n_local_heads: int = 2,
        head_dim: int = 64,
        intermediate_size: int = 4864,
        max_seq_len: int = 2048,
        rope_base: float = 1_000_000,
        norm_eps: float = 1e-6,
        dropout: float = 0.0,
        attention_qkv_bias: bool = True,
        attention_qk_norm: bool = False,
        attention_o_bias: bool = False,
        tie_word_embeddings: bool = True,
        codebook_size: int = 4096,
        num_codebooks: int = 10,
        semantic_begin_id: int = 151678,
        semantic_end_id: int = 155773,
        n_fast_layer: int = 4,
        fast_dim: int = 896,
        fast_n_head: int = 14,
        fast_n_local_heads: int = 2,
        fast_head_dim: int = 64,
        fast_intermediate_size: int = 4864,
        fast_attention_qkv_bias: bool = False,
        fast_attention_qk_norm: bool = False,
        fast_attention_o_bias: bool = False,
        norm_fastlayer_input: bool = True,
        initializer_range: float = 0.02,
        use_gradient_checkpointing: bool = False,
        codec_filename: str = "codec.pth",
        codec_sample_rate: int = 44100,
        codec_frame_size: int = 2048,
        ras_window_size: int = 10,
        ras_temperature: float = 1.0,
        ras_top_p: float = 0.9,
        eos_token_id: int = 151645,
        pad_token_id: int = 151643,
        **kwargs,
    ):
        kwargs.pop("model_type", None)
        kwargs.pop("audio_embed_dim", None)
        kwargs.pop("is_reward_model", None)
        super().__init__(
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = int(vocab_size)
        self.dim = int(dim)
        self.hidden_size = self.dim
        self.n_layer = int(n_layer)
        self.num_hidden_layers = self.n_layer
        self.n_head = int(n_head)
        self.num_attention_heads = self.n_head
        self.n_local_heads = int(n_local_heads)
        self.num_key_value_heads = self.n_local_heads
        self.head_dim = int(head_dim)
        self.intermediate_size = int(intermediate_size)
        self.max_seq_len = int(max_seq_len)
        self.max_position_embeddings = self.max_seq_len
        self.rope_base = float(rope_base)
        self.rope_theta = self.rope_base
        self.norm_eps = float(norm_eps)
        self.rms_norm_eps = self.norm_eps
        self.dropout = float(dropout)
        self.attention_qkv_bias = bool(attention_qkv_bias)
        self.attention_qk_norm = bool(attention_qk_norm)
        self.attention_o_bias = bool(attention_o_bias)
        self.codebook_size = int(codebook_size)
        self.num_codebooks = int(num_codebooks)
        self.semantic_begin_id = int(semantic_begin_id)
        self.semantic_end_id = int(semantic_end_id)
        self.n_fast_layer = int(n_fast_layer)
        self.fast_dim = int(fast_dim)
        self.fast_n_head = int(fast_n_head)
        self.fast_n_local_heads = int(fast_n_local_heads)
        self.fast_head_dim = int(fast_head_dim)
        self.fast_intermediate_size = int(fast_intermediate_size)
        self.fast_attention_qkv_bias = bool(fast_attention_qkv_bias)
        self.fast_attention_qk_norm = bool(fast_attention_qk_norm)
        self.fast_attention_o_bias = bool(fast_attention_o_bias)
        self.norm_fastlayer_input = bool(norm_fastlayer_input)
        self.initializer_range = float(initializer_range)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)
        self.codec_filename = str(codec_filename)
        self.codec_sample_rate = int(codec_sample_rate)
        self.codec_frame_size = int(codec_frame_size)
        self.ras_window_size = int(ras_window_size)
        self.ras_temperature = float(ras_temperature)
        self.ras_top_p = float(ras_top_p)


__all__ = ["ArkttsConfig"]

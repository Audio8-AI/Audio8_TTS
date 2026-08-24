//! DualAR (slow AR + fast AR) forward pass in candle, ported from
//! model/audio8_tts_0_6B_preview/modeling_arktts.py (ArkttsModel) and
//! cross-checked against onnx_runtime/arktts_runtime/runtime.py's
//! _slow_step/_fast_step and rust_runtime/src/runtime.rs's slow_step/fast_step.
//!
//! Architecture constants (config.json): dim=fast_dim=896 (fast_project_in is
//! Identity), n_head=fast_n_head=14, n_local_heads=fast_n_local_heads=2,
//! head_dim=fast_head_dim=64, intermediate_size=fast_intermediate_size=4864,
//! n_layer=24, n_fast_layer=4, rope_base=1e6, norm_eps=1e-6,
//! attention_qk_norm=false (both), attention_qkv_bias=true (slow only),
//! attention_o_bias=false (both), norm_fastlayer_input=true (fast AR gets the
//! post-norm hidden state), tie_word_embeddings=true (but the repacked GGUF
//! carries a separate lm_head.weight tensor - see verify_gguf_weights.rs -
//! which is bit-identical to embeddings.weight under tying, so it is used
//! directly rather than re-deriving from the embedding table).
//!
//! RoPE uses the adjacent-pair (GPT-J-style) rotation: _apply_rope in
//! modeling_arktts.py reshapes the head dim as (..., head_dim/2, 2) and
//! rotates each pair - this is candle_nn::rotary_emb's "rope_i" family, not
//! the NeoX/half-split "rope" family.

use candle_core::quantized::{gguf_file, QMatMul};
use candle_core::{DType, Device, IndexOp, Module, Result, Tensor, Var, D};
use candle_nn::rotary_emb::rope_i;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

pub const DIM: usize = 896;
pub const N_HEAD: usize = 14;
pub const N_LOCAL_HEADS: usize = 2;
pub const HEAD_DIM: usize = 64;
pub const INTERMEDIATE_SIZE: usize = 4864;
pub const N_LAYER: usize = 24;
pub const N_FAST_LAYER: usize = 4;
pub const FAST_DIM: usize = 896;
pub const CODEBOOK_SIZE: usize = 4096;
pub const NUM_CODEBOOKS: usize = 10;
pub const VOCAB_SIZE: usize = 155776;
pub const SLOW_LOGITS_SIZE: usize = 4097;
pub const NORM_EPS: f64 = 1e-6;
pub const ROPE_BASE: f32 = 1_000_000.0;
pub const MAX_SEQ_LEN: usize = 2048;

/// A linear layer whose weight may be Q4_0-quantized (candle QMatMul) or
/// plain F32 (norms/embeddings are never routed through this type; only
/// nn.Linear-shaped 2D matrices are). Bias, when present, stays F32.
struct QLinear {
    weight: QMatMul,
    bias: Option<Tensor>,
}

impl QLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let ys = self.weight.forward(xs)?;
        match &self.bias {
            Some(b) => ys.broadcast_add(b),
            None => Ok(ys),
        }
    }
}

struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let in_dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let x_normed = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        x_normed.to_dtype(in_dtype)?.broadcast_mul(&self.weight)
    }
}

/// Precomputed RoPE cos/sin tables, one row per position, head_dim/2 columns
/// each - matches _precompute_rope's polar(1, freqs) real/imag split, stored
/// separately here since candle's rope_i takes cos and sin as separate
/// tensors rather than modeling_arktts.py's stacked-last-dim layout.
struct RopeTable {
    cos: Tensor,
    sin: Tensor,
}

fn precompute_rope(len: usize, head_dim: usize, base: f32, device: &Device) -> Result<RopeTable> {
    let half = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1f32 / base.powf((2 * i) as f32 / head_dim as f32))
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, half, device)?;
    let t: Vec<f32> = (0..len).map(|i| i as f32).collect();
    let t = Tensor::from_vec(t, len, device)?;
    let freqs = t.reshape((len, 1))?.broadcast_mul(&inv_freq.reshape((1, half))?)?;
    let cos = freqs.cos()?;
    let sin = freqs.sin()?;
    Ok(RopeTable { cos, sin })
}

/// GQA attention. wqkv projects to (n_head + 2*n_local_heads)*head_dim, split
/// into q/k/v; k/v are repeat_interleave'd from n_local_heads to n_head
/// before the dot-product (matches ArkttsAttention.forward exactly, minus
/// qk_norm which config.json confirms is false for both slow and fast AR).
struct Attention {
    wqkv: QLinear,
    wo: QLinear,
    n_head: usize,
    n_local_heads: usize,
    head_dim: usize,
}

impl Attention {
    /// x: [B,T,dim]. cos/sin: [T, head_dim/2] rows selected for this call's
    /// positions. kv_cache: persistent [B, n_local_heads, max_len, head_dim]
    /// Var pair, written in place at `positions` (write-only past what's
    /// already there; contents at other positions must already be correct).
    /// causal_len is the number of valid key positions to attend over
    /// (0..=max cache_position for this call).
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        kv_cache: &mut FixedKvCache,
        positions: &[usize],
        causal_len: usize,
    ) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let qkv = self.wqkv.forward(x)?;
        let q_size = self.n_head * self.head_dim;
        let kv_size = self.n_local_heads * self.head_dim;
        let q = qkv.narrow(D::Minus1, 0, q_size)?;
        let k = qkv.narrow(D::Minus1, q_size, kv_size)?;
        let v = qkv.narrow(D::Minus1, q_size + kv_size, kv_size)?;

        let q = q.reshape((b, t, self.n_head, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b, t, self.n_local_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b, t, self.n_local_heads, self.head_dim))?.transpose(1, 2)?;

        let q = rope_i(&q.contiguous()?, cos, sin)?;
        let k = rope_i(&k.contiguous()?, cos, sin)?;

        kv_cache.write(&k, &v, positions)?;
        let (k_full, v_full) = kv_cache.read(causal_len)?;

        let repeats = self.n_head / self.n_local_heads;
        let k_full = repeat_interleave_heads(&k_full, repeats)?;
        let v_full = repeat_interleave_heads(&v_full, repeats)?;

        let scale = 1f64 / (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k_full.transpose(2, 3)?.contiguous()?)? * scale)?;

        let attn_weights = if t == causal_len {
            let mask = causal_mask(t, x.device())?;
            attn_weights.broadcast_add(&mask)?
        } else {
            // decode step: t=1 new query position at the end of causal_len
            // keys, attends to everything (no masking needed).
            attn_weights
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let out = attn_weights.matmul(&v_full.contiguous()?)?;
        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, t, q_size))?;
        self.wo.forward(&out)
    }
}

fn causal_mask(t: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<f32> = (0..t)
        .flat_map(|i| (0..t).map(move |j| if j <= i { 0f32 } else { f32::NEG_INFINITY }))
        .collect();
    Tensor::from_vec(mask, (1, 1, t, t), device)
}

fn repeat_interleave_heads(x: &Tensor, repeats: usize) -> Result<Tensor> {
    if repeats == 1 {
        return Ok(x.clone());
    }
    let (b, h, t, d) = x.dims4()?;
    x.reshape((b, h, 1, t, d))?
        .broadcast_as((b, h, repeats, t, d))?
        .reshape((b, h * repeats, t, d))
}

/// Fixed-shape, address-stable KV cache (Var-backed) sized to `max_len` so
/// its storage never reallocates across decode steps - required for CUDA
/// graph replay, and mirrors rust_runtime's ArrayD-cache-with-delta-write
/// approach (there done on the host/ORT side, here directly on device).
pub struct FixedKvCache {
    k: Var,
    v: Var,
    n_local_heads: usize,
    head_dim: usize,
    max_len: usize,
}

impl FixedKvCache {
    fn new(batch: usize, n_local_heads: usize, max_len: usize, head_dim: usize, device: &Device) -> Result<Self> {
        let k = Var::zeros((batch, n_local_heads, max_len, head_dim), DType::F32, device)?;
        let v = Var::zeros((batch, n_local_heads, max_len, head_dim), DType::F32, device)?;
        Ok(Self { k, v, n_local_heads, head_dim, max_len })
    }

    fn reset(&mut self) -> Result<()> {
        let shape = (self.k.dims()[0], self.n_local_heads, self.max_len, self.head_dim);
        self.k.set(&Tensor::zeros(shape, DType::F32, self.k.device())?)?;
        self.v.set(&Tensor::zeros(shape, DType::F32, self.v.device())?)?;
        Ok(())
    }

    /// Writes new_k/new_v (shape [B,H,len(positions),D]) into the cache at
    /// the given absolute positions via slice_assign. positions must be
    /// contiguous ascending (true for both prefill and single-step decode
    /// here) so a single range assign suffices.
    ///
    /// slice_assign, not slice_scatter: slice_scatter's CUDA path
    /// (alloc_uninit + two back-to-back copy_strided_src calls into a fresh
    /// scratch buffer) is NOT reliable under CUDA graph capture with
    /// CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH - verified via
    /// src/bin/graph_smoke_test.rs's test2 vs test5: slice_scatter silently
    /// replayed stale/zero data in 3 of 8 repeated capture+replay runs (same
    /// binary, same GPU, no crash - values just didn't update), while
    /// slice_assign's pad_with_zeros+where_cond composition was correct in
    /// 8 of 8. Root cause not narrowed further than "the alloc_uninit
    /// scratch buffer's address becomes unreliable across AUTO_FREE_ON_LAUNCH
    /// replay cycles"; slice_assign avoids that scratch-buffer pattern
    /// entirely, so it's used everywhere a KV cache write can end up inside
    /// a captured graph (graph_decode.rs), not just in the T=1 decode path.
    fn write(&mut self, new_k: &Tensor, new_v: &Tensor, positions: &[usize]) -> Result<()> {
        let start = positions[0];
        let len = positions.len();
        debug_assert!(positions.iter().enumerate().all(|(i, &p)| p == start + i));
        let batch = self.k.dims()[0];
        let ranges = [0..batch, 0..self.n_local_heads, start..start + len, 0..self.head_dim];
        let k_full = self.k.as_tensor();
        let v_full = self.v.as_tensor();
        let k_new = k_full.slice_assign(&ranges, new_k)?;
        let v_new = v_full.slice_assign(&ranges, new_v)?;
        debug_assert_eq!(len, new_k.dim(2)?);
        self.k.set(&k_new)?;
        self.v.set(&v_new)?;
        Ok(())
    }

    fn read(&self, causal_len: usize) -> Result<(Tensor, Tensor)> {
        let k = self.k.as_tensor().narrow(2, 0, causal_len)?;
        let v = self.v.as_tensor().narrow(2, 0, causal_len)?;
        Ok((k, v))
    }

    /// Zeroes just one absolute position's slot in both k and v, in place
    /// (storage address unchanged) - used by CUDA graph capture to undo the
    /// warm-up/capture-time writes to a position before the graph's own
    /// replay becomes the sole writer of that slot.
    pub fn zero_position(&mut self, position: usize) -> Result<()> {
        let batch = self.k.dims()[0];
        let shape = (batch, self.n_local_heads, 1, self.head_dim);
        let zeros = Tensor::zeros(shape, DType::F32, self.k.device())?;
        let ranges = [0..batch, 0..self.n_local_heads, position..position + 1, 0..self.head_dim];
        let k_new = self.k.as_tensor().slice_assign(&ranges, &zeros)?;
        let v_new = self.v.as_tensor().slice_assign(&ranges, &zeros)?;
        self.k.set(&k_new)?;
        self.v.set(&v_new)?;
        Ok(())
    }
}

struct FeedForward {
    w1: QLinear,
    w2: QLinear,
    w3: QLinear,
}

impl FeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.w1.forward(x)?)?;
        let up = self.w3.forward(x)?;
        self.w2.forward(&(gate * up)?)
    }
}

struct TransformerBlock {
    attention: Attention,
    feed_forward: FeedForward,
    attention_norm: RmsNorm,
    ffn_norm: RmsNorm,
}

impl TransformerBlock {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        kv_cache: &mut FixedKvCache,
        positions: &[usize],
        causal_len: usize,
    ) -> Result<Tensor> {
        let normed = self.attention_norm.forward(x)?;
        let attn_out = self.attention.forward(&normed, cos, sin, kv_cache, positions, causal_len)?;
        let hidden = (x + attn_out)?;
        let normed2 = self.ffn_norm.forward(&hidden)?;
        let ff_out = self.feed_forward.forward(&normed2)?;
        hidden + ff_out
    }
}

fn load_qlinear(content: &gguf_file::Content, file: &mut std::fs::File, name: &str, device: &Device) -> Result<QMatMul> {
    let qtensor = content.tensor(file, name, device)?;
    QMatMul::from_qtensor(qtensor)
}

fn load_f32(content: &gguf_file::Content, file: &mut std::fs::File, name: &str, device: &Device) -> Result<Tensor> {
    let qtensor = content.tensor(file, name, device)?;
    qtensor.dequantize(device)
}

fn load_rmsnorm(content: &gguf_file::Content, file: &mut std::fs::File, name: &str, device: &Device) -> Result<RmsNorm> {
    let weight = load_f32(content, file, name, device)?;
    Ok(RmsNorm { weight, eps: NORM_EPS })
}

fn load_block(
    content: &gguf_file::Content,
    file: &mut std::fs::File,
    prefix: &str,
    n_head: usize,
    n_local_heads: usize,
    head_dim: usize,
    qkv_bias: bool,
    device: &Device,
) -> Result<TransformerBlock> {
    let wqkv = load_qlinear(content, file, &format!("{prefix}.attention.wqkv.weight"), device)?;
    let wqkv_bias = if qkv_bias {
        Some(load_f32(content, file, &format!("{prefix}.attention.wqkv.bias"), device)?)
    } else {
        None
    };
    let wo = load_qlinear(content, file, &format!("{prefix}.attention.wo.weight"), device)?;
    let attention = Attention {
        wqkv: QLinear { weight: wqkv, bias: wqkv_bias },
        wo: QLinear { weight: wo, bias: None },
        n_head,
        n_local_heads,
        head_dim,
    };
    let feed_forward = FeedForward {
        w1: QLinear { weight: load_qlinear(content, file, &format!("{prefix}.feed_forward.w1.weight"), device)?, bias: None },
        w2: QLinear { weight: load_qlinear(content, file, &format!("{prefix}.feed_forward.w2.weight"), device)?, bias: None },
        w3: QLinear { weight: load_qlinear(content, file, &format!("{prefix}.feed_forward.w3.weight"), device)?, bias: None },
    };
    let attention_norm = load_rmsnorm(content, file, &format!("{prefix}.attention_norm.weight"), device)?;
    let ffn_norm = load_rmsnorm(content, file, &format!("{prefix}.ffn_norm.weight"), device)?;
    Ok(TransformerBlock { attention, feed_forward, attention_norm, ffn_norm })
}

pub struct SlowAr {
    embeddings: Tensor,          // [VOCAB_SIZE, DIM] F32 (dequantized; also used for embedding lookups)
    codebook_embeddings: Tensor, // [CODEBOOK_SIZE*NUM_CODEBOOKS, DIM] F32
    layers: Vec<TransformerBlock>,
    norm: RmsNorm,
    lm_head: QLinear,
    rope: RopeTable,
    semantic_begin_id: i64,
    semantic_end_id: i64,
    device: Device,
}

pub struct FastAr {
    fast_embeddings: Tensor, // [CODEBOOK_SIZE, FAST_DIM] F32
    layers: Vec<TransformerBlock>,
    fast_norm: RmsNorm,
    fast_output: QLinear,
    rope: RopeTable,
    device: Device,
}

impl SlowAr {
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        let mut file = std::fs::File::open(path)?;
        let content = gguf_file::Content::read(&mut file)?;

        let embeddings = load_f32(&content, &mut file, "embeddings.weight", device)?;
        let codebook_embeddings = load_f32(&content, &mut file, "codebook_embeddings.weight", device)?;
        let mut layers = Vec::with_capacity(N_LAYER);
        for i in 0..N_LAYER {
            layers.push(load_block(
                &content,
                &mut file,
                &format!("layers.{i}"),
                N_HEAD,
                N_LOCAL_HEADS,
                HEAD_DIM,
                true, // attention_qkv_bias
                device,
            )?);
        }
        let norm = load_rmsnorm(&content, &mut file, "norm.weight", device)?;
        let lm_head_w = load_qlinear(&content, &mut file, "lm_head.weight", device)?;
        let lm_head = QLinear { weight: lm_head_w, bias: None };
        let rope = precompute_rope(MAX_SEQ_LEN, HEAD_DIM, ROPE_BASE, device)?;

        Ok(Self {
            embeddings,
            codebook_embeddings,
            layers,
            norm,
            lm_head,
            rope,
            semantic_begin_id: 151678,
            semantic_end_id: 155773,
            device: device.clone(),
        })
    }

    pub fn new_kv_caches(&self, batch: usize) -> Result<Vec<FixedKvCache>> {
        (0..N_LAYER)
            .map(|_| FixedKvCache::new(batch, N_LOCAL_HEADS, MAX_SEQ_LEN, HEAD_DIM, &self.device))
            .collect()
    }

    /// input_ids: [B, num_codebooks+1, T] i64. Embeds token row 0 via the
    /// vocab embedding table, sums in codebook embeddings for rows 1.. gated
    /// by row 0 falling in [semantic_begin_id, semantic_end_id] - matches
    /// ArkttsModel._embed exactly.
    fn embed(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (b, _, t) = input_ids.dims3()?;
        let row0 = input_ids.i((.., 0, ..))?.contiguous()?; // [B,T]
        let tok_embed = self.embeddings.index_select(&row0.flatten_all()?.to_dtype(DType::U32)?, 0)?.reshape((b, t, DIM))?;

        let mut codebook_sum = Tensor::zeros((b, t, DIM), DType::F32, &self.device)?;
        for cb in 0..NUM_CODEBOOKS {
            let row = input_ids.i((.., cb + 1, ..))?.contiguous()?; // [B,T]
            let offset = (cb * CODEBOOK_SIZE) as f64;
            let idx = (row.to_dtype(DType::F32)? + offset)?.to_dtype(DType::U32)?;
            let emb = self.codebook_embeddings.index_select(&idx.flatten_all()?, 0)?.reshape((b, t, DIM))?;
            codebook_sum = (codebook_sum + emb)?;
        }
        let row0_f32 = row0.to_dtype(DType::F32)?;
        let begin = self.semantic_begin_id as f64;
        let end = self.semantic_end_id as f64;
        let ge = row0_f32.ge(begin)?;
        let le = row0_f32.le(end)?;
        let semantic_mask = (ge * le)?.to_dtype(DType::F32)?.reshape((b, t, 1))?; // [B,T,1]
        let gated = codebook_sum.broadcast_mul(&semantic_mask)?;
        tok_embed + gated
    }

    /// One slow_step call: input_ids [B, num_codebooks+1, T], positions are
    /// the absolute cache positions this call's T columns occupy (ascending,
    /// contiguous). causal_len = positions.last()+1 (valid key count).
    /// Returns (logits [B, SLOW_LOGITS_SIZE] for the LAST position only,
    /// fast_hidden [B, 1, DIM] - the normalized hidden state fed to fast AR
    /// since norm_fastlayer_input=true).
    pub fn slow_step(
        &self,
        input_ids: &Tensor,
        positions: &[usize],
        kv_caches: &mut [FixedKvCache],
    ) -> Result<(Tensor, Tensor)> {
        let mut hidden = self.embed(input_ids)?;
        let t = positions.len();
        let cos = self.rope.cos.narrow(0, positions[0], t)?;
        let sin = self.rope.sin.narrow(0, positions[0], t)?;
        let causal_len = positions[positions.len() - 1] + 1;
        for (layer, cache) in self.layers.iter().zip(kv_caches.iter_mut()) {
            hidden = layer.forward(&hidden, &cos, &sin, cache, positions, causal_len)?;
        }
        let last_hidden = hidden.narrow(1, t - 1, 1)?; // [B,1,DIM]
        let normalized = self.norm.forward(&last_hidden)?;
        let logits = self.lm_head.forward(&normalized)?; // [B,1,SLOW_LOGITS_SIZE]
        let logits = logits.squeeze(1)?; // [B,SLOW_LOGITS_SIZE]
        Ok((logits, normalized)) // norm_fastlayer_input=true -> pass normalized hidden
    }
}

impl FastAr {
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        let mut file = std::fs::File::open(path)?;
        let content = gguf_file::Content::read(&mut file)?;

        let fast_embeddings = load_f32(&content, &mut file, "fast_embeddings.weight", device)?;
        let mut layers = Vec::with_capacity(N_FAST_LAYER);
        for i in 0..N_FAST_LAYER {
            layers.push(load_block(
                &content,
                &mut file,
                &format!("fast_layers.{i}"),
                N_HEAD,
                N_LOCAL_HEADS,
                HEAD_DIM,
                false, // fast_attention_qkv_bias = false
                device,
            )?);
        }
        let fast_norm = load_rmsnorm(&content, &mut file, "fast_norm.weight", device)?;
        let fast_output_w = load_qlinear(&content, &mut file, "fast_output.weight", device)?;
        let fast_output = QLinear { weight: fast_output_w, bias: None };
        let rope = precompute_rope(NUM_CODEBOOKS, HEAD_DIM, ROPE_BASE, device)?;

        Ok(Self { fast_embeddings, layers, fast_norm, fast_output, rope, device: device.clone() })
    }

    pub fn new_kv_caches(&self, batch: usize) -> Result<Vec<FixedKvCache>> {
        (0..N_FAST_LAYER)
            .map(|_| FixedKvCache::new(batch, N_LOCAL_HEADS, NUM_CODEBOOKS, HEAD_DIM, &self.device))
            .collect()
    }

    /// hidden: [B,1,FAST_DIM] (either the slow AR's fast_hidden at position
    /// 0, or fast_embeddings(token) at position>0 - matches
    /// ArkttsModel._fast_step exactly, fast_project_in is Identity since
    /// dim==fast_dim==896 for this model). position: absolute codebook
    /// index (0..NUM_CODEBOOKS). Returns logits [B, CODEBOOK_SIZE].
    pub fn fast_step(&self, hidden: &Tensor, position: usize, kv_caches: &mut [FixedKvCache]) -> Result<Tensor> {
        self.fast_step_n_layers(hidden, position, kv_caches, self.layers.len())
    }

    /// Bisection helper for CUDA-graph-capture debugging: same as fast_step
    /// but only runs the first n_layers transformer blocks. Not used outside
    /// diagnostics.
    pub fn fast_step_n_layers(&self, hidden: &Tensor, position: usize, kv_caches: &mut [FixedKvCache], n_layers: usize) -> Result<Tensor> {
        let cos = self.rope.cos.narrow(0, position, 1)?;
        let sin = self.rope.sin.narrow(0, position, 1)?;
        let causal_len = position + 1;
        let mut hidden = hidden.clone();
        for (layer, cache) in self.layers.iter().zip(kv_caches.iter_mut()).take(n_layers) {
            hidden = layer.forward(&hidden, &cos, &sin, cache, &[position], causal_len)?;
        }
        if n_layers == self.layers.len() {
            let normed = self.fast_norm.forward(&hidden)?;
            let logits = self.fast_output.forward(&normed)?; // [B,1,CODEBOOK_SIZE]
            logits.squeeze(1)
        } else {
            hidden.squeeze(1)
        }
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Runs exactly ONE transformer block (layer_idx) - the per-layer unit
    /// graph_decode.rs's ChainedFastStepGraph captures as its own small CUDA
    /// graph. hidden: [B,1,FAST_DIM] in, [B,1,FAST_DIM] out (kept 3D, unlike
    /// fast_step's final squeeze, since this feeds the next layer/head as
    /// another 3D input).
    pub fn fast_layer_step(&self, layer_idx: usize, hidden: &Tensor, position: usize, cache: &mut FixedKvCache) -> Result<Tensor> {
        let cos = self.rope.cos.narrow(0, position, 1)?;
        let sin = self.rope.sin.narrow(0, position, 1)?;
        let causal_len = position + 1;
        self.layers[layer_idx].forward(hidden, &cos, &sin, cache, &[position], causal_len)
    }

    /// fast_norm + fast_output, the tail after all transformer blocks.
    /// hidden: [B,1,FAST_DIM] in, returns [B,CODEBOOK_SIZE].
    pub fn fast_output_head(&self, hidden: &Tensor) -> Result<Tensor> {
        let normed = self.fast_norm.forward(hidden)?;
        let logits = self.fast_output.forward(&normed)?;
        logits.squeeze(1)
    }

    pub fn embed_token(&self, token: u32) -> Result<Tensor> {
        let idx = Tensor::new(&[token], &self.device)?;
        self.fast_embeddings.index_select(&idx, 0)?.reshape((1, 1, FAST_DIM))
    }

    pub fn reset_caches(caches: &mut [FixedKvCache]) -> Result<()> {
        for c in caches.iter_mut() {
            c.reset()?;
        }
        Ok(())
    }
}

/// Loads runtime_manifest.json constants used to sanity-check the hardcoded
/// architecture consts above against the shipped model at startup.
pub struct ManifestCheck;

impl ManifestCheck {
    pub fn verify(manifest_path: &Path) -> Result<()> {
        let text = std::fs::read_to_string(manifest_path).map_err(candle_core::Error::wrap)?;
        let v: HashMap<String, serde_json::Value> =
            serde_json::from_str(&text).map_err(candle_core::Error::wrap)?;
        let check = |k: &str, expect: i64| -> Result<()> {
            let got = v.get(k).and_then(|x| x.as_i64()).unwrap_or(-1);
            if got != expect {
                candle_core::bail!("manifest {k}={got} does not match hardcoded {expect}");
            }
            Ok(())
        };
        check("num_layers", N_LAYER as i64)?;
        check("num_fast_layers", N_FAST_LAYER as i64)?;
        check("num_codebooks", NUM_CODEBOOKS as i64)?;
        check("n_local_heads", N_LOCAL_HEADS as i64)?;
        check("head_dim", HEAD_DIM as i64)?;
        check("codebook_size", CODEBOOK_SIZE as i64)?;
        check("slow_logits_size", SLOW_LOGITS_SIZE as i64)?;
        check("vocab_size", VOCAB_SIZE as i64)?;
        Ok(())
    }
}

pub fn device_cuda_or_cpu(prefer_cuda: bool) -> Result<Device> {
    if prefer_cuda {
        match Device::new_cuda(0) {
            Ok(d) => return Ok(d),
            Err(e) => eprintln!("[warn] CUDA device unavailable ({e}), falling back to CPU"),
        }
    }
    Ok(Device::Cpu)
}

pub type SharedTensor = Arc<Tensor>;

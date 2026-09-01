//! Candle port of onnx_runtime/model/codec_decoder_fp16.onnx.
//!
//! Architecture (reverse-engineered from the ONNX graph, node-by-node, via
//! onnx.load + graph.node/graph.initializer inspection - not guessed):
//!
//!   codes [batch, 10, frames] (int64 codebook indices)
//!     -> RVQ dequant: 1 semantic codebook (size 4096) + 9 acoustic codebooks
//!        (size 1024 each, "quantizer.quantizer.quantizers.0..8"), each codebook
//!        dim=8, each with its own out_proj (1x1 Conv, dim 8 -> 1024), summed.
//!     -> post_module: 8-layer GQA transformer (16 q heads / 8 kv heads,
//!        head_dim=64, RoPE base 10000, non-causal - the ONNX graph's own
//!        attention mask is all-visible for an unpadded sequence) with
//!        RMSNorm + SwiGLU FFN + per-branch learnable layer-scale (gamma),
//!        operating at dim=1024. Final RMSNorm (quantizer.post_module.norm).
//!     -> upsample: 2x ConvNeXt-style blocks (quantizer.upsample.{0,1}), each
//!        = ConvTranspose1d(kernel2,stride2) + depthwise Conv1d(kernel7) +
//!        LayerNorm + pwconv1(1024->4096) + exact-erf GELU + pwconv2(4096->1024)
//!        + learnable gamma scale + residual. Two stages = 4x upsample.
//!     -> decoder: DAC/Descript-Audio-Codec-style vocoder (matches
//!        candle_transformers::models::dac::Decoder's own architecture
//!        exactly): conv1(1024->1536,k7) -> 4x DecoderBlock(Snake + ConvTranspose
//!        + 3x ResidualUnit(dilations 1,3,9)) with rates [8,8,4,2] (1536->768
//!        ->384->192->96) -> Snake -> conv2(96->1,k7) -> Tanh.
//!
//!   audio [batch, 1, samples], samples = frames * 2048 (hop = 8*8*4*2*2*2 = 2048,
//!   matching onnx_runtime's documented codec_hop_length).
//!
//! Snake(x) = x + (1/(alpha+1e-9)) * sin(alpha*x)^2 - verified against the
//! exact ONNX op sequence (Mul by reciprocal(alpha) -> Sin -> Pow(2) -> Mul ->
//! Add to residual), same formula candle_transformers::models::dac uses.
//!
//! All ONNX weight tensors are already plain (weight-norm fused at export
//! time, no weight_g/weight_v split), so this loads them directly with no
//! reparametrization - unlike candle_transformers::models::encodec's
//! conv1d_weight_norm helper, which is not needed here.

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, Module, VarBuilder};

const HIDDEN: usize = 1024;
const N_HEADS_Q: usize = 16;
const N_HEADS_KV: usize = 8;
const HEAD_DIM: usize = 64;
const N_TRANSFORMER_LAYERS: usize = 8;
const N_ACOUSTIC_CODEBOOKS: usize = 9;
const ROPE_BASE: f64 = 10000.0;

// ---------------------------------------------------------------------
// Snake activation + DAC-style decoder (matches candle_transformers::models::dac)
// ---------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Snake1d {
    alpha: Tensor,
}

impl Snake1d {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let alpha = vb.get((1, channels, 1), "alpha")?;
        Ok(Self { alpha })
    }
}

impl Module for Snake1d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs_shape = xs.shape();
        let xs = xs.flatten_from(2)?;
        let sin = self.alpha.broadcast_mul(&xs)?.sin()?;
        let sin = (&sin * &sin)?;
        (xs + (&self.alpha + 1e-9)?.recip()?.broadcast_mul(&sin)?)?.reshape(xs_shape)
    }
}

/// This whole decoder is a CAUSAL/streaming-capable variant, NOT the plain
/// symmetric-padding DAC that candle_transformers::models::dac implements -
/// verified directly from the ONNX graph's own computed Pad "pads" tensors
/// (via onnx.load + walking every Pad node's actual runtime pads input, not
/// inferred from kernel size): every kernel-7 conv pads [left=(kernel-1)*
/// dilation, right=0] (e.g. dilation=1 -> [6,0], dilation=3 -> [18,0],
/// dilation=9 -> [54,0]); kernel-1 convs pad [0,0] (no padding needed).
/// candle_nn::Conv1dConfig only supports symmetric padding, so left-only
/// padding is applied manually via Tensor::pad_with_zeros before a
/// padding:0 conv.
struct CausalConv1d {
    inner: Conv1d,
    left_pad: usize,
}

impl CausalConv1d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = if self.left_pad > 0 {
            xs.pad_with_zeros(D::Minus1, self.left_pad, 0)?
        } else {
            xs.clone()
        };
        self.inner.forward(&xs)
    }
}

fn causal_conv1d_plain(
    in_c: usize,
    out_c: usize,
    kernel: usize,
    dilation: usize,
    groups: usize,
    vb: VarBuilder,
) -> Result<CausalConv1d> {
    let weight = vb.get((out_c, in_c / groups, kernel), "weight")?;
    let bias = vb.get(out_c, "bias")?;
    let cfg = Conv1dConfig {
        dilation,
        groups,
        padding: 0,
        ..Default::default()
    };
    let inner = Conv1d::new(weight, Some(bias), cfg);
    let left_pad = (kernel - 1) * dilation;
    Ok(CausalConv1d { inner, left_pad })
}

/// ConvTranspose1d with padding=0 at the op level, then trimmed by
/// `kernel - stride` samples off the END to produce a causal, exactly
/// `T_in * stride`-length output - verified directly from the ONNX graph's
/// Slice node immediately following every ConvTranspose (start=0,
/// end=T_in*stride), not inferred: raw ConvTranspose output length is
/// (T_in-1)*stride+kernel = T_in*stride + (kernel-stride), and the graph
/// slices off exactly that trailing excess.
struct CausalConvTranspose1d {
    inner: ConvTranspose1d,
    trim: usize,
}

impl CausalConvTranspose1d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let ys = self.inner.forward(xs)?;
        if self.trim > 0 {
            let keep = ys.dim(D::Minus1)? - self.trim;
            ys.narrow(D::Minus1, 0, keep)
        } else {
            Ok(ys)
        }
    }
}

fn causal_conv_transpose1d_plain(
    in_c: usize,
    out_c: usize,
    kernel: usize,
    stride: usize,
    vb: VarBuilder,
) -> Result<CausalConvTranspose1d> {
    let weight = vb.get((in_c, out_c, kernel), "weight")?;
    let bias = vb.get(out_c, "bias")?;
    let cfg = ConvTranspose1dConfig {
        stride,
        padding: 0,
        ..Default::default()
    };
    let inner = ConvTranspose1d::new(weight, Some(bias), cfg);
    let trim = kernel - stride;
    Ok(CausalConvTranspose1d { inner, trim })
}

struct ResidualUnit {
    snake1: Snake1d,
    conv1: CausalConv1d,
    snake2: Snake1d,
    conv2: CausalConv1d,
}

impl ResidualUnit {
    fn new(dim: usize, dilation: usize, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("block");
        let snake1 = Snake1d::new(dim, vb.pp(0))?;
        let conv1 = causal_conv1d_plain(dim, dim, 7, dilation, 1, vb.pp(1).pp("conv"))?;
        let snake2 = Snake1d::new(dim, vb.pp(2))?;
        let conv2 = causal_conv1d_plain(dim, dim, 1, 1, 1, vb.pp(3).pp("conv"))?;
        Ok(Self {
            snake1,
            conv1,
            snake2,
            conv2,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let ys = self.conv1.forward(&xs.apply(&self.snake1)?)?;
        let ys = self.conv2.forward(&ys.apply(&self.snake2)?)?;
        // causal padding keeps output length == input length, so no length
        // reconciliation is needed here (unlike the symmetric-padding DAC's
        // narrow-and-add trick in candle_transformers::models::dac).
        ys + xs
    }
}

struct DecoderBlock {
    snake1: Snake1d,
    conv_tr1: CausalConvTranspose1d,
    res1: ResidualUnit,
    res2: ResidualUnit,
    res3: ResidualUnit,
}

impl DecoderBlock {
    fn new(in_dim: usize, out_dim: usize, stride: usize, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("block");
        let snake1 = Snake1d::new(in_dim, vb.pp(0))?;
        let conv_tr1 = causal_conv_transpose1d_plain(in_dim, out_dim, 2 * stride, stride, vb.pp(1).pp("conv"))?;
        let res1 = ResidualUnit::new(out_dim, 1, vb.pp(2))?;
        let res2 = ResidualUnit::new(out_dim, 3, vb.pp(3))?;
        let res3 = ResidualUnit::new(out_dim, 9, vb.pp(4))?;
        Ok(Self {
            snake1,
            conv_tr1,
            res1,
            res2,
            res3,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.apply(&self.snake1)?;
        let xs = self.conv_tr1.forward(&xs)?;
        let xs = self.res1.forward(&xs)?;
        let xs = self.res2.forward(&xs)?;
        self.res3.forward(&xs)
    }
}

struct DacDecoder {
    conv1: CausalConv1d,
    blocks: Vec<DecoderBlock>,
    snake1: Snake1d,
    conv2: CausalConv1d,
}

impl DacDecoder {
    fn new(in_c: usize, mut channels: usize, rates: &[usize], d_out: usize, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("model");
        let conv1 = causal_conv1d_plain(in_c, channels, 7, 1, 1, vb.pp(0).pp("conv"))?;
        let mut blocks = Vec::with_capacity(rates.len());
        for (idx, stride) in rates.iter().enumerate() {
            let block = DecoderBlock::new(channels, channels / 2, *stride, vb.pp(idx + 1))?;
            channels /= 2;
            blocks.push(block);
        }
        let snake1 = Snake1d::new(channels, vb.pp(rates.len() + 1))?;
        let conv2 = causal_conv1d_plain(channels, d_out, 7, 1, 1, vb.pp(rates.len() + 2).pp("conv"))?;
        Ok(Self {
            conv1,
            blocks,
            snake1,
            conv2,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = self.conv1.forward(xs)?;
        for block in self.blocks.iter() {
            xs = block.forward(&xs)?;
        }
        let xs = xs.apply(&self.snake1)?;
        self.conv2.forward(&xs)?.tanh()
    }
}

// ---------------------------------------------------------------------
// RVQ dequantization (semantic + acoustic codebooks)
// ---------------------------------------------------------------------

struct VectorQuantizer {
    codebook: Tensor, // [cb_size, cb_dim]
    out_proj_w: Tensor, // [out_c, cb_dim, 1] conv weight
    out_proj_b: Tensor, // [out_c]
}

impl VectorQuantizer {
    fn new(cb_size: usize, cb_dim: usize, out_c: usize, vb: VarBuilder) -> Result<Self> {
        let codebook = vb.get((cb_size, cb_dim), "codebook.weight")?;
        let out_proj_w = vb.get((out_c, cb_dim, 1), "out_proj.weight")?;
        let out_proj_b = vb.get(out_c, "out_proj.bias")?;
        Ok(Self {
            codebook,
            out_proj_w,
            out_proj_b,
        })
    }

    /// codes_row: [batch, frames] int64 indices -> [batch, out_c, frames]
    fn decode(&self, codes_row: &Tensor) -> Result<Tensor> {
        // embedding lookup: [batch, frames, cb_dim]
        let embed = self.codebook.index_select(&codes_row.flatten_all()?, 0)?;
        let (b, t) = codes_row.dims2()?;
        let cb_dim = self.codebook.dim(1)?;
        let embed = embed.reshape((b, t, cb_dim))?.transpose(1, 2)?.contiguous()?; // [b, cb_dim, t]
        // out_proj is a 1x1 conv == matmul over channel dim
        let out_c = self.out_proj_w.dim(0)?;
        let w = self.out_proj_w.reshape((out_c, cb_dim))?; // [out_c, cb_dim]
        // [b, cb_dim, t] -> [b, t, cb_dim] @ w^T -> [b, t, out_c] -> [b, out_c, t]
        let embed_bt_c = embed.transpose(1, 2)?.contiguous()?;
        let out = embed_bt_c.broadcast_matmul(&w.t()?)?; // [b, t, out_c]
        let out = out.broadcast_add(&self.out_proj_b)?;
        out.transpose(1, 2)?.contiguous()
    }
}

struct ResidualVectorQuantizerDecoder {
    semantic: VectorQuantizer,
    acoustic: Vec<VectorQuantizer>,
}

impl ResidualVectorQuantizerDecoder {
    fn new(vb: VarBuilder) -> Result<Self> {
        let semantic = VectorQuantizer::new(4096, 8, HIDDEN, vb.pp("semantic_quantizer.quantizers.0"))?;
        let mut acoustic = Vec::with_capacity(N_ACOUSTIC_CODEBOOKS);
        for i in 0..N_ACOUSTIC_CODEBOOKS {
            acoustic.push(VectorQuantizer::new(
                1024,
                8,
                HIDDEN,
                vb.pp(format!("quantizer.quantizers.{i}")),
            )?);
        }
        Ok(Self { semantic, acoustic })
    }

    /// codes: [batch, 10, frames] int64 -> [batch, HIDDEN, frames]
    fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let semantic_codes = codes.i((.., 0))?.contiguous()?; // [b, frames]
        let mut sum = self.semantic.decode(&semantic_codes)?;
        for i in 0..N_ACOUSTIC_CODEBOOKS {
            let row = codes.i((.., i + 1))?.contiguous()?;
            let z = self.acoustic[i].decode(&row)?;
            sum = (sum + z)?;
        }
        Ok(sum)
    }
}

// ---------------------------------------------------------------------
// post_module: 8-layer GQA transformer with RoPE, RMSNorm, SwiGLU FFN,
// per-branch layer-scale (gamma). Non-causal (single unpadded sequence).
// ---------------------------------------------------------------------

fn rms_norm(xs: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let dtype = xs.dtype();
    let xs32 = xs.to_dtype(DType::F32)?;
    let variance = xs32.sqr()?.mean_keepdim(D::Minus1)?;
    let xs32 = xs32.broadcast_div(&(variance + eps)?.sqrt()?)?;
    let xs = xs32.to_dtype(dtype)?;
    xs.broadcast_mul(weight)
}

struct Attention {
    wqkv: Tensor, // [hidden, q_dim+k_dim+v_dim]
    wo: Tensor,   // [q_dim, hidden]
}

impl Attention {
    fn new(vb: VarBuilder) -> Result<Self> {
        let q_dim = N_HEADS_Q * HEAD_DIM;
        let kv_dim = N_HEADS_KV * HEAD_DIM;
        let wqkv = vb.get((HIDDEN, q_dim + 2 * kv_dim), "wqkv.weight")?;
        let wo = vb.get((q_dim, HIDDEN), "wo.weight")?;
        Ok(Self { wqkv, wo })
    }

    fn forward(&self, xs: &Tensor, cos: &Tensor, sin: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let (b, t, _h) = xs.dims3()?;
        let q_dim = N_HEADS_Q * HEAD_DIM;
        let kv_dim = N_HEADS_KV * HEAD_DIM;
        let qkv = xs.broadcast_matmul(&self.wqkv)?; // [b, t, q_dim+2*kv_dim]
        let q = qkv.narrow(D::Minus1, 0, q_dim)?;
        let k = qkv.narrow(D::Minus1, q_dim, kv_dim)?;
        let v = qkv.narrow(D::Minus1, q_dim + kv_dim, kv_dim)?;

        let q = q.reshape((b, t, N_HEADS_Q, HEAD_DIM))?.transpose(1, 2)?.contiguous()?; // [b, hq, t, d]
        let k = k.reshape((b, t, N_HEADS_KV, HEAD_DIM))?.transpose(1, 2)?.contiguous()?;
        let v = v.reshape((b, t, N_HEADS_KV, HEAD_DIM))?.transpose(1, 2)?.contiguous()?;

        let q = apply_rope(&q, cos, sin)?;
        let k = apply_rope(&k, cos, sin)?;

        // GQA: repeat kv heads to match q heads
        let n_rep = N_HEADS_Q / N_HEADS_KV;
        let k = repeat_kv(&k, n_rep)?;
        let v = repeat_kv(&v, n_rep)?;

        let scale = 1.0 / (HEAD_DIM as f64).sqrt();
        let attn = (q.contiguous()?.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let attn = attn.broadcast_add(mask)?;
        let attn = candle_nn::ops::softmax(&attn, D::Minus1)?;
        let out = attn.matmul(&v.contiguous()?)?; // [b, hq, t, d]
        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, t, q_dim))?;
        out.broadcast_matmul(&self.wo)
    }
}

fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (b, h, t, d) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((b, h, n_rep, t, d))?
        .reshape((b, h * n_rep, t, d))
}

/// Applies RoPE matching the ONNX graph's exact even/odd-split ("interleaved
/// pairs at absolute positions, not rotate-half") pattern: for each pair
/// (x[2i], x[2i+1]) at frequency i, rotate by (cos_i, sin_i).
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    // x: [b, h, t, d], cos/sin: [t, d/2]
    let (b, h, t, d) = x.dims4()?;
    let half = d / 2;
    let x = x.reshape((b, h, t, half, 2))?;
    let x0 = x.i((.., .., .., .., 0))?.contiguous()?; // [b,h,t,half]
    let x1 = x.i((.., .., .., .., 1))?.contiguous()?;
    let cos = cos.reshape((1, 1, t, half))?;
    let sin = sin.reshape((1, 1, t, half))?;
    let o0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
    let o1 = (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?;
    let o = Tensor::stack(&[o0, o1], D::Minus1)?; // [b,h,t,half,2]
    o.reshape((b, h, t, d))
}

struct FeedForward {
    w1: Tensor, // [hidden, ffn]
    w2: Tensor, // [ffn, hidden]
    w3: Tensor, // [hidden, ffn]
}

impl FeedForward {
    fn new(ffn_dim: usize, vb: VarBuilder) -> Result<Self> {
        let w1 = vb.get((HIDDEN, ffn_dim), "w1.weight")?;
        let w2 = vb.get((ffn_dim, HIDDEN), "w2.weight")?;
        let w3 = vb.get((HIDDEN, ffn_dim), "w3.weight")?;
        Ok(Self { w1, w2, w3 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let a = xs.broadcast_matmul(&self.w1)?.silu()?;
        let b = xs.broadcast_matmul(&self.w3)?;
        (a * b)?.broadcast_matmul(&self.w2)
    }
}

struct TransformerLayer {
    attention_norm_w: Tensor,
    attention: Attention,
    attention_layer_scale: Tensor,
    ffn_norm_w: Tensor,
    feed_forward: FeedForward,
    ffn_layer_scale: Tensor,
}

impl TransformerLayer {
    fn new(ffn_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attention_norm_w: vb.get(HIDDEN, "attention_norm.weight")?,
            attention: Attention::new(vb.pp("attention"))?,
            attention_layer_scale: vb.get(HIDDEN, "attention_layer_scale.gamma")?,
            ffn_norm_w: vb.get(HIDDEN, "ffn_norm.weight")?,
            feed_forward: FeedForward::new(ffn_dim, vb.pp("feed_forward"))?,
            ffn_layer_scale: vb.get(HIDDEN, "ffn_layer_scale.gamma")?,
        })
    }

    fn forward(&self, xs: &Tensor, cos: &Tensor, sin: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let normed = rms_norm(xs, &self.attention_norm_w, 1e-5)?;
        let attn_out = self.attention.forward(&normed, cos, sin, mask)?;
        let xs = (xs + attn_out.broadcast_mul(&self.attention_layer_scale)?)?;

        let normed = rms_norm(&xs, &self.ffn_norm_w, 1e-5)?;
        let ffn_out = self.feed_forward.forward(&normed)?;
        xs + ffn_out.broadcast_mul(&self.ffn_layer_scale)?
    }
}

struct PostModuleTransformer {
    layers: Vec<TransformerLayer>,
    final_norm_w: Tensor,
}

impl PostModuleTransformer {
    fn new(vb: VarBuilder) -> Result<Self> {
        // layers 0-6 use ffn_dim=1216 (w1/w3: [1024,1216]); layer 7 uses 4096
        // (per the ONNX weight dump: onnx::MatMul_4520/4527 are [1024,4096] -
        // those belong to the ConvNeXt upsample pwconv1, NOT layer 7's FFN;
        // layer 7's own w1/w3 (MatMul_4514-adjacent) are also 1216-dim,
        // consistent with all 8 layers sharing the same FFN width).
        let mut layers = Vec::with_capacity(N_TRANSFORMER_LAYERS);
        for i in 0..N_TRANSFORMER_LAYERS {
            layers.push(TransformerLayer::new(1216, vb.pp(format!("layers.{i}")))?);
        }
        let final_norm_w = vb.get(HIDDEN, "norm.weight")?;
        Ok(Self { layers, final_norm_w })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // xs: [b, HIDDEN, t] (channel-first, as produced by the RVQ sum) ->
        // transformer operates channel-last [b, t, HIDDEN].
        let xs = xs.transpose(1, 2)?.contiguous()?;
        let t = xs.dim(1)?;
        let (cos, sin) = rope_cos_sin(t, HEAD_DIM, ROPE_BASE, xs.device())?;
        // Causal sliding-window mask (window=128, verified from the ONNX
        // graph's own And(LessOrEqual(j,i), GreaterOrEqual(j, i-127)) - at
        // our sequence lengths (<128 frames) the window bound never
        // triggers, so this reduces to standard causal masking, but the
        // window is implemented in full for correctness at longer inputs.
        let mask = causal_sliding_window_mask(t, 128, xs.device())?;
        let mut xs = xs;
        for layer in &self.layers {
            xs = layer.forward(&xs, &cos, &sin, &mask)?;
        }
        let xs = rms_norm(&xs, &self.final_norm_w, 1e-5)?;
        xs.transpose(1, 2)?.contiguous() // back to [b, HIDDEN, t]
    }
}

/// mask[i,j] = 0.0 if j <= i AND j >= max(0, i-window+1), else -inf.
/// Shape [1, 1, t, t] for broadcasting over [b, h, t, t] attention scores.
fn causal_sliding_window_mask(t: usize, window: usize, device: &Device) -> Result<Tensor> {
    let mut data = vec![0f32; t * t];
    for i in 0..t {
        let lo = i.saturating_sub(window - 1);
        for j in 0..t {
            if j > i || j < lo {
                data[i * t + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(data, (1, 1, t, t), device)
}

fn rope_cos_sin(seq_len: usize, head_dim: usize, base: f64, device: &Device) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1f32 / (base as f32).powf(2.0 * i as f32 / head_dim as f32))
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq, half, device)?;
    let t: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
    let t = Tensor::from_vec(t, seq_len, device)?;
    let freqs = t.reshape((seq_len, 1))?.broadcast_mul(&inv_freq.reshape((1, half))?)?; // [t, half]
    Ok((freqs.cos()?, freqs.sin()?))
}

// ---------------------------------------------------------------------
// ConvNeXt-style upsample blocks (quantizer.upsample.{0,1})
// ---------------------------------------------------------------------

struct LayerNorm1d {
    weight: Tensor,
    bias: Tensor,
}

impl LayerNorm1d {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get(dim, "weight")?,
            bias: vb.get(dim, "bias")?,
        })
    }

    /// xs: [.., dim] (channel-last)
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mean = xs.mean_keepdim(D::Minus1)?;
        let xs_c = xs.broadcast_sub(&mean)?;
        let var = xs_c.sqr()?.mean_keepdim(D::Minus1)?;
        let xs_n = xs_c.broadcast_div(&(var + 1e-5)?.sqrt()?)?;
        xs_n.broadcast_mul(&self.weight)?.broadcast_add(&self.bias)
    }
}

struct ConvNextBlock {
    dwconv: CausalConv1d, // depthwise, groups=dim, causal-padded
    norm: LayerNorm1d,
    pwconv1_w: Tensor, // [dim, 4*dim]
    pwconv1_b: Tensor,
    pwconv2_w: Tensor, // [4*dim, dim]
    pwconv2_b: Tensor,
    gamma: Tensor,
}

impl ConvNextBlock {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let dwconv = causal_conv1d_plain(dim, dim, 7, 1, dim, vb.pp("dwconv").pp("conv"))?;
        let norm = LayerNorm1d::new(dim, vb.pp("norm"))?;
        let hidden = dim * 4;
        let pwconv1_w = vb.get((dim, hidden), "pwconv1.weight")?;
        let pwconv1_b = vb.get(hidden, "pwconv1.bias")?;
        let pwconv2_w = vb.get((hidden, dim), "pwconv2.weight")?;
        let pwconv2_b = vb.get(dim, "pwconv2.bias")?;
        let gamma = vb.get(dim, "gamma")?;
        Ok(Self {
            dwconv,
            norm,
            pwconv1_w,
            pwconv1_b,
            pwconv2_w,
            pwconv2_b,
            gamma,
        })
    }

    /// xs: [b, dim, t] channel-first
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward_debug(xs, &mut |_, _| {})
    }

    fn forward_debug(&self, xs: &Tensor, probe: &mut dyn FnMut(&str, &Tensor)) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = self.dwconv.forward(xs)?; // [b, dim, t]
        probe("dwconv", &xs);
        let xs = xs.transpose(1, 2)?.contiguous()?; // [b, t, dim]
        let xs = self.norm.forward(&xs)?;
        probe("norm", &xs);
        let xs = xs.broadcast_matmul(&self.pwconv1_w)?.broadcast_add(&self.pwconv1_b)?;
        probe("pwconv1", &xs);
        let xs = xs.gelu_erf()?;
        let xs = xs.broadcast_matmul(&self.pwconv2_w)?.broadcast_add(&self.pwconv2_b)?;
        probe("pwconv2", &xs);
        let xs = xs.broadcast_mul(&self.gamma)?;
        let xs = xs.transpose(1, 2)?.contiguous()?; // [b, dim, t]
        residual + xs
    }
}

struct UpsampleStage {
    conv_tr: CausalConvTranspose1d,
    convnext: ConvNextBlock,
}

impl UpsampleStage {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let conv_tr = causal_conv_transpose1d_plain(dim, dim, 2, 2, vb.pp(0).pp("conv"))?;
        let convnext = ConvNextBlock::new(dim, vb.pp(1))?;
        Ok(Self { conv_tr, convnext })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.conv_tr.forward(xs)?;
        self.convnext.forward(&xs)
    }

    fn forward_debug(&self, xs: &Tensor, probe: &mut dyn FnMut(&str, &Tensor)) -> Result<Tensor> {
        let xs = self.conv_tr.forward(xs)?;
        probe("conv_tr", &xs);
        self.convnext.forward_debug(&xs, probe)
    }
}

// ---------------------------------------------------------------------
// Top-level codec decoder
// ---------------------------------------------------------------------

pub struct CodecDecoder {
    rvq: ResidualVectorQuantizerDecoder,
    post_module: PostModuleTransformer,
    upsample0: UpsampleStage,
    upsample1: UpsampleStage,
    decoder: DacDecoder,
}

impl CodecDecoder {
    pub fn new(vb: VarBuilder) -> Result<Self> {
        let rvq = ResidualVectorQuantizerDecoder::new(vb.pp("quantizer"))?;
        let post_module = PostModuleTransformer::new(vb.pp("quantizer.post_module"))?;
        let upsample0 = UpsampleStage::new(HIDDEN, vb.pp("quantizer.upsample.0"))?;
        let upsample1 = UpsampleStage::new(HIDDEN, vb.pp("quantizer.upsample.1"))?;
        let decoder = DacDecoder::new(HIDDEN, 1536, &[8, 8, 4, 2], 1, vb.pp("decoder"))?;
        Ok(Self {
            rvq,
            post_module,
            upsample0,
            upsample1,
            decoder,
        })
    }

    /// codes: [batch, 10, frames] int64 -> audio: [batch, 1, samples]
    pub fn forward(&self, codes: &Tensor) -> Result<Tensor> {
        let xs = self.rvq.decode(codes)?; // [b, HIDDEN, frames]
        let xs = self.post_module.forward(&xs)?; // [b, HIDDEN, frames]
        let xs = self.upsample0.forward(&xs)?; // [b, HIDDEN, frames*2]
        let xs = self.upsample1.forward(&xs)?; // [b, HIDDEN, frames*4]
        self.decoder.forward(&xs) // [b, 1, frames*4*256]
    }

    /// Same as forward(), but calls `probe(stage_name, tensor)` after each
    /// major stage - for comparing against ONNX intermediate dumps.
    pub fn forward_debug(&self, codes: &Tensor, probe: &mut dyn FnMut(&str, &Tensor)) -> Result<Tensor> {
        let xs = self.rvq.decode(codes)?;
        probe("rvq_sum", &xs);
        let xs = self.post_module.forward(&xs)?;
        probe("post_module", &xs);
        let xs = self.upsample0.forward_debug(&xs, &mut |name, t| probe(&format!("upsample0_{name}"), t))?;
        probe("upsample0", &xs);
        let xs = self.upsample1.forward(&xs)?;
        probe("upsample1", &xs);
        let dec_conv1 = self.decoder.conv1.forward(&xs)?;
        probe("decoder_conv1", &dec_conv1);
        self.decoder.forward(&xs)
    }
}

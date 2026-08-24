//! Real verification for candle-weight-repacking-quantization: loads the
//! .gguf files produced by repack_quantized_weights via candle_core's own
//! quantized::gguf_file::Content::read + .tensor(), prints shapes/dtypes for
//! a set of key layers, and cross-checks them against
//! onnx_runtime/model/runtime_manifest.json's documented architecture
//! (24 slow layers, 4 fast layers, width 896, 14 attention heads, 2 KV
//! heads, head_dim 64, 10 codebooks of 4096 entries each). Also dequantizes
//! a couple of Q4_0 tensors back to F32 and compares against the original
//! extracted FP32 values to report the real requantization error
//! introduced by the dequant-then-Q4_0-requant path (the documented
//! precision tradeoff), not just "it loaded".

use candle_core::quantized::gguf_file::Content;
use candle_core::{DType, Device, Tensor};
use std::fs::File;
use std::path::{Path, PathBuf};

const DIM: usize = 896;
const N_HEAD: usize = 14;
const N_LOCAL_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const N_LAYER: usize = 24;
const N_FAST_LAYER: usize = 4;
const FAST_DIM: usize = 896;
const CODEBOOK_SIZE: usize = 4096;
const NUM_CODEBOOKS: usize = 10;
const VOCAB_SIZE: usize = 155776;

fn check(cond: bool, msg: &str) {
    if cond {
        println!("  [OK] {msg}");
    } else {
        println!("  [FAIL] {msg}");
        std::process::exit(1);
    }
}

fn load_and_report(path: &Path, device: &Device, checks: &[(&str, Vec<usize>)]) -> anyhow::Result<()> {
    println!("=== loading {path:?} ===");
    let mut file = File::open(path)?;
    let content = Content::read(&mut file)?;
    println!("  tensor count in file: {}", content.tensor_infos.len());

    for (name, expected_shape) in checks {
        let info = content
            .tensor_infos
            .get(*name)
            .unwrap_or_else(|| panic!("tensor {name} not found in {path:?}"));
        let qtensor = content.tensor(&mut file, name, device)?;
        let dims = qtensor.shape().dims().to_vec();
        println!(
            "  {name}: shape={:?} dtype={:?} (gguf ggml_dtype={:?})",
            dims,
            qtensor.dtype(),
            info.ggml_dtype
        );
        check(&dims == expected_shape, &format!("{name} shape matches expected {expected_shape:?}"));
    }
    Ok(())
}

fn requant_error_report(path: &Path, device: &Device, tensor_name: &str, npz_path: &Path) -> anyhow::Result<()> {
    println!("=== requantization error check: {tensor_name} ===");
    let mut file = File::open(path)?;
    let content = Content::read(&mut file)?;
    let qtensor = content.tensor(&mut file, tensor_name, device)?;
    let dequant = qtensor.dequantize(device)?.to_dtype(DType::F32)?;

    let originals: Vec<(String, Tensor)> = Tensor::read_npz(npz_path)?;
    let original = originals
        .iter()
        .find(|(n, _)| n == tensor_name)
        .map(|(_, t)| t.clone())
        .expect("original tensor not found in npz");
    let original = original.to_dtype(DType::F32)?;

    let diff = (&dequant - &original)?.abs()?;
    let max_abs_diff: f32 = diff.max_all()?.to_scalar()?;
    let mean_abs_diff: f32 = diff.mean_all()?.to_scalar()?;
    let mean_abs_val: f32 = original.abs()?.mean_all()?.to_scalar()?;
    let rel_error = mean_abs_diff / mean_abs_val.max(1e-9);

    println!("  original mean_abs={mean_abs_val:.6}");
    println!("  requant  max_abs_diff={max_abs_diff:.6} mean_abs_diff={mean_abs_diff:.6} rel_error={rel_error:.4}");
    check(max_abs_diff.is_finite() && mean_abs_diff.is_finite(), "requant error is finite (no NaN/Inf)");
    check(rel_error < 0.15, "mean relative requantization error < 15% (int4 weight-only quant, expected lossy)");
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let root: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("weights");

    let qkv_dim = (N_HEAD + 2 * N_LOCAL_HEADS) * HEAD_DIM; // 1152
    let fast_qkv_dim = (N_HEAD + 2 * N_LOCAL_HEADS) * HEAD_DIM; // fast attn uses same head config per config.json

    // --- slow_ar_q4_0.gguf ---
    load_and_report(
        &root.join("slow_ar_q4_0.gguf"),
        &device,
        &[
            ("embeddings.weight", vec![VOCAB_SIZE, DIM]),
            ("codebook_embeddings.weight", vec![CODEBOOK_SIZE * NUM_CODEBOOKS, DIM]),
            ("layers.0.attention.wqkv.weight", vec![qkv_dim, DIM]),
            ("layers.0.attention.wo.weight", vec![N_HEAD * HEAD_DIM, DIM]),
            ("layers.0.feed_forward.w1.weight", vec![4864, DIM]),
            ("layers.0.feed_forward.w2.weight", vec![DIM, 4864]),
            ("layers.0.feed_forward.w3.weight", vec![4864, DIM]),
            ("layers.0.attention_norm.weight", vec![DIM]),
            ("layers.0.ffn_norm.weight", vec![DIM]),
            ("layers.23.attention.wqkv.weight", vec![qkv_dim, DIM]),
            ("layers.23.feed_forward.w2.weight", vec![DIM, 4864]),
            ("norm.weight", vec![DIM]),
            ("lm_head.weight", vec![4097, DIM]),
        ],
    )?;
    // confirm all 24 layers present
    {
        let mut file = File::open(root.join("slow_ar_q4_0.gguf"))?;
        let content = Content::read(&mut file)?;
        for i in 0..N_LAYER {
            let key = format!("layers.{i}.attention.wqkv.weight");
            check(content.tensor_infos.contains_key(&key), &format!("layer {i} present ({key})"));
        }
        check(!content.tensor_infos.contains_key(&format!("layers.{N_LAYER}.attention.wqkv.weight")), "no layer 24 (exactly N_LAYER=24 layers)");
    }

    // --- fast_ar_q4_0.gguf ---
    load_and_report(
        &root.join("fast_ar_q4_0.gguf"),
        &device,
        &[
            ("fast_embeddings.weight", vec![CODEBOOK_SIZE, FAST_DIM]),
            ("fast_layers.0.attention.wqkv.weight", vec![fast_qkv_dim, FAST_DIM]),
            ("fast_layers.0.attention.wo.weight", vec![N_HEAD * HEAD_DIM, FAST_DIM]),
            ("fast_layers.3.feed_forward.w1.weight", vec![4864, FAST_DIM]),
            ("fast_norm.weight", vec![FAST_DIM]),
            ("fast_output.weight", vec![CODEBOOK_SIZE, FAST_DIM]),
        ],
    )?;
    {
        let mut file = File::open(root.join("fast_ar_q4_0.gguf"))?;
        let content = Content::read(&mut file)?;
        for i in 0..N_FAST_LAYER {
            let key = format!("fast_layers.{i}.attention.wqkv.weight");
            check(content.tensor_infos.contains_key(&key), &format!("fast layer {i} present ({key})"));
        }
        check(!content.tensor_infos.contains_key(&format!("fast_layers.{N_FAST_LAYER}.attention.wqkv.weight")), "no fast layer 4 (exactly N_FAST_LAYER=4 layers)");
    }

    // --- requantization error, a few representative tensors ---
    requant_error_report(
        &root.join("slow_ar_q4_0.gguf"),
        &device,
        "layers.0.attention.wqkv.weight",
        &root.join("slow_ar_fp32.npz"),
    )?;
    requant_error_report(
        &root.join("slow_ar_q4_0.gguf"),
        &device,
        "layers.0.feed_forward.w2.weight",
        &root.join("slow_ar_fp32.npz"),
    )?;
    requant_error_report(
        &root.join("fast_ar_q4_0.gguf"),
        &device,
        "fast_output.weight",
        &root.join("fast_ar_fp32.npz"),
    )?;

    println!("\nALL CHECKS PASSED.");
    Ok(())
}

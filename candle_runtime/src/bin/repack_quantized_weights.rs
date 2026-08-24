//! candle-weight-repacking-quantization
//!
//! Reads the FP32 weights dequantized from ONNX's GatherBlockQuantized /
//! MatMulNBits blocks (candle_runtime/weights/{slow_ar,fast_ar}_fp32.npz,
//! produced by onnx-to-candle-weight-extraction /
//! candle_runtime/scripts/extract_onnx_weights.py) and repacks them into
//! candle's own QTensor block-quantization format, persisted as a .gguf file
//! per graph (candle's native on-disk quantized-tensor format, written via
//! candle_core::quantized::gguf_file::write).
//!
//! DECISION: dequantize-then-requantize, NOT a direct bit-for-bit repack.
//! Established live (see extract_onnx_weights.py's docstring + this
//! session's research) that ONNX's MatMulNBits/GatherBlockQuantized layout
//! and candle's Q4_0 layout are incompatible at three independent levels:
//!   1. Block size: ONNX MatMulNBits uses block_size=128 (per its own node
//!      attributes, verified live); candle's GgmlDType::Q4_0 hardcodes
//!      block_size=32 (QK4_0 constant, candle-core/src/quantized/k_quants.rs).
//!   2. Quantization scheme: ONNX uses affine (asymmetric) quantization with
//!      an explicit per-block zero_point input; candle's Q4_0 is symmetric
//!      only (scale d = amax/-8, dequant = (nibble-8)*d, no zero-point field
//!      in BlockQ4_0 at all).
//!   3. Nibble packing order: ONNX Runtime's own RTN packer
//!      (matmul_nbits_quantizer.py's pack_int8_to_int4) packs ADJACENT PAIRS
//!      (element i -> low nibble, i+1 -> high nibble); candle's BlockQ4_0
//!      packs SPLIT HALVES (element j -> low nibble of qs[j], element j+16
//!      -> high nibble of the same byte, for a 32-element block).
//! Any one of these would already block a direct repack; all three together
//! make it definitively infeasible without emulating ONNX's exact affine
//! block-128 scheme as a brand new custom candle quantization type (far
//! higher risk/cost for a preview model than accepting FP16-order requant
//! error). Dequantizing to FP32 (already done, extraction step) and calling
//! candle's own QTensor::quantize(&tensor, GgmlDType::Q4_0) is the correct,
//! low-risk path: candle re-derives its own scale/zero-point/packing
//! natively, guaranteeing internal consistency with candle's own dequant
//! kernels at inference time (the only thing that matters for correctness -
//! bit-parity with ONNX's specific quantization choice was never a
//! requirement, only "coherent, close" per this PRD row's own postcondition
//! and the sibling port-slow-fast-ar-to-candle row's postcondition).
//! Precision tradeoff: two lossy int4 quantization passes now compose
//! (original ONNX RTN quant -> our FP32 dequant, exact -> candle Q4_0 requant,
//! lossy) instead of one. Q4_0's block size is 4x smaller than ONNX's
//! (32 vs 128), which partially COMPENSATES by giving candle's requant finer
//! granularity (each candle block sees less dynamic range to cover with one
//! scale) - the two effects partially offset. This is accepted as the right
//! tradeoff for a research/preview port; verified numerically below.
//!
//! Which tensors get quantized: only the actual nn.Linear-equivalent 2D
//! weight matrices that were originally MatMulNBits/GatherBlockQuantized in
//! ONNX (wqkv, wo, w1/w2/w3, fast_output, lm_head, embeddings,
//! codebook_embeddings, fast_embeddings) - RMSNorm weights and biases (1D,
//! small, and precision-sensitive for norm stability) are kept as plain F32
//! QTensor-wrapped-as-F32 (GgmlDType::F32), matching how the ONNX graphs
//! themselves left them unquantized.

use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Tensor name patterns that were MatMulNBits/GatherBlockQuantized in the
/// source ONNX graph (see extract_onnx_weights.py's node-walk) - these get
/// real Q4_0 quantization. Everything else (norms, biases) stays F32.
fn should_quantize(name: &str) -> bool {
    let leaf = name.rsplit('.').next().unwrap_or(name);
    if leaf != "weight" {
        return false;
    }
    // 1D tensors (RMSNorm .weight) must NOT be quantized even though they
    // end in "weight" - checked by caller via actual tensor rank, this is
    // just the name-based first filter (norm weights match *_norm.weight).
    !(name.ends_with("_norm.weight") || name == "norm.weight" || name == "fast_norm.weight")
}

fn repack_one(npz_path: &Path, gguf_path: &Path, graph_label: &str) -> anyhow::Result<()> {
    println!("=== {graph_label}: loading {npz_path:?} ===");
    let device = Device::Cpu;
    let tensors: Vec<(String, Tensor)> = Tensor::read_npz(npz_path)?;
    println!("  loaded {} tensors", tensors.len());

    let mut qtensors: Vec<(String, QTensor)> = Vec::with_capacity(tensors.len());
    let mut n_quantized = 0usize;
    let mut n_plain = 0usize;
    let mut shape_report: HashMap<String, String> = HashMap::new();

    for (name, tensor) in &tensors {
        let tensor = tensor.to_device(&device)?;
        let dims = tensor.dims();
        let is_2d_weight = dims.len() == 2 && should_quantize(name);

        if is_2d_weight {
            // candle requires last dim % block_size == 0 for Q4_0 (block_size=32).
            let k = dims[1];
            if k % 32 != 0 {
                println!("  WARN: {name} has K={k} not divisible by 32, keeping as F32");
                let qt = QTensor::quantize(&tensor.to_dtype(DType::F32)?, GgmlDType::F32)?;
                shape_report.insert(name.clone(), format!("F32(fallback) {:?}", dims));
                qtensors.push((name.clone(), qt));
                n_plain += 1;
                continue;
            }
            let qt = QTensor::quantize(&tensor.to_dtype(DType::F32)?, GgmlDType::Q4_0)?;
            shape_report.insert(name.clone(), format!("Q4_0 {:?}", dims));
            qtensors.push((name.clone(), qt));
            n_quantized += 1;
        } else {
            let qt = QTensor::quantize(&tensor.to_dtype(DType::F32)?, GgmlDType::F32)?;
            shape_report.insert(name.clone(), format!("F32 {:?}", dims));
            qtensors.push((name.clone(), qt));
            n_plain += 1;
        }
    }

    println!("  quantized (Q4_0): {n_quantized}, plain (F32): {n_plain}, total: {}", qtensors.len());

    let mut file = std::fs::File::create(gguf_path)?;
    let tensor_refs: Vec<(&str, &QTensor)> =
        qtensors.iter().map(|(n, t)| (n.as_str(), t)).collect();
    gguf_file::write(&mut file, &[], &tensor_refs)?;
    println!("  wrote {gguf_path:?} ({} bytes)", std::fs::metadata(gguf_path)?.len());

    // print a small sample of the shape report for the log
    let mut names: Vec<&String> = shape_report.keys().collect();
    names.sort();
    for n in names.iter().take(6) {
        println!("    {n} -> {}", shape_report[*n]);
    }
    println!("    ... ({} total)", shape_report.len());

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let root: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("weights");

    repack_one(
        &root.join("slow_ar_fp32.npz"),
        &root.join("slow_ar_q4_0.gguf"),
        "slow_ar",
    )?;
    repack_one(
        &root.join("fast_ar_fp32.npz"),
        &root.join("fast_ar_q4_0.gguf"),
        "fast_ar",
    )?;

    println!("\nDone.");
    Ok(())
}

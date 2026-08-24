//! Correctness check for port-slow-fast-ar-to-candle: loads the repacked
//! Q4_0 GGUF weights, builds a real prompt from onnx_runtime/voices/probe_voice
//! (same voice used throughout this project's benchmarking), runs one
//! slow_step + a full fast_step codebook pass, and dumps the resulting
//! top-k semantic logits plus generated codebook tokens as JSON so a
//! companion Python script (running the exact same input through the
//! ONNX reference) can be diffed against it.

use audio8_candle_runtime::model::{FastAr, ManifestCheck, SlowAr, NUM_CODEBOOKS};
use audio8_candle_runtime::prompt::PromptBuilder;
use candle_core::{DType, Device, IndexOp, Tensor};

fn top_k_indices(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    idx.into_iter().take(k).map(|i| (i, logits[i])).collect()
}

fn main() -> anyhow::Result<()> {
    let use_cuda = std::env::var("ARKTTS_CUDA").is_ok();
    let device = if use_cuda {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };
    println!("[model_check] device: {:?}", device);

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let manifest_path = root.join("../onnx_runtime/model/runtime_manifest.json");
    ManifestCheck::verify(&manifest_path)?;
    println!("[model_check] manifest constants verified OK");

    let t0 = std::time::Instant::now();
    let slow = SlowAr::load(&root.join("weights/slow_ar_q4_0.gguf"), &device)?;
    let fast = FastAr::load(&root.join("weights/fast_ar_q4_0.gguf"), &device)?;
    println!("[model_check] weights loaded in {:?}", t0.elapsed());

    let tokenizer_dir = root.join("../onnx_runtime/model/tokenizer");
    let prompt_builder = PromptBuilder::new(&tokenizer_dir, 151678, NUM_CODEBOOKS)?;

    // probe_voice reference codes: [10, 65], stored as uint16 per meta.json.
    let codes_path = root.join("../onnx_runtime/voices/probe_voice/codes.npy");
    let codes_u16 = audio8_candle_runtime::npy::read_npy_u16_2d(&codes_path)?;
    let reference_codes = codes_u16.mapv(|v| v as i64);
    let reference_text = "This is a synthetic probe reference used only for timing benchmarks.";
    let target_text = "The quick brown fox jumps over the lazy dog.";

    let prompt_arr = prompt_builder.build(target_text, reference_text, &reference_codes)?;
    let (b, rows, t) = (prompt_arr.shape()[0], prompt_arr.shape()[1], prompt_arr.shape()[2]);
    println!("[model_check] prompt shape: [{b},{rows},{t}]");

    let flat: Vec<i64> = prompt_arr.iter().copied().collect();
    let prompt = Tensor::from_vec(flat, (b, rows, t), &device)?;

    let mut slow_caches = slow.new_kv_caches(1)?;
    let positions: Vec<usize> = (0..t).collect();

    let t_slow = std::time::Instant::now();
    let (logits, fast_hidden) = slow.slow_step(&prompt, &positions, &mut slow_caches)?;
    println!("[model_check] slow_step took {:?}", t_slow.elapsed());

    let logits_vec: Vec<f32> = logits.i(0)?.to_dtype(DType::F32)?.to_vec1()?;
    println!("[model_check] slow logits shape: {:?} (SLOW_LOGITS_SIZE)", logits_vec.len());
    let top10 = top_k_indices(&logits_vec, 10);
    println!("[model_check] slow logits top-10 (index_within_semantic_then_eos, value):");
    for (i, v) in &top10 {
        println!("    {i}: {v:.4}");
    }
    let nan_count = logits_vec.iter().filter(|x| x.is_nan()).count();
    let max_abs = logits_vec.iter().fold(0f32, |a, &b| a.max(b.abs()));
    println!("[model_check] slow logits nan_count={nan_count} max_abs={max_abs:.4}");

    // Take the argmax semantic-then-eos index as our "sampled" token for the
    // fast AR pass (greedy, not stochastic - deterministic for comparison).
    let semantic_then_eos_argmax = top10[0].0;
    println!("[model_check] greedy semantic_then_eos index: {semantic_then_eos_argmax}");

    let mut fast_caches = fast.new_kv_caches(1)?;
    FastAr::reset_caches(&mut fast_caches)?;

    let t_fast = std::time::Instant::now();
    let first_logits = fast.fast_step(&fast_hidden, 0, &mut fast_caches)?;
    let first_vec: Vec<f32> = first_logits.i(0)?.to_dtype(DType::F32)?.to_vec1()?;
    let first_top = top_k_indices(&first_vec, 5);
    println!("[model_check] fast_step(position=0, using slow_hidden) top-5: {:?}", first_top);

    // Codebook 0 token: derived from the semantic index directly, matching
    // ArkttsModel._generate_codebooks (current = semantic - begin, clamped).
    let codebook_size = 4096i64;
    let semantic_begin_id = 151678i64;
    let semantic_id = semantic_then_eos_argmax as i64 + semantic_begin_id; // approx: only valid if layout=semantic_then_eos and index<end-begin
    let token0 = (semantic_id - semantic_begin_id).clamp(0, codebook_size - 1) as u32;
    let mut codebooks = vec![token0];

    for pos in 1..NUM_CODEBOOKS {
        let tok = *codebooks.last().unwrap();
        let hidden = fast.embed_token(tok)?;
        let logits = fast.fast_step(&hidden, pos, &mut fast_caches)?;
        let vec: Vec<f32> = logits.i(0)?.to_dtype(DType::F32)?.to_vec1()?;
        let argmax = vec.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
        codebooks.push(argmax);
    }
    println!("[model_check] fast AR full pass took {:?}", t_fast.elapsed());
    println!("[model_check] generated codebook tokens (greedy): {:?}", codebooks);

    // Dump JSON for the Python reference comparison script.
    let out = serde_json::json!({
        "device": format!("{:?}", device),
        "prompt_shape": [b, rows, t],
        "slow_logits": logits_vec,
        "slow_top10": top10,
        "fast_first_logits_top5": first_top,
        "codebooks_greedy": codebooks,
    });
    let out_path = root.join("weights/candle_model_check_out.json");
    std::fs::write(&out_path, serde_json::to_string_pretty(&out)?)?;
    println!("[model_check] wrote {:?}", out_path);

    Ok(())
}

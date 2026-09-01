//! End-to-end candle-based synthesis: text -> tokens -> DualAR autoregressive
//! generation -> codec decode -> WAV, mirroring rust_runtime/src/bin/synth.rs's
//! CLI shape (--cuda flag, positional text arg, --repeat for warm-run
//! benchmarking) for direct comparison against the ort-based engine.

use std::path::PathBuf;
use std::time::Instant;

use audio8_candle_runtime::model::{FastAr, SlowAr, IM_END_ID, NUM_CODEBOOKS, SEMANTIC_BEGIN_ID, SEMANTIC_END_ID};
use audio8_candle_runtime::prompt::PromptBuilder;
use audio8_candle_runtime::sampling::sample;
use audio8_candle_runtime::{codec_decoder::CodecDecoder, npy};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn model_dir() -> PathBuf {
    PathBuf::from(std::env::var("ARKTTS_MODEL_DIR").unwrap_or_else(|_| "../onnx_runtime/model".to_string()))
}

fn voices_dir() -> PathBuf {
    PathBuf::from(std::env::var("ARKTTS_VOICES_DIR").unwrap_or_else(|_| "../onnx_runtime/voices".to_string()))
}

fn weights_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("weights")
}

#[derive(serde::Deserialize)]
struct VoiceMeta {
    reference_text: String,
}

fn load_voice(name: &str) -> anyhow::Result<(ndarray::Array2<i64>, VoiceMeta)> {
    let dir = voices_dir().join(name);
    let meta: VoiceMeta = serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json"))?)?;
    let codes_u16 = npy::read_npy_u16_2d(&dir.join("codes.npy"))?;
    Ok((codes_u16.mapv(|v| v as i64), meta))
}

/// Semantic-token sampling exactly matching onnx_runtime/arktts_runtime/
/// runtime.py's _sample_semantic and rust_runtime/src/runtime.rs's
/// sample_semantic: slow_logits_layout is semantic_then_eos (verified in
/// runtime_manifest.json), so the model's raw logits ARE already the
/// compact [semantic_begin..semantic_end, im_end] allowed-token space in
/// order - no indexing by raw token id into a larger vocab array.
#[allow(clippy::too_many_arguments)]
fn sample_semantic(logits: &[f32], previous: &[i64], temperature: f32, top_p: f32, top_k: usize, rng: &mut StdRng) -> i64 {
    let begin = SEMANTIC_BEGIN_ID;
    let end = SEMANTIC_END_ID;
    let stop = IM_END_ID;
    let allowed_ids: Vec<i64> = (begin..=end).chain(std::iter::once(stop)).collect();
    debug_assert_eq!(allowed_ids.len(), logits.len());

    let normal_idx = sample(logits, temperature, top_p, top_k, rng);
    let normal = allowed_ids[normal_idx];
    let high_idx = sample(logits, 1.0, 0.9, top_k, rng);
    let high = allowed_ids[high_idx];

    if normal >= begin && normal <= end && previous.contains(&normal) {
        high
    } else {
        normal
    }
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let use_cuda = args.iter().any(|a| a == "--cuda");
    let text = args
        .iter()
        .skip(1)
        .find(|a| !a.starts_with("--"))
        .cloned()
        .unwrap_or_else(|| "Welcome to Audio8 TTS, running from a native Rust engine built on candle.".to_string());
    let repeats: usize = args
        .iter()
        .position(|a| a == "--repeat")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    let device = if use_cuda { Device::new_cuda(0)? } else { Device::Cpu };
    println!("[synth] device: {:?}", device);

    let load_start = Instant::now();
    let slow = SlowAr::load(&weights_dir().join("slow_ar_q4_0.gguf"), &device)?;
    let fast = FastAr::load(&weights_dir().join("fast_ar_q4_0.gguf"), &device)?;
    let codec_vb = VarBuilder::from_npz(weights_dir().join("codec_decoder.npz"), DType::F32, &device)?;
    let codec = CodecDecoder::new(codec_vb)?;
    let prompt_builder = PromptBuilder::new(&model_dir().join("tokenizer"), SEMANTIC_BEGIN_ID, NUM_CODEBOOKS)?;
    println!("[synth] models loaded in {:?}", load_start.elapsed());

    let (reference_codes, meta) = load_voice("probe_voice")?;
    println!("[synth] loaded voice probe_voice, reference codes shape {:?}", reference_codes.shape());

    let sample_rate = 44100u32;
    let mut audio_samples: Vec<f32> = Vec::new();

    for run in 0..repeats {
        let synth_start = Instant::now();

        let prompt = prompt_builder.build(&text, &meta.reference_text, &reference_codes)?;
        let prompt_len = prompt.shape()[2];
        if prompt_len >= audio8_candle_runtime::model::MAX_SEQ_LEN {
            anyhow::bail!("prompt length {prompt_len} exceeds max sequence length");
        }

        let flat: Vec<i64> = prompt.iter().copied().collect();
        let prompt_tensor = Tensor::from_vec(flat, (1, NUM_CODEBOOKS + 1, prompt_len), &device)?;

        let mut slow_caches = slow.new_kv_caches(1)?;
        let mut fast_caches = fast.new_kv_caches(1)?;

        let positions: Vec<usize> = (0..prompt_len).collect();
        let (mut logits_t, mut fast_hidden) = slow.slow_step(&prompt_tensor, &positions, &mut slow_caches)?;

        let mut rng = StdRng::seed_from_u64(42 + run as u64);
        let mut previous: Vec<i64> = Vec::new();
        let max_new_tokens = 256usize.min(audio8_candle_runtime::model::MAX_SEQ_LEN - prompt_len);
        let mut frames: Vec<Vec<i64>> = Vec::new();

        for step in 0..max_new_tokens {
            let logits: Vec<f32> = logits_t.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;
            let semantic = sample_semantic(&logits, &previous, 0.8, 0.95, 50, &mut rng);
            if semantic == IM_END_ID {
                break;
            }
            previous.push(semantic);
            if previous.len() > 10 {
                previous.remove(0);
            }

            let mut codebooks: Vec<i64> = Vec::with_capacity(NUM_CODEBOOKS);
            let token0 = (semantic - SEMANTIC_BEGIN_ID).clamp(0, 4095);
            codebooks.push(token0);

            let mut hidden = fast_hidden.clone();
            let fast_logits0 = fast.fast_step(&hidden, 0, &mut fast_caches)?;
            let _ = fast_logits0; // position 0 uses fast_hidden directly per fast_project semantics; token0 already fixed from semantic id
            for fast_pos in 1..NUM_CODEBOOKS {
                let token_tensor = fast.embed_token(*codebooks.last().unwrap() as u32)?;
                hidden = token_tensor;
                let fast_logits = fast.fast_step(&hidden, fast_pos, &mut fast_caches)?;
                let fl: Vec<f32> = fast_logits.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;
                let token = sample(&fl, 0.8, 0.95, 50, &mut rng) as i64;
                codebooks.push(token);
            }

            frames.push(codebooks.clone());
            if step + 1 >= max_new_tokens {
                break;
            }

            let mut column = vec![semantic];
            column.extend(&codebooks);
            let column_tensor = Tensor::from_vec(column, (1, NUM_CODEBOOKS + 1, 1), &device)?;
            let position = prompt_len + step;
            let (next_logits, next_hidden) = slow.slow_step(&column_tensor, &[position], &mut slow_caches)?;
            logits_t = next_logits;
            fast_hidden = next_hidden;
        }

        if frames.is_empty() {
            anyhow::bail!("model produced no codec frames");
        }

        let t = frames.len();
        let mut codes_flat = vec![0i64; NUM_CODEBOOKS * t];
        for (col, frame) in frames.iter().enumerate() {
            for row in 0..NUM_CODEBOOKS {
                codes_flat[row * t + col] = frame[row];
            }
        }
        let codes_tensor = Tensor::from_vec(codes_flat, (1, NUM_CODEBOOKS, t), &device)?;

        let audio = codec.forward(&codes_tensor)?.to_dtype(DType::F32)?;
        audio_samples = audio.flatten_all()?.to_vec1()?;

        let audio_duration_s = audio_samples.len() as f64 / sample_rate as f64;
        let total_elapsed = synth_start.elapsed();
        let rtf = total_elapsed.as_secs_f64() / audio_duration_s;
        println!(
            "[synth] run={run} frames={t} audio_duration={:.3}s synthesis_time={:.3}s RTF={rtf:.4}",
            audio_duration_s,
            total_elapsed.as_secs_f64()
        );
    }

    let spec = hound::WavSpec { channels: 1, sample_rate, bits_per_sample: 32, sample_format: hound::SampleFormat::Float };
    let out_path = "candle_probe_output.wav";
    let mut writer = hound::WavWriter::create(out_path, spec)?;
    for &s in &audio_samples {
        writer.write_sample(s)?;
    }
    writer.finalize()?;
    println!("[synth] wrote {out_path}");

    Ok(())
}

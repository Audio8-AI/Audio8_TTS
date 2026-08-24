use std::path::PathBuf;
use std::time::Instant;

use audio8_rust_runtime::runtime::{ArkTtsRuntime, ExecutionProviderChoice};
use ndarray::Array2;

fn model_dir() -> PathBuf {
    PathBuf::from(std::env::var("ARKTTS_MODEL_DIR").unwrap_or_else(|_| "../onnx_runtime/model".to_string()))
}

fn voices_dir() -> PathBuf {
    PathBuf::from(std::env::var("ARKTTS_VOICES_DIR").unwrap_or_else(|_| "../onnx_runtime/voices".to_string()))
}

#[derive(serde::Deserialize)]
struct VoiceMeta {
    reference_text: String,
}

fn load_voice(name: &str) -> anyhow::Result<(Array2<i64>, VoiceMeta)> {
    let dir = voices_dir().join(name);
    let meta: VoiceMeta = serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json"))?)?;
    let codes_u16 = audio8_rust_runtime::npy::read_npy_u16_2d(&dir.join("codes.npy"))?;
    let codes = codes_u16.mapv(|v| v as i64);
    Ok((codes, meta))
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env().add_directive(tracing::Level::WARN.into()))
        .init();

    let args: Vec<String> = std::env::args().collect();
    let use_cuda = args.iter().any(|a| a == "--cuda");
    let text = args
        .iter()
        .skip(1)
        .find(|a| !a.starts_with("--"))
        .cloned()
        .unwrap_or_else(|| "Welcome to Audio8 TTS, running from Rust.".to_string());

    let ep = if use_cuda { ExecutionProviderChoice::Cuda } else { ExecutionProviderChoice::Cpu };
    println!("[synth] loading runtime from {:?} (ep={:?})", model_dir(), ep);
    let load_start = Instant::now();
    let mut runtime = ArkTtsRuntime::load_with_ep(&model_dir(), ep)?;
    println!("[synth] runtime loaded in {:?}", load_start.elapsed());

    let (reference_codes, meta) = load_voice("probe_voice")?;
    println!("[synth] loaded voice probe_voice, reference codes shape {:?}", reference_codes.shape());

    let repeats: usize = args
        .iter()
        .position(|a| a == "--repeat")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);

    let sample_rate = 44100u32;
    let mut audio = Vec::new();
    for run in 0..repeats {
        let synth_start = Instant::now();
        let codes = runtime.synthesize_no_reference(&text, &meta.reference_text, &reference_codes, 256, 0.8, 0.95, 50, 42 + run as u64)?;
        let generation_elapsed = synth_start.elapsed();

        let decode_start = Instant::now();
        audio = runtime.decode_codes(&codes)?;
        let decode_elapsed = decode_start.elapsed();

        let audio_duration_s = audio.len() as f64 / sample_rate as f64;
        let total_elapsed = synth_start.elapsed();
        let rtf = total_elapsed.as_secs_f64() / audio_duration_s;
        println!(
            "[synth] run={run} frames={} gen={:?} decode={:?} audio_duration={:.3}s synthesis_time={:.3}s RTF={:.4}",
            codes.shape()[1],
            generation_elapsed,
            decode_elapsed,
            audio_duration_s,
            total_elapsed.as_secs_f64(),
            rtf
        );
    }

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let out_path = "rust_probe_output.wav";
    let mut writer = hound::WavWriter::create(out_path, spec)?;
    for sample in &audio {
        writer.write_sample(*sample)?;
    }
    writer.finalize()?;
    println!("[synth] wrote {}", out_path);

    Ok(())
}
